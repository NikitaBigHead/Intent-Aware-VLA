#!/usr/bin/env python3
"""Local RGB inference: read-only robot camera subscription or recorded video replay."""
import argparse
import json
import os
from pathlib import Path
import time

# The exported checkpoint includes the encoder; inference never needs HF downloads.
os.environ.setdefault('HF_HUB_OFFLINE', '1')
import cv2
import numpy as np
import torch
from hri_video.config import load_config, resolve_temporal_cfg
from hri_video.data import clip_indices
from hri_video.model import VideoModel
from hri_video.streaming import StreamingPredictor
from sources import RobotZmqSource

ROOT = Path(__file__).resolve().parent
MODELS = sorted(json.loads((ROOT / 'MODELS.json').read_text()))


def load_model(device_name='auto', model_name='tinyvit_videolightformer'):
    device = torch.device(('cuda' if torch.cuda.is_available() else 'cpu') if device_name == 'auto' else device_name)
    model_dir = ROOT / 'models' / model_name
    cfg = load_config(model_dir / 'config.yaml')
    cfg.model.encoder.pretrained = False
    temporal = resolve_temporal_cfg(cfg)
    if not temporal['causal'] or temporal['pool'] != 'last':
        raise ValueError('Expected causal=true, pool=last')
    model = VideoModel(cfg).to(device).eval()
    state = torch.load(model_dir / 'model.pt', map_location='cpu', weights_only=True)
    model.load_state_dict(state, strict=True)
    return model, cfg, device


def replay_pairs(head, ego, target_fps):
    caps = [cv2.VideoCapture(str(p)) for p in (head, ego)]
    try:
        if not all(c.isOpened() for c in caps):
            raise RuntimeError('Cannot open both input videos')
        fps = [c.get(cv2.CAP_PROP_FPS) for c in caps]
        counts = [int(c.get(cv2.CAP_PROP_FRAME_COUNT)) for c in caps]
        if min(fps) <= 0 or min(counts) <= 0:
            raise RuntimeError('Missing video fps/frame count')
        last = min((n-1)/f for n,f in zip(counts,fps))
        indices = [-1,-1]
        frames = [None,None]
        for k in range(int(np.floor(last*target_fps))+1):
            t = k/target_fps
            for v,c in enumerate(caps):
                # Same past-only resampling as the training extractor.
                want = int(np.searchsorted(np.arange(counts[v])/fps[v], t, side='right')-1)
                while indices[v] < want:
                    ok,frames[v] = c.read()
                    if not ok:
                        raise RuntimeError(f'Truncated camera {v} at frame {indices[v]+1}')
                    indices[v] += 1
            yield dict(person=frames[0],hand=frames[1],t_person=t,t_hand=t,frame_index=k,has_timestamps=True)
    finally:
        for c in caps:c.release()


@torch.no_grad()
def verify(model, cfg, device, amp):
    rng = np.random.default_rng(42)
    n = 20
    frames = {v:rng.integers(0,256,(n,224,224,3),dtype=np.uint8) for v in model.views}
    sp = StreamingPredictor(model,cfg,device,amp=amp)
    live = np.stack([sp.step({v:frames[v][i] for v in model.views},i/cfg.data.fps)['probs'] for i in range(n)])
    with torch.autocast(device.type,dtype=torch.float16,enabled=amp and device.type=='cuda'):
        tokens = {v:model.encoder_for(v)(torch.from_numpy(frames[v]).permute(0,3,1,2).to(device).float()/255).float() for v in model.views}
        offline=[]
        for i in range(n):
            idx=clip_indices(i,cfg.data.num_frames,cfg.data.frame_stride)
            out=model.head_from_tokens({v:tokens[v][idx][None] for v in model.views},torch.ones(1,2,device=device,dtype=torch.bool))
            offline.append(out['logits']['action'].float().softmax(-1)[0].cpu().numpy())
    err=float(np.max(np.abs(live-np.stack(offline))))
    tolerance=.005 if amp and device.type=='cuda' else .0001
    if not np.isfinite(live).all() or err>tolerance:
        raise AssertionError(f'live/offline mismatch: {err} > {tolerance}')
    report=dict(ok=True,device=str(device),amp=amp and device.type=='cuda',max_live_offline_error=err,tolerance=tolerance,frames=n,classes=list(cfg.data.classes),torch=torch.__version__)
    print(json.dumps(report,indent=2))
    return report


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('mode',choices=['robot','replay','verify'])
    ap.add_argument('--model',choices=MODELS,default='tinyvit_videolightformer')
    ap.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
    ap.add_argument('--no-amp',action='store_true')
    ap.add_argument('--threads',type=int,default=4)
    ap.add_argument('--host');ap.add_argument('--port',type=int,default=5555)
    ap.add_argument('--head-key',default='head');ap.add_argument('--ego-key',default='ego_view')
    ap.add_argument('--head');ap.add_argument('--ego')
    ap.add_argument('--show',action='store_true')
    ap.add_argument('--log',type=Path)
    ap.add_argument('--duration',type=float,default=0,help='Stop after this many wall seconds; 0 = unlimited')
    ap.add_argument('--max-frames',type=int,default=0)
    ap.add_argument('--timeout',type=float,default=15)
    ap.add_argument('--max-skew',type=float,default=.1,help='Skip pairs exceeding this timestamp skew, seconds')
    ap.add_argument('--reset-every',type=float,default=0,help='Optional independent trial length in seconds; 0 = manual reset only')
    a=ap.parse_args()
    if a.mode=='robot' and not a.host:ap.error('robot requires --host')
    if a.mode=='replay' and (not a.head or not a.ego):ap.error('replay requires --head and --ego')
    if a.max_skew<0 or a.timeout<=0 or a.threads<1:ap.error('Invalid skew/timeout/threads')
    torch.set_num_threads(a.threads)
    cv2.setNumThreads(1)
    model,cfg,device=load_model(a.device,a.model)
    if a.mode=='verify':
        verify(model,cfg,device,not a.no_amp);return
    logpath=a.log or ROOT/'logs'/f'{a.mode}_{time.strftime("%Y%m%d_%H%M%S")}.jsonl'
    logpath.parent.mkdir(parents=True,exist_ok=True)
    sp=StreamingPredictor(model,cfg,device,amp=not a.no_amp)
    source=None
    frames_seen=0;epoch=0;last_t=None;trial_start=None;previous_id=None
    started=last_received=time.monotonic()
    try:
        with logpath.open('w',buffering=1) as log:
            def write(x):log.write(json.dumps(x,ensure_ascii=False,allow_nan=False)+'\n')
            write(dict(type='session',mode=a.mode,model=a.model,device=str(device),classes=list(cfg.data.classes),threshold=sp.thr,consecutive=sp.k,fps=sp.fps,T=sp.T,head_key=a.head_key,ego_key=a.ego_key,max_skew=a.max_skew,reset_every=a.reset_every))
            if a.mode=='robot':source=RobotZmqSource(a.host,a.port,a.head_key,a.ego_key)
            iterator=iter(replay_pairs(a.head,a.ego,sp.fps)) if a.mode=='replay' else None
            print(f'Ready: {device}, {sp.fps:g} FPS, log={logpath}',flush=True)
            while True:
                if a.duration and time.monotonic()-started>=a.duration:break
                if a.mode=='robot':
                    if getattr(source,'error',None):raise RuntimeError(source.error)
                    pair=source.get()
                    if pair is None or pair['frame_index']==previous_id or time.monotonic()-pair['received']>.5:
                        if time.monotonic()-last_received>a.timeout:
                            raise TimeoutError(f'No fresh camera pairs; keys seen: {sorted(source.keys_seen)}')
                        time.sleep(.005);continue
                    previous_id=pair['frame_index'];last_received=time.monotonic()
                else:
                    try:pair=next(iterator)
                    except StopIteration:break
                tp,th=float(pair['t_person']),float(pair['t_hand'])
                if not np.isfinite([tp,th]).all():raise ValueError('Nonfinite camera timestamps')
                skew=abs(tp-th)
                if skew>a.max_skew:
                    write(dict(type='skipped_desync',skew_s=skew));continue
                t=max(tp,th)  # Neither input camera frame lies in the future of t.
                reset=(last_t is not None and (t<=last_t or t-last_t>.5)) or (a.reset_every and trial_start is not None and t-trial_start>=a.reset_every)
                if reset:
                    sp.reset();trial_start=t;epoch+=1
                    write(dict(type='reset',trial=epoch,t=t,reason='timestamp discontinuity or configured interval'))
                if trial_start is None:trial_start=t
                last_t=t
                rgb={'head':cv2.cvtColor(pair['person'],cv2.COLOR_BGR2RGB),'ego':cv2.cvtColor(pair['hand'],cv2.COLOR_BGR2RGB)}
                tic=time.perf_counter();out=sp.step(rgb,t)
                if out is None:continue
                ms=(time.perf_counter()-tic)*1000
                frames_seen+=1
                committed=out['committed']
                record=dict(type='prediction',trial=epoch,t=t,elapsed_trial_s=t-trial_start,label=out['label'],probabilities=dict(zip(sp.classes,map(float,out['probs']))),first_commit=list(committed) if committed else None,view_mask=out['view_mask'],skew_s=skew,has_capture_timestamps=bool(pair.get('has_timestamps')),processing_ms=ms)
                write(record)
                if frames_seen==1 or frames_seen%15==0:
                    print(f"trial={epoch} t={t-trial_start:.2f}s label={out['label']} p={max(out['probs']):.3f} first_commit={committed[0] if committed else '-'} processing={ms:.1f}ms",flush=True)
                if a.show:
                    thumbs=[cv2.resize(pair[k],(640,360)) for k in ('person','hand')]
                    canvas=np.hstack(thumbs)
                    cv2.putText(canvas,f"{out['label']} {max(out['probs']):.2f} | first: {committed[0] if committed else '-'} | R reset, Q quit",(15,30),cv2.FONT_HERSHEY_SIMPLEX,.65,(0,255,0),2)
                    cv2.imshow('HRI RGB: head | ego',canvas)
                    key=cv2.waitKey(1)&255
                    if key in (27,ord('q')):break
                    if key==ord('r'):
                        sp.reset();trial_start=None;last_t=None;epoch+=1
                        write(dict(type='reset',trial=epoch,reason='keyboard'))
                if a.max_frames and frames_seen>=a.max_frames:break
            write(dict(type='summary',predictions=frames_seen,wall_seconds=time.monotonic()-started))
    finally:
        if source:source.close()
        if a.show:cv2.destroyAllWindows()
    if not frames_seen:raise RuntimeError('No predictions produced')
    print(f'Saved {frames_seen} predictions to {logpath}',flush=True)


if __name__=='__main__':
    try:main()
    except KeyboardInterrupt:print('Stopped by user')
