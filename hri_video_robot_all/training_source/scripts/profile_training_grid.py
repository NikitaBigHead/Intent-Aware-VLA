"""Measure real training steps and validation throughput; never infer on test."""
import gc,itertools,json,math,sys,time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from hri_video.config import load_config
from hri_video.data import IGNORE,ClipDataset,collate,load_episodes,split_episodes
from hri_video.model import VideoModel
from hri_video.evaluation import episode_traces
from train import view_dropout,set_seed

cfgpath='configs/local_experiment.yaml'
rows=[]
for enc,tmp in itertools.product(['tinyvit','mobilevit','efficientformer'],['vtn','videolightformer']):
    set_seed(0)
    cfg=load_config(cfgpath,[f'model.encoder.name={enc}',f'model.temporal.name={tmp}'])
    sp=split_episodes(load_episodes(cfg),cfg)
    ds=ClipDataset(sp['train'],cfg,train=True)
    dl=DataLoader(ds,batch_size=cfg.train.batch_size,shuffle=True,drop_last=True,num_workers=cfg.data.num_workers,collate_fn=collate,pin_memory=True)
    model=VideoModel(cfg).cuda().train()
    opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=float(cfg.train.lr),weight_decay=float(cfg.train.weight_decay))
    counts=ds.action_counts(len(cfg.data.classes))
    w=torch.tensor(counts.sum()/np.maximum(counts,1)/len(counts),dtype=torch.float,device='cuda');w=w/w.mean()
    durations=[];torch.cuda.synchronize();prev=time.perf_counter()
    for step,b in enumerate(dl):
        if step>=15:break
        frames={v:x.cuda(non_blocking=True).float().div_(255) for v,x in b['frames'].items()}
        mask=view_dropout(b['view_mask'].cuda(),float(cfg.train.view_dropout))
        ya,yp=b['action'].cuda(),b['phase'].cuda()
        with torch.autocast('cuda',dtype=torch.bfloat16):o=model({'frames':frames,'view_mask':mask})
        la=o['logits']['action'].float()
        loss=F.cross_entropy(la,ya,weight=w,ignore_index=IGNORE,label_smoothing=float(cfg.train.label_smoothing)) if (ya!=IGNORE).any() else la.sum()*0
        if (yp!=IGNORE).any():loss=loss+float(cfg.train.phase_loss_weight)*F.cross_entropy(o['logits']['phase'].float(),yp,ignore_index=IGNORE)
        opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),float(cfg.train.grad_clip));opt.step()
        torch.cuda.synchronize();now=time.perf_counter();durations.append(now-prev);prev=now
    # Evenly spaced validation episodes, time inference only. No metric-based selection.
    subset=[sp['val'][i] for i in np.linspace(0,len(sp['val'])-1,8,dtype=int)]
    start=time.perf_counter();tr=episode_traces(model,subset,cfg,torch.device('cuda'));torch.cuda.synchronize();val_s=time.perf_counter()-start
    step_s=float(np.mean(durations[5:]));val_est=val_s*sum(e.length for e in sp['val'])/sum(e.length for e in subset)
    row=dict(encoder=enc,temporal=tmp,steps_per_epoch=len(dl),train_step_seconds=step_s,train_epoch_seconds=step_s*len(dl),val_epoch_seconds=val_est,epoch_seconds=step_s*len(dl)+val_est)
    rows.append(row);print(json.dumps(row),flush=True)
    Path('reports/grid_profile.json').write_text(json.dumps(rows,indent=2)+'\n')
    del model,opt,dl,ds,tr,b,o,loss,frames;gc.collect();torch.cuda.empty_cache()
