"""Deterministic visual audit samples; never changes source annotations."""
from pathlib import Path
import json
import cv2
import numpy as np
import pandas as pd

root=Path('reports')
out=root/'audit_frames'
out.mkdir(exist_ok=True)
df=pd.read_csv(root/'audit_episodes.csv',keep_default_na=False)
manifest=[]
for label, group in df.groupby('label'):
    for _,r in group.sample(2,random_state=0).iterrows():
        eid=int(r.episode_id)
        if label=='no_interaction':
            last=min(r.head_duration_s-1/r.head_fps,r.ego_duration_s-1/r.ego_fps)
            times=[('start',0.0),('middle',last/2),('end',last)]
        else:
            times=[('before_onset',max(0,r.t_onset-0.5)),('onset',r.t_onset),('contact',r.t_contact)]
        cells=[]
        for view in ['head','ego']:
            cap=cv2.VideoCapture(r[view+'_video'])
            for name,t in times:
                frame_index=int(round(t*r[view+'_fps']))
                cap.set(cv2.CAP_PROP_POS_FRAMES,frame_index)
                ok,frame=cap.read()
                if not ok: raise RuntimeError(f'Cannot decode {eid} {view} {t}')
                path=out/f'{label}_{eid:06d}_{view}_{name}.jpg'
                assert cv2.imwrite(str(path),frame)
                manifest.append(dict(episode_id=eid,label=label,view=view,marker=name,requested_time_s=t,frame_index=frame_index,frame_time_s=frame_index/r[view+'_fps'],path=str(path)))
                h,w=frame.shape[:2]
                thumb=cv2.resize(frame,(480,round(h*480/w)))
                canvas=np.zeros((310,480,3),np.uint8)
                canvas[:thumb.shape[0]]=thumb
                cv2.putText(canvas,f'{view} {name} t={t:.3f}s',(8,299),cv2.FONT_HERSHEY_SIMPLEX,.55,(255,255,255),1)
                cells.append(canvas)
            cap.release()
        sheet=np.vstack([np.hstack(cells[:3]),np.hstack(cells[3:])])
        assert cv2.imwrite(str(out/f'{label}_{eid:06d}_sheet.jpg'),sheet)
(root/'audit_samples.json').write_text(json.dumps(manifest,indent=2)+'\n')
print('Saved',len(manifest),'frames and 8 contact sheets in',out)
