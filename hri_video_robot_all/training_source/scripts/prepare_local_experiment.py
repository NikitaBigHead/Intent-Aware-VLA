"""Fixed episode-stratified train/val and a separate, untouched test dataset."""
from pathlib import Path
import json
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

root=Path(__file__).resolve().parents[1]
records=[]
for source,folder in [('development','annotations_from_mac'),('external_test','test-dataset')]:
    ds=root.parent/'datasets'/folder
    rows=[json.loads(s) for s in (ds/'meta/interaction_metadata.jsonl').read_text().splitlines() if s.strip()]
    for r in rows:
        e=r['episode_index']; ph=r.get('phase_timestamps') or {}
        label={'none':'no_interaction'}.get(r['interaction_class'],r['interaction_class'])
        row=dict(episode_id=f'{source}_{e:06d}',person_id=str(r['person']),label=label,source=source,t0=0.0)
        for target,key in [('t_onset','approach_start'),('t_contact','contact_active_start')]:
            value=ph.get(key)
            row[target]=value if label!='no_interaction' and value and value>0 else ''
        for view,key in [('head','observation.images.webcam'),('ego','observation.images.depth_camera_rgb')]:
            paths=list((ds/'videos').glob(f'chunk-*/{key}/episode_{e:06d}.mp4'))
            if len(paths)!=1: raise RuntimeError(f'{source} {e} {view}: {paths}')
            row[view+'_video']=str(paths[0].resolve())
        records.append(row)
df=pd.DataFrame(records)
dev=df.index[df.source=='development']
tr,va=train_test_split(dev,test_size=.2,random_state=0,stratify=df.loc[dev,'label'])
df['split']='test';df.loc[tr,'split']='train';df.loc[va,'split']='val'
raw=root/'raw_index.csv'
if raw.exists():
    old=pd.read_csv(raw,keep_default_na=False)
    assert old.episode_id.tolist()==df.episode_id.tolist() and old.split.tolist()==df.split.tolist(), 'Existing split differs'
df.to_csv(raw,index=False)
report={}
for name,g in df.groupby('split'):
    report[name]={'episodes':g.episode_id.tolist(),'persons':sorted(g.person_id.unique().tolist()),'class_counts':g.label.value_counts().to_dict()}
(root/'reports/fixed_split.json').write_text(json.dumps(report,indent=2)+'\n')
base=yaml.safe_load((root/'configs/base.yaml').read_text())
artifacts=Path('/workspace/hri_mvp_experiment');artifacts.mkdir(exist_ok=True)
base['data'].update(root=str(artifacts/'frames'),index=str(artifacts/'frames/index.csv'),num_workers=4)
base['split']['mode']='column'
base['experiment'].update(out_dir=str(artifacts/'runs'),name='tinyvit_vtn_f0_s0')
(root/'configs/local_experiment.yaml').write_text(yaml.safe_dump(base,sort_keys=False))
for k,v in report.items():print(k,len(v['episodes']),v['class_counts'],'persons:',v['persons'])
print('Onset/contact 0 means unannotated: exported blank; no episode removed or relabeled.')
