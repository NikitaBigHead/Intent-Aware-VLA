"""Read-only audit of local LeRobot annotations and the two RGB cameras."""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
import pandas as pd

p = argparse.ArgumentParser()
p.add_argument('--dataset', type=Path, required=True)
p.add_argument('--out', type=Path, required=True)
a = p.parse_args()
a.out.mkdir(parents=True, exist_ok=True)
rows = [json.loads(s) for s in (a.dataset/'meta/interaction_metadata.jsonl').read_text().splitlines() if s.strip()]
cameras = {'head': 'observation.images.webcam', 'ego': 'observation.images.depth_camera_rgb'}
records = []
for r in rows:
    e = r['episode_index']
    ph = r.get('phase_timestamps') or {}
    d = dict(episode_id=e, person_id=r.get('person'), label={'none':'no_interaction'}.get(r['interaction_class'], r['interaction_class']), t_onset=ph.get('approach_start'), t_contact=ph.get('contact_active_start'))
    problems = []
    for v, key in cameras.items():
        paths = list((a.dataset/'videos').glob(f'chunk-*/{key}/episode_{e:06d}.mp4'))
        d[v+'_video'] = str(paths[0].resolve()) if len(paths)==1 else ''
        if len(paths)!=1:
            problems.append(v+': missing/ambiguous video'); continue
        cap = cv2.VideoCapture(str(paths[0]))
        fps = cap.get(cv2.CAP_PROP_FPS)
        n = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        d[v+'_fps'], d[v+'_frames'] = fps, n
        d[v+'_duration_s'] = n/fps if fps else np.nan
        cap.release()
        if not fps or not n: problems.append(v+': unreadable video')
        for t in ('t_onset','t_contact'):
            if d[t] is not None and (d[t]<0 or (fps and d[t]>(n-1)/fps)): problems.append(v+': '+t+' outside video')
    if d['label']!='no_interaction':
        if not d['t_onset'] or not d['t_contact']: problems.append('missing time (zero means unannotated in annotation UI)')
        if d['t_onset'] is not None and d['t_contact'] is not None and d['t_onset']>=d['t_contact']: problems.append('onset >= contact')
    d['approach_s'] = d['t_contact']-d['t_onset'] if d['t_contact'] and d['t_onset'] else np.nan
    d['issues'] = '; '.join(problems)
    records.append(d)
df=pd.DataFrame(records)
df.to_csv(a.out/'audit_episodes.csv',index=False)
table=pd.crosstab(df.person_id,df.label)
table.to_csv(a.out/'person_class.csv')
def md(t):
    return t.to_markdown()
lines=['# Preliminary data audit', '', f'Source: `{a.dataset.resolve()}`.', '', 'The source was supplied by the dataset owner. Values person=none/outputs do not represent confirmed participant identities.', '', f'Episodes: {len(df)}; unique person values: {df.person_id.nunique()} (not a confirmed participant count).', '', '## Participant by class', '', md(table), '', '## Durations and contact minus onset, seconds', '', md(df[[c for c in df if c.endswith("duration_s")]+['approach_s']].describe(percentiles=[.05,.25,.5,.75,.95])), '', '## Problematic episodes', '', md(df.loc[df.issues!='',['episode_id','person_id','label','t_onset','t_contact','issues']]), '', '## Timing and synchronization', '', 'annotate_phases.py stores video position in seconds; zero means an unset annotation. The annotation source is external_view_camera. The side camera is not passed to the model. Alignment of these times with head/ego has not been confirmed.', '', 'The inspected parquet for episode 0 uses a shared 0 to 6.6 s timestamp scale with a 0.02 s step. Robot telemetry timestamps are not camera timestamps. Per-camera capture timestamps were not found, so equal FPS does not prove physical synchronization.', '', '## Ego minus head duration, seconds', '', str((df.ego_duration_s-df.head_duration_s).describe()), '', '## no_interaction by participant', '', md(table[['no_interaction']]), '', 'Source data was not modified. No split was created. Per-episode details: audit_episodes.csv.']
(a.out/'data_audit.md').write_text('\n'.join(lines)+'\n')
print('episodes',len(df),'issues',int((df.issues!='').sum()))
print(df.loc[df.issues!='',['episode_id','label','issues']].to_string(index=False))
