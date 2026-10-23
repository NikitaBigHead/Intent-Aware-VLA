"""Run the standard extractor on disjoint shards, then merge its checked indexes."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import subprocess,sys,time,json,os
import pandas as pd

start=time.time()
root=Path('/workspace/hri_mvp_experiment/frames')
root.mkdir(parents=True,exist_ok=True)
raw=pd.read_csv('raw_index.csv',keep_default_na=False)
def run(i):
    part=raw.iloc[i::4]; out=root/f'shard_{i:02d}'
    csv=Path(f'reports/raw_shard_{i:02d}.csv');part.to_csv(csv,index=False)
    env=os.environ.copy();env.update(OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENCV_FOR_THREADS_NUM='1')
    with open(f'reports/extract_shard_{i:02d}.log','w') as log:
        subprocess.run([sys.executable,'-u','scripts/extract_frames.py','--raw',str(csv),'--out',str(out),'--views','head','ego','--fps','15','--size','224'],stdout=log,stderr=subprocess.STDOUT,env=env,check=True)
    df=pd.read_csv(out/'index.csv',keep_default_na=False)
    for v in ['head','ego']:
        df[v+'_dir']=f'shard_{i:02d}/'+df[v+'_dir']
        for r in df.to_dict('records'):
            assert len(list((root/r[v+'_dir']).glob('*.jpg')))==r['num_frames']
    print('shard complete',i,len(df),flush=True)
    return df
with ThreadPoolExecutor(max_workers=4) as pool:
    frames=list(pool.map(run,range(4)))
combined=pd.concat(frames).sort_values('episode_id')
assert set(combined.episode_id)==set(raw.episode_id)
assert len(combined)==len(raw)
combined.to_csv(root/'index.csv',index=False)
result={'episodes':len(combined),'frames_per_camera':int(combined.num_frames.sum()),'seconds':time.time()-start,'index':str(root/'index.csv')}
Path('reports/extraction_result.json').write_text(json.dumps(result,indent=2)+'\n')
print(result,flush=True)
