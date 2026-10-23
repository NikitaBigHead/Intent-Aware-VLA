"""Supervised sanity -> measured pilot -> common-budget grid. Stops on failure."""
import json,os,re,subprocess,sys,time,traceback
from pathlib import Path
import pandas as pd
import yaml

ROOT=Path(__file__).resolve().parents[1]
os.chdir(ROOT)
REPORT=ROOT/'reports'
STATUS=REPORT/'training_status.json'
START=time.time()
PY=sys.executable
ENV=os.environ.copy()
for k in list(ENV):
    if k.startswith(('ROS_','AMENT_','COLCON_')) or k in ('PYTHONPATH','CMAKE_PREFIX_PATH','LD_LIBRARY_PATH'):
        ENV.pop(k,None)
ENV.update(OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENCV_FOR_THREADS_NUM='1',PYTEST_DISABLE_PLUGIN_AUTOLOAD='1',PYTHONUNBUFFERED='1')
ENV['PATH']=str(Path(PY).parent)+':/usr/local/bin:/usr/bin:/bin'

def status(stage,**kwargs):
    record=dict(stage=stage,pid=os.getpid(),elapsed_seconds=time.time()-START,**kwargs)
    tmp=STATUS.with_suffix('.tmp');tmp.write_text(json.dumps(record,indent=2)+'\n');tmp.replace(STATUS)
    print(json.dumps(record),flush=True)

def run(stage,args,log):
    status(stage,log=str(log))
    with open(log,'w') as f:
        subprocess.run(args,env=ENV,stdout=f,stderr=subprocess.STDOUT,check=True)

try:
    # No training until both required sanity checks pass.
    for enc,tmp in [('tinyvit','vtn'),('efficientformer','videolightformer')]:
        log=REPORT/f'sanity_{enc}_{tmp}.log'
        run(f'sanity_{enc}_{tmp}',[PY,'-u','scripts/sanity_overfit.py','--config','configs/local_experiment.yaml','--set',f'model.encoder.name={enc}',f'model.temporal.name={tmp}'],log)
        values=re.findall(r'train-acc ([0-9.]+)',log.read_text())
        if not values or float(values[-1])<.98:
            raise RuntimeError(f'Sanity check failed: {enc}/{tmp}, final accuracy {values[-1:]}; STOP')
    run('pilot_epoch',[PY,'-u','scripts/train.py','--config','configs/local_experiment.yaml','--skip-test','--set','experiment.name=timing_pilot_tinyvit_vtn','train.epochs=1'],REPORT/'pilot_epoch.log')
    run('profile_grid',[PY,'-u','scripts/profile_training_grid.py'],REPORT/'profile_grid.log')
    cfg=yaml.safe_load(Path('configs/local_experiment.yaml').read_text())
    pilot=pd.read_csv(Path(cfg['experiment']['out_dir'])/'timing_pilot_tinyvit_vtn/log.csv').iloc[-1]
    prof=json.loads((REPORT/'grid_profile.json').read_text())
    # Correct short-run throughput using one entire train+val epoch, and reserve 25%.
    train_factor=max(1.,float(pilot.train_seconds)/prof[0]['train_epoch_seconds'])
    val_factor=max(1.,float(pilot.val_seconds)/prof[0]['val_epoch_seconds'])
    for r in prof:
        r['adjusted_epoch_seconds']=r['train_epoch_seconds']*train_factor+r['val_epoch_seconds']*val_factor
    per_grid_epoch=sum(r['adjusted_epoch_seconds'] for r in prof)
    # Reserve for final val/test, bootstraps and process startup; no performance tuning on test.
    final_overhead=sum(r['val_epoch_seconds']*val_factor for r in prof)*4+600
    remaining=8*3600-(time.time()-START)-final_overhead
    epochs=min(30,int(remaining/(1.25*per_grid_epoch)))
    if epochs<1:raise RuntimeError('A full grid cannot fit the 8-hour budget; no partial grid launched')
    cfg['train']['epochs']=epochs
    Path('configs/local_experiment.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False))
    plan=dict(epochs=epochs,train_window_step=cfg['data']['train_window_step'],pilot_epoch_seconds=float(pilot.epoch_seconds),models=prof,estimated_grid_seconds=epochs*per_grid_epoch+final_overhead,conservative_grid_seconds=1.25*epochs*per_grid_epoch+final_overhead,estimated_tinyvit_seconds=epochs*prof[0]['adjusted_epoch_seconds'],created_unix=time.time())
    (REPORT/'training_plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    with open(REPORT/'REPORT.md','a') as f:
        f.write(f'\n## Measured training plan\n\nPilot tinyvit+vtn epoch: {pilot.epoch_seconds:.1f} s; the test set was not evaluated. The same epoch count is used for the full grid: {epochs} (initially 30); train_window_step={cfg["data"]["train_window_step"]}. The epoch count is based only on runtime and an eight-hour budget. Estimated grid time: {plan["estimated_grid_seconds"]/3600:.2f} h; conservative estimate with a 25% epoch margin: {plan["conservative_grid_seconds"]/3600:.2f} h. Details: training_plan.json.\n')
    ENV.update(CONFIG='configs/local_experiment.yaml',OUT_DIR=cfg['experiment']['out_dir'])
    run('training_grid',['bash','scripts/run_grid.sh','0','0'],REPORT/'training_grid.log')
    status('training_grid_complete',plan=plan,results_dir=cfg['experiment']['out_dir'])
except Exception as e:
    status('stopped',error=str(e))
    traceback.print_exc()
    sys.exit(1)
