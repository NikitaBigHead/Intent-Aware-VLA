"""Verify every exported model on available compute devices, without network access."""
import gc,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from run_video import ROOT,MODELS,load_model,verify
torch.set_num_threads(4)
results=[]
for device in ['cpu']+(['cuda'] if torch.cuda.is_available() else []):
    for name in MODELS:
        model,cfg,dev=load_model(device,name)
        result=verify(model,cfg,dev,device=='cuda');result['model']=name
        results.append(result)
        del model;gc.collect()
        if device=='cuda':torch.cuda.empty_cache()
(ROOT/'validation/verify_all.json').write_text(json.dumps(results,indent=2)+'\n')
print('Verified',len(results),'model/device combinations')
