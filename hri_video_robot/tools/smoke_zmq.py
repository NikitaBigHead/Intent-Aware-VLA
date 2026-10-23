"""Loopback-only test of all three camera encodings; no robot connection required."""
import base64,json,os,subprocess,sys,time
from pathlib import Path
import cv2
import msgpack
import msgpack_numpy
import numpy as np
import zmq
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from sources import _decode

ctx=zmq.Context()
results=[]
try:
    for encoding in ['b64','mjpeg','array']:
        rgb=np.zeros((240,320,3),np.uint8);rgb[:]=[210,70,25]
        bgr=cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR)
        if encoding=='b64':value=base64.b64encode(cv2.imencode('.jpg',rgb)[1]).decode()
        elif encoding=='mjpeg':value=cv2.imencode('.jpg',bgr)[1].tobytes()
        else:value=rgb
        assert np.max(np.abs(_decode(value).astype(float)-bgr))<=3, 'Color-channel mismatch'
        sock=ctx.socket(zmq.PUB);port=sock.bind_to_random_port('tcp://127.0.0.1')
        log=ROOT/'validation'/f'zmq_{encoding}.jsonl'
        output=open(ROOT/'validation'/f'zmq_{encoding}.log','w')
        process=subprocess.Popen([sys.executable,'-u',str(ROOT/'run_video.py'),'robot','--host','127.0.0.1','--port',str(port),'--device','cpu','--duration','12','--max-frames','10','--log',str(log)],stdout=output,stderr=subprocess.STDOUT,cwd=ROOT)
        deadline=time.monotonic()+25
        try:
            while process.poll() is None and time.monotonic()<deadline:
                now=time.time()
                sock.send(msgpack.packb({'timestamps':{'head':now,'ego_view':now},'images':{'head':value,'ego_view':value}},use_bin_type=True,default=msgpack_numpy.encode))
                time.sleep(.05)
            process.wait(timeout=2)
            assert process.returncode==0, f'{encoding}: see {output.name}'
            records=[json.loads(s) for s in log.read_text().splitlines()]
            predictions=[r for r in records if r['type']=='prediction']
            assert len(predictions)==10
            assert all(r['has_capture_timestamps'] and r['view_mask']==[True,True] for r in predictions)
            assert all(abs(sum(r['probabilities'].values())-1)<1e-5 for r in predictions)
            results.append({'encoding':encoding,'predictions':len(predictions),'ok':True})
        finally:
            if process.poll() is None:process.terminate();process.wait(timeout=5)
            output.close();sock.close(0)
finally:ctx.term()
(ROOT/'validation/zmq_smoke.json').write_text(json.dumps(results,indent=2)+'\n')
print(json.dumps(results,indent=2))
