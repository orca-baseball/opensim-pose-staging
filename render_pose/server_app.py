"""Pose-only OpenSim API for a separate Render staging service.
Does not replace Orca production analysis. Local Lab app.py is unchanged."""
from pathlib import Path
import os, sys, json, uuid, threading, subprocess, math
from flask import Flask, request, jsonify, send_file

ROOT=Path(__file__).resolve().parents[1]
os.chdir(ROOT)
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.matplotlib'))
JOBS=Path(os.environ.get('OPENSIM_JOBS_DIR') or (ROOT/'jobs'))
JOBS.mkdir(parents=True,exist_ok=True)
API_KEY=os.environ.get('OPENSIM_API_KEY','').strip()
app=Flask(__name__)
app.config['MAX_CONTENT_LENGTH']=32*1024*1024
lock=threading.Lock(); active={}

def job_dir(key):
    try: return JOBS/str(uuid.UUID(key))
    except ValueError: return None

@app.before_request
def guard():
    if request.path=='/api/health': return None
    if API_KEY:
        if request.headers.get('X-OpenSim-Key')!=API_KEY:
            return jsonify(error='인증이 필요합니다.'),401

@app.get('/api/health')
def health():
    return jsonify(app='orca-opensim-pose',version=1,mode='render_pose',opensim='4.6')

def monitor(key,process,log):
    try:
        process.wait(timeout=600)
        folder=JOBS/key
        path=folder/'status.json'
        if process.returncode and (not path.exists() or json.loads(path.read_text()).get('state')=='running'):
            path.write_text(json.dumps({'state':'error','message':'분석 엔진이 종료됐습니다.'},ensure_ascii=False))
    except subprocess.TimeoutExpired:
        process.kill(); process.wait()
        (JOBS/key/'status.json').write_text(json.dumps({'state':'error','message':'분석 시간이 10분을 초과했습니다.'},ensure_ascii=False))
    finally:
        log.close()
        with lock: active.clear()

def start_job(options,message):
    with lock:
        if active: return None, (jsonify(error='다른 분석을 처리 중입니다. 완료 후 다시 시도해주세요.'),409)
        key=str(uuid.uuid4()); folder=JOBS/key; folder.mkdir()
        (folder/'options.json').write_text(json.dumps(options,ensure_ascii=False))
        (folder/'status.json').write_text(json.dumps(dict(state='running',progress=1,message=message),ensure_ascii=False))
        log=(folder/'worker.log').open('w')
        try: process=subprocess.Popen([sys.executable,str(ROOT/'analyze_pose.py'),str(folder)],cwd=ROOT,stdout=log,stderr=log)
        except Exception:
            log.close(); return None, (jsonify(error='분석 엔진을 실행하지 못했습니다.'),500)
        active.update(key=key,process=process)
        threading.Thread(target=monitor,args=(key,process,log),daemon=True).start()
        return key, None

@app.post('/api/analyze-pose')
def analyze_pose_api():
    data=request.get_json(silent=True)
    if not isinstance(data,dict): return jsonify(error='JSON body가 필요합니다.'),400
    frames=data.get('frames')
    if not isinstance(frames,list) or len(frames)<12:
        return jsonify(error='frames는 최소 12개 장면이 필요합니다.'),400
    space=data.get('coordinateSpace','mediapipe_world')
    if space not in ('mediapipe_world','lab'):
        return jsonify(error='coordinateSpace는 mediapipe_world 또는 lab 이어야 합니다.'),400
    factor=data.get('factor',None)
    try:
        if factor is not None:
            if isinstance(factor,bool) or not isinstance(factor,(int,float)) or not math.isfinite(factor) or not 1<=float(factor)<=32:
                raise ValueError('슬로모션 배율을 확인해주세요.')
            factor=float(factor)
        joints=('leftHip','rightHip','leftShoulder','rightShoulder'); prev=None
        for i,frame in enumerate(frames):
            if not isinstance(frame,dict): raise ValueError(f'frames[{i}] 형식이 올바르지 않습니다.')
            t=frame.get('t')
            if isinstance(t,bool) or not isinstance(t,(int,float)) or not math.isfinite(t) or t<0:
                raise ValueError(f'frames[{i}].t 시간을 확인해주세요.')
            t=float(t)
            if prev is not None and t<=prev: raise ValueError('frames의 t는 증가하는 초 단위여야 합니다.')
            prev=t
            for name in joints:
                xyz=frame.get(name)
                if not isinstance(xyz,(list,tuple)) or len(xyz)!=3:
                    raise ValueError(f'frames[{i}].{name}는 [x,y,z] 배열이어야 합니다.')
                if any(isinstance(n,bool) or not isinstance(n,(int,float)) or not math.isfinite(n) for n in xyz):
                    raise ValueError(f'frames[{i}].{name} 좌표가 올바르지 않습니다.')
            if 'usable' in frame and not isinstance(frame['usable'],bool):
                raise ValueError(f'frames[{i}].usable는 true/false여야 합니다.')
        span=float(frames[-1]['t'])-float(frames[0]['t'])
        if span<=0 or span>30.1: raise ValueError('분석 구간은 0초보다 길고 최대 30초로 보내주세요.')
    except ValueError as e:
        return jsonify(error=str(e)),400
    key,err=start_job(dict(coordinateSpace=space,factor=factor,frames=frames),'관절 좌표 준비 중')
    if err: return err
    return jsonify(id=key)

@app.get('/api/jobs/<key>')
def status(key):
    folder=job_dir(key)
    if folder is None or not (folder/'status.json').exists(): return jsonify(error='분석 기록이 없습니다.'),404
    data=json.loads((folder/'status.json').read_text())
    if data.get('state')=='done': data['result']=json.loads((folder/'result.json').read_text())
    return jsonify(data)

@app.get('/api/jobs/<key>/files/<name>')
def download(key,name):
    folder=job_dir(key)
    allowed=('measurements.csv','model.osim','motion.mot','markers.trc','result.json')
    if folder is None or name not in allowed or not (folder/name).is_file(): return '',404
    return send_file(folder/name,as_attachment=True,conditional=True)

if __name__=='__main__':
    port=int(os.environ.get('PORT','10000'))
    app.run(host='0.0.0.0',port=port,threaded=True,debug=False)
