"""Isolated pose staging v2. Single-process bounded FIFO; not Orca production."""
from pathlib import Path
import os, sys, json, uuid, threading, subprocess, time, hmac, hashlib, base64
from collections import deque
from flask import Flask, request, jsonify, send_file, g
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from analyze_pose import load_frames
JOBS=Path(os.environ.get('OPENSIM_JOBS_DIR',str(ROOT/'jobs')))
JOBS.mkdir(parents=True,exist_ok=True)
KEY=os.environ.get('OPENSIM_API_KEY','').strip()
ORIGINS={o.strip() for o in os.environ.get('OPENSIM_ALLOWED_ORIGINS','capacitor://localhost,http://localhost,https://localhost,http://localhost:3000,http://127.0.0.1:3000').split(',') if o.strip()}
app=Flask(__name__); app.config['MAX_CONTENT_LENGTH']=2*1024*1024
lock=threading.RLock(); queue=deque(); active=None; started=False

def save(path,data):
    tmp=path.with_suffix('.tmp'); tmp.write_text(json.dumps(data,ensure_ascii=False,allow_nan=False)); tmp.replace(path)

def decode_token(token):
    body,sig=token.split('.')
    expected=hmac.new(KEY.encode(),body.encode(),hashlib.sha256).hexdigest()
    if not hmac.compare_digest(sig,expected): raise ValueError()
    data=json.loads(base64.urlsafe_b64decode(body+'='*(-len(body)%4)))
    if data.get('aud')!='opensim-staging' or not isinstance(data.get('sub'),str) or not data['sub']: raise ValueError()
    now=time.time()
    if not isinstance(data.get('exp'),(int,float)) or not now<data['exp']<=now+86400: raise ValueError()
    return data['sub']

@app.before_request
def guard():
    origin=request.headers.get('Origin')
    if origin and origin not in ORIGINS: return jsonify(error='허용되지 않은 앱 출처입니다.'),403
    if request.method=='OPTIONS': return '',204
    if request.path=='/api/health': return
    if not KEY: return jsonify(error='테스트 인증 설정이 필요합니다.'),503
    try:
        token=request.headers.get('Authorization','')
        if not token.startswith('Bearer '): raise ValueError()
        g.owner=decode_token(token[7:])
    except (ValueError,TypeError,KeyError): return jsonify(error='테스트 코드가 없거나 만료되었습니다.'),401

@app.after_request
def cors(response):
    origin=request.headers.get('Origin')
    if origin in ORIGINS:
        response.headers['Access-Control-Allow-Origin']=origin
        response.headers['Vary']='Origin'
        response.headers['Access-Control-Allow-Headers']='Content-Type, Authorization, X-Request-ID'
        response.headers['Access-Control-Allow-Methods']='GET, POST, DELETE, OPTIONS'
    response.headers['Cache-Control']='no-store'
    return response

@app.errorhandler(413)
def oversized(_): return jsonify(error='관절 데이터가 너무 큽니다.'),413

def ensure_worker():
    global started
    with lock:
        if started: return
        # Jobs use temporary storage: after restart explicitly fail unfinished requests.
        for p in JOBS.glob('*/status.json'):
            try:
                s=json.loads(p.read_text())
                if s.get('state') in ('running','queued'): save(p,dict(state='error',message='테스트 서버가 재시작되었습니다. 좌표 전송을 다시 눌러주세요.'))
            except (OSError,ValueError): pass
        started=True
        threading.Thread(target=worker,daemon=True).start()

def worker():
    global active
    while True:
        with lock:
            key=queue.popleft() if queue else None
            active=key
        if not key: time.sleep(.15); continue
        folder=JOBS/key
        save(folder/'status.json',dict(state='running',progress=1,message='OpenSim 계산 중'))
        try:
            with (folder/'worker.log').open('w') as log:
                p=subprocess.Popen([sys.executable,str(ROOT/'analyze_pose.py'),str(folder)],cwd=ROOT,stdout=log,stderr=log)
                try: p.wait(timeout=600)
                except subprocess.TimeoutExpired:
                    p.kill(); p.wait(); raise RuntimeError('계산 시간이 초과됐습니다.')
                status=json.loads((folder/'status.json').read_text())
                if p.returncode or status.get('state')!='done':
                    if status.get('state')!='error': raise RuntimeError('계산에 실패했습니다.')
        except Exception as e:
            save(folder/'status.json',dict(state='error',message=str(e) if isinstance(e,RuntimeError) else '계산 작업을 완료하지 못했습니다.'))
        finally:
            with lock: active=None

def owned(key):
    try: folder=JOBS/str(uuid.UUID(key)); meta=json.loads((folder/'meta.json').read_text())
    except (ValueError,OSError): return None
    return folder if meta.get('owner')==g.owner else None

@app.get('/api/health')
def health(): return jsonify(app='orca-opensim-pose',apiVersion=2,mode='render_pose',auth='short-lived-test-token',maxFrames=901)

@app.post('/api/analyze-pose')
def submit():
    data=request.get_json(silent=True)
    if not isinstance(data,dict): return jsonify(error='관절 JSON이 필요합니다.'),400
    if not isinstance(data.get('frames'),list) or not 12<=len(data['frames'])<=901: return jsonify(error='12~901개 장면이 필요합니다.'),400
    if any(not isinstance(f,dict) or not isinstance(f.get('usable'),bool) for f in data['frames']): return jsonify(error='모든 장면의 usable 품질 표시가 필요합니다.'),400
    try: load_frames(data)
    except (ValueError,TypeError,KeyError) as e: return jsonify(error=str(e)),400
    try: request_id=str(uuid.UUID(request.headers.get('X-Request-ID','')))
    except ValueError: return jsonify(error='유효한 요청 ID가 필요합니다.'),400
    ensure_worker()
    with lock:
        # Idempotent retries survive client disconnect and token refresh with same subject.
        for mp in JOBS.glob('*/meta.json'):
            try:
                m=json.loads(mp.read_text())
                if m.get('owner')==g.owner and m.get('requestId')==request_id: return jsonify(id=mp.parent.name)
            except (OSError,ValueError): pass
        pending=([active] if active else [])+list(queue)
        for key in pending:
            if owned(key): return jsonify(error='이미 요청한 분석이 있습니다.',id=key),409
        if len(pending)>=10: return jsonify(error='테스트 대기열이 가득 찼습니다. 잠시 후 다시 전송해주세요.'),429
        key=str(uuid.uuid4()); folder=JOBS/key; folder.mkdir()
        save(folder/'options.json',data)
        save(folder/'meta.json',dict(owner=g.owner,requestId=request_id,created=time.time()))
        save(folder/'status.json',dict(state='queued',progress=0,message='분석 대기 중'))
        queue.append(key)
    return jsonify(id=key),202

@app.get('/api/jobs/<key>')
def status(key):
    ensure_worker(); folder=owned(key)
    if not folder: return jsonify(error='분석 기록을 찾을 수 없습니다.'),404
    with lock:
        data=json.loads((folder/'status.json').read_text())
        if key in queue: data['queuePosition']=list(queue).index(key)+1+(1 if active else 0)
        elif key==active and data.get('state')=='queued': data['state']='running'
    if data.get('state')=='done': data['result']=json.loads((folder/'result.json').read_text())
    return jsonify(data)

@app.get('/api/jobs/<key>/files/<name>')
def files(key,name):
    folder=owned(key)
    if not folder or name not in ('measurements.csv','model.osim','motion.mot','markers.trc','result.json') or not (folder/name).is_file(): return '',404
    return send_file(folder/name,as_attachment=True,conditional=True)

if __name__=='__main__': app.run(host='127.0.0.1',port=int(os.environ.get('PORT','8766')),threaded=True)
