"""Pose JSON → OpenSim 2-axis IK. No MediaPipe, no video."""
from pathlib import Path
import os, sys, json, math, traceback, csv
ROOT=Path(__file__).resolve().parent
os.chdir(ROOT)
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.matplotlib'))
import numpy as np
from scipy.signal import savgol_filter
from kinematics import fit

JOINTS=('leftHip','rightHip','leftShoulder','rightShoulder')
# MediaPipe Pose world landmark order used by Lab video analyze: 23,24,11,12
# mediapipe_world → Lab: [x, -y, -z] (same as analyze.py)

def write_json(path,data):
    temporary=path.with_suffix('.tmp'); temporary.write_text(json.dumps(data,ensure_ascii=False,allow_nan=False)); temporary.replace(path)

def progress(job,pct,message,**extra):
    write_json(job/'status.json',dict(state='running',progress=pct,message=message,**extra))

def longest_run(mask):
    best=[]; run=[]
    for i,ok in enumerate(mask):
        if ok:
            run.append(i)
            if len(run)>len(best): best=run.copy()
        else: run=[]
    return np.array(best,dtype=int)

def to_lab(xyz, space):
    x,y,z=xyz
    if space=='mediapipe_world': return [float(x),-float(y),-float(z)]
    if space=='lab': return [float(x),float(y),float(z)]
    raise ValueError('coordinateSpace는 mediapipe_world 또는 lab 이어야 합니다.')

def parse_xyz(value,label):
    if not isinstance(value,(list,tuple)) or len(value)!=3:
        raise ValueError(f'{label}는 [x,y,z] 숫자 배열이어야 합니다.')
    out=[]
    for n in value:
        if isinstance(n,bool) or not isinstance(n,(int,float)) or not math.isfinite(n):
            raise ValueError(f'{label} 좌표가 올바르지 않습니다.')
        out.append(float(n))
    return out

def load_frames(options):
    space=options.get('coordinateSpace','mediapipe_world')
    if space not in ('mediapipe_world','lab'):
        raise ValueError('coordinateSpace는 mediapipe_world 또는 lab 이어야 합니다.')
    frames=options.get('frames')
    if not isinstance(frames,list) or len(frames)<12:
        raise ValueError('frames는 최소 12개 장면이 필요합니다.')
    times=[]; rows=[]; reliable=[]
    prev=None
    for i,frame in enumerate(frames):
        if not isinstance(frame,dict): raise ValueError(f'frames[{i}] 형식이 올바르지 않습니다.')
        t=frame.get('t')
        if isinstance(t,bool) or not isinstance(t,(int,float)) or not math.isfinite(t) or t<0:
            raise ValueError(f'frames[{i}].t 시간을 확인해주세요.')
        t=float(t)
        if prev is not None and t<=prev:
            raise ValueError('frames의 t는 증가하는 초 단위여야 합니다.')
        prev=t
        coords=[]
        finite=True
        for name in JOINTS:
            if name not in frame: raise ValueError(f'frames[{i}]에 {name}가 필요합니다.')
            xyz=parse_xyz(frame[name],f'frames[{i}].{name}')
            lab=to_lab(xyz,space)
            if not all(math.isfinite(v) for v in lab): finite=False
            coords.append(lab)
        usable=frame.get('usable')
        if usable is None: usable=finite
        elif not isinstance(usable,bool): raise ValueError(f'frames[{i}].usable는 true/false여야 합니다.')
        times.append(t); rows.append(coords); reliable.append(bool(usable and finite))
    span=times[-1]-times[0]
    if span<=0 or span>30.1:
        raise ValueError('분석 구간은 0초보다 길고 최대 30초로 보내주세요.')
    factor=options.get('factor')
    if factor is not None:
        if isinstance(factor,bool) or not isinstance(factor,(int,float)) or not math.isfinite(factor) or not 1<=float(factor)<=32:
            raise ValueError('슬로모션 배율을 확인해주세요.')
        factor=float(factor)
    return np.array(times),np.array(rows,dtype=float),np.array(reliable,dtype=bool),space,factor

def analyze_pose(job):
    options=json.loads((job/'options.json').read_text())
    progress(job,10,'관절 좌표를 확인하고 있습니다.')
    times,points,reliable,space,factor=load_frames(options)
    good=reliable.copy()
    centered=points-points[:,:2].mean(axis=1)[:,None,:]
    for i in range(1,len(good)):
        if reliable[i] and reliable[i-1] and np.max(np.linalg.norm(centered[i]-centered[i-1],axis=1))>.22:
            good[i]=False
    run=longest_run(good)
    if len(run)<12 or times[run[-1]]-times[run[0]]<.35:
        raise ValueError('어깨와 골반을 연속해서 확인할 수 없습니다. 유효 장면이 더 길게 이어진 좌표를 보내주세요.')
    t=times[run]; p=centered[run]
    if len(t)>=7: p=savgol_filter(p,5,2,axis=0)
    progress(job,55,'OpenSim으로 골반·몸통 회전 모델을 맞추고 있습니다.')
    fitted=fit(t,p,job)
    ts=fitted['times']; pelvis=fitted['pelvis']; trunk=fitted['trunk']; rel=fitted['relative']
    if not all(np.isfinite(a).all() for a in [pelvis,trunk,rel]):
        raise ValueError('모델 계산이 수렴하지 않았습니다. 다른 좌표로 시도해주세요.')
    error_ratio=float(np.median(fitted['residual'])/fitted['scale'])
    if error_ratio>.30:
        raise ValueError('상체 기울기 또는 좌표 흔들림이 커서 이번 회전 모델과 맞지 않습니다. 숫자를 표시하지 않았습니다.')
    pelvis=pelvis-pelvis[0]; trunk=trunk-trunk[0]; rel=rel-rel[0]
    speed_p=np.gradient(pelvis,ts); speed_t=np.gradient(trunk,ts)
    if len(ts)>=7:
        speed_p=savgol_filter(speed_p,7,2); speed_t=savgol_filter(speed_t,7,2)
    interior=slice(2,-2); ip=int(np.argmax(abs(speed_p[interior])))+2; it=int(np.argmax(abs(speed_t[interior])))+2
    delta=float(ts[it]-ts[ip]); peak_ok=np.ptp(pelvis)>10 and np.ptp(trunk)>10 and error_ratio<.15
    warnings=[
        '이 모델은 몸을 세운 상태의 골반·몸통 회전 두 가지만 맞추는 실험 모델입니다. 팔·다리, 근육의 힘, 관절 하중은 계산하지 않습니다.',
        '각도는 첫 유효 장면을 0°로 둔 변화량입니다. 카메라 기준의 추정 3D 좌표를 사용하며 해부학적 절대 각도가 아닙니다.',
        '입력은 관절 JSON입니다. MediaPipe world 원본은 coordinateSpace=mediapipe_world(기본)로 보내면 Lab과 동일하게 [x,-y,-z]로 변환합니다. 이미 Lab 축이면 lab을 지정하세요.',
    ]
    coverage=float(len(run)/len(times))
    if coverage<.9: warnings.append('가림 또는 추적 불안정 때문에 일부 장면을 제외했습니다. 아래 분석 구간이 스윙을 포함하는지 확인해주세요.')
    if factor is None: warnings.append('슬로모션 배율을 지정하지 않아 실제 회전 속도는 표시하지 않습니다. 시간은 입력 t 기준입니다.')
    if error_ratio>.15: warnings.append('모델과 관절 좌표의 차이가 큽니다. 회전 곡선을 참고용으로 확인해주세요.')
    progress(job,90,'결과 파일을 만들고 있습니다.')
    with (job/'measurements.csv').open('w',newline='') as f:
        writer=csv.writer(f); writer.writerow(['video_seconds','pelvis_change_deg','trunk_change_deg','trunk_minus_pelvis_change_deg','model_marker_residual_m'])
        writer.writerows(zip(ts,pelvis,trunk,rel,fitted['residual']))
    payload={
        'times':ts.tolist(),'pelvis':pelvis.tolist(),'trunk':trunk.tolist(),'relative':rel.tolist(),
        'overlays':[],'range':[float(ts[0]),float(ts[-1])],'duration':float(times[-1]-times[0]),
        'coverage':coverage,'fitRatio':error_ratio,'factor':factor,'warnings':warnings,
        'pelvisRange':float(np.ptp(pelvis)),'trunkRange':float(np.ptp(trunk)),
        'peakGap':delta/factor if factor else delta,'peakValid':bool(peak_ok),
        'peakTimes':[float(ts[ip]),float(ts[it])],
        'pelvisSpeed':float(max(abs(speed_p))*factor) if factor else None,
        'trunkSpeed':float(max(abs(speed_t))*factor) if factor else None,
        'engine':'OpenSim inverse kinematics / experimental 2-axis model',
        'source':'pose_json','coordinateSpace':space,
    }
    write_json(job/'result.json',payload)
    write_json(job/'status.json',dict(state='done',progress=100,message='분석 완료'))

if __name__=='__main__':
    job=Path(sys.argv[1]).resolve()
    try: analyze_pose(job)
    except Exception as e:
        traceback.print_exc()
        message=str(e) if isinstance(e,ValueError) else '관절 좌표 분석에 실패했습니다. 자세한 내용은 실험 폴더의 worker.log에 기록됐습니다.'
        write_json(job/'status.json',dict(state='error',progress=0,message=message))
        sys.exit(1)
