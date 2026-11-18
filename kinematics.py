"""Experimental upright two-segment yaw IK; not a full musculoskeletal model."""
from pathlib import Path
import math
import numpy as np
import opensim as o

def build_model(hip_width, shoulder_width, height):
    m=o.Model(); m.setName('experimental_upright_pelvis_trunk_yaw')
    m.setGravity(o.Vec3(0,-9.81,0))
    pelvis=o.Body('pelvis',1,o.Vec3(0),o.Inertia(1))
    trunk=o.Body('trunk',1,o.Vec3(0),o.Inertia(1))
    # Pin z axis rotated into camera-up y axis.
    orient=o.Vec3(-math.pi/2,0,0)
    j1=o.PinJoint('pelvis_yaw_joint',m.getGround(),o.Vec3(0),orient,pelvis,o.Vec3(0),orient)
    j2=o.PinJoint('trunk_yaw_joint',pelvis,o.Vec3(0,height,0),orient,trunk,o.Vec3(0),orient)
    j1.updCoordinate().setName('pelvis_yaw')
    j2.updCoordinate().setName('trunk_relative_yaw')
    for body in [pelvis,trunk]: m.addBody(body)
    for joint in [j1,j2]: m.addJoint(joint)
    for name,body,p in [('left_hip',pelvis,(-hip_width/2,0,0)),('right_hip',pelvis,(hip_width/2,0,0)),('left_shoulder',trunk,(-shoulder_width/2,0,0)),('right_shoulder',trunk,(shoulder_width/2,0,0))]:
        m.addMarker(o.Marker(name,body,o.Vec3(*p)))
    m.finalizeConnections()
    return m

def fit(times, points, folder):
    """points: N x [left hip,right hip,left shoulder,right shoulder] x XYZ, meters."""
    folder=Path(folder); folder.mkdir(exist_ok=True,parents=True)
    o.Logger.setLevelString('Warn')
    p=np.array(points,dtype=float)
    p-=p[:,:2].mean(axis=1)[:,None,:]
    hw=float(np.median(np.linalg.norm(p[:,1]-p[:,0],axis=1)))
    sw=float(np.median(np.linalg.norm(p[:,3]-p[:,2],axis=1)))
    height=float(np.median(p[:,2:,1].mean(axis=1)))
    if not (0.12<hw<0.65 and 0.15<sw<0.85 and 0.15<height<0.9):
        raise ValueError('신체 좌표의 크기 또는 방향이 불안정합니다. 몸통과 골반이 잘 보이는 영상을 사용해주세요.')
    m=build_model(hw,sw,height)
    hip=p[0,1]-p[0,0]; shoulder=p[0,3]-p[0,2]
    initial_p=math.atan2(-hip[2],hip[0]); initial_t=math.atan2(-shoulder[2],shoulder[0])
    m.updCoordinateSet().get('pelvis_yaw').setDefaultValue(initial_p)
    m.updCoordinateSet().get('trunk_relative_yaw').setDefaultValue(math.atan2(math.sin(initial_t-initial_p),math.cos(initial_t-initial_p)))
    m.printToXML(str(folder/'model.osim'))
    table=o.TimeSeriesTableVec3(); names=['left_hip','right_hip','left_shoulder','right_shoulder']
    labels=o.StdVectorString()
    for name in names: labels.append(name)
    table.setColumnLabels(labels); table.addTableMetaDataString('Units','m')
    table.addTableMetaDataString('DataRate',str(1/np.median(np.diff(times))))
    for t,frame in zip(times,p):
        row=o.RowVectorVec3(4)
        for k,xyz in enumerate(frame): row[k]=o.Vec3(*map(float,xyz))
        table.appendRow(float(t),row)
    trc=str(folder/'markers.trc'); o.TRCFileAdapter.write(table,trc)
    tool=o.InverseKinematicsTool(); tool.setModel(m)
    tool.setMarkerDataFileName(trc); tool.setStartTime(float(times[0])); tool.setEndTime(float(times[-1]))
    tool.setOutputMotionFileName(str(folder/'motion.mot')); tool.setResultsDir(str(folder))
    tool.set_report_errors(False)
    for name in names:
        task=o.IKMarkerTask(); task.setName(name); task.setWeight(1); task.setApply(True)
        tool.upd_IKTaskSet().cloneAndAppend(task)
    tool.set_accuracy(1e-5)
    tool.run()
    result=o.TimeSeriesTable(str(folder/'motion.mot'))
    ts=np.array(result.getIndependentColumn())
    pelvis=np.array(result.getDependentColumn('pelvis_yaw').to_numpy()).ravel()
    relative=np.array(result.getDependentColumn('trunk_relative_yaw').to_numpy()).ravel()
    if result.hasTableMetaDataKey('inDegrees') and result.getTableMetaDataAsString('inDegrees')=='no':
        pelvis=np.degrees(pelvis); relative=np.degrees(relative)
    pelvis=np.degrees(np.unwrap(np.radians(pelvis)))
    relative=np.degrees(np.unwrap(np.radians(relative)))
    # Check fit residual in the same inferred 3D coordinate space, not real-world accuracy.
    state=m.initSystem(); residual=[]
    for i,t in enumerate(ts):
        a=float(np.interp(t,times,pelvis)) if len(pelvis)==len(times) else float(pelvis[i])
        m.updCoordinateSet().get('pelvis_yaw').setValue(state,math.radians(a))
        m.updCoordinateSet().get('trunk_relative_yaw').setValue(state,math.radians(float(relative[i])))
        m.realizePosition(state)
        observed=np.array([[np.interp(t,times,p[:,k,d]) for d in range(3)] for k in range(4)])
        predicted=np.array([[m.getMarkerSet().get(n).getLocationInGround(state).get(d) for d in range(3)] for n in names])
        residual.append(float(np.sqrt(np.mean(np.sum((observed-predicted)**2,axis=1)))))
    return {'times':ts,'pelvis':pelvis,'trunk':pelvis+relative,'relative':relative,'residual':np.array(residual),'scale':height}
