import numpy as np
import json
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation   

# with open('/media/adminnio/Volume/jasjappan/blender/datacolmap/ouput/transforms.json', 'r') as file:
with open('/media/adminnio/Volume/jasjappan/blender/orginalposes/transforms_val.json', 'r') as file:
     transform_data = json.load(file)

def roationconvert_nt(phi,theta,psi,data):
    cphi = np.cos(phi)
    sphi = np.sin(phi)
    ctheta = np.cos(theta)
    stheta = np.sin(theta)
    cpsi = np.cos(psi)
    spsi = np.sin(psi)
    R = np.array([
        [cpsi * ctheta, -spsi * ctheta + cpsi * stheta * sphi, spsi * sphi + cpsi * stheta * cphi],
        [spsi * ctheta,  cpsi * ctheta + spsi * stheta * sphi, -cpsi * sphi + spsi * stheta * cphi],
        [-stheta,        ctheta * sphi,                         ctheta * cphi]
    ])
    # Rt = np.transpose(R)
    # Perform matrix multiplication (rotation)
    rotated_data = R @ data
    
    return rotated_data
def roationconvert(phi,theta,psi,data):
    cphi = np.cos(phi)
    sphi = np.sin(phi)
    ctheta = np.cos(theta)
    stheta = np.sin(theta)
    cpsi = np.cos(psi)
    spsi = np.sin(psi)
    R = np.array([
        [cpsi * ctheta, -spsi * ctheta + cpsi * stheta * sphi, spsi * sphi + cpsi * stheta * cphi],
        [spsi * ctheta,  cpsi * ctheta + spsi * stheta * sphi, -cpsi * sphi + spsi * stheta * cphi],
        [-stheta,        ctheta * sphi,                         ctheta * cphi]
    ])
    Rt = np.transpose(R)
    # Perform matrix multiplication (rotation)
    rotated_data = Rt @ data
    
    return rotated_data

poses = np.array([frame['transform_matrix'] for frame in transform_data['frames']])

origins = poses[:,:-1,-1]
rotation_matrix = poses[:,:-1,0:3]

#Extract position and translation matrix from poses

r =  Rotation.from_matrix(rotation_matrix)
angles = r.as_euler("zyx",degrees=True)
x = origins[:,0]
y = origins[:,1]
z = origins[:,2]
phi = angles[:,0]
theta = angles[:,1]
psi = angles[:,2]

#Simulate IMU

dt = 1
xdot = np.gradient(x, dt)
ydot = np.gradient(y,dt)
zdot = np.gradient(z,dt)
phidot = np.gradient(phi, dt)
thetadot = np.gradient(theta, dt)
psidot = np.gradient(psi, dt)
posdot=np.array([xdot,ydot,zdot])
rotdot = np.array([phidot,thetadot,psidot])
linearvelocity = []
angularvelocity = []
i = 0
for pos in posdot.T:  
    linearvelocity.append(roationconvert(phi[i], theta[i], psi[i], pos))
    i += 1

i = 0
for rot in rotdot.T: 
    angularvelocity.append(roationconvert(phi[i], theta[i], psi[i], rot))
    i += 1

linearvelocity = np.array(linearvelocity)
angularvelocity = np.array(angularvelocity)
u = linearvelocity[:,0]
v = linearvelocity[:,1]
w = linearvelocity[:,2]

p = angularvelocity[:,0]
q = angularvelocity[:,1]
r = angularvelocity[:,2]

xdotdot = np.gradient(u, dt)
ydotdot = np.gradient(v,dt)
zdotdot = np.gradient(w,dt)

linearacc = np.array([xdotdot,ydotdot,zdotdot])
angularvel = np.array([p,q,r])
#accelerometre data & gyroscope data


