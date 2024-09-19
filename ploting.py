import numpy as np
import json
import matplotlib.pyplot as plt

with open('/media/adminnio/Volume/jasjappan/blender/datacolmap/ouput/transforms.json', 'r') as file:
# with open('/media/adminnio/Volume/jasjappan/blender/orginalposes/transforms_val.json', 'r') as file:
     transform_data = json.load(file)


# with open('/media/adminnio/Volume/jasjappan/nerf/data/blender/chair/transforms_val.json', 'r') as file:
    # transform_data = json.load(file)
poses = np.array([frame['transform_matrix'] for frame in transform_data['frames']])

origins = poses[:,:-1,-1]
directions = poses[:,:-1,0:3]
origins_x = origins[:,0]
origins_y = origins[:,1]
origins_z = origins[:,2]
directions_x = directions[:,0]
directions_y = directions[:,1]
directions_z = directions[:,2]
 
# fig = plt.figure()
# ax = fig.add_subplot(projection='3d')

# n = 100

# # For each set of style and range settings, plot n random points in the box
# # defined by x in [23, 32], y in [0, 100], z in [zlow, zhigh].
# xs = origins_x
# ys = origins_y
# zs = origins_z
# ax.scatter(xs, ys, zs, marker=m)
# # ax.quiver(, origins_y[1] origins_z[1], directions_x[1].flatten(),directions_x[1].flatten(),directions_x[1].flatten(),,color=')
# # ax.quiver(origins_x[1], origins_y[1], origins_z[1], u, v, w, length=0.1, normalize=True)

# ax.set_xlabel('X Label')
# ax.set_ylabel('Y Label')
# ax.set_zlabel('Z Label')
# plt.savefig("path") 
# plt.show()
 
print(directions_x[1])
dx_list = []
dy_list = []
dz_list = []

for direction in directions:
    direction = direction*[0,0,-1]
    dx = direction[0].flatten()
    dy = direction[1].flatten()
    dz = direction[2].flatten()
    dx_list.append(dx)
    dy_list.append(dy)
    dz_list.append(dz)
print(dx_list[1])
dx1 = np.array(dx_list)
dy1 = np.array(dy_list)
dz1 = np.array(dz_list)
  
fig = plt.figure()
ax = fig.add_subplot(projection='3d')

n = 100

# For each set of style and range settings, plot n random points in the box
# defined by x in [23, 32], y in [0, 100], z in [zlow, zhigh].
xs = origins_x
ys = origins_y
zs = origins_z
ax.scatter(xs, ys, zs)

for i in range(xs.shape[0]):
    ax.quiver(origins_x[i], origins_y[i] ,origins_z[i],np.sum(dx1[i]),np.sum(dy1[i]),np.sum(dz1[i]),length=0.7,color='red')
 
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
plt.savefig("path") 
plt.show()


def plot_camera(ax, M, label, idx):
    R = M[:3, :3] 
    t = M[:3, 3]   

    camera_origin = t
    x_axis = R[:, 0]  
    y_axis = R[:, 1]  
    z_axis = R[:, 2]  
    ax.scatter(*camera_origin, color='red', label=f'{label} {idx} origin' if idx == 0 else None)

    ax.quiver(*camera_origin, *x_axis, color='blue', length=0.5, normalize=True, label=f'{label} x-axis' if idx == 0 else None)
    ax.quiver(*camera_origin, *y_axis, color='green', length=0.5, normalize=True, label=f'{label} y-axis' if idx == 0 else None)
    ax.quiver(*camera_origin, *z_axis, color='orange', length=0.5, normalize=True, label=f'{label} z-axis' if idx == 0 else None)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

for idx, pose in enumerate(poses):
    plot_camera(ax, pose, label='Camera', idx=idx)

# Set plot labels
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')
ax.set_title('Camera Poses')

# Set limits for better visualization
ax.set_xlim([-3, 3])
ax.set_ylim([-3, 3])
ax.set_zlim([-3, 3])

# Display legend and plot
plt.legend()
plt.show()

def getrays(H:int, W:int, focallength:float, extrensicmatrix):
    print(extrensicmatrix)
    posematrix = extrensicmatrix[:3,:3]
    camera_origin = extrensicmatrix[:-1,-1]
    print(posematrix)
    # print(extrensicmatrix)
    u , v = np.meshgrid(np.arange(W),np.arange(H))
    u = u.reshape(-1).astype(np.float32)
    v = v.reshape(-1).astype(np.float32)
    focal_length =1200
    d = np.stack((u-W/2,-(v-H/2),-np.ones_like(u)*focal_length),axis=-1)
    # print(np.shape(d))
    d = d / np.linalg.norm(d, axis=-1, keepdims=True)
    # print(np.shape(posematrix))
    d = np.dot(d,posematrix)
    return d,camera_origin
    ## 3*3 . 3*1

rayset,camerapoint = getrays(256,256,1200,poses[1])
origin = np.array(camerapoint)   
k = 1
num_rays = 100  
selected_rays = rayset[np.random.choice(rayset.shape[0], num_rays, replace=False)]
scaled_rays = selected_rays * k
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

for ray in scaled_rays:
    ax.quiver(origin[0], origin[1], origin[2], ray[0], ray[1], ray[2], length=k, color='b')
ax.set_xlim([-k, k])
ax.set_ylim([-k, k])
ax.set_zlim([-k, k])
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()