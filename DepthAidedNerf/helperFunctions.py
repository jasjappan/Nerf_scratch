import json
import numpy as np
import matplotlib.pyplot as plt
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from mpl_toolkits.mplot3d import Axes3D
import os
import math

seed = 42
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True #False
device = torch.device("cpu") if not torch.cuda.is_available() else torch.device("cuda:0")
#print("Using device", device)
gpu_name = torch.cuda.get_device_name(device)
#print(f"GPU Name: {gpu_name}")

class HelperFunctions:
    """
    A helper class for processing and visualizing camera poses.
    """
    
    @staticmethod
    def get_camera_poses(input_dir):
        """
           Reads the transforms.json file from the given directory and returns camera poses.
           Each pose contains origin (x, y, z) and direction (x, y, z).
           
           Args:
               input_dir (str): Path to the directory containing transforms.json.
           
           Returns:
               tuple: 
                   - np.ndarray: Array of shape (N, 6), where N is the number of frames.
                     Each row contains [origin_x, origin_y, origin_z, dir_x, dir_y, dir_z].
                   - np.ndarray: Array of shape (N, 4, 4), where N is the number of frames.
                     Each element is a 4x4 pose matrix.
           """
        json_path = f"{input_dir}/transforms.json"
        
        with open(json_path, 'r') as file:
            transform_data = json.load(file)
        
        poses = np.array([frame['transform_matrix'] for frame in transform_data['frames']])
        origins = poses[:, :-1, -1]
        directions = poses[:, :-1, 0:3]
        
        dx_list, dy_list, dz_list = [], [], []
        
        for direction in directions:
            direction = direction * [0, 0, -1]  # Flip z-axis
            dx_list.append(np.sum(direction[0]))
            dy_list.append(np.sum(direction[1]))
            dz_list.append(np.sum(direction[2]))
        
        origins_x, origins_y, origins_z = origins[:, 0], origins[:, 1], origins[:, 2]
        poses_output = np.column_stack((origins_x, origins_y, origins_z, dx_list, dy_list, dz_list))
        
        return poses_output, poses

    @staticmethod
    def plot_camera_poses(poses):
        """
        Plots the camera poses in 3D space.
        
        Args:
            poses (np.ndarray): Array of camera poses containing origin and direction.
        """
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        
        xs, ys, zs = poses[:, 0], poses[:, 1], poses[:, 2]
        ax.scatter(xs, ys, zs)
        
        for i in range(poses.shape[0]):
            ax.quiver(xs[i], ys[i], zs[i], poses[i, 3], poses[i, 4], poses[i, 5], length=0.7, color='red')
        
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
        plt.show()
    
    @staticmethod
    def generateRays(hn, hf, origins, H, W, focalLength, c2w):
        height = np.arange(H)
        width = np.arange(W)
        x_1, y_1 = np.meshgrid(height, width, indexing='ij')
        
        x_1 = x_1.reshape(-1).astype(np.float32)
        y_1 = y_1.reshape(-1).astype(np.float32)
    
        rays_direction = np.stack([(x_1 - W/2) / focalLength, 
       -(y_1 - H/2) / focalLength, 
                                                               -np.ones_like(x_1)], axis=-1)
    
        # Ensure all tensors are on the same device
        origins = torch.tensor(origins, dtype=torch.float32, device=device)
        rays_direction = torch.tensor(rays_direction, dtype=torch.float32, device=device)  # Moved to device
        c2w = torch.tensor(c2w, dtype=torch.float32, device=device)
    
        rays_origin = origins.expand(rays_direction.shape)   
        #print(np.shape(rays_direction[..., None, :]))
        #print(c2w)
        rays_d = torch.sum(rays_direction[..., None, :] * c2w[:3, :3], dim=-1)
    
        return rays_origin, rays_d
    

    @staticmethod
    def plot_rays(o: torch.Tensor, d: torch.Tensor, t=10, num_rays=10):
        """Plot 3D rays using PyTorch tensors, only plotting a random subset of rays on the GPU."""
        with torch.no_grad():
            # Ensure both tensors are on the same device (CUDA)
            device = torch.device("cuda")  # Use CUDA device
            o = o.to(device)
            d = d.to(device)
            
            # Randomly select `num_rays` indices
            total_rays = o.shape[0]
            selected_indices = torch.randint(0, total_rays, (num_rays,), device=device)
            
            # Select rays based on the random indices
            o_selected = o[selected_indices]
            d_selected = d[selected_indices]
            
            # Prepare for plotting on CPU (because matplotlib requires CPU data)
            pt1 = o_selected.cpu().numpy()  # Move to CPU for plotting
            pt2 = (o_selected + t * d_selected).cpu().numpy()
            
            # Plot each selected ray
            fig = plt.figure(figsize=(12, 12))
            ax = fig.add_subplot(111, projection='3d')
    
            # Plot the rays
            for p1, p2 in zip(pt1, pt2):
                ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], c="C0")
            
            plt.show()
    
    @staticmethod
    def samplePoints(o: torch.tensor, d:torch.tensor, numPoints, tn, tf):
        t = torch.linspace(tn, tf, numPoints).view(1, numPoints, 1)
        t = t.to(device)
        o = o.unsqueeze(1)  # Shape: [23924496, 1, 3]
        d = d.unsqueeze(1)  # Shape: [23924496, 1, 3]
        pt = o + t*d
        return pt
    @staticmethod
    def plot_ray(pts: torch.tensor, i: int):
        """
        Plots the sampled points along the i-th ray.
        
        Args:
            pts (torch.tensor): Tensor of shape [N, numPoints, 3], where N is the number of rays.
            i (int): Index of the ray to plot.
        """
        if i >= pts.shape[0]:
            print(f"Invalid index {i}, must be less than {pts.shape[0]}")
            return
    
        ray_pts = pts[i].cpu().numpy()  # Get the specific ray's points and move to CPU if needed
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, projection='3d')
    
        # Plot sampled points
        ax.scatter(ray_pts[:, 0], ray_pts[:, 1], ray_pts[:, 2], c='r', marker='o', label="Sampled Points")
    
        # Draw a line connecting the points
        ax.plot(ray_pts[:, 0], ray_pts[:, 1], ray_pts[:, 2], 'b-', label="Ray Path")
    
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'Ray {i}')
        ax.legend()
        plt.show()
    
    @staticmethod
    def plot_rays_and_points(o: torch.Tensor, d: torch.Tensor, pts: torch.Tensor, t=10, num_rays=10):
        """
        Plots 3D rays and their sampled points using PyTorch tensors.
        
        Args:
            o (torch.Tensor): Origins of shape [N, 3]
            d (torch.Tensor): Directions of shape [N, 3]
            pts (torch.Tensor): Sampled points of shape [N, numPoints, 3]
            t (float): Length of the ray to visualize
            num_rays (int): Number of rays to plot
        """
        with torch.no_grad():
            # Ensure both tensors are on the same device (CUDA)
            device = torch.device("cuda")  # Use CUDA device
            o = o.to(device)
            d = d.to(device)
            pts = pts.to(device)
            
            # Randomly select `num_rays` indices
            total_rays = o.shape[0]
            selected_indices = torch.randint(0, total_rays, (num_rays,), device=device)
            
            # Select rays and their sampled points
            o_selected = o[selected_indices]
            d_selected = d[selected_indices]
            pts_selected = pts[selected_indices].cpu().numpy()  # Move to CPU
            
            # Prepare for plotting on CPU (because matplotlib requires CPU data)
            pt1 = o_selected.cpu().numpy()  # Move to CPU for plotting
            pt2 = (o_selected + t * d_selected).cpu().numpy()
            
            # Plot
            fig = plt.figure(figsize=(12, 12))
            ax = fig.add_subplot(111, projection='3d')
            
            # Plot the rays
            for p1, p2, ray_pts in zip(pt1, pt2, pts_selected):
                ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], c="C0", label="Ray Path")
                ax.scatter(ray_pts[:, 0], ray_pts[:, 1], ray_pts[:, 2], c='r', marker='o', label="Sampled Points")
            
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_title(f'Visualization of {num_rays} Rays and Their Sampled Points')
            plt.show()
    @staticmethod
    def stratifiedSampling(o: torch.tensor, d:torch.tensor, numPoints, tn, tf):
        t = torch.linspace(tn, tf, numPoints ).view(1, numPoints, 1)
        bin_size = (tf -tn )/numPoints
        noise = torch.rand_like(t) * bin_size
        t = t + noise - bin_size/2
        t = torch.clamp(t, tn, tf)
        t = t.to(device)
        o = o.unsqueeze(1)  # Shape: [23924496, 1, 3]
        d = d.unsqueeze(1)  # Shape: [23924496, 1, 3]
        o = o.to(device)
        d = d.to(device)
        pts = o + t*d
        pts = pts.to(device)
        return pts 
    
    @staticmethod
    def positionalencoding(x,num_frequency=10,include_input = True):
           device = x.device
           results = []
       
           for f in range(num_frequency):
               value = (2 ** f )* x
               # value = (2 ** f) * x
               results.append(torch.sin(value))
               results.append(torch.cos(value))
           if include_input:
               pos_out = torch.cat([x] + results, dim=-1)
           else:
               pos_out = torch.cat(results, dim=-1)
           return pos_out
    @staticmethod
    def directionalencoding(x,num_frequency=4, include_input=True):
        evice = x.device
        results = []
    
        for f in range(num_frequency):
            value = (2 ** f )* x
            # value = (2 ** f) * x
            results.append(torch.sin(value))
            results.append(torch.cos(value))
        if include_input:
            pos_out = torch.cat([x] + results, dim=-1)
        else:
            pos_out = torch.cat(results, dim=-1)
        return pos_out