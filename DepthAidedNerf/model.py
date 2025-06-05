import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch
import torch
import torch.nn as nn
import torchvision.transforms.functional as F
import numpy as np
import kagglehub
from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
import torch
from torchvision import transforms
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm
import os
import glob
import cv2
from PIL import Image
import pandas as pd
from torchmetrics import JaccardIndex
from datasets import load_dataset
from IPython.display import display
from torch.optim.lr_scheduler import CyclicLR, OneCycleLR, CosineAnnealingWarmRestarts, LinearLR
from torchsummary import summary
from dataloader import RayDataLoader
import matplotlib.pyplot as plt
import numpy as np
import torch
import random
from matplotlib.colors import ListedColormap
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchmetrics import JaccardIndex

 
from torchvision.transforms import v2
from torch.utils.data import DataLoader

import json
from helperFunctions import HelperFunctions
from modelNerf import NeRFModel

seed = 42
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True #False
device = torch.device("cpu") if not torch.cuda.is_available() else torch.device("cuda:0")
#print("Using device", device)
gpu_name = torch.cuda.get_device_name(device)
#print(f"GPU Name: {gpu_name}")



##  MODEL TESTING  START

path = '/media/adminnio/Volume/Data_NerfRaw/Lakshwadeep/LAK/1/colmap'
poses, c2w = HelperFunctions.get_camera_poses(path)
#HelperFunctions.plot_camera_poses(poses)
with open((rf"{path}/transforms.json"), "r") as file:
    data = json.load(file)
    
    
W, H ,focalLength = data.get("w"), data.get("h"), data.get("fl_x")
W = 50 #Fixed for Testing 
H = 50 #Fixed for Testing should generate 50*50=2500 points
origins = poses[:,:-3]
camera_vector = poses[1,3:]
    
W, H ,focalLength = data.get("w"), data.get("h"), data.get("fl_x")
W = 50 #Fixed for Testing 
H = 50 #Fixed for Testing should generate 50*50=2500 points
origins = poses[:,:-3]
camera_vector = poses[1,3:]
originrays, dirrays  = HelperFunctions.generateRays(1, 10, origins[3], H, W, focalLength,c2w[1])
print(dirrays.size())
#Takes in near bound, farbound,origin cord, image dims, Focal Length, transformation matrix. generate rays for each origin
#originrays shape is [NRays ,3] tensor, all same values
# dirrays is also [NRays ,3] tensor
#pts = HelperFunctions.samplePoints(originrays, dirrays, numPoints=20, tn=1, tf=20)
#sample 20 points over each ray, returns cordinates [20,3] for each ray * NRays tensor
#HelperFunctions.plot_rays(originrays, dirrays,10,10)
#HelperFunctions.plot_ray(pts, 2)
#*HelperFunctions.plot_rays_and_points(originrays, dirrays, pts)
pts2 = HelperFunctions.stratifiedSampling(originrays, dirrays, numPoints=20, tn=1, tf=20)
#*HelperFunctions.plot_rays_and_points(originrays, dirrays, pts2)

#Normalise Dir Rays
dirrays = dirrays / torch.norm(dirrays, dim=-1, keepdim=True)
d1 = HelperFunctions.directionalencoding(dirrays) #Input is [2500,3] Out is {2500,27}
#*print(d1.size())
p1 = HelperFunctions.positionalencoding(pts2) # Input is [2500,20,3] Out [2500,20,63]
#*print(p1.size())


##  MODEL TESTING  END


## MODEL DECLARATION
model = NeRFModel(pos_encoding_dim=60, dir_encoding_dim=24)
model.to(device)

## MODEL DECLARATION TEST

# Training example with ground truth comparison
def train_nerf_with_gt():
    """
    Example training loop that compares rendered images with ground truth
    """
    # Dummy data - replace with your actual data
    num_images = 5
    H, W = 100, 100  # Small for demo
    focal_length = 100.0
    
    # Create dummy data
    origins = torch.randn(num_images, 3)
    c2w_matrices = torch.eye(4).unsqueeze(0).repeat(num_images, 1, 1)
    ground_truth_images = torch.rand(num_images, H, W, 3)  # RGB values [0,1]
    
    # Create dataloader
    ray_dataset = RayDataLoader(
        origins=origins,
        c2w_matrices=c2w_matrices,
        ground_truth_images=ground_truth_images,
        H=H, W=W,
        focal_length=focal_length,
        rays_per_batch=512,  # Sample 512 rays per batch
        num_points=32
    )
    
    dataloader = DataLoader(ray_dataset, batch_size=1, shuffle=True, num_workers=0)
    
    # Initialize model
    model = NeRFModel(pos_encoding_dim=63, dir_encoding_dim=27)
    model.to(device)
    
    # Training loop
    for epoch in range(2):
        print(f"\n=== Epoch {epoch} ===")
        
        for batch_idx, batch in enumerate(dataloader):
            # Extract batch data
            rays = batch['rays'].squeeze(0).to(device)  # [rays_per_batch*num_points, encoding_dim]
            target_rgb = batch['target_rgb'].squeeze(0).to(device)  # [rays_per_batch, 3]
            image_idx = batch['image_index'].item()
            batch_size = batch['batch_size'].item()
            num_points = batch['num_points'].item()
            
            # Forward pass through NeRF
            with torch.no_grad():  # Remove for actual training
                output = model(rays)  # [rays_per_batch*num_points, 4]
            
            # Reshape output: [rays_per_batch, num_points, 4]
            output = output.reshape(batch_size, num_points, 4).to(device)
            rgb_samples = output[..., :3].to(device)    # [rays_per_batch, num_points, 3]
            density_samples = output[..., 3].to(device) # [rays_per_batch, num_points]
            
            # Volume rendering (simplified - you'd use HelperFunctions.renderImage)
            # For demo, just average RGB along ray
            rendered_rgb = torch.mean(rgb_samples, dim=1)  # [rays_per_batch, 3]
            
            # Compute loss (MSE between rendered and ground truth RGB)
            mse_loss = torch.mean((rendered_rgb - target_rgb) ** 2)
            
            if batch_idx % 20 == 0:
                print(f"  Batch {batch_idx}, Image {image_idx}")
                print(f"    RGB rendered range: [{rendered_rgb.min():.3f}, {rendered_rgb.max():.3f}]")
                print(f"    RGB target range: [{target_rgb.min():.3f}, {target_rgb.max():.3f}]")
                print(f"    MSE Loss: {mse_loss.item():.6f}")
            
            # Here you would:
            # loss.backward()
            # optimizer.step()
            # optimizer.zero_grad()
            
            if batch_idx >= 10:  # Just a few batches for demo
                break
    
    # Example: Render full image for validation
    print(f"\n=== Full Image Rendering Example ===")
    full_batch = ray_dataset.get_full_image_batch(0)
    print(f"Full image batch shape: {full_batch['rays'].shape}")
    print(f"Target image shape: {full_batch['target_rgb'].shape}")
    print("You can now use HelperFunctions.renderImage() with this data")

if __name__ == "__main__":
    train_nerf_with_gt()