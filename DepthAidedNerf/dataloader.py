import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
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

class RayDataLoader(Dataset):
    def __init__(self, origins, c2w_matrices, ground_truth_images, H, W, focal_length, 
                 num_points=64, tn=2.0, tf=6.0, rays_per_batch=1024):
        """
        NeRF Ray DataLoader for training - samples rays from single images
        
        Args:
            origins: List/tensor of camera origins [N_images, 3]
            c2w_matrices: Camera-to-world transformation matrices [N_images, 4, 4]
            ground_truth_images: Ground truth images [N_images, H, W, 3]
            H, W: Image height and width
            focal_length: Camera focal length
            num_points: Number of points to sample along each ray
            tn, tf: Near and far bounds for sampling
            rays_per_batch: Number of rays to sample from each image per batch
        """
        self.origins = origins
        self.c2w_matrices = c2w_matrices
        self.ground_truth_images = ground_truth_images
        self.H = H
        self.W = W
        self.focal_length = focal_length
        self.num_points = num_points
        self.tn = tn
        self.tf = tf
        self.rays_per_batch = rays_per_batch
        
        self.num_images = len(origins)
        
        # Pre-generate pixel coordinates for efficient sampling
        self.i_coords, self.j_coords = torch.meshgrid(
            torch.arange(H, dtype=torch.float32),
            torch.arange(W, dtype=torch.float32),
            indexing='ij'
        )
        self.pixel_coords = torch.stack([self.i_coords, self.j_coords], dim=-1)  # [H, W, 2]
        
        print(f"DataLoader initialized with {self.num_images} images")
        print(f"Each batch samples {rays_per_batch} rays from a single image")
        
    def __getitem__(self, index):
        """
        Returns a batch of rays sampled from a single image
        
        Returns:
            dict containing:
                - 'rays': Combined position + direction encoding [rays_per_batch*num_points, pos_dim + dir_dim]
                - 'target_rgb': Ground truth RGB values for sampled rays [rays_per_batch, 3]
                - 'ray_origins': Ray origins [rays_per_batch, 3]
                - 'ray_directions': Ray directions [rays_per_batch, 3]
                - 'pixel_coords': Pixel coordinates of sampled rays [rays_per_batch, 2]
                - 'image_index': Which image this batch is from
                - 'batch_size': Number of rays in batch
                - 'num_points': Number of points per ray
        """
        # Select which image to sample from (cycle through images)
        img_idx = index % self.num_images
        
        # Randomly sample pixel coordinates from this image
        total_pixels = self.H * self.W
        sampled_indices = torch.randperm(total_pixels)[:self.rays_per_batch]
        
        # Convert flat indices to (i, j) coordinates
        sampled_i = sampled_indices // self.W
        sampled_j = sampled_indices % self.W
        pixel_coords = torch.stack([sampled_i, sampled_j], dim=-1).float()  # [rays_per_batch, 2]
        
        # Get ground truth RGB values for sampled pixels
        target_rgb = self.ground_truth_images[img_idx][sampled_i, sampled_j]  # [rays_per_batch, 3]
        
        # Generate rays for sampled pixels
        ray_origins, ray_directions = self._generate_rays_for_pixels(
            img_idx, pixel_coords
        )
        
        # Normalize ray directions
        ray_directions = ray_directions / torch.norm(ray_directions, dim=-1, keepdim=True)
        
        # Sample points along rays using stratified sampling
        sampled_points = HelperFunctions.stratifiedSampling(
            ray_origins, ray_directions,
            numPoints=self.num_points, tn=self.tn, tf=self.tf
        )  # [rays_per_batch, num_points, 3]
        sampled_points = sampled_points.to(device)
        # Position encoding of sampled points
        points_flat = sampled_points.reshape(-1, 3)  # [rays_per_batch*num_points, 3]
        pos_encoded = HelperFunctions.positionalencoding(points_flat)  # [rays_per_batch*num_points, pos_dim]
        pos_encoded = pos_encoded.to(device)
        # Direction encoding (same direction for all points on a ray)
        directions_expanded = ray_directions.unsqueeze(1).expand(-1, self.num_points, -1).reshape(-1, 3)
        dir_encoded = HelperFunctions.directionalencoding(directions_expanded)  # [rays_per_batch*num_points, dir_dim]
        dir_encoded = dir_encoded.to(device)
        # Combine position and direction encodings
        combined_encoding = torch.cat([pos_encoded, dir_encoded], dim=-1)
        
        return {
            'rays': combined_encoding,
            'target_rgb': target_rgb,
            'ray_origins': ray_origins,
            'ray_directions': ray_directions,
            'sampled_points': sampled_points,
            'pixel_coords': pixel_coords,
            'image_index': img_idx,
            'batch_size': self.rays_per_batch,
            'num_points': self.num_points
        }
    
    def _generate_rays_for_pixels(self, img_idx, pixel_coords):
        """
        Generate rays for specific pixel coordinates
        
        Args:
            img_idx: Image index
            pixel_coords: Pixel coordinates [N, 2] (i, j format)
            
        Returns:
            ray_origins: [N, 3]
            ray_directions: [N, 3]
        """
        i_coords = pixel_coords[:, 0]  # [N]
        j_coords = pixel_coords[:, 1]  # [N]
        
        # Convert pixel coordinates to camera coordinates
        # Assuming camera is at center of image
        x = (j_coords - self.W * 0.5) / self.focal_length
        y = -(i_coords - self.H * 0.5) / self.focal_length  # Negative for image coordinate system
        z = -torch.ones_like(x)  # Forward direction in camera space
        
        # Ray directions in camera space
        dirs_camera = torch.stack([x, y, z], dim=-1)  # [N, 3]
        
        # Transform to world space using camera-to-world matrix
        c2w = self.c2w_matrices[img_idx]  # [4, 4]
        
        # Apply rotation (top-left 3x3 of c2w matrix)
        dirs_world = torch.sum(dirs_camera[..., None, :] * c2w[:3, :3], dim=-1)  # [N, 3]
        
        # Ray origins (camera position in world space)
        ray_origins = c2w[:3, 3].expand(dirs_world.shape[0], -1)  # [N, 3]
        
        return ray_origins, dirs_world
    
    def get_full_image_batch(self, img_idx):
        """
        Get all rays for a complete image (for validation/testing)
        
        Args:
            img_idx: Image index
            
        Returns:
            Same format as __getitem__ but for all pixels of the image
        """
        # All pixel coordinates for this image
        pixel_coords = self.pixel_coords.reshape(-1, 2)  # [H*W, 2]
        total_pixels = pixel_coords.shape[0]
        
        # Ground truth for all pixels
        target_rgb = self.ground_truth_images[img_idx].reshape(-1, 3)  # [H*W, 3]
        
        # Generate rays for all pixels
        ray_origins, ray_directions = self._generate_rays_for_pixels(img_idx, pixel_coords)
        ray_directions = ray_directions / torch.norm(ray_directions, dim=-1, keepdim=True)
        
        # Sample points along rays
        sampled_points = HelperFunctions.stratifiedSampling(
            ray_origins, ray_directions,
            numPoints=self.num_points, tn=self.tn, tf=self.tf
        )
        sampled_points = sampled_points.to(device)
        # Encodings
        points_flat = sampled_points.reshape(-1, 3)
        pos_encoded = HelperFunctions.positionalencoding(points_flat)
        pos_encoded = pos_encoded.to(device)
        directions_expanded = ray_directions.unsqueeze(1).expand(-1, self.num_points, -1).reshape(-1, 3)
        dir_encoded = HelperFunctions.directionalencoding(directions_expanded)
        dir_encoded = dir_encoded.to(device)
        combined_encoding = torch.cat([pos_encoded, dir_encoded], dim=-1)
        
        return {
            'rays': combined_encoding,
            'target_rgb': target_rgb,
            'ray_origins': ray_origins,
            'ray_directions': ray_directions,
            'sampled_points': sampled_points,
            'pixel_coords': pixel_coords,
            'image_index': img_idx,
            'batch_size': total_pixels,
            'num_points': self.num_points
        }
    
    def __len__(self):
        """Return number of training iterations (can be arbitrary for continuous training)"""
        return self.num_images * 100  # 100 batches per image for example

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