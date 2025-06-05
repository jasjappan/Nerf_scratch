import torch
import torch.nn as nn
import torch.nn.functional as F

class NeRFModel(nn.Module):
    def __init__(self, pos_encoding_dim=60, dir_encoding_dim=24):
        super(NeRFModel, self).__init__()
        
        # Position encoding input dimension (3D position + encoding)
        self.pos_encoding_dim = pos_encoding_dim
        # Direction encoding input dimension (3D direction + encoding) 
        self.dir_encoding_dim = dir_encoding_dim
        
        # Main MLP for processing ONLY position-encoded coordinates (like original NeRF)
        self.fc1 = nn.Linear(pos_encoding_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.fc4 = nn.Linear(256, 256)
        self.fc5 = nn.Linear(256 + pos_encoding_dim, 256)  # Skip connection at layer 5
        self.fc6 = nn.Linear(256, 256)
        self.fc7 = nn.Linear(256, 256)
        self.fc8 = nn.Linear(256, 256)
        
        # Density output head (sigma/alpha) - from 256D features
        self.density_head = nn.Linear(256, 1)
        
        # Feature extraction for view-dependent color (256D -> 256D)
        self.feature_linear = nn.Linear(256, 256)
        
        # View-dependent color MLP (features + view directions)
        self.view_fc1 = nn.Linear(256 + dir_encoding_dim, 128)
        self.color_head = nn.Linear(128, 3)  # RGB output
        
    def forward(self, x):
        # Split input: position encoding + direction encoding
        pos_encoded = x[:, :self.pos_encoding_dim]
        dir_encoded = x[:, self.pos_encoding_dim:]
        
        # Main MLP processing ONLY position-encoded coordinates
        h = F.relu(self.fc1(pos_encoded))
        h = F.relu(self.fc2(h))
        h = F.relu(self.fc3(h))
        h = F.relu(self.fc4(h))
        
        # Skip connection: concatenate original position encoding at layer 5
        h = torch.cat([pos_encoded, h], dim=-1)
        h = F.relu(self.fc5(h))
        h = F.relu(self.fc6(h))
        h = F.relu(self.fc7(h))
        h = F.relu(self.fc8(h))
        
        # Volume density output (always positive)
        density = F.relu(self.density_head(h))
        
        # Extract features for view-dependent color computation
        feature = self.feature_linear(h)
        
        # View-dependent color MLP: combine features with view directions
        h_color = torch.cat([feature, dir_encoded], dim=-1)
        h_color = F.relu(self.view_fc1(h_color))
        color = torch.sigmoid(self.color_head(h_color))  # RGB in [0,1]
        
        # Output format: [R, G, B, density] as in original NeRF
        output = torch.cat([color, density], dim=-1)
        
        return output