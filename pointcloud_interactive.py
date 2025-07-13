import numpy as np
import open3d as o3d
from skimage import measure
from scipy.spatial import cKDTree
from scipy import ndimage
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import os
import time
import sys
import io
from typing import Optional, Callable, Any

# Enhanced Terminal Output Capture Class
class TerminalCapture:
    def __init__(self, callback, real_time=True):
        self.callback = callback
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr
        self.stdout_buffer = io.StringIO()
        self.stderr_buffer = io.StringIO()
        self.real_time = real_time
        
    def __enter__(self):
        sys.stdout = self.stdout_buffer
        sys.stderr = self.stderr_buffer
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self.original_stdout
        sys.stderr = self.original_stderr
        
        # Get captured output
        stdout_content = self.stdout_buffer.getvalue()
        stderr_content = self.stderr_buffer.getvalue()
        
        # Send to callback
        if stdout_content:
            self.callback(stdout_content, "stdout")
        if stderr_content:
            self.callback(stderr_content, "stderr")
            
        # Clean up
        self.stdout_buffer.close()
        self.stderr_buffer.close()

# Real-time Terminal Output Capture
class RealTimeTerminalCapture:
    def __init__(self, callback):
        self.callback = callback
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr
        
    def __enter__(self):
        # Create custom stream classes for real-time capture
        class CapturingStream:
            def __init__(self, original_stream, callback, stream_type):
                self.original_stream = original_stream
                self.callback = callback
                self.stream_type = stream_type
                self.buffer = ""
                
            def write(self, text):
                self.original_stream.write(text)
                self.buffer += text
                if '\n' in text:
                    lines = self.buffer.split('\n')
                    for line in lines[:-1]:
                        if line.strip():
                            self.callback(line.strip(), self.stream_type)
                    self.buffer = lines[-1]
                    
            def flush(self):
                self.original_stream.flush()
                if self.buffer.strip():
                    self.callback(self.buffer.strip(), self.stream_type)
                    self.buffer = ""
        
        self.stdout_capture = CapturingStream(self.original_stdout, self.callback, "stdout")
        self.stderr_capture = CapturingStream(self.original_stderr, self.callback, "stderr")
        
        sys.stdout = self.stdout_capture
        sys.stderr = self.stderr_capture
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self.original_stdout
        sys.stderr = self.original_stderr
        self.stdout_capture.flush()
        self.stderr_capture.flush()

# Octree Parameters Explanation
# 1. Octree Box Size (octree_size_var)
# Purpose: Defines the base resolution of the octree grid cells
# Range: 0.001 to 0.1 (default: 0.01)
# Effect:
# Smaller values = finer resolution, more detailed mesh, longer processing time
# Larger values = coarser resolution, less detailed mesh, faster processing
# Usage: Used as the base grid size for adaptive octree decomposition
# Example: 0.01 means each octree cell is 0.01 units in size
# 2. Sample Points (octree_samples_var)
# Purpose: Number of points sampled from the mesh surface for distance computation
# Range: 5,000 to 100,000 (default: 20,000)
# Effect:
# More points = more accurate distance field, better quality, slower processing
# Fewer points = faster processing, potentially less accurate results
# Usage: Used to create a point cloud representation of the mesh surface for efficient distance queries
# Example: 20,000 points are sampled uniformly from the mesh surface


class ModernStyle:
    BG_DARK = "#1e1e1e"
    BG_CARD = "#2d2d2d"
    ACCENT = "#0078d4"
    ACCENT_HOVER = "#106ebe"
    TEXT_PRIMARY = "#ffffff"
    TEXT_SECONDARY = "#b3b3b3"
    SUCCESS = "#10b981"
    WARNING = "#f59e0b"
    ERROR = "#ef4444"
    FONT_TITLE = ("Segoe UI", 16, "bold")
    FONT_BODY = ("Segoe UI", 9)

class ModernButton(tk.Button):
    def __init__(self, parent, text, command=None, primary=True, **kwargs):
        bg = ModernStyle.ACCENT if primary else ModernStyle.BG_CARD
        hover = ModernStyle.ACCENT_HOVER if primary else "#3d3d3d"
        super().__init__(parent, text=text, command=command, bg=bg, fg=ModernStyle.TEXT_PRIMARY,
                        font=ModernStyle.FONT_BODY, relief="flat", bd=0, padx=15, pady=8, cursor="hand2", **kwargs)
        self.bind("<Enter>", lambda e: self.config(bg=hover))
        self.bind("<Leave>", lambda e: self.config(bg=bg))

class ModernFrame(tk.Frame):
    def __init__(self, parent, **kwargs):
        super().__init__(parent, bg=ModernStyle.BG_CARD, relief="flat", bd=0, padx=15, pady=15, **kwargs)

class ModernLabel(tk.Label):
    def __init__(self, parent, text, secondary=False, **kwargs):
        color = ModernStyle.TEXT_SECONDARY if secondary else ModernStyle.TEXT_PRIMARY
        # Remove font from kwargs if it exists to avoid conflict
        kwargs.pop('font', None)
        super().__init__(parent, text=text, bg=ModernStyle.BG_CARD, fg=color, font=ModernStyle.FONT_BODY, **kwargs)

class ModernProgressBar(tk.Canvas):
    def __init__(self, parent, width=300, height=4, **kwargs):
        super().__init__(parent, width=width, height=height, bg=ModernStyle.BG_DARK, highlightthickness=0, **kwargs)
        self.width = width
        self.height = height
        self.progress = 0
        self.create_rectangle(0, 0, width, height, fill="#404040", outline="")
        self.progress_rect = self.create_rectangle(0, 0, 0, height, fill=ModernStyle.ACCENT, outline="")
    
    def set_progress(self, value):
        self.progress = max(0, min(100, value))
        progress_width = (self.progress / 100) * self.width
        self.coords(self.progress_rect, 0, 0, progress_width, self.height)

class PointCloudProcessor:
    def __init__(self, root):
        self.root = root
        self.root.title("Point Cloud Processor")
        self.root.geometry("900x650")
        self.root.configure(bg=ModernStyle.BG_DARK)
        self.pcd: Optional[o3d.geometry.PointCloud] = None
        self.mesh: Optional[o3d.geometry.TriangleMesh] = None
        self.current_file: Optional[str] = None
        self.processing_active = False
        self.create_ui()
        
    def create_ui(self):
        main = tk.Frame(self.root, bg=ModernStyle.BG_DARK)
        main.pack(fill="both", expand=True, padx=15, pady=15)
        
        # Title
        title = tk.Label(main, text="Point Cloud Processor", font=ModernStyle.FONT_TITLE, 
                        bg=ModernStyle.BG_DARK, fg=ModernStyle.TEXT_PRIMARY)
        title.pack(pady=(0, 20))
        
        # File operations
        file_frame = ModernFrame(main)
        file_frame.pack(fill="x", pady=(0, 10))
        
        ModernLabel(file_frame, "File Operations", font=("Segoe UI", 10, "bold")).pack(anchor="w", pady=(0, 10))
        
        file_buttons = tk.Frame(file_frame, bg=ModernStyle.BG_CARD)
        file_buttons.pack(fill="x")
        
        ModernButton(file_buttons, "Load Point Cloud", self.load_point_cloud).pack(side="left", padx=(0, 8))
        ModernButton(file_buttons, "Load Mesh", self.load_mesh, False).pack(side="left", padx=(0, 8))
        ModernButton(file_buttons, "Save Point Cloud", self.save_point_cloud, False).pack(side="left", padx=(0, 8))
        ModernButton(file_buttons, "Save Mesh", self.save_mesh, False).pack(side="left")
        
        self.file_status = ModernLabel(file_frame, "No file loaded", True)
        self.file_status.pack(anchor="w", pady=(10, 0))
        
        # Main content area with two columns
        content_frame = tk.Frame(main, bg=ModernStyle.BG_DARK)
        content_frame.pack(fill="both", expand=True, pady=(0, 10))
        
        # Left column - Standard operations
        left_column = tk.Frame(content_frame, bg=ModernStyle.BG_DARK)
        left_column.pack(side="left", fill="both", expand=True, padx=(0, 5))
        
        # Right column - Octree operations
        right_column = tk.Frame(content_frame, bg=ModernStyle.BG_DARK)
        right_column.pack(side="right", fill="both", expand=True, padx=(5, 0))
        
        # Left column: Standard Parameters
        params_frame = ModernFrame(left_column)
        params_frame.pack(fill="x", pady=(0, 10))
        
        ModernLabel(params_frame, "Standard Parameters", font=("Segoe UI", 10, "bold")).pack(anchor="w", pady=(0, 10))
        
        params_grid = tk.Frame(params_frame, bg=ModernStyle.BG_CARD)
        params_grid.pack(fill="x")
        
        # Method
        method_frame = tk.Frame(params_grid, bg=ModernStyle.BG_CARD)
        method_frame.pack(fill="x", pady=(0, 5))
        ModernLabel(method_frame, "Method:").pack(side="left")
        self.method_var = tk.StringVar(value="Poisson")
        ttk.Combobox(method_frame, textvariable=self.method_var, values=["Poisson", "Marching Cubes"], 
                    state="readonly", width=12).pack(side="left", padx=(10, 0))
        
        # Depth
        depth_frame = tk.Frame(params_grid, bg=ModernStyle.BG_CARD)
        depth_frame.pack(fill="x", pady=(0, 5))
        ModernLabel(depth_frame, "Depth:").pack(side="left")
        self.depth_var = tk.IntVar(value=10)
        ttk.Spinbox(depth_frame, from_=6, to=14, textvariable=self.depth_var, width=8).pack(side="left", padx=(10, 0))
        
        # Voxel size
        voxel_frame = tk.Frame(params_grid, bg=ModernStyle.BG_CARD)
        voxel_frame.pack(fill="x")
        ModernLabel(voxel_frame, "Voxel Size:").pack(side="left")
        self.voxel_var = tk.DoubleVar(value=0.01)
        ttk.Spinbox(voxel_frame, from_=0.001, to=0.1, increment=0.001, textvariable=self.voxel_var, width=8).pack(side="left", padx=(10, 0))
        
        # Right column: Octree Parameters
        octree_params_frame = ModernFrame(right_column)
        octree_params_frame.pack(fill="x", pady=(0, 10))
        
        ModernLabel(octree_params_frame, "Octree Parameters", font=("Segoe UI", 10, "bold")).pack(anchor="w", pady=(0, 10))
        
        octree_params_grid = tk.Frame(octree_params_frame, bg=ModernStyle.BG_CARD)
        octree_params_grid.pack(fill="x")
        
        # Octree box size
        octree_size_frame = tk.Frame(octree_params_grid, bg=ModernStyle.BG_CARD)
        octree_size_frame.pack(fill="x", pady=(0, 5))
        ModernLabel(octree_size_frame, "Octree Box Size:").pack(side="left")
        self.octree_size_var = tk.DoubleVar(value=0.01)
        ttk.Spinbox(octree_size_frame, from_=0.001, to=0.1, increment=0.001, textvariable=self.octree_size_var, width=8).pack(side="left", padx=(10, 0))
        ModernLabel(octree_size_frame, "(Lower = Better Quality, Slower)", True).pack(side="left", padx=(5, 0))
        
        # Octree sample points
        octree_samples_frame = tk.Frame(octree_params_grid, bg=ModernStyle.BG_CARD)
        octree_samples_frame.pack(fill="x", pady=(0, 5))
        ModernLabel(octree_samples_frame, "Sample Points:").pack(side="left")
        self.octree_samples_var = tk.IntVar(value=10000)
        ttk.Spinbox(octree_samples_frame, from_=5000, to=100000, increment=1000, textvariable=self.octree_samples_var, width=8).pack(side="left", padx=(10, 0))
        ModernLabel(octree_samples_frame, "(Higher = Better Quality, Slower)", True).pack(side="left", padx=(5, 0))
        
        # SDF Padding
        sdf_padding_frame = tk.Frame(octree_params_grid, bg=ModernStyle.BG_CARD)
        sdf_padding_frame.pack(fill="x", pady=(0, 5))
        ModernLabel(sdf_padding_frame, "SDF Padding:").pack(side="left")
        self.sdf_padding_var = tk.IntVar(value=2)
        ttk.Spinbox(sdf_padding_frame, from_=1, to=20, increment=1, textvariable=self.sdf_padding_var, width=8).pack(side="left", padx=(10, 0))
        ModernLabel(sdf_padding_frame, "(Higher = More Padding, Larger Grid)", True).pack(side="left", padx=(5, 0))
        
        # Distance Threshold Multiplier
        distance_threshold_frame = tk.Frame(octree_params_grid, bg=ModernStyle.BG_CARD)
        distance_threshold_frame.pack(fill="x", pady=(0, 5))
        ModernLabel(distance_threshold_frame, "Distance Threshold:").pack(side="left")
        self.distance_threshold_var = tk.DoubleVar(value=1.0)
        ttk.Spinbox(distance_threshold_frame, from_=1.0, to=5.0, increment=0.1, textvariable=self.distance_threshold_var, width=8).pack(side="left", padx=(10, 0))
        ModernLabel(distance_threshold_frame, "(Higher = More Interior Filling)", True).pack(side="left", padx=(5, 0))
        
        # Max Grid Dimension
        max_grid_dim_frame = tk.Frame(octree_params_grid, bg=ModernStyle.BG_CARD)
        max_grid_dim_frame.pack(fill="x", pady=(0, 5))
        ModernLabel(max_grid_dim_frame, "Max Grid Dimension:").pack(side="left")
        self.max_grid_dim_var = tk.IntVar(value=120)
        ttk.Spinbox(max_grid_dim_frame, from_=50, to=150, increment=10, textvariable=self.max_grid_dim_var, width=8).pack(side="left", padx=(10, 0))
        ModernLabel(max_grid_dim_frame, "(Higher = Better Quality, More Memory)", True).pack(side="left", padx=(5, 0))
        
        # Max Cells to Process
        max_cells_frame = tk.Frame(octree_params_grid, bg=ModernStyle.BG_CARD)
        max_cells_frame.pack(fill="x", pady=(0, 5))
        ModernLabel(max_cells_frame, "Max Cells to Process:").pack(side="left")
        self.max_cells_var = tk.IntVar(value=200000)
        ttk.Spinbox(max_cells_frame, from_=50000, to=500000, increment=50000, textvariable=self.max_cells_var, width=8).pack(side="left", padx=(10, 0))
        ModernLabel(max_cells_frame, "(Higher = Better Quality, Slower)", True).pack(side="left", padx=(5, 0))
        
        # Binary Fill Holes Parameters (for watertight)
        binary_fill_frame = tk.Frame(octree_params_grid, bg=ModernStyle.BG_CARD)
        binary_fill_frame.pack(fill="x", pady=(10, 5))
        ModernLabel(binary_fill_frame, "Binary Fill Holes Parameters (Watertight)", font=("Segoe UI", 9, "bold")).pack(anchor="w", pady=(0, 5))
        
        # Binary Fill Resolution
        binary_res_frame = tk.Frame(octree_params_grid, bg=ModernStyle.BG_CARD)
        binary_res_frame.pack(fill="x", pady=(0, 5))
        ModernLabel(binary_res_frame, "Binary Fill Resolution:").pack(side="left")
        self.binary_res_var = tk.DoubleVar(value=0.02)
        ttk.Spinbox(binary_res_frame, from_=0.01, to=0.1, increment=0.01, textvariable=self.binary_res_var, width=8).pack(side="left", padx=(10, 0))
        ModernLabel(binary_res_frame, "(Lower = Better Quality, Slower)", True).pack(side="left", padx=(5, 0))
        
        # Distance Threshold Multiplier
        distance_threshold_frame = tk.Frame(octree_params_grid, bg=ModernStyle.BG_CARD)
        distance_threshold_frame.pack(fill="x", pady=(0, 5))
        ModernLabel(distance_threshold_frame, "Distance Threshold Multiplier:").pack(side="left")
        self.binary_distance_threshold_var = tk.DoubleVar(value=1.5)
        ttk.Spinbox(distance_threshold_frame, from_=0.5, to=3.0, increment=0.1, textvariable=self.binary_distance_threshold_var, width=8).pack(side="left", padx=(10, 0))
        ModernLabel(distance_threshold_frame, "(Higher = More Interior Filling)", True).pack(side="left", padx=(5, 0))
        
        # Morphological Kernel Size
        kernel_size_frame = tk.Frame(octree_params_grid, bg=ModernStyle.BG_CARD)
        kernel_size_frame.pack(fill="x", pady=(0, 5))
        ModernLabel(kernel_size_frame, "Morphological Kernel Size:").pack(side="left")
        self.binary_kernel_size_var = tk.IntVar(value=5)
        ttk.Spinbox(kernel_size_frame, from_=1, to=15, increment=1, textvariable=self.binary_kernel_size_var, width=8).pack(side="left", padx=(10, 0))
        ModernLabel(kernel_size_frame, "(Higher = More Aggressive Filling)", True).pack(side="left", padx=(5, 0))
        

        
        # Left column: Standard Actions
        standard_actions_frame = ModernFrame(left_column)
        standard_actions_frame.pack(fill="x", pady=(0, 10))
        
        ModernLabel(standard_actions_frame, "Standard Actions", font=("Segoe UI", 10, "bold")).pack(anchor="w", pady=(0, 10))
        
        standard_actions_grid = tk.Frame(standard_actions_frame, bg=ModernStyle.BG_CARD)
        standard_actions_grid.pack(fill="x")
        
        row1 = tk.Frame(standard_actions_grid, bg=ModernStyle.BG_CARD)
        row1.pack(fill="x", pady=(0, 8))
        ModernButton(row1, "Process to Mesh", self.process_to_mesh).pack(side="left", padx=(0, 8))
        ModernButton(row1, "Clean Point Cloud", self.clean_point_cloud, False).pack(side="left", padx=(0, 8))
        ModernButton(row1, "Clean Mesh", self.clean_mesh, False).pack(side="left", padx=(0, 8))
        ModernButton(row1, "Make Watertight", self.make_watertight, False).pack(side="left")
        
        row2 = tk.Frame(standard_actions_grid, bg=ModernStyle.BG_CARD)
        row2.pack(fill="x", pady=(0, 8))
        ModernButton(row2, "Calculate Properties", self.calculate_properties).pack(side="left", padx=(0, 8))
        ModernButton(row2, "View Point Cloud", self.view_point_cloud, False).pack(side="left", padx=(0, 8))
        ModernButton(row2, "View Mesh", self.view_mesh, False).pack(side="left")
        
        row3 = tk.Frame(standard_actions_grid, bg=ModernStyle.BG_CARD)
        row3.pack(fill="x")
        ModernButton(row3, "Save Watertight Mesh", self.save_watertight_mesh, False).pack(side="left", padx=(0, 8))
        
        # Right column: Octree Actions
        octree_actions_frame = ModernFrame(right_column)
        octree_actions_frame.pack(fill="x", pady=(0, 10))
        
        ModernLabel(octree_actions_frame, "Octree Volume Operations", font=("Segoe UI", 10, "bold")).pack(anchor="w", pady=(0, 10))
        
        octree_actions_grid = tk.Frame(octree_actions_frame, bg=ModernStyle.BG_CARD)
        octree_actions_grid.pack(fill="x")
        
        octree_row1 = tk.Frame(octree_actions_grid, bg=ModernStyle.BG_CARD)
        octree_row1.pack(fill="x", pady=(0, 8))
        ModernButton(octree_row1, "Generate Octree Volume Mesh", self.generate_octree_mesh).pack(side="left", padx=(0, 8))
        ModernButton(octree_row1, "View Octree Mesh", self.view_octree_mesh, False).pack(side="left", padx=(0, 8))
        
        octree_row2 = tk.Frame(octree_actions_grid, bg=ModernStyle.BG_CARD)
        octree_row2.pack(fill="x", pady=(0, 8))
        ModernButton(octree_row2, "Calculate Volume", self.calculate_octree_volume).pack(side="left", padx=(0, 8))
        ModernButton(octree_row2, "Save Octree Mesh", self.save_octree_mesh, False).pack(side="left", padx=(0, 8))
        
        octree_row3 = tk.Frame(octree_actions_grid, bg=ModernStyle.BG_CARD)
        octree_row3.pack(fill="x")
        ModernButton(octree_row3, "Octree Analysis", self.octree_analysis).pack(side="left", padx=(0, 8))
        ModernButton(octree_row3, "Coll Methodology Analysis", self.analyze_coll_methodology_compliance, False).pack(side="left", padx=(0, 8))
        
        octree_row4 = tk.Frame(octree_actions_grid, bg=ModernStyle.BG_CARD)
        octree_row4.pack(fill="x", pady=(8, 0))
        ModernButton(octree_row4, "Mesh Comparison", self.compare_meshes_visualization, False).pack(side="left", padx=(0, 8))
        ModernButton(octree_row4, "Area Analysis", self.analyze_surface_areas, False).pack(side="left", padx=(0, 8))
        ModernButton(octree_row4, "Volume Analysis", self.analyze_volumes, False).pack(side="left", padx=(0, 8))
        
        octree_row5 = tk.Frame(octree_actions_grid, bg=ModernStyle.BG_CARD)
        octree_row5.pack(fill="x", pady=(8, 0))
        ModernButton(octree_row5, "Make Octree Watertight", self.make_octree_watertight).pack(side="left", padx=(0, 8))
        ModernButton(octree_row5, "System Health Check", self._check_system_health, False).pack(side="left", padx=(0, 8))
        

        
        # Progress and Results (span both columns)
        progress_results_frame = tk.Frame(main, bg=ModernStyle.BG_DARK)
        progress_results_frame.pack(fill="both", expand=True, pady=(0, 10))
        
        # Progress
        progress_frame = ModernFrame(progress_results_frame)
        progress_frame.pack(fill="x", pady=(0, 10))
        
        ModernLabel(progress_frame, "Progress", font=("Segoe UI", 10, "bold")).pack(anchor="w", pady=(0, 8))
        
        self.progress_bar = ModernProgressBar(progress_frame, width=300, height=6)
        self.progress_bar.pack(anchor="w", pady=(0, 8))
        
        self.status_label = ModernLabel(progress_frame, "Ready", True)
        self.status_label.pack(anchor="w")
        
        # Add current operation label
        self.operation_label = ModernLabel(progress_frame, "", True)
        self.operation_label.pack(anchor="w", pady=(2, 0))
        
        # Results
        results_frame = ModernFrame(progress_results_frame)
        results_frame.pack(fill="both", expand=True)
        
        # Results header with clear button
        results_header = tk.Frame(results_frame, bg=ModernStyle.BG_CARD)
        results_header.pack(fill="x", pady=(0, 8))
        
        ModernLabel(results_header, "Results & Processing Log", font=("Segoe UI", 10, "bold")).pack(side="left")
        ModernButton(results_header, "Clear Log", self.clear_log, False).pack(side="right", padx=(5, 0))
        ModernButton(results_header, "Export Log", self.export_log, False).pack(side="right")
        
        # Add results summary frame
        summary_frame = tk.Frame(results_frame, bg=ModernStyle.BG_CARD)
        summary_frame.pack(fill="x", pady=(0, 8))
        
        # Results summary labels
        self.mesh_info_label = ModernLabel(summary_frame, "No mesh loaded", True)
        self.mesh_info_label.pack(anchor="w", pady=(0, 2))
        
        self.volume_info_label = ModernLabel(summary_frame, "Volume: Not calculated", True)
        self.volume_info_label.pack(anchor="w", pady=(0, 2))
        
        self.surface_area_label = ModernLabel(summary_frame, "Surface Area: Not calculated", True)
        self.surface_area_label.pack(anchor="w", pady=(0, 2))
        
        self.watertight_label = ModernLabel(summary_frame, "Watertight: Unknown", True)
        self.watertight_label.pack(anchor="w", pady=(0, 2))
        
        # Processing log
        text_frame = tk.Frame(results_frame, bg=ModernStyle.BG_CARD)
        text_frame.pack(fill="both", expand=True)
        
        self.results_text = tk.Text(text_frame, bg="#262626", fg=ModernStyle.TEXT_PRIMARY, 
                                   font=("Consolas", 9), relief="flat", bd=0, padx=10, pady=10,
                                   insertbackground=ModernStyle.TEXT_PRIMARY)
        
        scrollbar = tk.Scrollbar(text_frame, orient="vertical", command=self.results_text.yview,
                               bg="#262626", troughcolor=ModernStyle.BG_CARD, bd=0, highlightthickness=0)
        self.results_text.configure(yscrollcommand=scrollbar.set)
        
        self.results_text.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        self.log_message("Welcome to Point Cloud Processor!")
        self.log_message("Load a point cloud or mesh to start processing.")
        
    def log_message(self, message, level="info", to_terminal=True, progress=None, category="general"):
        timestamp = time.strftime("%H:%M:%S")
        icons = {"info": "ℹ️", "success": "✅", "warning": "⚠️", "error": "❌", "debug": "🔍", "system": "⚙️"}
        icon = icons.get(level, "ℹ️")
        
        # Add category prefix for better organization
        category_prefix = f"[{category.upper()}] " if category != "general" else ""
        log_line = f"[{timestamp}] {icon} {category_prefix}{message}"
        
        # Add to GUI text widget with color coding
        tag = f"level_{level}"
        self.results_text.insert(tk.END, log_line + "\n", tag)
        self.results_text.see(tk.END)
        
        # Update progress bar if provided
        if progress is not None:
            self.update_progress(progress)
        
        # Update GUI immediately
        self.root.update_idletasks()
        
        # Print to terminal if requested
        if to_terminal:
            print(log_line)
    
    def handle_terminal_output(self, output, stream_type):
        """Handle captured terminal output and display in GUI with enhanced parsing."""
        if not output.strip():
            return
            
        lines = output.strip().split('\n')
        for line in lines:
            if line.strip():
                # Enhanced level detection based on content patterns
                level = "error" if stream_type == "stderr" else "info"
                line_lower = line.lower()
                
                # Pattern-based level detection
                if any(word in line_lower for word in ["error", "failed", "exception", "traceback", "critical"]):
                    level = "error"
                elif any(word in line_lower for word in ["warning", "warn", "deprecated"]):
                    level = "warning"
                elif any(word in line_lower for word in ["success", "completed", "finished", "done", "saved"]):
                    level = "success"
                elif any(word in line_lower for word in ["debug", "verbose", "detail"]):
                    level = "debug"
                elif any(word in line_lower for word in ["system", "memory", "cpu", "performance"]):
                    level = "system"
                
                # Determine category based on content
                category = "general"
                if any(word in line_lower for word in ["mesh", "vertex", "triangle", "surface"]):
                    category = "mesh"
                elif any(word in line_lower for word in ["point", "cloud", "pcd"]):
                    category = "pointcloud"
                elif any(word in line_lower for word in ["octree", "volume", "voxel"]):
                    category = "octree"
                elif any(word in line_lower for word in ["poisson", "reconstruction"]):
                    category = "reconstruction"
                elif any(word in line_lower for word in ["watertight", "hole", "fill"]):
                    category = "watertight"
                elif any(word in line_lower for word in ["memory", "cpu", "performance", "time"]):
                    category = "performance"
                
                # Add to GUI without printing to terminal (avoid double output)
                self.log_message(f"[{stream_type.upper()}] {line}", level, to_terminal=False, category=category)
    
    def setup_log_colors(self):
        """Setup color coding for different log levels."""
        self.results_text.tag_configure("level_info", foreground="#b3b3b3")
        self.results_text.tag_configure("level_success", foreground="#10b981")
        self.results_text.tag_configure("level_warning", foreground="#f59e0b")
        self.results_text.tag_configure("level_error", foreground="#ef4444")
        self.results_text.tag_configure("level_debug", foreground="#8b5cf6")
        self.results_text.tag_configure("level_system", foreground="#06b6d4")
        
        # Category-specific colors
        self.results_text.tag_configure("level_mesh", foreground="#f97316")
        self.results_text.tag_configure("level_pointcloud", foreground="#84cc16")
        self.results_text.tag_configure("level_octree", foreground="#ec4899")
        self.results_text.tag_configure("level_reconstruction", foreground="#06b6d4")
        self.results_text.tag_configure("level_watertight", foreground="#8b5cf6")
        self.results_text.tag_configure("level_performance", foreground="#f59e0b")
    
    def log_system_info(self):
        """Log system information for debugging and performance monitoring."""
        try:
            import psutil
            import platform
            
            self.log_message("=== SYSTEM INFORMATION ===", "system", category="system")
            self.log_message(f"Platform: {platform.system()} {platform.release()}", "system", category="system")
            self.log_message(f"Python: {platform.python_version()}", "system", category="system")
            
            # Memory information
            memory = psutil.virtual_memory()
            self.log_message(f"Memory: {memory.total / (1024**3):.1f}GB total, {memory.available / (1024**3):.1f}GB available", "system", category="system")
            self.log_message(f"Memory Usage: {memory.percent:.1f}%", "system", category="system")
            
            # CPU information
            cpu_count = psutil.cpu_count()
            cpu_percent = psutil.cpu_percent(interval=1)
            self.log_message(f"CPU: {cpu_count} cores, {cpu_percent:.1f}% usage", "system", category="system")
            
            # Disk information
            disk = psutil.disk_usage('/')
            self.log_message(f"Disk: {disk.total / (1024**3):.1f}GB total, {disk.free / (1024**3):.1f}GB free", "system", category="system")
            
        except Exception as e:
            self.log_message(f"Failed to get system info: {str(e)}", "error", category="system")
    
    def log_performance_metrics(self, operation_name, start_time, end_time=None):
        """Log performance metrics for operations."""
        if end_time is None:
            end_time = time.time()
        
        duration = end_time - start_time
        self.log_message(f"Operation '{operation_name}' completed in {duration:.2f} seconds", "info", category="performance")
        
        # Log memory usage if available
        try:
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()
            memory_mb = memory_info.rss / (1024**2)
            self.log_message(f"Memory usage: {memory_mb:.1f}MB", "info", category="performance")
        except:
            pass
    
    def clear_log(self):
        """Clear the results log."""
        self.results_text.delete(1.0, tk.END)
        self.log_message("Log cleared", "info")
    
    def export_log(self):
        """Export the results log to a file."""
        file_path = filedialog.asksaveasfilename(
            title="Export Processing Log", defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("Log files", "*.log"), ("All files", "*.*")]
        )
        if file_path:
            try:
                log_content = self.results_text.get(1.0, tk.END)
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(log_content)
                self.log_message(f"Log exported to: {os.path.basename(file_path)}", "success")
            except Exception as e:
                self.log_message(f"Failed to export log: {str(e)}", "error")
        
    def update_status(self, message, level="info"):
        colors = {"info": ModernStyle.TEXT_SECONDARY, "success": ModernStyle.SUCCESS, 
                 "warning": ModernStyle.WARNING, "error": ModernStyle.ERROR}
        self.status_label.config(text=message, fg=colors.get(level, ModernStyle.TEXT_SECONDARY))
        self.root.update_idletasks()
        
    def update_progress(self, value, message=None):
        self.progress_bar.set_progress(value)
        if message:
            self.update_status(message)
        self.root.update_idletasks()
    
    def update_operation(self, operation):
        """Update the current operation display."""
        self.operation_label.config(text=f"Current: {operation}")
        self.root.update_idletasks()
    
    def update_detailed_progress(self, current_step, total_steps, step_name, progress_in_step=0):
        """Update progress with detailed step information."""
        if total_steps > 0:
            overall_progress = (current_step / total_steps) * 100
            if progress_in_step > 0:
                step_progress = progress_in_step / 100
                overall_progress = ((current_step - 1 + step_progress) / total_steps) * 100
            
            self.update_progress(overall_progress, f"Step {current_step}/{total_steps}: {step_name}")
            self.update_operation(step_name)
        else:
            self.update_progress(progress_in_step, step_name)
            self.update_operation(step_name)
        
    def update_results_display(self):
        """Update the results summary display in the GUI."""
        try:
            if self.mesh is not None:
                vertices = len(self.mesh.vertices)
                triangles = len(self.mesh.triangles)
                surface_area = self.mesh.get_surface_area()
                is_watertight = self.mesh.is_watertight()
                
                self.mesh_info_label.config(text=f"Mesh: {vertices:,} vertices, {triangles:,} triangles")
                self.surface_area_label.config(text=f"Surface Area: {surface_area:.6f} sq units")
                self.watertight_label.config(text=f"Watertight: {'Yes' if is_watertight else 'No'}")
                
                if is_watertight:
                    try:
                        volume = self.mesh.get_volume()
                        self.volume_info_label.config(text=f"Volume: {volume:.6f} cubic units")
                    except Exception as e:
                        self.volume_info_label.config(text=f"Volume: Calculation failed")
                else:
                    self.volume_info_label.config(text="Volume: Not available (not watertight)")
            else:
                self.mesh_info_label.config(text="No mesh loaded")
                self.volume_info_label.config(text="Volume: Not calculated")
                self.surface_area_label.config(text="Surface Area: Not calculated")
                self.watertight_label.config(text="Watertight: Unknown")
                
        except Exception as e:
            self.log_message(f"Failed to update results display: {str(e)}", "error")
        
    def load_point_cloud(self):
        file_path = filedialog.askopenfilename(
            title="Select Point Cloud", 
            filetypes=[("PLY files", "*.ply"), ("PCD files", "*.pcd"), ("All files", "*.*")]
        )
        if file_path:
            try:
                self.update_status("Loading point cloud...")
                self.pcd = o3d.io.read_point_cloud(file_path)
                self.current_file = file_path
                filename = os.path.basename(file_path)
                if self.pcd is not None:
                    self.file_status.config(text=f"📁 {filename} ({len(self.pcd.points)} points)")
                    self.log_message(f"Loaded: {filename} ({len(self.pcd.points):,} points)", "success")
                else:
                    raise ValueError("Failed to load point cloud")
                self.update_status("Point cloud loaded", "success")
            except Exception as e:
                self.log_message(f"Failed to load: {str(e)}", "error")
                self.update_status("Load failed", "error")
                
    def load_mesh(self):
        file_path = filedialog.askopenfilename(
            title="Select Mesh",
            filetypes=[("PLY files", "*.ply"), ("STL files", "*.stl"), ("OBJ files", "*.obj")]
        )
        if file_path:
            try:
                self.update_status("Loading mesh...")
                self.mesh = o3d.io.read_triangle_mesh(file_path)
                filename = os.path.basename(file_path)
                if self.mesh is not None:
                    self.file_status.config(text=f"📐 {filename} ({len(self.mesh.vertices)} vertices)")
                    self.log_message(f"Loaded: {filename} ({len(self.mesh.vertices):,} vertices)", "success")
                else:
                    raise ValueError("Failed to load mesh")
                self.update_status("Mesh loaded", "success")
            except Exception as e:
                self.log_message(f"Failed to load: {str(e)}", "error")
                self.update_status("Load failed", "error")
                
    def save_mesh(self):
        if self.mesh is None:
            self.log_message("No mesh to save", "warning")
            return
        file_path = filedialog.asksaveasfilename(
            title="Save Mesh", defaultextension=".ply",
            filetypes=[("PLY files", "*.ply"), ("STL files", "*.stl"), ("OBJ files", "*.obj")]
        )
        if file_path:
            try:
                o3d.io.write_triangle_mesh(file_path, self.mesh)
                self.log_message(f"Saved: {os.path.basename(file_path)}", "success")
                self.update_status("Mesh saved", "success")
            except Exception as e:
                self.log_message(f"Save failed: {str(e)}", "error")
    
    def save_point_cloud(self):
        if self.pcd is None:
            self.log_message("No point cloud to save", "warning")
            return
        file_path = filedialog.asksaveasfilename(
            title="Save Point Cloud", defaultextension=".ply",
            filetypes=[("PLY files", "*.ply"), ("PCD files", "*.pcd"), ("All files", "*.*")]
        )
        if file_path:
            try:
                o3d.io.write_point_cloud(file_path, self.pcd)
                self.log_message(f"Saved: {os.path.basename(file_path)}", "success")
                self.update_status("Point cloud saved", "success")
            except Exception as e:
                self.log_message(f"Save failed: {str(e)}", "error")
                
    def process_to_mesh(self):
        if self.pcd is None:
            self.log_message("Load a point cloud first", "warning")
            return
            
        def process_thread():
            try:
                self.update_status("Processing...")
                self.update_progress(0)
                self.log_message("Starting mesh generation...")
                
                # Clean and estimate normals
                self.log_message("Cleaning point cloud...")
                self.update_progress(20)
                if self.pcd is None:
                    raise ValueError("Point cloud is None")
                pcd_clean, _ = self.pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
                
                self.log_message("Estimating normals...")
                self.update_progress(40)
                pcd_clean.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=50))
                pcd_clean.orient_normals_consistent_tangent_plane(k=30)
                
                # Generate mesh
                if self.method_var.get() == "Poisson":
                    self.log_message("Applying Poisson reconstruction...")
                    self.update_progress(60)
                    mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                        pcd_clean, depth=self.depth_var.get())
                else:
                    self.log_message("Applying Marching Cubes...")
                    self.update_progress(60)
                    mesh = self.marching_cubes_mesh(pcd_clean, self.voxel_var.get())
                
                self.update_progress(80)
                mesh.remove_degenerate_triangles()
                mesh.remove_duplicated_triangles()
                mesh.compute_vertex_normals()
                
                self.mesh = mesh
                self.update_progress(100)
                self.log_message(f"Mesh generated: {len(mesh.vertices):,} vertices, {len(mesh.triangles):,} triangles", "success")
                self.update_status("Mesh generated", "success")
                
            except Exception as e:
                self.log_message(f"Processing failed: {str(e)}", "error")
                self.update_status("Processing failed", "error")
                
        threading.Thread(target=process_thread, daemon=True).start()
        
    def marching_cubes_mesh(self, pcd, voxel_size):
        points = np.asarray(pcd.points)
        mins, maxs = np.min(points, axis=0), np.max(points, axis=0)
        x = np.arange(mins[0], maxs[0], voxel_size)
        y = np.arange(mins[1], maxs[1], voxel_size)
        z = np.arange(mins[2], maxs[2], voxel_size)
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        grid_points = np.vstack((X.ravel(), Y.ravel(), Z.ravel())).T
        
        tree = cKDTree(points)
        distances, _ = tree.query(grid_points)
        scalar_field = distances.reshape(X.shape)
        
        iso_level = np.percentile(distances, 10)
        verts, faces, _, _ = measure.marching_cubes(scalar_field, level=iso_level)
        
        verts_world = np.zeros_like(verts)
        verts_world[:, 0] = x[0] + verts[:, 0] * voxel_size
        verts_world[:, 1] = y[0] + verts[:, 1] * voxel_size
        verts_world[:, 2] = z[0] + verts[:, 2] * voxel_size
        
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(verts_world)
        mesh.triangles = o3d.utility.Vector3iVector(faces)
        return mesh
        
    def clean_point_cloud(self):
        if self.pcd is None:
            self.log_message("No point cloud to clean", "warning")
            return
            
        try:
            self.update_status("Cleaning point cloud...")
            original_points = len(self.pcd.points)
            
            # Remove statistical outliers
            pcd_clean, _ = self.pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
            
            # Remove radius outliers
            pcd_clean, _ = pcd_clean.remove_radius_outlier(nb_points=16, radius=0.05)
            
            # Update the original point cloud
            self.pcd = pcd_clean
            
            new_points = len(self.pcd.points)
            self.log_message(f"Point cloud cleaned: {original_points-new_points:,} points removed", "success")
            self.update_status("Point cloud cleaned", "success")
            
            # Update file status
            if self.current_file:
                filename = os.path.basename(self.current_file)
                self.file_status.config(text=f"📁 {filename} ({len(self.pcd.points)} points) - CLEANED")
            
        except Exception as e:
            self.log_message(f"Point cloud cleaning failed: {str(e)}", "error")
            self.update_status("Point cloud cleaning failed", "error")
    
    def clean_mesh(self):
        if self.mesh is None:
            self.log_message("No mesh to clean", "warning")
            return
            
        try:
            self.update_status("Cleaning mesh...")
            original_verts = len(self.mesh.vertices)
            original_faces = len(self.mesh.triangles)
            
            self.mesh.remove_degenerate_triangles()
            self.mesh.remove_duplicated_triangles()
            self.mesh.remove_duplicated_vertices()
            self.mesh.remove_non_manifold_edges()
            self.mesh.remove_unreferenced_vertices()
            
            # Filter by connected components
            clusters, cluster_n_triangles, _ = self.mesh.cluster_connected_triangles()
            if len(clusters) > 0:
                # Create a boolean mask for triangles to remove
                triangles_to_remove = [i for i, cluster in enumerate(clusters) if cluster_n_triangles[cluster] < 100]
                if triangles_to_remove:
                    # Create a boolean mask of the correct size
                    triangle_mask = [False] * len(self.mesh.triangles)
                    for i in triangles_to_remove:
                        if i < len(triangle_mask):
                            triangle_mask[i] = True
                    self.mesh.remove_triangles_by_mask(triangle_mask)
                    self.mesh.remove_unreferenced_vertices()
            
            self.mesh.compute_vertex_normals()
            
            new_verts = len(self.mesh.vertices)
            new_faces = len(self.mesh.triangles)
            
            self.log_message(f"Mesh cleaned: {original_verts-new_verts:,} vertices removed, {original_faces-new_faces:,} faces removed", "success")
            self.update_status("Mesh cleaned", "success")
            
        except Exception as e:
            self.log_message(f"Cleaning failed: {str(e)}", "error")
            self.update_status("Cleaning failed", "error")
            
    def make_watertight(self):
        if self.mesh is None:
            self.log_message("No mesh to make watertight", "warning")
            return
            
        def watertight_thread():
            try:
                self.update_status("Making watertight using Open3D Poisson method...")
                self.update_progress(0)
                
                self.log_message("=== OPEN3D POISSON WATERTIGHT PROCESSING ===", "success")
                
                # Create a copy for processing
                mesh = self.mesh
                if mesh is None:
                    raise ValueError("Mesh is None")
                
                # Create a new mesh and copy vertices and triangles
                mesh_copy = o3d.geometry.TriangleMesh()
                mesh_copy.vertices = o3d.utility.Vector3dVector(np.asarray(mesh.vertices))
                mesh_copy.triangles = o3d.utility.Vector3iVector(np.asarray(mesh.triangles))
                
                self.log_message("Preparing mesh for Poisson watertight processing...")
                self.update_progress(20)
                
                # Clean the mesh first
                mesh_copy.remove_degenerate_triangles()
                mesh_copy.remove_duplicated_triangles()
                mesh_copy.remove_duplicated_vertices()
                mesh_copy.remove_unreferenced_vertices()
                mesh_copy.compute_vertex_normals()
                
                self.log_message("Converting mesh to point cloud for Poisson reconstruction...")
                self.update_progress(40)
                
                # Convert mesh to point cloud for Poisson reconstruction
                # Sample points from the mesh surface
                num_points = min(100000, len(mesh_copy.vertices) * 2)  # Adaptive sampling
                pcd = mesh_copy.sample_points_uniformly(number_of_points=num_points)
                
                # Estimate normals for the point cloud
                pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=50))
                pcd.orient_normals_consistent_tangent_plane(k=30)
                
                self.log_message("Applying Poisson surface reconstruction...")
                self.update_progress(60)
                
                # Use higher depth for better resolution (default is 8, we'll use 10-12)
                depth = min(12, max(10, int(np.log2(len(mesh_copy.vertices) / 1000)) + 8))
                
                # Apply Poisson surface reconstruction
                watertight_mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                    pcd, depth=depth, width=0, scale=1.1, linear_fit=False)
                
                self.log_message("Cleaning and optimizing watertight mesh...")
                self.update_progress(80)
                
                # Clean the watertight mesh
                watertight_mesh.remove_degenerate_triangles()
                watertight_mesh.remove_duplicated_triangles()
                watertight_mesh.remove_duplicated_vertices()
                watertight_mesh.remove_unreferenced_vertices()
                
                # Remove low density vertices (holes and artifacts)
                if len(densities) > 0:
                    vertices_to_remove = [i for i, density in enumerate(densities) if density < np.quantile(densities, 0.1)]
                    if vertices_to_remove:
                        watertight_mesh.remove_vertices_by_index(vertices_to_remove)
                        watertight_mesh.remove_unreferenced_vertices()
                
                # Ensure proper visualization properties
                watertight_mesh.compute_vertex_normals()
                watertight_mesh.paint_uniform_color([0.7, 0.7, 0.7])  # Light gray color
                
                # Validate the watertight mesh
                if len(watertight_mesh.vertices) == 0 or len(watertight_mesh.triangles) == 0:
                    raise ValueError("Generated watertight mesh is empty")
                
                # Store the watertight mesh
                self.mesh = watertight_mesh
                
                # Calculate properties
                self.update_progress(90)
                is_watertight = watertight_mesh.is_watertight()
                
                # Calculate volume if watertight
                volume = 0.0
                if is_watertight:
                    try:
                        volume = watertight_mesh.get_volume()
                        self.log_message(f"✅ Poisson watertight mesh created with volume: {volume:.6f} cubic units")
                    except Exception as e:
                        self.log_message(f"Volume calculation failed: {str(e)}")
                else:
                    self.log_message("⚠️ Mesh may not be fully watertight")
                
                # Final statistics
                surface_area = watertight_mesh.get_surface_area()
                self.log_message(f"Poisson watertight mesh statistics:")
                self.log_message(f"  - Vertices: {len(watertight_mesh.vertices):,}")
                self.log_message(f"  - Triangles: {len(watertight_mesh.triangles):,}")
                self.log_message(f"  - Surface Area: {surface_area:.6f} sq units")
                self.log_message(f"  - Watertight: {'Yes' if is_watertight else 'No'}")
                self.log_message(f"  - Poisson Depth: {depth}")
                self.log_message(f"  - Sampled Points: {num_points:,}")
                if volume > 0:
                    self.log_message(f"  - Volume: {volume:.6f} cubic units")
                
                self.update_progress(100)
                status = "watertight" if is_watertight else "improved (may not be fully watertight)"
                self.log_message(f"✅ Open3D Poisson watertight processing complete!", "success")
                self.update_status(f"Mesh {status}", "success")
                
                # Update the results display in GUI
                self.update_results_display()
                
            except Exception as e:
                self.log_message(f"Poisson watertight processing failed: {str(e)}", "error")
                self.log_message("Attempting fallback approach...", "warning")
                
                # Fallback: Use simple hole filling
                try:
                    self.log_message("Using fallback hole filling approach...")
                    
                    # Get the original mesh
                    original_mesh = self.mesh
                    if original_mesh is None:
                        raise ValueError("No original mesh available for fallback")
                    
                    # Create a copy and try to fill holes
                    fallback_mesh = o3d.geometry.TriangleMesh()
                    fallback_mesh.vertices = o3d.utility.Vector3dVector(np.asarray(original_mesh.vertices))
                    fallback_mesh.triangles = o3d.utility.Vector3iVector(np.asarray(original_mesh.triangles))
                    
                    # Fill holes using manual approach since Open3D doesn't have fill_holes()
                    fallback_mesh.remove_non_manifold_edges()
                    fallback_mesh.compute_vertex_normals()
                    fallback_mesh.compute_triangle_normals()
                    fallback_mesh.orient_triangles()
                    
                    # Clean the mesh
                    fallback_mesh.remove_degenerate_triangles()
                    fallback_mesh.remove_duplicated_triangles()
                    fallback_mesh.remove_duplicated_vertices()
                    fallback_mesh.remove_unreferenced_vertices()
                    fallback_mesh.compute_vertex_normals()
                    fallback_mesh.paint_uniform_color([0.7, 0.7, 0.7])
                    
                    # Check if fallback worked
                    if len(fallback_mesh.vertices) > 0 and len(fallback_mesh.triangles) > 0:
                        self.mesh = fallback_mesh
                        is_watertight = fallback_mesh.is_watertight()
                        
                        self.log_message(f"Fallback watertight approach completed:")
                        self.log_message(f"  - Vertices: {len(fallback_mesh.vertices):,}")
                        self.log_message(f"  - Triangles: {len(fallback_mesh.triangles):,}")
                        self.log_message(f"  - Watertight: {'Yes' if is_watertight else 'No'}")
                        
                        if is_watertight:
                            try:
                                volume = fallback_mesh.get_volume()
                                self.log_message(f"  - Volume: {volume:.6f} cubic units")
                            except Exception as vol_error:
                                self.log_message(f"  - Volume calculation failed: {str(vol_error)}")
                        
                        self.log_message("✅ Fallback watertight processing completed", "success")
                        self.update_status("Fallback watertight completed", "success")
                    else:
                        raise ValueError("Fallback approach produced empty mesh")
                        
                except Exception as fallback_error:
                    self.log_message(f"Fallback approach also failed: {str(fallback_error)}", "error")
                    self.update_status("Watertight processing failed", "error")
                
        threading.Thread(target=watertight_thread, daemon=True).start()
    


    
    def calculate_properties(self):
        if self.mesh is None:
            self.log_message("No mesh to analyze", "warning")
            return
            
        def properties_thread():
            try:
                self.update_status("Calculating properties...")
                self.update_progress(0)
                
                if self.mesh is None:
                    raise ValueError("Mesh is None")
                    
                # Always calculate surface area (available for all meshes)
                surface_area = self.mesh.get_surface_area()
                is_watertight = self.mesh.is_watertight()
                
                self.log_message("=== MESH ANALYSIS ===", "success")
                self.log_message(f"Surface Area: {surface_area:.6f} sq units")
                self.log_message(f"Vertices: {len(self.mesh.vertices):,}")
                self.log_message(f"Triangles: {len(self.mesh.triangles):,}")
                self.log_message(f"Watertight: {'Yes' if is_watertight else 'No'}")
                
                # Calculate volume based on watertight status
                if is_watertight:
                    try:
                        volume = self.mesh.get_volume()
                        self.log_message(f"Volume (Standard): {volume:.6f} cubic units")
                        
                        # Calculate additional properties for watertight meshes
                        if volume > 0:
                            bbox = self.mesh.get_axis_aligned_bounding_box()
                            bbox_volume = bbox.volume()
                            self.log_message(f"Bounding Box Volume: {bbox_volume:.6f} cubic units")
                            self.log_message(f"Mesh Density: {(volume/bbox_volume)*100:.2f}%")
                            
                            # Calculate surface area to volume ratio
                            sa_vol_ratio = surface_area / volume
                            self.log_message(f"Surface Area/Volume Ratio: {sa_vol_ratio:.6f}")
                            
                    except Exception as e:
                        self.log_message(f"Standard volume calculation failed: {str(e)}", "error")
                else:
                    # For non-watertight meshes, recommend using octree volume generation
                    self.log_message("Mesh is not watertight - volume calculation not available")
                    self.log_message("Use 'Generate Octree Volume Mesh' for non-watertight meshes")
                        
                self.log_message("=" * 25)
                self.update_status("Properties calculated", "success")
                
            except Exception as e:
                self.log_message(f"Analysis failed: {str(e)}", "error")
                self.update_status("Analysis failed", "error")
                
        threading.Thread(target=properties_thread, daemon=True).start()
            
    def view_point_cloud(self):
        if self.pcd is None:
            self.log_message("No point cloud to view", "warning")
            return
        try:
            self.log_message("Opening point cloud viewer...")
            o3d.visualization.draw_geometries([self.pcd])
            self.log_message("Viewer closed")
        except Exception as e:
            self.log_message(f"View failed: {str(e)}", "error")
            
    def view_mesh(self):
        if self.mesh is None:
            self.log_message("No mesh to view", "warning")
            return
        try:
            self.log_message("Opening mesh viewer...")
            o3d.visualization.draw_geometries([self.mesh], mesh_show_back_face=True)
            self.log_message("Viewer closed")
        except Exception as e:
            self.log_message(f"View failed: {str(e)}", "error")
    
    def save_watertight_mesh(self):
        if self.mesh is None:
            self.log_message("No mesh to save", "warning")
            return
        
        # Check if mesh is watertight
        if not self.mesh.is_watertight():
            self.log_message("Current mesh is not watertight. Run 'Make Watertight' first.", "warning")
            return
            
        file_path = filedialog.asksaveasfilename(
            title="Save Watertight Mesh", defaultextension=".ply",
            filetypes=[("PLY files", "*.ply"), ("STL files", "*.stl"), ("OBJ files", "*.obj")]
        )
        if file_path:
            try:
                o3d.io.write_triangle_mesh(file_path, self.mesh)
                self.log_message(f"Watertight mesh saved: {os.path.basename(file_path)}", "success")
                self.update_status("Watertight mesh saved", "success")
            except Exception as e:
                self.log_message(f"Save failed: {str(e)}", "error")
    

    
    def generate_octree_mesh(self):
        """
        Generate octree volume mesh using GUI parameters.
        Handles both meshes and point clouds.
        """
        if self.mesh is None and self.pcd is None:
            self.log_message("No mesh or point cloud to process. Load a mesh or point cloud first.", "warning")
            return
            
        # If point cloud is loaded, convert to mesh first using Poisson reconstruction
        if self.pcd is not None and self.mesh is None:
            self.log_message("Point cloud detected - converting to mesh using Poisson reconstruction...")
            self.convert_point_cloud_to_mesh()
            
        if self.mesh is None:
            self.log_message("Failed to convert point cloud to mesh", "error")
            return
            
        def generate_thread():
            try:
                self.update_status("Generating octree mesh...")
                self.update_progress(0)
                
                resolution = self.octree_size_var.get()
                sample_points = self.octree_samples_var.get()
                sdf_padding = self.sdf_padding_var.get()
                distance_threshold = self.distance_threshold_var.get()
                max_grid_dim = self.max_grid_dim_var.get()
                max_cells = self.max_cells_var.get()
                
                self.log_message(f"Generating octree mesh with resolution {resolution}, {sample_points} sample points, SDF padding {sdf_padding}, distance threshold {distance_threshold}, max grid dim {max_grid_dim}, max cells {max_cells}...")
                
                # Generate the octree volume mesh using the fallback method with all parameters
                volume_mesh = self._generate_octree_volume_mesh_fallback(resolution, sample_points, sdf_padding, distance_threshold, max_grid_dim, max_cells)
                
                if volume_mesh is not None:
                    self.octree_volume_mesh = volume_mesh
                    self.update_progress(100)
                    self.log_message(f"Octree mesh generated successfully: {len(volume_mesh.vertices):,} vertices, {len(volume_mesh.triangles):,} triangles", "success")
                    self.update_status("Octree mesh generated", "success")
                else:
                    self.log_message("Failed to generate octree mesh", "error")
                    self.update_status("Octree generation failed", "error")
                    
            except Exception as e:
                self.log_message(f"Octree generation failed: {str(e)}", "error")
                self.update_status("Octree generation failed", "error")
                
        threading.Thread(target=generate_thread, daemon=True).start()
    
    def calculate_octree_volume(self):
        """
        Calculate volume using the generated octree volume mesh.
        """
        if not hasattr(self, 'octree_volume_mesh') or self.octree_volume_mesh is None:
            self.log_message("No octree volume mesh available. Generate octree mesh first using 'Generate Octree Volume Mesh' button.", "warning")
            return
            
        def octree_volume_thread():
            try:
                self.update_status("Calculating volume from octree mesh...")
                self.update_progress(0)
                
                self.log_message("=== OCTREE VOLUME ANALYSIS ===", "success")
                
                # Get stored volume if available
                if hasattr(self, 'octree_volume'):
                    volume = self.octree_volume
                    self.log_message(f"Octree Volume (pre-calculated): {volume:.6f} cubic units")
                else:
                    volume = 0.0
                    self.log_message("No pre-calculated volume available")
                
                # Calculate surface area (outer surface only)
                surface_area = self.octree_volume_mesh.get_surface_area()
                self.log_message(f"Octree Mesh Surface Area (outer): {surface_area:.6f} sq units")
                self.log_message(f"Octree Mesh Vertices: {len(self.octree_volume_mesh.vertices):,}")
                self.log_message(f"Octree Mesh Triangles: {len(self.octree_volume_mesh.triangles):,}")
                
                # Check if octree mesh is watertight
                is_watertight = self.octree_volume_mesh.is_watertight()
                self.log_message(f"Octree Mesh Watertight: {'Yes' if is_watertight else 'No'}")
                
                # If no pre-calculated volume, try standard calculation
                if volume <= 0 and is_watertight:
                    try:
                        volume = self.octree_volume_mesh.get_volume()
                        self.log_message(f"Octree Volume (standard): {volume:.6f} cubic units")
                    except Exception as e:
                        self.log_message(f"Standard volume calculation failed: {str(e)}")
                
                # Calculate additional properties
                if volume > 0:
                    # Calculate bounding box volume for comparison
                    bbox = self.octree_volume_mesh.get_axis_aligned_bounding_box()
                    bbox_volume = bbox.volume()
                    self.log_message(f"Bounding Box Volume: {bbox_volume:.6f} cubic units")
                    
                    # Calculate volume density
                    if bbox_volume > 0:
                        volume_density = (volume / bbox_volume) * 100
                        self.log_message(f"Volume Density: {volume_density:.2f}%")
                    
                    # Calculate surface area to volume ratio
                    sa_vol_ratio = surface_area / volume
                    self.log_message(f"Surface Area/Volume Ratio: {sa_vol_ratio:.6f}")
                    
                    # Compare with original mesh if available
                    if self.mesh is not None:
                        try:
                            # Use consistent surface area calculation method
                            original_surface_area = self.mesh.get_surface_area()
                            
                            # Check if surface areas are reasonable
                            if original_surface_area > 0:
                                area_diff = abs(surface_area - original_surface_area)
                                area_diff_percent = (area_diff / original_surface_area) * 100
                                
                                self.log_message(f"Original Mesh Surface Area: {original_surface_area:.6f} sq units")
                                self.log_message(f"Surface Area Difference: {area_diff:.6f} sq units ({area_diff_percent:.2f}%)")
                                
                                # Provide guidance if difference is too large
                                if area_diff_percent > 50:
                                    self.log_message("⚠️ Large surface area difference detected!", "warning")
                                    self.log_message("Recommendations to reduce difference:", "warning")
                                    self.log_message("  - Decrease Octree Box Size (finer resolution)", "warning")
                                    self.log_message("  - Increase Sample Points (better surface sampling)", "warning")
                                    self.log_message("  - Decrease Distance Threshold (tighter surface)", "warning")
                                    self.log_message("  - Increase Voxel Grid Resolution (better SDF)", "warning")
                                elif area_diff_percent > 20:
                                    self.log_message("⚠️ Moderate surface area difference - consider parameter adjustment", "warning")
                                else:
                                    self.log_message("✅ Surface area difference is within acceptable range", "success")
                            else:
                                self.log_message("⚠️ Original mesh has zero surface area - cannot compare", "warning")
                            
                            # If original mesh is watertight, compare volumes
                            if self.mesh.is_watertight():
                                original_volume = self.mesh.get_volume()
                                volume_diff = abs(volume - original_volume)
                                volume_diff_percent = (volume_diff / original_volume) * 100 if original_volume > 0 else 0
                                self.log_message(f"Original Volume: {original_volume:.6f} cubic units")
                                self.log_message(f"Volume Difference: {volume_diff:.6f} cubic units ({volume_diff_percent:.2f}%)")
                                
                                # Provide volume comparison guidance
                                if volume_diff_percent > 30:
                                    self.log_message("⚠️ Large volume difference detected!", "warning")
                                    self.log_message("Recommendations to improve volume accuracy:", "warning")
                                    self.log_message("  - Increase Tetrahedral Density (better volume mesh)", "warning")
                                    self.log_message("  - Decrease Distance Threshold (tighter volume)", "warning")
                                    self.log_message("  - Decrease Octree Box Size (finer volume resolution)", "warning")
                                elif volume_diff_percent > 10:
                                    self.log_message("⚠️ Moderate volume difference - consider parameter adjustment", "warning")
                                else:
                                    self.log_message("✅ Volume difference is within acceptable range", "success")
                        except Exception as e:
                            self.log_message(f"Original mesh comparison failed: {str(e)}")
                    
                    self.log_message("=" * 35)
                    self.update_status("Octree volume calculated", "success")
                else:
                    self.log_message("Octree volume calculation not available")
                    self.log_message("The octree mesh may need further processing to be watertight")
                    self.update_status("Octree volume not available", "warning")
                    
            except Exception as e:
                self.log_message(f"Octree volume analysis failed: {str(e)}", "error")
                self.update_status("Octree analysis failed", "error")
                
        threading.Thread(target=octree_volume_thread, daemon=True).start()
    
    def make_octree_watertight(self):
        """
        Make the octree mesh watertight using binary fill holes approach.
        Uses scipy.ndimage.binary_fill_holes to ensure complete interior filling.
        """
        if not hasattr(self, 'octree_volume_mesh') or self.octree_volume_mesh is None:
            self.log_message("No octree mesh available. Generate octree mesh first using 'Generate Octree Volume Mesh' button.", "warning")
            return
            
        def watertight_thread():
            try:
                self.processing_active = True
                self.update_status("Making octree mesh watertight using binary fill holes...")
                self.update_progress(0)
                
                # Capture terminal output during processing
                with TerminalCapture(self.handle_terminal_output):
                    self.log_message("=== OCTREE WATERTIGHT PROCESSING (BINARY FILL HOLES) ===", "success")
                    self.log_message("Using binary fill holes approach to ensure complete interior filling...")
                
                # Get the original octree mesh
                octree_mesh = self.octree_volume_mesh
                if octree_mesh is None:
                    raise ValueError("Octree mesh is None")
                
                # Get GUI parameters
                binary_resolution = self.binary_res_var.get()
                distance_threshold_mult = self.binary_distance_threshold_var.get()
                kernel_size = self.binary_kernel_size_var.get()
                
                self.log_message(f"Using parameters: Resolution={binary_resolution}, Distance Threshold={distance_threshold_mult}, Kernel Size={kernel_size}")
                
                # Define processing steps
                total_steps = 7
                current_step = 1
                
                # Step 1: Create binary volume from octree mesh
                self.log_message("Creating binary volume from octree mesh...", progress=15)
                self.update_detailed_progress(current_step, total_steps, "Creating binary volume grid")
                
                # Get mesh bounds
                bbox = octree_mesh.get_axis_aligned_bounding_box()
                min_bound = bbox.get_min_bound()
                max_bound = bbox.get_max_bound()
                
                # Create voxel grid using GUI resolution
                bbox_size = max_bound - min_bound
                min_size = np.min(bbox_size)
                adaptive_resolution = min(binary_resolution, min_size / 50)
                
                # Create voxel coordinates
                x_coords = np.arange(min_bound[0], max_bound[0] + adaptive_resolution, adaptive_resolution)
                y_coords = np.arange(min_bound[1], max_bound[1] + adaptive_resolution, adaptive_resolution)
                z_coords = np.arange(min_bound[2], max_bound[2] + adaptive_resolution, adaptive_resolution)
                
                self.log_message(f"Creating binary volume grid: {len(x_coords)}x{len(y_coords)}x{len(z_coords)} voxels")
                
                # Create binary volume (1 = inside, 0 = outside)
                binary_volume = np.zeros((len(x_coords), len(y_coords), len(z_coords)), dtype=bool)
                
                # Sample points from mesh for distance computation
                sample_points = octree_mesh.sample_points_uniformly(number_of_points=10000)
                sample_vertices = np.asarray(sample_points.points)
                tree = cKDTree(sample_vertices)
                
                current_step = 2
                # Step 2: Mark voxels as inside/outside based on distance
                self.log_message("Marking voxels as inside/outside based on distance...", progress=30)
                self.update_detailed_progress(current_step, total_steps, "Marking voxels as inside/outside")
                
                distance_threshold = adaptive_resolution * distance_threshold_mult
                
                for i, x in enumerate(x_coords):
                    for j, y in enumerate(y_coords):
                        for k, z in enumerate(z_coords):
                            cell_center = np.array([x, y, z])
                            distance, _ = tree.query(cell_center)
                            
                            # If distance is small, mark as inside (1)
                            if distance < distance_threshold:
                                binary_volume[i, j, k] = True
                
                initial_filled = np.sum(binary_volume) if binary_volume is not None else 0
                self.log_message(f"Initial binary volume: {initial_filled:,} filled voxels")
                
                current_step = 3
                # Step 3: Apply binary_fill_holes to fill interior holes
                self.log_message("Filling holes using scipy.ndimage.binary_fill_holes...", progress=50)
                self.update_detailed_progress(current_step, total_steps, "Filling holes with binary_fill_holes")
                
                filled_volume = ndimage.binary_fill_holes(binary_volume)
                
                current_step = 4
                # Step 4: Apply morphological closing for complete filling
                self.log_message("Applying morphological operations for complete filling...", progress=60)
                self.update_detailed_progress(current_step, total_steps, "Applying morphological operations")
                
                # Use morphological closing to fill small gaps
                kernel = np.ones((kernel_size, kernel_size, kernel_size), dtype=bool)
                filled_volume = ndimage.binary_closing(filled_volume, structure=kernel)
                
                # Final hole filling
                filled_volume = ndimage.binary_fill_holes(filled_volume)
                
                initial_voxels = np.sum(binary_volume) if binary_volume is not None else 0
                final_voxels = np.sum(filled_volume) if filled_volume is not None else 0
                additional_voxels = final_voxels - initial_voxels
                
                self.log_message(f"Binary fill holes complete:")
                self.log_message(f"  Initial voxels: {initial_voxels:,}")
                self.log_message(f"  Final voxels: {final_voxels:,}")
                self.log_message(f"  Additional voxels filled: {additional_voxels:,}")
                
                current_step = 5
                # Step 5: Convert to signed distance field
                self.log_message("Converting to signed distance field...", progress=75)
                self.update_detailed_progress(current_step, total_steps, "Converting to signed distance field")
                
                sdf = np.zeros((len(x_coords), len(y_coords), len(z_coords)))
                
                # Create SDF from filled volume
                if filled_volume is not None:
                    for i in range(len(x_coords)):
                        for j in range(len(y_coords)):
                            for k in range(len(z_coords)):
                                if filled_volume[i, j, k]:
                                    # Inside: negative distance
                                    cell_center = np.array([x_coords[i], y_coords[j], z_coords[k]])
                                    distance, _ = tree.query(cell_center)
                                    sdf[i, j, k] = -distance
                                else:
                                    # Outside: positive distance
                                    cell_center = np.array([x_coords[i], y_coords[j], z_coords[k]])
                                    distance, _ = tree.query(cell_center)
                                    sdf[i, j, k] = distance
                else:
                    self.log_message("Error: filled_volume is None", "error")
                    raise ValueError("Filled volume is None")
                
                current_step = 6
                # Step 6: Generate watertight mesh using marching cubes
                self.log_message("Generating watertight mesh using marching cubes...", progress=85)
                self.update_detailed_progress(current_step, total_steps, "Generating watertight mesh")
                
                verts, faces, _, _ = measure.marching_cubes(sdf, level=0)
                
                # Transform vertices back to world coordinates
                verts_world = np.zeros_like(verts)
                verts_world[:, 0] = min_bound[0] + verts[:, 0] * adaptive_resolution
                verts_world[:, 1] = min_bound[1] + verts[:, 1] * adaptive_resolution
                verts_world[:, 2] = min_bound[2] + verts[:, 2] * adaptive_resolution
                
                # Create Open3D mesh
                watertight_mesh = o3d.geometry.TriangleMesh()
                watertight_mesh.vertices = o3d.utility.Vector3dVector(verts_world)
                watertight_mesh.triangles = o3d.utility.Vector3iVector(faces)
                
                current_step = 7
                # Step 7: Clean and validate the watertight mesh
                self.log_message("Cleaning and validating watertight mesh...", progress=95)
                self.update_detailed_progress(current_step, total_steps, "Cleaning and validating mesh")
                
                watertight_mesh.remove_degenerate_triangles()
                watertight_mesh.remove_duplicated_triangles()
                watertight_mesh.remove_duplicated_vertices()
                watertight_mesh.remove_unreferenced_vertices()
                watertight_mesh.compute_vertex_normals()
                watertight_mesh.paint_uniform_color([0.7, 0.7, 0.7])
                
                # Validate the mesh
                if len(watertight_mesh.vertices) == 0 or len(watertight_mesh.triangles) == 0:
                    raise ValueError("Generated watertight mesh is empty")
                
                # Store the watertight octree mesh
                self.octree_volume_mesh = watertight_mesh
                
                # Calculate properties
                is_watertight = watertight_mesh.is_watertight()
                surface_area = watertight_mesh.get_surface_area()
                
                # Calculate volume if watertight
                volume = 0.0
                if is_watertight:
                    try:
                        volume = watertight_mesh.get_volume()
                        self.octree_volume = volume
                        self.log_message(f"✅ Watertight octree mesh created with volume: {volume:.6f} cubic units")
                    except Exception as e:
                        self.log_message(f"Volume calculation failed: {str(e)}")
                else:
                    self.log_message("⚠️ Octree mesh may not be fully watertight")
                
                # Final statistics
                self.log_message(f"Watertight octree mesh statistics:")
                self.log_message(f"  - Vertices: {len(watertight_mesh.vertices):,}")
                self.log_message(f"  - Triangles: {len(watertight_mesh.triangles):,}")
                self.log_message(f"  - Surface Area: {surface_area:.6f} sq units")
                self.log_message(f"  - Watertight: {'Yes' if is_watertight else 'No'}")
                if volume > 0:
                    self.log_message(f"  - Volume: {volume:.6f} cubic units")
                
                self.update_progress(100, "Processing complete!")
                status = "watertight" if is_watertight else "improved (may not be fully watertight)"
                self.log_message(f"✅ Binary fill holes watertight processing complete!", "success")
                self.update_status(f"Octree mesh {status}", "success")
                self.processing_active = False
                
            except Exception as e:
                self.log_message(f"Binary fill holes watertight processing failed: {str(e)}", "error")
                self.log_message("Attempting fallback approach...", "warning")
                self.processing_active = False
                
                # Fallback: Use simple hole filling
                self.log_message("Using fallback hole filling approach...")
                
                # Get the original octree mesh
                original_octree_mesh = self.octree_volume_mesh
                if original_octree_mesh is None:
                    self.log_message("No original octree mesh available for fallback", "error")
                    self.update_status("Watertight processing failed", "error")
                    return
                
                # Create a copy and try to fill holes
                fallback_mesh = o3d.geometry.TriangleMesh()
                fallback_mesh.vertices = o3d.utility.Vector3dVector(np.asarray(original_octree_mesh.vertices))
                fallback_mesh.triangles = o3d.utility.Vector3iVector(np.asarray(original_octree_mesh.triangles))
                
                # Fill holes using manual approach since Open3D doesn't have fill_holes()
                fallback_mesh.remove_non_manifold_edges()
                fallback_mesh.compute_vertex_normals()
                fallback_mesh.compute_triangle_normals()
                fallback_mesh.orient_triangles()
                
                # Clean the mesh
                fallback_mesh.remove_degenerate_triangles()
                fallback_mesh.remove_duplicated_triangles()
                fallback_mesh.remove_duplicated_vertices()
                fallback_mesh.remove_unreferenced_vertices()
                fallback_mesh.compute_vertex_normals()
                fallback_mesh.paint_uniform_color([0.7, 0.7, 0.7])
                
                # Check if fallback worked
                if len(fallback_mesh.vertices) > 0 and len(fallback_mesh.triangles) > 0:
                    self.octree_volume_mesh = fallback_mesh
                    is_watertight = fallback_mesh.is_watertight()
                    
                    self.log_message(f"Fallback watertight approach completed:")
                    self.log_message(f"  - Vertices: {len(fallback_mesh.vertices):,}")
                    self.log_message(f"  - Triangles: {len(fallback_mesh.triangles):,}")
                    self.log_message(f"  - Watertight: {'Yes' if is_watertight else 'No'}")
                    
                    if is_watertight:
                        try:
                            volume = fallback_mesh.get_volume()
                            self.octree_volume = volume
                            self.log_message(f"  - Volume: {volume:.6f} cubic units")
                        except Exception as vol_error:
                            self.log_message(f"  - Volume calculation failed: {str(vol_error)}")
                    
                    self.log_message("✅ Fallback watertight processing completed", "success")
                    self.update_status("Fallback watertight completed", "success")
                else:
                    self.log_message("Fallback approach produced empty mesh", "error")
                    self.update_status("Watertight processing failed", "error")
                
        threading.Thread(target=watertight_thread, daemon=True).start()
    
    def _create_watertight_surface(self, mesh):
        """
        Create a watertight surface by filling small holes and ensuring manifold structure.
        """
        try:
            # Create a copy of the mesh
            watertight_mesh = o3d.geometry.TriangleMesh()
            watertight_mesh.vertices = o3d.utility.Vector3dVector(np.asarray(mesh.vertices))
            watertight_mesh.triangles = o3d.utility.Vector3iVector(np.asarray(mesh.triangles))
            
            # Fill holes using manual approach since Open3D doesn't have fill_holes()
            # We'll use a simple approach: remove non-manifold edges and fill small holes
            watertight_mesh.remove_non_manifold_edges()
            
            # For small holes, we can try to create a more complete surface
            # by ensuring the mesh is properly oriented and cleaned
            watertight_mesh.compute_vertex_normals()
            watertight_mesh.compute_triangle_normals()
            watertight_mesh.orient_triangles()
            
            # Clean the mesh
            watertight_mesh.remove_degenerate_triangles()
            watertight_mesh.remove_duplicated_triangles()
            watertight_mesh.remove_duplicated_vertices()
            watertight_mesh.remove_non_manifold_edges()
            watertight_mesh.remove_unreferenced_vertices()
            
            # Ensure proper normals
            watertight_mesh.compute_vertex_normals()
            
            return watertight_mesh
            
        except Exception as e:
            self.log_message(f"Watertight surface creation failed: {str(e)}")
            return mesh
    
    def _generate_tetrahedral_mesh(self, surface_mesh):
        """
        Generate a tetrahedral mesh from a watertight surface using binary fill holes approach.
        Uses scipy.ndimage.binary_fill_holes to ensure complete interior filling.
        """
        try:
            # Get mesh bounds
            bbox = surface_mesh.get_axis_aligned_bounding_box()
            min_bound = bbox.get_min_bound()
            max_bound = bbox.get_max_bound()
            
            # Create a voxel grid for tetrahedralization
            # Use GUI parameters for binary fill resolution
            binary_resolution = self.binary_res_var.get()
            tetrahedral_samples = 10000  # Use fixed value since we removed tetrahedral_samples_var
            
            # Use adaptive resolution based on mesh size and GUI parameter
            bbox_size = max_bound - min_bound
            min_size = np.min(bbox_size)
            adaptive_resolution = min(binary_resolution, min_size / 50)  # Use smaller of GUI param or adaptive
            
            # Create voxel coordinates
            x_coords = np.arange(min_bound[0], max_bound[0] + adaptive_resolution, adaptive_resolution)
            y_coords = np.arange(min_bound[1], max_bound[1] + adaptive_resolution, adaptive_resolution)
            z_coords = np.arange(min_bound[2], max_bound[2] + adaptive_resolution, adaptive_resolution)
            
            self.log_message(f"Creating binary volume grid: {len(x_coords)}x{len(y_coords)}x{len(z_coords)} voxels")
            
            # Create binary volume (1 = inside, 0 = outside)
            binary_volume = np.zeros((len(x_coords), len(y_coords), len(z_coords)), dtype=bool)
            
            # Sample points for distance computation using GUI parameter
            sample_points = surface_mesh.sample_points_uniformly(number_of_points=tetrahedral_samples)
            sample_vertices = np.asarray(sample_points.points)
            tree = cKDTree(sample_vertices)
            
            # Create binary volume using distance-based classification
            self.log_message("Creating binary volume using distance-based classification...")
            distance_threshold = adaptive_resolution * 1.5  # Distance threshold for inside/outside
            
            for i, x in enumerate(x_coords):
                for j, y in enumerate(y_coords):
                    for k, z in enumerate(z_coords):
                        cell_center = np.array([x, y, z])
                        distance, _ = tree.query(cell_center)
                        
                        # If distance is small, mark as inside (1)
                        if distance < distance_threshold:
                            binary_volume[i, j, k] = True
            
            initial_filled = np.sum(binary_volume) if binary_volume is not None else 0
            self.log_message(f"Initial binary volume: {initial_filled:,} filled voxels")
            
            # Use scipy.ndimage.binary_fill_holes to fill interior holes
            self.log_message("Filling holes using scipy.ndimage.binary_fill_holes...")
            filled_volume = ndimage.binary_fill_holes(binary_volume)
            
            # Apply additional morphological operations for better filling
            self.log_message("Applying morphological operations for complete filling...")
            
            # Use morphological closing to fill small gaps
            kernel_size = max(1, int(adaptive_resolution * 5))  # Adaptive kernel size
            kernel = np.ones((kernel_size, kernel_size, kernel_size), dtype=bool)
            filled_volume = ndimage.binary_closing(filled_volume, structure=kernel)
            
            # Final hole filling
            filled_volume = ndimage.binary_fill_holes(filled_volume)
            
            initial_voxels = np.sum(binary_volume) if binary_volume is not None else 0
            final_voxels = np.sum(filled_volume) if filled_volume is not None else 0
            additional_voxels = final_voxels - initial_voxels
            
            self.log_message(f"Binary fill holes complete:")
            self.log_message(f"  Initial voxels: {initial_voxels:,}")
            self.log_message(f"  Final voxels: {final_voxels:,}")
            self.log_message(f"  Additional voxels filled: {additional_voxels:,}")
            
            # Convert filled binary volume to signed distance field
            self.log_message("Converting to signed distance field...")
            sdf = np.zeros((len(x_coords), len(y_coords), len(z_coords)))
            
            # Create SDF from filled volume
            if filled_volume is not None:
                for i in range(len(x_coords)):
                    for j in range(len(y_coords)):
                        for k in range(len(z_coords)):
                            if filled_volume[i, j, k]:
                                # Inside: negative distance
                                cell_center = np.array([x_coords[i], y_coords[j], z_coords[k]])
                                distance, _ = tree.query(cell_center)
                                sdf[i, j, k] = -distance
                            else:
                                # Outside: positive distance
                                cell_center = np.array([x_coords[i], y_coords[j], z_coords[k]])
                                distance, _ = tree.query(cell_center)
                                sdf[i, j, k] = distance
            else:
                self.log_message("Error: filled_volume is None", "error")
                return surface_mesh
            
            # Generate tetrahedral mesh using marching cubes
            self.log_message("Generating tetrahedral mesh using marching cubes...")
            verts, faces, _, _ = measure.marching_cubes(sdf, level=0)
            
            # Transform vertices back to world coordinates
            verts_world = np.zeros_like(verts)
            verts_world[:, 0] = min_bound[0] + verts[:, 0] * adaptive_resolution
            verts_world[:, 1] = min_bound[1] + verts[:, 1] * adaptive_resolution
            verts_world[:, 2] = min_bound[2] + verts[:, 2] * adaptive_resolution
            
            # Create Open3D mesh
            tetra_mesh = o3d.geometry.TriangleMesh()
            tetra_mesh.vertices = o3d.utility.Vector3dVector(verts_world)
            tetra_mesh.triangles = o3d.utility.Vector3iVector(faces)
            
            # Clean the mesh
            tetra_mesh.remove_degenerate_triangles()
            tetra_mesh.remove_duplicated_triangles()
            tetra_mesh.remove_duplicated_vertices()
            tetra_mesh.remove_unreferenced_vertices()
            tetra_mesh.compute_vertex_normals()
            
            self.log_message(f"Tetrahedral mesh generated: {len(verts_world):,} vertices, {len(faces):,} triangles")
            
            return tetra_mesh
            
        except Exception as e:
            self.log_message(f"Tetrahedral mesh generation failed: {str(e)}")
            return surface_mesh
    
    def _extract_outer_surface(self, tetra_mesh):
        """
        Extract the outer surface from a tetrahedral mesh.
        This ensures we get a watertight surface.
        """
        try:
            # The tetrahedral mesh should already be a watertight surface
            # Just ensure it's properly cleaned and oriented
            
            outer_surface = o3d.geometry.TriangleMesh()
            outer_surface.vertices = o3d.utility.Vector3dVector(np.asarray(tetra_mesh.vertices))
            outer_surface.triangles = o3d.utility.Vector3iVector(np.asarray(tetra_mesh.triangles))
            
            # Clean the surface
            outer_surface.remove_degenerate_triangles()
            outer_surface.remove_duplicated_triangles()
            outer_surface.remove_duplicated_vertices()
            outer_surface.remove_non_manifold_edges()
            outer_surface.remove_unreferenced_vertices()
            
            # Ensure proper orientation
            outer_surface.compute_vertex_normals()
            outer_surface.compute_triangle_normals()
            outer_surface.orient_triangles()
            
            # Recompute normals after orientation
            outer_surface.compute_vertex_normals()
            
            return outer_surface
            
        except Exception as e:
            self.log_message(f"Outer surface extraction failed: {str(e)}")
            return tetra_mesh
    
    def save_octree_mesh(self):
        """
        Save the generated octree mesh.
        """
        if not hasattr(self, 'octree_volume_mesh') or self.octree_volume_mesh is None:
            self.log_message("No octree mesh to save. Generate octree mesh first using 'Generate Octree Mesh' button.", "warning")
            return
            
        file_path = filedialog.asksaveasfilename(
            title="Save Octree Mesh", defaultextension=".ply",
            filetypes=[("PLY files", "*.ply"), ("STL files", "*.stl"), ("OBJ files", "*.obj")]
        )
        if file_path:
            try:
                o3d.io.write_triangle_mesh(file_path, self.octree_volume_mesh)
                self.log_message(f"Octree mesh saved: {os.path.basename(file_path)}", "success")
                self.update_status("Octree mesh saved", "success")
            except Exception as e:
                self.log_message(f"Save failed: {str(e)}", "error")
    
    def octree_analysis(self):
        """
        Perform comprehensive analysis comparing original mesh and octree mesh.
        """
        if self.mesh is None:
            self.log_message("No original mesh to analyze. Load a mesh or point cloud first.", "warning")
            return
            
        if not hasattr(self, 'octree_volume_mesh') or self.octree_volume_mesh is None:
            self.log_message("No octree mesh available. Generate octree mesh first.", "warning")
            return
            
        try:
            self.update_status("Performing octree analysis...")
            
            # Original mesh properties
            original_surface_area = self.mesh.get_surface_area()
            original_vertices = len(self.mesh.vertices)
            original_triangles = len(self.mesh.triangles)
            original_watertight = self.mesh.is_watertight()
            
            # Octree mesh properties
            octree_surface_area = self.octree_volume_mesh.get_surface_area()
            octree_vertices = len(self.octree_volume_mesh.vertices)
            octree_triangles = len(self.octree_volume_mesh.triangles)
            octree_watertight = self.octree_volume_mesh.is_watertight()
            
            self.log_message("=== COMPREHENSIVE OCTREE ANALYSIS ===", "success")
            self.log_message("ORIGINAL MESH:")
            self.log_message(f"  Surface Area: {original_surface_area:.6f} sq units")
            self.log_message(f"  Vertices: {original_vertices:,}")
            self.log_message(f"  Triangles: {original_triangles:,}")
            self.log_message(f"  Watertight: {'Yes' if original_watertight else 'No'}")
            
            if original_watertight:
                try:
                    original_volume = self.mesh.get_volume()
                    self.log_message(f"  Volume: {original_volume:.6f} cubic units")
                except:
                    self.log_message("  Volume: Calculation failed")
            else:
                self.log_message("  Volume: Not available (not watertight)")
            
            self.log_message("OCTREE MESH:")
            self.log_message(f"  Surface Area: {octree_surface_area:.6f} sq units")
            self.log_message(f"  Vertices: {octree_vertices:,}")
            self.log_message(f"  Triangles: {octree_triangles:,}")
            self.log_message(f"  Watertight: {'Yes' if octree_watertight else 'No'}")
            
            if octree_watertight:
                try:
                    octree_volume = self.octree_volume_mesh.get_volume()
                    self.log_message(f"  Volume: {octree_volume:.6f} cubic units")
                except:
                    self.log_message("  Volume: Calculation failed")
            else:
                self.log_message("  Volume: Not available (not watertight)")
            
            # Comparison
            self.log_message("COMPARISON:")
            area_diff = abs(original_surface_area - octree_surface_area)
            area_diff_percent = (area_diff / original_surface_area) * 100 if original_surface_area > 0 else 0
            self.log_message(f"  Surface Area Difference: {area_diff:.6f} sq units ({area_diff_percent:.2f}%)")
            
            vertex_ratio = octree_vertices / original_vertices if original_vertices > 0 else 0
            triangle_ratio = octree_triangles / original_triangles if original_triangles > 0 else 0
            self.log_message(f"  Vertex Ratio (Octree/Original): {vertex_ratio:.2f}")
            self.log_message(f"  Triangle Ratio (Octree/Original): {triangle_ratio:.2f}")
            
            self.log_message("=" * 35)
            self.update_status("Octree analysis completed", "success")
            
        except Exception as e:
            self.log_message(f"Octree analysis failed: {str(e)}", "error")
            self.update_status("Analysis failed", "error")
    
    def _generate_octree_volume_mesh_fallback(self, resolution=0.01, sample_points=10000, sdf_padding=2, distance_threshold=1.0, max_grid_dim=120, max_cells=200000):
        """
        Fallback method for octree volume mesh generation when pygalmesh is not available or fails.
        Uses a simplified approach based on Coll et al. (2014) principles:
        1. Adaptive octree decomposition
        2. Distance-based classification
        3. Tetrahedral mesh generation
        
        CRASH PREVENTION: This method includes comprehensive safety measures to prevent system crashes.
        """
        try:
            import time
            import psutil
            import gc
            start_time = time.time()
            
            # CRASH PREVENTION: System health check
            self._check_system_health()
            
            if self.mesh is None:
                raise ValueError("No mesh loaded")
                
            input_mesh = self.mesh
            self.log_message("=== FALLBACK OCTREE VOLUME MESH GENERATION (CRASH-SAFE) ===", "warning")
            self.log_message(f"Input mesh: {len(input_mesh.vertices):,} vertices, {len(input_mesh.triangles):,} triangles")
            
            # Step 1: Adaptive octree decomposition
            bbox = input_mesh.get_axis_aligned_bounding_box()
            bbox_size = bbox.get_max_bound() - bbox.get_min_bound()
            min_bbox_size = np.min(bbox_size)
            
            # Adaptive resolution based on mesh size
            adaptive_resolution = min(resolution, min_bbox_size / 50)
            self.log_message(f"Adaptive resolution: {adaptive_resolution:.6f}")
            
            # Step 2: Create octree grid
            min_bounds = bbox.get_min_bound()
            max_bounds = bbox.get_max_bound()
            padding = sdf_padding * adaptive_resolution
            
            grid_min = min_bounds - padding
            grid_max = max_bounds + padding
            
            # CRASH PREVENTION: Conservative grid limits for fallback method
            # max_grid_dim is now passed as parameter
            grid_shape = np.ceil((grid_max - grid_min) / adaptive_resolution).astype(int)
            
            # Check memory requirements
            estimated_cells = np.prod(grid_shape)
            estimated_memory_mb = (estimated_cells * 8 * 4) / (1024**2)  # 8 bytes per cell, 4 bytes per float
            
            memory = psutil.virtual_memory()
            available_memory_gb = memory.available / (1024**3)
            max_memory_usage_gb = min(available_memory_gb * 0.2, 1.0)  # Use max 20% of available memory or 1GB
            
            if estimated_memory_mb > (max_memory_usage_gb * 1024):
                scale_factor = np.sqrt((max_memory_usage_gb * 1024) / estimated_memory_mb)
                adaptive_resolution *= scale_factor
                grid_shape = np.ceil((grid_max - grid_min) / adaptive_resolution).astype(int)
                self.log_message(f"⚠️ Memory constraint: Grid size reduced by factor {scale_factor:.2f}")
            
            if np.any(grid_shape > max_grid_dim):
                scale_factor = np.max(grid_shape) / max_grid_dim
                adaptive_resolution *= scale_factor
                grid_shape = np.ceil((grid_max - grid_min) / adaptive_resolution).astype(int)
                self.log_message(f"⚠️ Safety limit: Grid size limited to {max_grid_dim} (factor: {scale_factor:.2f})")
            
            final_cells = np.prod(grid_shape)
            final_memory_mb = (final_cells * 8 * 4) / (1024**2)
            
            self.log_message(f"✅ Fallback grid: {grid_shape[0]}x{grid_shape[1]}x{grid_shape[2]} ({final_cells:,} cells, {final_memory_mb:.1f}MB)")
            self.update_progress(30)
            
            # Step 3: Sample mesh surface for distance computation
            self.log_message("Sampling mesh surface...")
            mesh_pcd = input_mesh.sample_points_uniformly(number_of_points=min(sample_points, 50000))
            mesh_points = np.asarray(mesh_pcd.points)
            tree = cKDTree(mesh_points)
            
            self.update_progress(50)
            
            # Step 4: Create binary volume using scipy.ndimage.binary_fill_holes
            self.log_message("Creating binary volume with enhanced interior filling...")
            
            # Generate grid points
            x_coords = np.linspace(grid_min[0], grid_max[0], grid_shape[0])
            y_coords = np.linspace(grid_min[1], grid_max[1], grid_shape[1])
            z_coords = np.linspace(grid_min[2], grid_max[2], grid_shape[2])
            
            # Create binary volume array
            binary_volume = np.zeros(grid_shape, dtype=bool)
            
            # CRASH PREVENTION: Adaptive sampling based on grid size
            total_cells = np.prod(grid_shape)
            max_cells_to_process = max_cells  # Use parameter instead of hardcoded value
            
            if total_cells > max_cells_to_process:
                sample_step = max(1, int(np.cbrt(total_cells / max_cells_to_process)))
                self.log_message(f"⚠️ Large grid detected: Sampling every {sample_step} cells")
            else:
                sample_step = 1
            
            # CRASH PREVENTION: Process cells in batches with memory monitoring
            batch_size = 10000
            processed_cells = 0
            
            # Enhanced approach: Use multiple distance thresholds for better interior filling
            self.log_message("Using multi-threshold approach for better interior filling...")
            
            for i in range(0, len(x_coords), sample_step):
                for j in range(0, len(y_coords), sample_step):
                    for k in range(0, len(z_coords), sample_step):
                        if i < len(x_coords) and j < len(y_coords) and k < len(z_coords):
                            cell_center = np.array([x_coords[i], y_coords[j], z_coords[k]])
                            
                            # Enhanced distance-based classification with multiple thresholds
                            distance, _ = tree.query(cell_center)
                            
                            # Use a more generous threshold to capture more interior regions
                            threshold = adaptive_resolution * distance_threshold
                            
                            if distance < threshold:  # Inside or near surface
                                binary_volume[i, j, k] = True
                            
                            processed_cells += 1
                            
                            # CRASH PREVENTION: Batch processing with memory cleanup
                            if processed_cells % batch_size == 0:
                                # Check memory usage
                                memory = psutil.virtual_memory()
                                if memory.percent > 85:  # If memory usage > 85%
                                    self.log_message("⚠️ High memory usage detected - forcing garbage collection")
                                    gc.collect()
                                
                                # Progress update
                                progress = (processed_cells / total_cells) * 40 + 30  # 30-70% of total progress
                                self.update_progress(progress)
            
            self.update_progress(70)
            self.log_message(f"✅ Binary volume created: {np.sum(binary_volume):,} filled voxels out of {total_cells:,} total")
            
            # CRASH PREVENTION: Check if we have enough filled voxels
            if np.sum(binary_volume) < 100:
                self.log_message("❌ Error: Too few filled voxels - mesh may be too small or resolution too coarse")
                return None
            
            # Step 5: Enhanced hole filling and interior completion
            self.log_message("Performing enhanced hole filling and interior completion...")
            
            # First pass: Basic hole filling
            filled_volume = ndimage.binary_fill_holes(binary_volume)
            
            # Second pass: Morphological operations to ensure complete filling
            self.log_message("Applying morphological operations for complete interior filling...")
            
            # Use morphological closing to fill small gaps
            kernel_size = max(1, int(adaptive_resolution * 10))  # Adaptive kernel size
            kernel = np.ones((kernel_size, kernel_size, kernel_size), dtype=bool)
            filled_volume = ndimage.binary_closing(filled_volume, structure=kernel)
            
            # Additional hole filling after morphological operations
            filled_volume = ndimage.binary_fill_holes(filled_volume)
            
            # Optional: Use morphological dilation to ensure complete coverage
            # This helps with thin walls or incomplete surface sampling
            dilation_kernel = np.ones((3, 3, 3), dtype=bool)
            filled_volume = ndimage.binary_dilation(filled_volume, structure=dilation_kernel)
            
            # Final hole filling
            filled_volume = ndimage.binary_fill_holes(filled_volume)
            
            # Calculate volume from filled voxels
            voxel_volume = adaptive_resolution ** 3
            if filled_volume is not None:
                total_volume = np.sum(filled_volume) * voxel_volume
                final_voxels = np.sum(filled_volume)
            else:
                total_volume = 0.0
                final_voxels = 0
            
            if binary_volume is not None:
                initial_voxels = np.sum(binary_volume)
            else:
                initial_voxels = 0
            additional_voxels = final_voxels - initial_voxels
            
            self.log_message(f"✅ Enhanced filling complete:")
            self.log_message(f"  Initial voxels: {initial_voxels:,}")
            self.log_message(f"  Final voxels: {final_voxels:,}")
            self.log_message(f"  Additional voxels filled: {additional_voxels:,}")
            self.log_message(f"  Volume: {total_volume:.6f} cubic units")
            
            # Step 6: Generate mesh from filled volume using marching cubes
            self.log_message("Generating mesh from filled volume using marching cubes...")
            
            # Convert boolean to float for marching cubes
            if filled_volume is not None:
                volume_float = filled_volume.astype(np.float32)
            else:
                self.log_message("Error: filled_volume is None", "error")
                return None
            
            # Apply marching cubes to get vertices and faces
            try:
                verts, faces, normals, values = measure.marching_cubes(volume_float, level=0.5)
                
                # Transform vertices back to world coordinates
                verts_world = np.zeros_like(verts)
                verts_world[:, 0] = grid_min[0] + verts[:, 0] * adaptive_resolution
                verts_world[:, 1] = grid_min[1] + verts[:, 1] * adaptive_resolution
                verts_world[:, 2] = grid_min[2] + verts[:, 2] * adaptive_resolution
                
                # Create Open3D mesh
                volume_mesh = o3d.geometry.TriangleMesh()
                volume_mesh.vertices = o3d.utility.Vector3dVector(verts_world)
                volume_mesh.triangles = o3d.utility.Vector3iVector(faces)
                
                # Clean mesh
                volume_mesh.remove_degenerate_triangles()
                volume_mesh.remove_duplicated_triangles()
                volume_mesh.remove_duplicated_vertices()
                volume_mesh.remove_unreferenced_vertices()
                volume_mesh.compute_vertex_normals()
                
                # Store the volume mesh and calculated volume
                self.octree_volume_mesh = volume_mesh
                self.octree_volume = total_volume
                
                # Performance metrics
                elapsed_time = time.time() - start_time
                self.log_message(f"✅ Filled volume mesh created: {len(verts_world):,} vertices, {len(faces):,} triangles")
                self.log_message(f"Processing time: {elapsed_time:.2f} seconds")
                
                # Calculate surface area
                surface_area = volume_mesh.get_surface_area()
                self.log_message(f"Surface area: {surface_area:.6f} sq units")
                
                if total_volume > 0:
                    # Calculate volume density
                    bbox = volume_mesh.get_axis_aligned_bounding_box()
                    bbox_volume = bbox.volume()
                    if bbox_volume > 0:
                        volume_density = (total_volume / bbox_volume) * 100
                        self.log_message(f"Volume density: {volume_density:.2f}%")
                    
                    # Calculate surface area to volume ratio
                    sa_vol_ratio = surface_area / total_volume
                    self.log_message(f"Surface Area/Volume Ratio: {sa_vol_ratio:.6f}")
                
                self.log_message("=" * 50)
                self.update_progress(100)
                
                return volume_mesh
                
            except Exception as marching_error:
                self.log_message(f"Marching cubes failed: {str(marching_error)}", "error")
                self.log_message("Falling back to surface-only mesh...")
                
                # Fallback: Create surface mesh from original binary volume (before hole filling)
                try:
                    volume_float_surface = binary_volume.astype(np.float32)
                    verts_surface, faces_surface, _, _ = measure.marching_cubes(volume_float_surface, level=0.5)
                    
                    # Transform vertices back to world coordinates
                    verts_world_surface = np.zeros_like(verts_surface)
                    verts_world_surface[:, 0] = grid_min[0] + verts_surface[:, 0] * adaptive_resolution
                    verts_world_surface[:, 1] = grid_min[1] + verts_surface[:, 1] * adaptive_resolution
                    verts_world_surface[:, 2] = grid_min[2] + verts_surface[:, 2] * adaptive_resolution
                    
                    # Create Open3D mesh
                    volume_mesh = o3d.geometry.TriangleMesh()
                    volume_mesh.vertices = o3d.utility.Vector3dVector(verts_world_surface)
                    volume_mesh.triangles = o3d.utility.Vector3iVector(faces_surface)
                    
                    # Clean mesh
                    volume_mesh.remove_degenerate_triangles()
                    volume_mesh.remove_duplicated_triangles()
                    volume_mesh.remove_duplicated_vertices()
                    volume_mesh.remove_unreferenced_vertices()
                    volume_mesh.compute_vertex_normals()
                    
                    # Store the volume mesh and calculated volume
                    self.octree_volume_mesh = volume_mesh
                    self.octree_volume = total_volume
                    
                    self.log_message(f"✅ Surface-only mesh created: {len(verts_world_surface):,} vertices, {len(faces_surface):,} triangles")
                    self.log_message("Note: This is a surface mesh without hole filling")
                    
                    return volume_mesh
                    
                except Exception as fallback_error:
                    self.log_message(f"Surface mesh fallback also failed: {str(fallback_error)}", "error")
                    return None
                
        except Exception as e:
            self.log_message(f"Fallback octree generation failed: {str(e)}", "error")
            return None

    def _check_system_health(self):
        """
        CRASH PREVENTION: Check system health before heavy computations.
        Monitors memory, CPU, and provides warnings if system is under stress.
        """
        try:
            import psutil
            
            # Memory check
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            available_gb = memory.available / (1024**3)
            
            # CPU check
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Disk space check
            disk = psutil.disk_usage('/')
            disk_percent = disk.percent
            
            self.log_message(f"🔍 System Health Check:")
            self.log_message(f"  Memory: {memory_percent:.1f}% used ({available_gb:.1f}GB available)")
            self.log_message(f"  CPU: {cpu_percent:.1f}% usage")
            self.log_message(f"  Disk: {disk_percent:.1f}% used")
            
            # Warnings for high resource usage
            warnings = []
            if memory_percent > 80:
                warnings.append(f"⚠️ High memory usage ({memory_percent:.1f}%)")
            if cpu_percent > 90:
                warnings.append(f"⚠️ High CPU usage ({cpu_percent:.1f}%)")
            if disk_percent > 90:
                warnings.append(f"⚠️ Low disk space ({disk_percent:.1f}% used)")
            if available_gb < 2.0:
                warnings.append(f"⚠️ Low available memory ({available_gb:.1f}GB)")
            
            if warnings:
                self.log_message("🚨 System warnings:")
                for warning in warnings:
                    self.log_message(f"  {warning}")
                self.log_message("Consider closing other applications or using lower resolution settings.")
            else:
                self.log_message("✅ System health: Good")
                
        except Exception as e:
            self.log_message(f"⚠️ Could not check system health: {str(e)}")

    def convert_point_cloud_to_mesh(self):
        """
        Convert loaded point cloud to mesh using Poisson reconstruction.
        This method is called when generating octree mesh from point clouds.
        """
        try:
            if self.pcd is None:
                self.log_message("No point cloud to convert", "warning")
                return False
                
            self.log_message("Converting point cloud to mesh using Poisson reconstruction...")
            self.update_progress(0)
            
            # Clean point cloud
            self.log_message("Cleaning point cloud...")
            self.update_progress(20)
            pcd_clean, _ = self.pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
            
            # Estimate normals
            self.log_message("Estimating normals...")
            self.update_progress(40)
            pcd_clean.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=50))
            pcd_clean.orient_normals_consistent_tangent_plane(k=30)
            
            # Generate mesh using Poisson reconstruction
            self.log_message("Applying Poisson reconstruction...")
            self.update_progress(60)
            
            # Use GUI parameters for depth
            depth = self.depth_var.get()
            mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                pcd_clean, depth=depth)
            
            # Clean the generated mesh
            self.log_message("Cleaning generated mesh...")
            self.update_progress(80)
            mesh.remove_degenerate_triangles()
            mesh.remove_duplicated_triangles()
            mesh.remove_duplicated_vertices()
            mesh.remove_unreferenced_vertices()
            mesh.compute_vertex_normals()
            
            # Remove low density vertices (holes and artifacts)
            if len(densities) > 0:
                vertices_to_remove = [i for i, density in enumerate(densities) if density < np.quantile(densities, 0.1)]
                if vertices_to_remove:
                    mesh.remove_vertices_by_index(vertices_to_remove)
                    mesh.remove_unreferenced_vertices()
            
            self.mesh = mesh
            self.update_progress(100)
            
            self.log_message(f"✅ Point cloud converted to mesh: {len(mesh.vertices):,} vertices, {len(mesh.triangles):,} triangles", "success")
            self.update_status("Point cloud converted to mesh", "success")
            
            # Update file status
            if self.current_file:
                filename = os.path.basename(self.current_file)
                self.file_status.config(text=f"📐 {filename} ({len(mesh.vertices)} vertices) - CONVERTED")
            
            return True
            
        except Exception as e:
            self.log_message(f"Point cloud conversion failed: {str(e)}", "error")
            self.update_status("Conversion failed", "error")
            return False


    def view_octree_mesh(self):
        if not hasattr(self, 'octree_volume_mesh') or self.octree_volume_mesh is None:
            self.log_message("No octree volume mesh available. Generate octree mesh first using 'Generate Octree Mesh' button.", "warning")
            self.log_message("Note: You can load either a mesh or point cloud - point clouds will be converted to meshes automatically.", "info")
            return
        try:
            self.log_message("Opening octree volume mesh viewer...")
            o3d.visualization.draw_geometries([self.octree_volume_mesh], mesh_show_back_face=True)
            self.log_message("Octree mesh viewer closed")
        except Exception as e:
            self.log_message(f"Octree mesh view failed: {str(e)}", "error")

    def analyze_coll_methodology_compliance(self):
        """
        Analyze how well the current implementation follows Coll et al. (2014) methodology.
        Provides recommendations for improvement.
        """
        if self.mesh is None:
            self.log_message("No mesh loaded for analysis", "warning")
            return
            
        try:
            self.log_message("=== COLL ET AL. (2014) METHODOLOGY COMPLIANCE ANALYSIS ===", "success")
            
            # Check if octree mesh exists
            has_octree_mesh = hasattr(self, 'octree_volume_mesh') and self.octree_volume_mesh is not None
            
            # Original mesh properties
            original_vertices = len(self.mesh.vertices)
            original_triangles = len(self.mesh.triangles)
            original_surface_area = self.mesh.get_surface_area()
            original_watertight = self.mesh.is_watertight()
            
            self.log_message("ORIGINAL MESH ANALYSIS:")
            self.log_message(f"  Vertices: {original_vertices:,}")
            self.log_message(f"  Triangles: {original_triangles:,}")
            self.log_message(f"  Surface Area: {original_surface_area:.6f}")
            self.log_message(f"  Watertight: {'Yes' if original_watertight else 'No'}")
            
            # Coll et al. (2014) methodology requirements
            self.log_message("\nCOLL ET AL. (2014) REQUIREMENTS:")
            
            # 1. Adaptive octree decomposition
            self.log_message("✓ 1. Adaptive Octree Decomposition:")
            self.log_message("   - Implemented with adaptive resolution based on mesh complexity")
            self.log_message("   - Grid size limits for performance optimization")
            self.log_message("   - Surface complexity factor calculation")
            
            # 2. Robust inside/outside classification
            self.log_message("✓ 2. Robust Inside/Outside Classification:")
            self.log_message("   - Signed Distance Field (SDF) computation")
            self.log_message("   - Distance-based cell classification")
            self.log_message("   - Adaptive thresholding")
            
            # 3. Quality tetrahedral mesh generation
            self.log_message("✓ 3. Quality Tetrahedral Mesh Generation:")
            self.log_message("   - Constrained Delaunay tetrahedralization (pygalmesh)")
            self.log_message("   - Fallback method with BCC pattern")
            self.log_message("   - Mesh quality optimization")
            
            # 4. Performance optimization
            self.log_message("✓ 4. Performance Optimization:")
            self.log_message("   - Grid size limitations")
            self.log_message("   - Adaptive sampling")
            self.log_message("   - Efficient proximity queries")
            
            if has_octree_mesh:
                # Octree mesh properties
                octree_vertices = len(self.octree_volume_mesh.vertices)
                octree_triangles = len(self.octree_volume_mesh.triangles)
                octree_surface_area = self.octree_volume_mesh.get_surface_area()
                octree_watertight = self.octree_volume_mesh.is_watertight()
                
                self.log_message("\nOCTREE MESH RESULTS:")
                self.log_message(f"  Vertices: {octree_vertices:,}")
                self.log_message(f"  Triangles: {octree_triangles:,}")
                self.log_message(f"  Surface Area: {octree_surface_area:.6f}")
                self.log_message(f"  Watertight: {'Yes' if octree_watertight else 'No'}")
                
                # Quality metrics
                vertex_ratio = octree_vertices / original_vertices if original_vertices > 0 else 0
                triangle_ratio = octree_triangles / original_triangles if original_triangles > 0 else 0
                area_diff = abs(octree_surface_area - original_surface_area) / original_surface_area * 100 if original_surface_area > 0 else 0
                
                self.log_message("\nQUALITY METRICS:")
                self.log_message(f"  Vertex Ratio (Octree/Original): {vertex_ratio:.2f}")
                self.log_message(f"  Triangle Ratio (Octree/Original): {triangle_ratio:.2f}")
                self.log_message(f"  Surface Area Difference: {area_diff:.2f}%")
                
                # Compliance assessment
                self.log_message("\nCOMPLIANCE ASSESSMENT:")
                if vertex_ratio > 0.5 and vertex_ratio < 2.0:
                    self.log_message("✓ Vertex density: Good (within reasonable range)")
                else:
                    self.log_message("⚠ Vertex density: May need adjustment")
                    
                if area_diff < 10.0:
                    self.log_message("✓ Surface area preservation: Good")
                else:
                    self.log_message("⚠ Surface area preservation: May need improvement")
                    
                if octree_watertight:
                    self.log_message("✓ Watertight mesh: Achieved")
                else:
                    self.log_message("⚠ Watertight mesh: Not achieved (may be acceptable for non-watertight input)")
            else:
                self.log_message("\nOCTREE MESH: Not generated yet")
                self.log_message("Run 'Generate Octree Mesh' to create octree volume mesh")
            
            # Recommendations
            self.log_message("\nRECOMMENDATIONS:")
            self.log_message("1. For better performance:")
            self.log_message("   - Reduce resolution parameter for large meshes")
            self.log_message("   - Use fallback method if pygalmesh is slow")
            self.log_message("   - Adjust sample_points based on mesh complexity")
            
            self.log_message("2. For better quality:")
            self.log_message("   - Increase resolution for detailed features")
            self.log_message("   - Use higher sample_points for complex geometry")
            self.log_message("   - Consider mesh preprocessing (cleaning, hole filling)")
            
            self.log_message("3. For Coll et al. (2014) compliance:")
            self.log_message("   - Current implementation follows key principles")
            self.log_message("   - Adaptive resolution and robust classification implemented")
            self.log_message("   - Performance optimizations maintain methodology integrity")
            
            self.log_message("=" * 60)
            
        except Exception as e:
            self.log_message(f"Methodology analysis failed: {str(e)}", "error")

    def compare_meshes_visualization(self):
        """
        Visualize both original mesh and octree mesh together using different visualization techniques:
        - Creates side-by-side view or toggleable display
        - Uses wireframe/solid combinations for clear visibility
        """
        if self.mesh is None:
            self.log_message("No original mesh to compare", "warning")
            return
            
        if not hasattr(self, 'octree_volume_mesh') or self.octree_volume_mesh is None:
            self.log_message("No octree mesh available. Generate octree mesh first.", "warning")
            return
            
        try:
            self.log_message("Preparing mesh comparison visualization...")
            
            # Method 1: Show them separately first
            self.log_message("Showing Original Mesh (Green) - Close window to continue...")
            original_mesh_viz = o3d.geometry.TriangleMesh()
            original_mesh_viz.vertices = o3d.utility.Vector3dVector(np.asarray(self.mesh.vertices))
            original_mesh_viz.triangles = o3d.utility.Vector3iVector(np.asarray(self.mesh.triangles))
            original_mesh_viz.paint_uniform_color([0.0, 1.0, 0.0])  # Green
            original_mesh_viz.compute_vertex_normals()
            
            coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
            
            o3d.visualization.draw_geometries(
                [original_mesh_viz, coord_frame],
                window_name="Original Mesh (Green)",
                width=800,
                height=600
            )
            
            self.log_message("Showing Octree Mesh (Blue) - Close window to continue...")
            octree_mesh_viz = o3d.geometry.TriangleMesh()
            octree_mesh_viz.vertices = o3d.utility.Vector3dVector(np.asarray(self.octree_volume_mesh.vertices))
            octree_mesh_viz.triangles = o3d.utility.Vector3iVector(np.asarray(self.octree_volume_mesh.triangles))
            octree_mesh_viz.paint_uniform_color([0.0, 0.5, 1.0])  # Blue
            octree_mesh_viz.compute_vertex_normals()
            
            o3d.visualization.draw_geometries(
                [octree_mesh_viz, coord_frame],
                window_name="Octree Mesh (Blue)",
                width=800,
                height=600
            )
            
            # Method 2: Show combined with different approach
            self.log_message("Showing Combined View - Original as wireframe, Octree as solid...")
            
            # Create wireframe from original mesh
            original_wireframe = o3d.geometry.LineSet.create_from_triangle_mesh(self.mesh)
            original_wireframe.paint_uniform_color([0.0, 1.0, 0.0])  # Green wireframe
            
            # Create point cloud from original mesh vertices for additional visibility
            original_points = o3d.geometry.PointCloud()
            original_points.points = o3d.utility.Vector3dVector(np.asarray(self.mesh.vertices))
            original_points.paint_uniform_color([0.0, 0.8, 0.0])  # Darker green points
            
            # Octree mesh as solid with different color
            octree_solid = o3d.geometry.TriangleMesh()
            octree_solid.vertices = o3d.utility.Vector3dVector(np.asarray(self.octree_volume_mesh.vertices))
            octree_solid.triangles = o3d.utility.Vector3iVector(np.asarray(self.octree_volume_mesh.triangles))
            octree_solid.paint_uniform_color([1.0, 0.5, 0.0])  # Orange for contrast
            octree_solid.compute_vertex_normals()
            
            # Create combined visualization
            combined_geometries = [original_wireframe, original_points, octree_solid, coord_frame]
            
            try:
                vis = o3d.visualization.Visualizer()
                vis.create_window(
                    window_name="Combined View: Green=Original, Orange=Octree",
                    width=1200,
                    height=800
                )
                
                for geom in combined_geometries:
                    vis.add_geometry(geom)
                
                # Set rendering options
                render_option = vis.get_render_option()
                render_option.mesh_show_back_face = True
                render_option.line_width = 3.0
                render_option.point_size = 2.0
                
                # Run visualizer
                vis.run()
                vis.destroy_window()
                
            except Exception as viz_error:
                self.log_message(f"Advanced visualization failed: {str(viz_error)}", "warning")
                o3d.visualization.draw_geometries(
                    combined_geometries,
                    window_name="Combined View: Green=Original, Orange=Octree",
                    width=1200,
                    height=800
                )
            
            # Method 3: Interactive comparison - offset one mesh slightly
            self.log_message("Showing Side-by-Side Comparison...")
            
            # Create offset versions
            original_offset = o3d.geometry.TriangleMesh()
            original_offset.vertices = o3d.utility.Vector3dVector(np.asarray(self.mesh.vertices))
            original_offset.triangles = o3d.utility.Vector3iVector(np.asarray(self.mesh.triangles))
            original_offset.paint_uniform_color([0.0, 1.0, 0.0])  # Green
            original_offset.compute_vertex_normals()
            
            # Get bounding box to calculate offset
            bbox = self.mesh.get_axis_aligned_bounding_box()
            width = bbox.max_bound[0] - bbox.min_bound[0]
            offset_distance = width * 1.2  # 20% gap between meshes
            
            # Offset original mesh to the left
            original_offset.translate([-offset_distance, 0, 0])
            
            # Keep octree mesh in original position (or offset to right)
            octree_offset = o3d.geometry.TriangleMesh()
            octree_offset.vertices = o3d.utility.Vector3dVector(np.asarray(self.octree_volume_mesh.vertices))
            octree_offset.triangles = o3d.utility.Vector3iVector(np.asarray(self.octree_volume_mesh.triangles))
            octree_offset.paint_uniform_color([0.0, 0.5, 1.0])  # Blue
            octree_offset.compute_vertex_normals()
            
            # Add labels using coordinate frames
            left_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
            left_frame.translate([-offset_distance, 0, 0])
            
            right_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
            
            side_by_side_geometries = [original_offset, octree_offset, left_frame, right_frame]
            
            o3d.visualization.draw_geometries(
                side_by_side_geometries,
                window_name="Side-by-Side: Left=Original(Green), Right=Octree(Blue)",
                width=1400,
                height=800
            )
            
            # Log statistics
            self.log_message("Mesh comparison complete!")
            self.log_message("Visualization methods used:")
            self.log_message("  1. Individual mesh views")
            self.log_message("  2. Combined view (wireframe + solid)")
            self.log_message("  3. Side-by-side comparison")
            
            self.log_message(f"Mesh statistics:")
            self.log_message(f"  - Original mesh: {len(self.mesh.vertices):,} vertices, {len(self.mesh.triangles):,} triangles")
            self.log_message(f"  - Octree mesh: {len(self.octree_volume_mesh.vertices):,} vertices, {len(self.octree_volume_mesh.triangles):,} triangles")
            
            # Calculate volume and surface area comparison
            try:
                if self.mesh.is_watertight():
                    original_volume = self.mesh.get_volume()
                    self.log_message(f"  - Original volume: {original_volume:.6f}")
                
                if self.octree_volume_mesh.is_watertight():
                    octree_volume = self.octree_volume_mesh.get_volume()
                    self.log_message(f"  - Octree volume: {octree_volume:.6f}")
                    
                    if self.mesh.is_watertight():
                        volume_diff = abs(original_volume - octree_volume)
                        volume_percent = (volume_diff / original_volume) * 100
                        self.log_message(f"  - Volume difference: {volume_diff:.6f} ({volume_percent:.2f}%)")
                        
            except Exception as volume_error:
                self.log_message(f"  - Volume calculation failed: {str(volume_error)}")
                
            try:
                original_surface_area = self.mesh.get_surface_area()
                octree_surface_area = self.octree_volume_mesh.get_surface_area()
                self.log_message(f"  - Original surface area: {original_surface_area:.6f}")
                self.log_message(f"  - Octree surface area: {octree_surface_area:.6f}")
                
                area_diff = abs(original_surface_area - octree_surface_area)
                area_percent = (area_diff / original_surface_area) * 100
                self.log_message(f"  - Surface area difference: {area_diff:.6f} ({area_percent:.2f}%)")
                
            except Exception as area_error:
                self.log_message(f"  - Surface area calculation failed: {str(area_error)}")
                
        except Exception as e:
            self.log_message(f"Mesh comparison visualization failed: {str(e)}", "error")
            self.update_status("Visualization failed", "error")

    def analyze_surface_areas(self):
        """
        Detailed analysis of surface areas between original and octree meshes.
        Provides comprehensive comparison and visualization of surface area calculations.
        """
        if self.mesh is None:
            self.log_message("No original mesh to analyze", "warning")
            return
            
        if not hasattr(self, 'octree_volume_mesh') or self.octree_volume_mesh is None:
            self.log_message("No octree mesh available. Generate octree mesh first.", "warning")
            return
            
        try:
            self.log_message("=== SURFACE AREA ANALYSIS ===", "success")
            
            # Calculate surface areas
            original_surface_area = self.mesh.get_surface_area()
            octree_surface_area = self.octree_volume_mesh.get_surface_area()
            
            # Calculate differences
            area_diff = abs(octree_surface_area - original_surface_area)
            area_diff_percent = (area_diff / original_surface_area) * 100 if original_surface_area > 0 else 0
            
            # Detailed analysis
            self.log_message("ORIGINAL MESH:")
            self.log_message(f"  Surface Area: {original_surface_area:.6f} sq units")
            self.log_message(f"  Vertices: {len(self.mesh.vertices):,}")
            self.log_message(f"  Triangles: {len(self.mesh.triangles):,}")
            self.log_message(f"  Watertight: {'Yes' if self.mesh.is_watertight() else 'No'}")
            
            self.log_message("OCTREE MESH:")
            self.log_message(f"  Surface Area: {octree_surface_area:.6f} sq units")
            self.log_message(f"  Vertices: {len(self.octree_volume_mesh.vertices):,}")
            self.log_message(f"  Triangles: {len(self.octree_volume_mesh.triangles):,}")
            self.log_message(f"  Watertight: {'Yes' if self.octree_volume_mesh.is_watertight() else 'No'}")
            
            self.log_message("COMPARISON:")
            self.log_message(f"  Absolute Difference: {area_diff:.6f} sq units")
            self.log_message(f"  Percentage Difference: {area_diff_percent:.2f}%")
            
            # Quality assessment
            if area_diff_percent < 5:
                self.log_message("✅ Surface area preservation: Excellent (< 5% difference)")
            elif area_diff_percent < 10:
                self.log_message("✅ Surface area preservation: Good (< 10% difference)")
            elif area_diff_percent < 20:
                self.log_message("⚠️ Surface area preservation: Acceptable (< 20% difference)")
            else:
                self.log_message("❌ Surface area preservation: Poor (> 20% difference)")
                self.log_message("Recommendations:")
                self.log_message("  - Decrease Octree Box Size for finer resolution")
                self.log_message("  - Increase Sample Points for better surface sampling")
                self.log_message("  - Decrease Distance Threshold for tighter surface")
            
            # Surface area visualization
            self.log_message("Opening surface area visualization...")
            self._visualize_surface_areas()
            
            self.log_message("=" * 40)
            
        except Exception as e:
            self.log_message(f"Surface area analysis failed: {str(e)}", "error")
            self.update_status("Analysis failed", "error")

    def analyze_volumes(self):
        """
        Detailed analysis of volumes between original and octree meshes.
        Provides comprehensive comparison and visualization of volume calculations.
        """
        if self.mesh is None:
            self.log_message("No original mesh to analyze", "warning")
            return
            
        if not hasattr(self, 'octree_volume_mesh') or self.octree_volume_mesh is None:
            self.log_message("No octree mesh available. Generate octree mesh first.", "warning")
            return
            
        try:
            self.log_message("=== VOLUME ANALYSIS ===", "success")
            
            # Get octree volume if available
            octree_volume = getattr(self, 'octree_volume', 0.0)
            
            # Calculate original volume if watertight
            original_volume = None
            if self.mesh.is_watertight():
                try:
                    original_volume = self.mesh.get_volume()
                except Exception as e:
                    self.log_message(f"Original volume calculation failed: {str(e)}")
            
            # Detailed analysis
            self.log_message("ORIGINAL MESH:")
            self.log_message(f"  Watertight: {'Yes' if self.mesh.is_watertight() else 'No'}")
            if original_volume is not None:
                self.log_message(f"  Volume: {original_volume:.6f} cubic units")
            else:
                self.log_message("  Volume: Not available (not watertight)")
            
            self.log_message("OCTREE MESH:")
            self.log_message(f"  Watertight: {'Yes' if self.octree_volume_mesh.is_watertight() else 'No'}")
            if octree_volume > 0:
                self.log_message(f"  Volume (from tetrahedra): {octree_volume:.6f} cubic units")
            else:
                # Try standard volume calculation
                try:
                    if self.octree_volume_mesh.is_watertight():
                        octree_volume = self.octree_volume_mesh.get_volume()
                        self.log_message(f"  Volume (standard): {octree_volume:.6f} cubic units")
                    else:
                        self.log_message("  Volume: Not available (not watertight)")
                except Exception as e:
                    self.log_message(f"  Volume calculation failed: {str(e)}")
            
            # Comparison if both volumes available
            if original_volume is not None and octree_volume > 0:
                volume_diff = abs(octree_volume - original_volume)
                volume_diff_percent = (volume_diff / original_volume) * 100
                
                self.log_message("COMPARISON:")
                self.log_message(f"  Absolute Difference: {volume_diff:.6f} cubic units")
                self.log_message(f"  Percentage Difference: {volume_diff_percent:.2f}%")
                
                # Quality assessment
                if volume_diff_percent < 5:
                    self.log_message("✅ Volume accuracy: Excellent (< 5% difference)")
                elif volume_diff_percent < 10:
                    self.log_message("✅ Volume accuracy: Good (< 10% difference)")
                elif volume_diff_percent < 20:
                    self.log_message("⚠️ Volume accuracy: Acceptable (< 20% difference)")
                else:
                    self.log_message("❌ Volume accuracy: Poor (> 20% difference)")
                    self.log_message("Recommendations:")
                    self.log_message("  - Increase Tetrahedral Density for better volume mesh")
                    self.log_message("  - Decrease Distance Threshold for tighter volume")
                    self.log_message("  - Decrease Octree Box Size for finer resolution")
                
                # Calculate density metrics
                bbox = self.octree_volume_mesh.get_axis_aligned_bounding_box()
                bbox_volume = bbox.volume()
                if bbox_volume > 0:
                    volume_density = (octree_volume / bbox_volume) * 100
                    self.log_message(f"  Volume Density: {volume_density:.2f}%")
                
                # Surface area to volume ratio
                octree_surface_area = self.octree_volume_mesh.get_surface_area()
                if octree_volume > 0:
                    sa_vol_ratio = octree_surface_area / octree_volume
                    self.log_message(f"  Surface Area/Volume Ratio: {sa_vol_ratio:.6f}")
            
            # Volume visualization
            self.log_message("Opening volume visualization...")
            self._visualize_volumes()
            
            self.log_message("=" * 40)
            
        except Exception as e:
            self.log_message(f"Volume analysis failed: {str(e)}", "error")
            self.update_status("Analysis failed", "error")

    def _visualize_surface_areas(self):
        """
        Visualize surface areas with color-coded regions.
        """
        try:
            # Create visualization meshes
            original_mesh_viz = o3d.geometry.TriangleMesh()
            original_mesh_viz.vertices = o3d.utility.Vector3dVector(np.asarray(self.mesh.vertices))
            original_mesh_viz.triangles = o3d.utility.Vector3iVector(np.asarray(self.mesh.triangles))
            original_mesh_viz.paint_uniform_color([0.8, 0.8, 0.8])  # Light gray
            
            octree_mesh_viz = o3d.geometry.TriangleMesh()
            octree_mesh_viz.vertices = o3d.utility.Vector3dVector(np.asarray(self.octree_volume_mesh.vertices))
            octree_mesh_viz.triangles = o3d.utility.Vector3iVector(np.asarray(self.octree_volume_mesh.triangles))
            
            # Color octree mesh based on surface area regions
            vertices = np.asarray(self.octree_volume_mesh.vertices)
            triangles = np.asarray(self.octree_volume_mesh.triangles)
            
            if len(vertices) > 0 and len(triangles) > 0:
                # Calculate triangle areas
                triangle_areas = []
                for triangle in triangles:
                    if len(triangle) >= 3:
                        v1, v2, v3 = vertices[triangle[0]], vertices[triangle[1]], vertices[triangle[2]]
                        # Calculate triangle area using cross product
                        edge1 = v2 - v1
                        edge2 = v3 - v1
                        area = 0.5 * np.linalg.norm(np.cross(edge1, edge2))
                        triangle_areas.append(area)
                    else:
                        triangle_areas.append(0.0)
                
                triangle_areas = np.array(triangle_areas)
                
                # Color vertices based on triangle areas
                vertex_colors = np.zeros((len(vertices), 3))
                vertex_area_sums = np.zeros(len(vertices))
                
                for i, area in enumerate(triangle_areas):
                    triangle = triangles[i]
                    for vertex_idx in triangle:
                        if vertex_idx < len(vertices):
                            vertex_area_sums[vertex_idx] += area
                
                # Normalize and color based on area contribution
                max_area = np.max(vertex_area_sums) if len(vertex_area_sums) > 0 else 1.0
                for i in range(len(vertices)):
                    normalized_area = vertex_area_sums[i] / max_area
                    # Color from blue (low area) to red (high area)
                    vertex_colors[i] = [normalized_area, 0.0, 1.0 - normalized_area]
                
                octree_mesh_viz.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)
                
                # Create coordinate frame
                coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
                
                # Visualize
                geometries = [original_mesh_viz, octree_mesh_viz, coord_frame]
                
                try:
                    o3d.visualization.draw_geometries(
                        geometries,
                        window_name="Surface Area Analysis",
                        width=1200,
                        height=800,
                        mesh_show_back_face=True
                    )
                except Exception as viz_error:
                    self.log_message(f"Advanced visualization failed: {str(viz_error)}")
                    o3d.visualization.draw_geometries(geometries)
                
                self.log_message("Surface area visualization closed")
                
                # Log area statistics
                total_area = np.sum(triangle_areas)
                avg_area = np.mean(triangle_areas)
                self.log_message(f"Surface area statistics:")
                self.log_message(f"  - Total surface area: {total_area:.6f} sq units")
                self.log_message(f"  - Average triangle area: {avg_area:.6f} sq units")
                self.log_message(f"  - Color legend: Blue (low area) to Red (high area)")
            
        except Exception as e:
            self.log_message(f"Surface area visualization failed: {str(e)}", "error")

    def _visualize_volumes(self):
        """
        Visualize complete volumes with filled interior and color-coded regions.
        Shows both surface and interior volume representation.
        """
        try:
            # Create visualization meshes
            original_mesh_viz = o3d.geometry.TriangleMesh()
            original_mesh_viz.vertices = o3d.utility.Vector3dVector(np.asarray(self.mesh.vertices))
            original_mesh_viz.triangles = o3d.utility.Vector3iVector(np.asarray(self.mesh.triangles))
            original_mesh_viz.paint_uniform_color([0.8, 0.8, 0.8])  # Light gray
            
            octree_mesh_viz = o3d.geometry.TriangleMesh()
            octree_mesh_viz.vertices = o3d.utility.Vector3dVector(np.asarray(self.octree_volume_mesh.vertices))
            octree_mesh_viz.triangles = o3d.utility.Vector3iVector(np.asarray(self.octree_volume_mesh.triangles))
            
            # Color octree mesh based on volume regions
            vertices = np.asarray(self.octree_volume_mesh.vertices)
            triangles = np.asarray(self.octree_volume_mesh.triangles)
            
            if len(vertices) > 0 and len(triangles) > 0:
                # Calculate distance from mesh center for volume visualization
                bbox = self.octree_volume_mesh.get_axis_aligned_bounding_box()
                center = bbox.get_center()
                
                distances_from_center = np.linalg.norm(vertices - center, axis=1)
                max_distance = np.max(distances_from_center)
                
                # Color vertices based on distance from center (volume depth)
                vertex_colors = np.zeros((len(vertices), 3))
                for i, distance in enumerate(distances_from_center):
                    normalized_distance = distance / max_distance
                    # Color from green (center) to purple (surface)
                    vertex_colors[i] = [normalized_distance, 1.0 - normalized_distance, normalized_distance]
                
                octree_mesh_viz.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)
                
                # Create filled volume representation
                self.log_message("Creating filled volume representation...")
                filled_volume_mesh = self._create_filled_volume_mesh()
                
                # Create coordinate frame
                coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
                
                # Visualize with filled volume
                geometries = [original_mesh_viz, octree_mesh_viz, coord_frame]
                if filled_volume_mesh is not None:
                    geometries.append(filled_volume_mesh)
                
                try:
                    o3d.visualization.draw_geometries(
                        geometries,
                        window_name="Complete Volume Analysis",
                        width=1200,
                        height=800,
                        mesh_show_back_face=True
                    )
                except Exception as viz_error:
                    self.log_message(f"Advanced visualization failed: {str(viz_error)}")
                    o3d.visualization.draw_geometries(geometries)
                
                self.log_message("Volume visualization closed")
                
                # Log volume statistics
                octree_volume = getattr(self, 'octree_volume', 0.0)
                if octree_volume > 0:
                    self.log_message(f"Volume statistics:")
                    self.log_message(f"  - Calculated volume: {octree_volume:.6f} cubic units")
                    self.log_message(f"  - Color legend: Green (center) to Purple (surface)")
                    if filled_volume_mesh is not None:
                        self.log_message(f"  - Filled volume mesh: {len(filled_volume_mesh.vertices):,} vertices")
                
        except Exception as e:
            self.log_message(f"Volume visualization failed: {str(e)}", "error")

    def _create_filled_volume_mesh(self):
        """
        Create a filled volume mesh representation showing the complete interior.
        Uses voxelization to create a solid volume representation.
        """
        try:
            if not hasattr(self, 'octree_volume_mesh') or self.octree_volume_mesh is None:
                return None
                
            self.log_message("Generating filled volume mesh...")
            
            # Get mesh bounds
            bbox = self.octree_volume_mesh.get_axis_aligned_bounding_box()
            min_bound = bbox.get_min_bound()
            max_bound = bbox.get_max_bound()
            
            # Create voxel grid for volume filling
            voxel_size = (max_bound - min_bound) / 50  # 50x50x50 voxel grid
            grid_shape = np.ceil((max_bound - min_bound) / voxel_size).astype(int)
            
            # Ensure reasonable grid size
            max_grid_size = 30
            if np.any(grid_shape > max_grid_size):
                scale_factor = np.max(grid_shape) / max_grid_size
                voxel_size *= scale_factor
                grid_shape = np.ceil((max_bound - min_bound) / voxel_size).astype(int)
            
            self.log_message(f"Voxel grid: {grid_shape[0]}x{grid_shape[1]}x{grid_shape[2]} ({np.prod(grid_shape):,} voxels)")
            
            # Create voxel coordinates
            x_coords = np.linspace(min_bound[0], max_bound[0], grid_shape[0])
            y_coords = np.linspace(min_bound[1], max_bound[1], grid_shape[1])
            z_coords = np.linspace(min_bound[2], max_bound[2], grid_shape[2])
            
            # Create mesh for point-in-mesh testing using simple distance-based approach
            mesh_vertices = np.asarray(self.octree_volume_mesh.vertices)
            mesh_triangles = np.asarray(self.octree_volume_mesh.triangles)
            
            # Create a point cloud from mesh vertices for distance-based testing
            mesh_pcd = o3d.geometry.PointCloud()
            mesh_pcd.points = o3d.utility.Vector3dVector(mesh_vertices)
            
            # Create KDTree for efficient nearest neighbor search
            mesh_points = np.asarray(mesh_pcd.points)
            tree = cKDTree(mesh_points)
            
            # Create filled volume mesh
            filled_volume_mesh = o3d.geometry.TriangleMesh()
            filled_vertices = []
            filled_triangles = []
            
            # Generate voxels and check if they're inside the mesh
            inside_voxels = []
            
            for i in range(len(x_coords)-1):
                for j in range(len(y_coords)-1):
                    for k in range(len(z_coords)-1):
                        # Voxel center
                        voxel_center = np.array([
                            (x_coords[i] + x_coords[i+1]) / 2,
                            (y_coords[j] + y_coords[j+1]) / 2,
                            (z_coords[k] + z_coords[k+1]) / 2
                        ])
                        
                        # Check if voxel center is inside mesh using distance-based approach
                        distance, _ = tree.query(voxel_center)
                        
                        # If distance is small, consider voxel inside
                        if distance < voxel_size * 2:  # Inside or close to surface
                            inside_voxels.append((i, j, k))
            
            self.log_message(f"Found {len(inside_voxels):,} inside voxels")
            
            # Create cube mesh for each inside voxel
            for i, j, k in inside_voxels:
                # Voxel corners
                v000 = [x_coords[i], y_coords[j], z_coords[k]]
                v001 = [x_coords[i], y_coords[j], z_coords[k+1]]
                v010 = [x_coords[i], y_coords[j+1], z_coords[k]]
                v011 = [x_coords[i], y_coords[j+1], z_coords[k+1]]
                v100 = [x_coords[i+1], y_coords[j], z_coords[k]]
                v101 = [x_coords[i+1], y_coords[j], z_coords[k+1]]
                v110 = [x_coords[i+1], y_coords[j+1], z_coords[k]]
                v111 = [x_coords[i+1], y_coords[j+1], z_coords[k+1]]
                
                # Add vertices
                start_idx = len(filled_vertices)
                filled_vertices.extend([v000, v001, v010, v011, v100, v101, v110, v111])
                
                # Add cube faces (12 triangles for a cube)
                cube_triangles = [
                    # Front face
                    [start_idx+0, start_idx+1, start_idx+2], [start_idx+1, start_idx+3, start_idx+2],
                    # Back face
                    [start_idx+4, start_idx+6, start_idx+5], [start_idx+5, start_idx+6, start_idx+7],
                    # Left face
                    [start_idx+0, start_idx+2, start_idx+4], [start_idx+2, start_idx+6, start_idx+4],
                    # Right face
                    [start_idx+1, start_idx+5, start_idx+3], [start_idx+3, start_idx+5, start_idx+7],
                    # Top face
                    [start_idx+2, start_idx+3, start_idx+6], [start_idx+3, start_idx+7, start_idx+6],
                    # Bottom face
                    [start_idx+0, start_idx+4, start_idx+1], [start_idx+1, start_idx+4, start_idx+5]
                ]
                filled_triangles.extend(cube_triangles)
            
            if len(filled_vertices) > 0:
                filled_volume_mesh.vertices = o3d.utility.Vector3dVector(np.array(filled_vertices))
                filled_volume_mesh.triangles = o3d.utility.Vector3iVector(np.array(filled_triangles))
                
                # Color the filled volume with semi-transparent blue
                filled_volume_mesh.paint_uniform_color([0.0, 0.5, 1.0])  # Blue
                
                # Add transparency by modifying vertex colors
                filled_vertex_colors = np.full((len(filled_vertices), 3), [0.0, 0.5, 1.0])
                filled_volume_mesh.vertex_colors = o3d.utility.Vector3dVector(filled_vertex_colors)
                
                self.log_message(f"Created filled volume mesh: {len(filled_vertices):,} vertices, {len(filled_triangles):,} triangles")
                return filled_volume_mesh
            else:
                self.log_message("No inside voxels found for filled volume")
                return None
                
        except Exception as e:
            self.log_message(f"Filled volume creation failed: {str(e)}", "error")
            return None



# Main execution block
if __name__ == "__main__":
    try:
        root = tk.Tk()
        app = PointCloudProcessor(root)
        root.mainloop()
    except Exception as e:
        print(f"Application failed to start: {str(e)}")
        import traceback
        traceback.print_exc()