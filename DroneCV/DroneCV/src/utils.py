"""
Utility Functions
Shared helper functions for the project
"""

import cv2
import numpy as np
import yaml
import time
from typing import Tuple, Dict, Any
from pathlib import Path

def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """
    Load configuration from YAML file
    
    Args:
        config_path: Path to config file
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def save_config(config: Dict[str, Any], config_path: str = "config.yaml"):
    """Save configuration to YAML file"""
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)


def feet_to_meters(feet: float) -> float:
    """Convert feet to meters"""
    return feet * 0.3048


def meters_to_feet(meters: float) -> float:
    """Convert meters to feet"""
    return meters * 3.28084


def draw_text_with_background(image: np.ndarray, 
                               text: str, 
                               position: Tuple[int, int],
                               font_scale: float = 0.6,
                               thickness: int = 2,
                               text_color: Tuple[int, int, int] = (255, 255, 255),
                               bg_color: Tuple[int, int, int] = (0, 0, 0),
                               padding: int = 5) -> np.ndarray:
    """
    Draw text with background rectangle for better visibility
    
    Args:
        image: Input image
        text: Text to draw
        position: (x, y) position
        font_scale: Size of text
        thickness: Thickness of text
        text_color: RGB color of text
        bg_color: RGB color of background
        padding: Padding around text
        
    Returns:
        Image with text drawn
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    # Get text size
    (text_width, text_height), baseline = cv2.getTextSize(
        text, font, font_scale, thickness
    )
    
    x, y = position
    
    # Draw background rectangle
    cv2.rectangle(
        image,
        (x - padding, y - text_height - padding),
        (x + text_width + padding, y + baseline + padding),
        bg_color,
        -1  # Filled
    )
    
    # Draw text
    cv2.putText(
        image,
        text,
        (x, y),
        font,
        font_scale,
        text_color,
        thickness
    )
    
    return image


class FPSCounter:
    """
    Calculate frames per second
    
    Usage:
        fps = FPSCounter()
        while True:
            # ... process frame ...
            fps.update()
            print(f"FPS: {fps.get_fps()}")
    """
    
    def __init__(self, buffer_size: int = 30):
        self.buffer_size = buffer_size
        self.timestamps = []
        
    def update(self):
        """Call this once per frame"""
        self.timestamps.append(time.time())
        
        # Keep only recent timestamps
        if len(self.timestamps) > self.buffer_size:
            self.timestamps.pop(0)
    
    def get_fps(self) -> float:
        """Get current FPS"""
        if len(self.timestamps) < 2:
            return 0.0
        
        elapsed = self.timestamps[-1] - self.timestamps[0]
        if elapsed == 0:
            return 0.0
        
        return (len(self.timestamps) - 1) / elapsed


class VideoWriter:
    """
    Easy video recording
    
    Usage:
        writer = VideoWriter("output.mp4", fps=30)
        writer.write(frame)
        writer.release()
    """
    
    def __init__(self, output_path: str, fps: int = 30, codec: str = 'mp4v'):
        self.output_path = output_path
        self.fps = fps
        self.codec = codec
        self.writer = None
        self.frame_size = None
        
    def write(self, frame: np.ndarray):
        """Write a frame to video"""
        if self.writer is None:
            # Initialize writer on first frame
            self.frame_size = (frame.shape[1], frame.shape[0])
            fourcc = cv2.VideoWriter_fourcc(*self.codec)
            self.writer = cv2.VideoWriter(
                self.output_path, fourcc, self.fps, self.frame_size
            )
        
        self.writer.write(frame)
    
    def release(self):
        """Close the video file"""
        if self.writer:
            self.writer.release()


def ensure_directory(path: str):
    """Create directory if it doesn't exist"""
    Path(path).mkdir(parents=True, exist_ok=True)


def get_project_root() -> Path:
    """Get the project root directory"""
    return Path(__file__).parent.parent


def calculate_iou(box1: Tuple[int, int, int, int], 
                  box2: Tuple[int, int, int, int]) -> float:
    """
    Calculate Intersection over Union (IoU) between two bounding boxes
    
    Args:
        box1, box2: Bounding boxes as (x1, y1, x2, y2)
        
    Returns:
        IoU value between 0 and 1
    """
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    # Calculate intersection
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)
    
    if x2_i < x1_i or y2_i < y1_i:
        return 0.0
    
    intersection = (x2_i - x1_i) * (y2_i - y1_i)
    
    # Calculate union
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0


# Test utilities
if __name__ == "__main__":
    # Test config loading
    config = load_config()
    print("Config loaded:", config.keys())
    
    # Test conversions
    print(f"5 feet = {feet_to_meters(5):.2f} meters")
    print(f"2 meters = {meters_to_feet(2):.2f} feet")
    
    # Test FPS counter
    fps = FPSCounter()
    for _ in range(10):
        time.sleep(0.033)  # ~30 FPS
        fps.update()
    print(f"FPS: {fps.get_fps():.1f}")