# First, install system dependencies (run these commands in terminal):
'''
# For Ubuntu/Debian:
sudo apt-get update
sudo apt-get install -y libgl1-mesa-glx
sudo apt-get install -y libglib2.0-0

# Then install Python dependencies:
pip install numpy opencv-python-headless scikit-learn pandas
'''

import numpy as np
try:
    import cv2
except ImportError:
    print("Attempting to install opencv-python-headless instead of opencv-python...")
    import subprocess
    subprocess.check_call(["pip", "install", "opencv-python-headless"])
    import cv2

from sklearn.preprocessing import StandardScaler
import pandas as pd
from pathlib import Path

class WaterQualityAnalyzer:
    def __init__(self):
        # Define typical ranges for water quality parameters based on color values
        self.parameter_ranges = {
            'turbidity': {
                'low': [150, 180, 200],    # RGB values for clear water
                'high': [100, 120, 140]    # RGB values for turbid water
            },
            'chlorophyll': {
                'low': [50, 150, 150],     # RGB values for low chlorophyll
                'high': [20, 180, 60]      # RGB values for high chlorophyll
            },
            'dissolved_oxygen': {
                'low': [60, 100, 140],     # RGB values for low DO
                'high': [100, 150, 200]    # RGB values for high DO
            }
        }

    def load_image(self, image_path):
        """Load and preprocess the satellite image."""
        try:
            # Try reading with OpenCV first
            img = cv2.imread(str(image_path))
            if img is None:
                raise ValueError(f"Could not load image at {image_path}")
            
            # Convert BGR to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return img
        except Exception as e:
            print(f"Error loading image with OpenCV: {e}")
            # Fallback to using PIL if OpenCV fails
            try:
                from PIL import Image
                import numpy as np
                
                img = Image.open(image_path)
                img = np.array(img)
                return img
            except Exception as e2:
                raise Exception(f"Failed to load image with both OpenCV and PIL: {e2}")
    
    def extract_water_pixels(self, image):
        """Extract pixels that are likely to be water based on color thresholds."""
        try:
            # Normalize image values to 0-1 range
            image_norm = image.astype(float) / 255.0
            
            # More robust water detection using multiple criteria
            # Water typically has higher blue values and lower red values
            blue_dominant = image_norm[:,:,2] > image_norm[:,:,0]
            
            # Water typically has lower overall brightness
            brightness = np.mean(image_norm, axis=2)
            not_too_bright = brightness < 0.8
            
            # Combine criteria
            water_mask = blue_dominant & not_too_bright
            
            water_pixels = image[water_mask]
            
            if len(water_pixels) == 0:
                raise ValueError("No water pixels detected in the image")
                
            return water_pixels
            
        except Exception as e:
            raise Exception(f"Error in water pixel extraction: {e}")

    def calculate_color_metrics(self, water_pixels):
        """Calculate various color-based metrics from water pixels."""
        metrics = {
            'mean_rgb': np.mean(water_pixels, axis=0),
            'std_rgb': np.std(water_pixels, axis=0),
            'median_rgb': np.median(water_pixels, axis=0),
            'min_rgb': np.min(water_pixels, axis=0),
            'max_rgb': np.max(water_pixels, axis=0)
        }
        return metrics

    def estimate_parameter(self, color_metrics, parameter):
        """Estimate water quality parameter based on color metrics."""
        try:
            mean_rgb = color_metrics['mean_rgb']
            low_range = np.array(self.parameter_ranges[parameter]['low'])
            high_range = np.array(self.parameter_ranges[parameter]['high'])
            
            # Calculate similarity to reference colors
            low_dist = np.linalg.norm(mean_rgb - low_range)
            high_dist = np.linalg.norm(mean_rgb - high_range)
            
            # Normalize to 0-100 scale
            total_dist = low_dist + high_dist
            if total_dist == 0:
                return 50
            
            score = (low_dist / total_dist) * 100
            
            # Ensure score is within 0-100 range
            return max(0, min(100, score))
            
        except Exception as e:
            raise Exception(f"Error in parameter estimation: {e}")

    def analyze_image(self, image_path):
        """Perform complete water quality analysis on an image."""
        try:
            # Load and process image
            image = self.load_image(image_path)
            water_pixels = self.extract_water_pixels(image)
            
            # Calculate color metrics
            color_metrics = self.calculate_color_metrics(water_pixels)
            
            # Estimate parameters
            results = {
                'turbidity': self.estimate_parameter(color_metrics, 'turbidity'),
                'chlorophyll': self.estimate_parameter(color_metrics, 'chlorophyll'),
                'dissolved_oxygen': self.estimate_parameter(color_metrics, 'dissolved_oxygen'),
                'color_metrics': {
                    'mean_rgb': color_metrics['mean_rgb'].tolist(),
                    'std_rgb': color_metrics['std_rgb'].tolist(),
                    'median_rgb': color_metrics['median_rgb'].tolist(),
                    'min_rgb': color_metrics['min_rgb'].tolist(),
                    'max_rgb': color_metrics['max_rgb'].tolist()
                }
            }
            
            # Add quality indicators
            results['water_quality_index'] = np.mean([
                results['turbidity'],
                results['chlorophyll'],
                results['dissolved_oxygen']
            ])
            
            return results
            
        except Exception as e:
            return {'error': str(e)}

def main():
    # Example usage
    analyzer = WaterQualityAnalyzer()
    
    # Replace with your image path
    image_path = Path("satellite-image.jpg")
    
    try:
        results = analyzer.analyze_image(image_path)
        
        if 'error' in results:
            print(f"Error: {results['error']}")
            return
        
        print("\nWater Quality Analysis Results:")
        print("-" * 30)
        print(f"Overall Water Quality Index: {results['water_quality_index']:.2f}/100")
        print(f"Turbidity Score: {results['turbidity']:.2f}/100")
        print(f"Chlorophyll Level Score: {results['chlorophyll']:.2f}/100")
        print(f"Dissolved Oxygen Score: {results['dissolved_oxygen']:.2f}/100")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
