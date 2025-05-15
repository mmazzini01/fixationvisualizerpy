# Fixation Visualizer

A Python package for visualizing eye tracking fixation data. This package provides tools to create scanpath visualizations and saliency maps from eye tracking data.

## Features

- Generate scanpath visualizations showing the sequence of fixations
- Create saliency maps with customizable Gaussian blur
- Support for both normalized and absolute coordinates
- Command-line interface for easy usage
- Support for multiple eye-tracking devices (Tobii and Gazepoint)
- Saliency map comparison and evaluation tools

## Installation

### From Git Repository

```bash
pip install git+https://github.com/mmazzini01/fixationvisualizerpy.git
```

## Usage

### Basic Usage

```python
from fixation_visualizer import FixationVisualizer

# Create visualization
visualizer = FixationVisualizer(
    image_path="path/to/image.jpg",
    fixation_df=your_fixation_dataframe,  # DataFrame with x, y, duration columns
    output_path="path/to/output",
    mode="both",  # Options: "both", "scanpath", "saliency"
    sigma=10,     # Gaussian blur sigma
    alpha=0.6,    # Transparency of the overlay
    normalized=True  # Whether coordinates are normalized
)
```

### Multiple Device Support

```python
from fixation_visualizer import DualDeviceFixationVisualizer

# Process data from multiple eye-tracking devices
visualizer = DualDeviceFixationVisualizer(
    users_folder='users',
    images_folder='images',
    n_prompt=45,
    fixation_time=5000
)
```

### Saliency Map Evaluation

```python
from fixation_visualizer import SaliencyEvaluator

# Evaluate saliency maps
evaluator = SaliencyEvaluator(
    user_root="users",
    synthetic_salmaps_root="salmaps",
    original_images_root="images",
    output_dir="salmaps_vis_avg",
    n_prompts=45
)
```

## Input Format

The fixation data should be provided as a pandas DataFrame with the following columns:
- `x`: X-coordinate of fixation (normalized or absolute)
- `y`: Y-coordinate of fixation (normalized or absolute)
- `duration`: Duration of fixation in milliseconds

## Output

The visualizations are saved in the specified output directory:
- `{image_name}_scanpath.png`: Visualization of fixation sequence
- `{image_name}_saliency_map.png`: Heat map of fixation density
- `{image_name}_saliency_map_vis.png`: Overlay of saliency map on original image

## Dependencies

- numpy >= 1.19.0
- pandas >= 1.2.0
- opencv-python >= 4.5.0
- matplotlib >= 3.3.0
- scipy >= 1.7.0

## Author

Matteo Mazzini (matteo.mazzini@estudiantat.upc.edu)

## License

This project is licensed under the MIT License - see the LICENSE file for details.
