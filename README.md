# Fixation Visualizer

A Python package for visualizing eye tracking fixation data. This package provides tools to create scanpath visualizations and saliency maps from eye tracking data.

## Features

- Generate scanpath visualizations showing the sequence of fixations
- Create saliency maps with customizable Gaussian blur
- Support for both normalized and absolute coordinates
- Command-line interface for easy usage

## Installation

```bash
pip install fixation_visualizer
```

Or install from source:

```bash
git clone https://github.com/mmazzini01/fixationvisualizerpy.git
cd fixationvisualizerpy
pip install -e .
```

## Usage

### As a Python Package

```python
from fixation_visualizer import FixationVisualizer

# Create visualization
visualizer = FixationVisualizer(
    image_path="path/to/image.jpg",
    csv_path="path/to/fixations.csv",
    mode="both",  # Options: "both", "scanpath", "saliency"
    sigma=10,     # Gaussian blur sigma
    alpha=0.6,    # Transparency of the overlay
    normalized=True  # Whether coordinates are normalized
)
```

### Command Line Interface

```bash
python examples/run_visualizer.py -i path/to/image.jpg -c path/to/fixations.csv
```

Optional arguments:
- `-m, --mode`: Visualization mode (both/scanpath/saliency)
- `-s, --sigma`: Sigma value for Gaussian blur
- `-a, --alpha`: Alpha value for overlay transparency
- `-n, --normalized`: Flag for normalized coordinates

## Example Usage

The package includes example files in the `examples` directory:

```bash
# Run with example files
python examples/run_visualizer.py -i examples/face.jpg -c examples/eye_tracking_raw2_fixations.csv
```

The results will be saved in the `results` directory.

## Input Format

The CSV file should contain the following columns:
- `x`: X-coordinate of fixation (normalized or absolute)
- `y`: Y-coordinate of fixation (normalized or absolute)
- `duration`: Duration of fixation in milliseconds

## Output

The visualizations are saved in the `results` directory:
- `{image_name}_scanpath.png`: Visualization of fixation sequence
- `{image_name}_saliency_map.png`: Heat map of fixation density

## Author

Matteo Mazzini (matteo.mazzini@estudiantat.upc.edu)
