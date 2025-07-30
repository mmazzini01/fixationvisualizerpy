# Fixation Visualizer

A Python package for visualizing eye tracking fixation data. This package provides tools to create scanpath visualizations and saliency maps from eye tracking data.

## Features

- Generate scanpath visualizations showing the sequence of fixations
- Create saliency maps with customizable Gaussian blur
- Support for multiple eye-tracking devices (Tobii and Gazepoint)
- Saliency map comparison and evaluation tools with the synthetically generated ones

## Installation

### From Git Repository

```bash
pip install git+https://github.com/mmazzini01/fixationvisualizerpy.git
```

## Usage

### Basic Usage

The package provides a convenient script `run_visualizer.py` that can be used to process eye tracking data and generate visualizations. The script performs two main functions:

1. Generates fixation visualizations for multiple users using the `DualDeviceFixationVisualizer`
2. Evaluates saliency maps using the `SaliencyEvaluator`

To use the script:

1. Ensure your data is organized in the following structure in te example folder:
   ```
   .
   ├── users/
   │   └── 1/
   │       ├── fixation_tobii.csv       # Tobii eye tracking data
   │       └── fixation_gazepoint.csv   # Gazepoint eye tracking data
   ├── images/                          # Directory containing stimulus images
   └── salmaps/                         # Directory containing saliency maps synthetically generated
   ```

2. Run the script:
   ```bash
   python run_visualizer.py
   ```

The script will:
- Process all users' data and generate fixation visualizations
- Compare saliency maps and generate evaluation metrics
- Save the results in the specified output directories

## Author

Matteo Mazzini (matteo.mazzini@estudiantat.upc.edu)
