#!/usr/bin/env python3
import argparse
from fixation_visualizer import FixationVisualizer

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected (true/false).')

def main():
    parser = argparse.ArgumentParser(description='Visualize eye tracking fixation data')

    parser.add_argument('--image', '-i', required=True, help='Path to the image file')
    parser.add_argument('--csv', '-c', required=True, help='Path to the CSV file containing fixation data')
    parser.add_argument('--mode', '-m', choices=['both', 'scanpath', 'saliency'], default='both',
                        help='Visualization mode (default: both)')
    parser.add_argument('--sigma', '-s', type=float, default=10,
                        help='Sigma value for Gaussian blur (default: 10)')
    parser.add_argument('--alpha', '-a', type=float, default=0.6,
                        help='Alpha value for overlay transparency (default: 0.6)')
    parser.add_argument('--normalized', '-n', type=str2bool, nargs='?', const=True, default=True,
                        help='Whether the coordinates in CSV are normalized (default: True). '
                             'Usage: --normalized true / false (or just --normalized to set True)')

    args = parser.parse_args()

    try:
        FixationVisualizer(
            image_path=args.image,
            csv_path=args.csv,
            mode=args.mode,
            sigma=args.sigma,
            alpha=args.alpha,
            normalized=args.normalized
        )
        print("Visualization completed successfully. Results saved in the 'results' directory.")
    except Exception as e:
        print(f"Error: {str(e)}")
        return 1

    return 0

if __name__ == "__main__":
    exit(main())
