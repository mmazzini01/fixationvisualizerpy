from fixation_visualizer import FixationVisualizer

FixationVisualizer(
    image_path="face.jpg",
    csv_path="eye_tracking_raw2_fixations.csv",
    mode="both",  # o "saliency" o "scanpath"
    sigma=10,
    alpha=0.6,
    normalized=True
)
