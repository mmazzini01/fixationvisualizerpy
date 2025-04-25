import os
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt

class FixationVisualizer:
    """
    A class for visualizing eye tracking fixation data by generating scanpath and saliency map visualizations.

    This class takes eye tracking data from a CSV file and creates visualizations to help analyze where users looked at an image.
    It can generate two types of visualizations:
    1. A scanpath showing the sequence and order of fixations as connected points
    2. A heatmap (saliency map) showing the density of visual attention across the image

    The scanpath visualization uses different colors to show the order of fixations, with point sizes indicating fixation duration.
    The saliency map creates a heatmap overlay on the original image using Gaussian blurs weighted by fixation durations.

    Args:
        image_path (str): Path to the image file to analyze
        csv_path (str): Path to CSV file with fixation data (columns: x, y, duration)
        mode (str, optional): Type of visualization to create - "both", "scanpath", or "saliency". Defaults to "both"
        sigma (int, optional): Controls the spread of the Gaussian blur for the saliency map. Defaults to 10
        alpha (float, optional): Controls the transparency of the saliency map overlay (0-1). Defaults to 0.6
        normalized (bool, optional): Whether input coordinates are normalized (0-1) or in pixels. Defaults to True

    Raises:
        FileNotFoundError: If the specified image file cannot be found/loaded
        ValueError: If the CSV file is missing required columns (x, y, duration)
    """
    def __init__(self, image_path, csv_path, mode="both", sigma=10, alpha=0.6, normalized=True):
        self.image_path = image_path
        self.csv_path = csv_path
        self.mode = mode
        self.sigma = sigma
        self.alpha = alpha
        self.normalized = normalized

        self.image = cv2.imread(image_path)
        if self.image is None:
            raise FileNotFoundError(f"Could not load image: {image_path}")
        self.fixations = pd.read_csv(csv_path)
        required_cols = {'x', 'y', 'duration'}
        if not required_cols.issubset(self.fixations.columns):
                raise ValueError(f"CSV must contain columns: {required_cols}")
        self.height, self.width = self.image.shape[:2]

        if self.normalized:
            # 🔁 Se fissazioni sono normalizzate, convertile in 'height' units e filtra quelle dentro immagine
            self.fixations = self._convert_and_filter_fixations_psychopy(
                self.fixations,
                img_width_px=self.width,
                img_height_px=self.height,
                img_size_height_units=0.5,
                screen_aspect=16 / 9  # oppure calcola dinamicamente se vuoi
            )
        else:
            # 🔁 Se sono già in pixel, riscalale in base alla dimensione dell'immagine
            self.fixations = self._scale_fixations(self.fixations, self.width, self.height)
        if self.mode in ["scanpath", "both"]:
            self._plot_scanpath()

        if self.mode in ["saliency", "both"]:
            self._generate_saliency_map()

    def _convert_fixations_to_numpy(self, fixations):
        """
        Converts fixation data from pandas DataFrame to structured numpy array.
        
        Args:
            fixations (pd.DataFrame): DataFrame containing x, y coordinates and durations
            
        Returns:
            np.ndarray: Structured array with x, y coordinates and normalized weights
        """
        x = fixations['x'].astype(int).to_numpy()
        y = fixations['y'].astype(int).to_numpy()
        durations = fixations['duration'].astype(float).to_numpy()
        norm_durations = self._normalize_durations(durations)
        return np.array(list(zip(x, y, norm_durations)), dtype=[('x', 'i4'), ('y', 'i4'), ('w', 'f4')])

    def _normalize_durations(self, durations):
        """
        Normalizes fixation durations to range [0,1]. Returns array of 1s if all durations are equal.
        
        Args:
            durations (np.ndarray): Array of fixation durations
            
        Returns:
            np.ndarray: Normalized durations between 0 and 1
        """
        if durations.max() == durations.min():
            return np.ones_like(durations)
        return (durations - durations.min()) / (durations.max() - durations.min())

    def _scale_fixations(self, df, width, height):
        """
        Scales normalized fixation coordinates (0-1) to image pixel coordinates.
        
        Args:
            df (pd.DataFrame): DataFrame with normalized coordinates
            width (int): Image width in pixels
            height (int): Image height in pixels
            
        Returns:
            pd.DataFrame: DataFrame with scaled coordinates
        """
        df = df.copy()
        df['x'] = df['x'] * width
        df['y'] = df['y'] * height
        return df
    
    def _convert_and_filter_fixations_psychopy(self, fixations_norm, img_width_px, img_height_px, img_size_height_units=0.5, screen_aspect=16/9):
        """
        Converte fissazioni normalizzate (0-1) in coordinate 'height' di PsychoPy,
        e filtra solo quelle dentro l'immagine centrata (0,0) con size img_size_height_units.
        """
        img_height_units = img_size_height_units
        img_width_units = img_height_units * (img_width_px / img_height_px)

        def norm_to_height_units(x_norm, y_norm):
            x_centered = x_norm - 0.5
            y_centered = 0.5 - y_norm  # inverte asse Y
            x = x_centered * screen_aspect
            y = y_centered
            return x, y

        filtered = []
        for _, row in fixations_norm.iterrows():
            x, y = norm_to_height_units(row['x'], row['y'])

            if abs(x) <= img_width_units / 2 and abs(y) <= img_height_units / 2:
                filtered.append({'x': x, 'y': y, 'duration': row['duration']})

        return pd.DataFrame(filtered)


    def _plot_scanpath(self):
        """
        Creates scanpath visualization showing sequence of fixations.
        Points are sized by duration and colored by order.
        Saves plot to results/{image_name}_scanpath.png.
        """
        durations = self.fixations['duration'].to_numpy()
        norm_durations = self._normalize_durations(durations)
        img_rgb = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.imshow(img_rgb)

        num_fix = len(self.fixations)
        colors = plt.cm.plasma(np.linspace(0, 1, num_fix))

        for i in range(num_fix):
            x = self.fixations.iloc[i]['x']
            y = self.fixations.iloc[i]['y']
            size = 50 + norm_durations[i] * 150
            ax.scatter(x, y, s=size, color=colors[i], edgecolors='black', zorder=3)
            if i > 0:
                x_prev = self.fixations.iloc[i - 1]['x']
                y_prev = self.fixations.iloc[i - 1]['y']
                ax.plot([x_prev, x], [y_prev, y], color=colors[i], linewidth=2, zorder=2)

        ax.axis("off")
        os.makedirs("results", exist_ok=True)
        image_name = os.path.basename(self.image_path)
        plt.savefig(f"results/{image_name}_scanpath.png", bbox_inches='tight', dpi=300)
        plt.close(fig)

    def _generate_saliency_map(self):
        """
        Creates saliency map visualization showing density of visual attention.
        Uses Gaussian blurs weighted by fixation duration.
        Saves map to results/{image_name}_saliency_map.png.
        """
        fix_np = self._convert_fixations_to_numpy(self.fixations)
        saliency_map = np.zeros((self.height, self.width), dtype=np.float32)
        for point in fix_np:
            x, y, weight = point['x'], point['y'], point['w']

            if 0 <= x < self.width and 0 <= y < self.height:
                gaussian = np.zeros((self.height, self.width), dtype=np.float32)
                intensity = float(255 * 3.0 * weight)
                radius = int(self.sigma)
                cv2.circle(gaussian, (x, y), radius, intensity, -1)
                gaussian = cv2.GaussianBlur(gaussian, (0, 0), self.sigma)
                saliency_map += gaussian

        if np.max(saliency_map) > 0:
            saliency_map = np.nan_to_num(saliency_map)
            saliency_map = np.power(saliency_map, 1.5)
            saliency_map = cv2.normalize(saliency_map, None, 0, 255, cv2.NORM_MINMAX)
            saliency_map = np.nan_to_num(saliency_map).astype(np.uint8)
        else:
            saliency_map = np.zeros_like(saliency_map, dtype=np.uint8)

        heatmap = cv2.applyColorMap(saliency_map, cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(self.image, 1 - self.alpha, heatmap, self.alpha, 0)

        os.makedirs("results", exist_ok=True)
        image_name = os.path.basename(self.image_path)
        cv2.imwrite(f"results/{image_name}_saliency_map.png", overlay)
