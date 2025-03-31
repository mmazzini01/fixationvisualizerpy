import os
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt

class FixationVisualizer:
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
            self.fixations = self._scale_fixations(self.fixations, self.width, self.height)
        if self.mode in ["scanpath", "both"]:
            self._plot_scanpath()

        if self.mode in ["saliency", "both"]:
            self._generate_saliency_map()

    def _convert_fixations_to_numpy(self, fixations):
        x = fixations['x'].astype(int).to_numpy()
        y = fixations['y'].astype(int).to_numpy()
        durations = fixations['duration'].astype(float).to_numpy()
        norm_durations = self._normalize_durations(durations)
        return np.array(list(zip(x, y, norm_durations)), dtype=[('x', 'i4'), ('y', 'i4'), ('w', 'f4')])

    def _normalize_durations(self, durations):
        if durations.max() == durations.min():
            return np.ones_like(durations)
        return (durations - durations.min()) / (durations.max() - durations.min())

    def _scale_fixations(self, df, width, height):
        df = df.copy()
        df['x'] = df['x'] * width
        df['y'] = df['y'] * height
        return df

    def _plot_scanpath(self):
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
        fix_np = self._convert_fixations_to_numpy(self.fixations)
        saliency_map = np.zeros((self.height, self.width), dtype=np.float32)
        for point in fix_np:
            x, y, weight = point['x'], point['y'], point['w']

            if 0 <= x < self.width and 0 <= y < self.height:
                gaussian = np.zeros((self.height, self.width), dtype=np.float32)
                intensity = 255 * 3.0 * weight
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