import os
import cv2
import numpy as np
import pandas as pd
from scipy.stats import pearsonr


class SaliencyEvaluator:
    def __init__(self, user_root="users", synthetic_salmaps_root="salmaps", original_images_root="images", output_dir="salmaps_vis_avg", n_prompts=45):
        """
        A class for evaluating and comparing saliency maps from different eye-tracking devices (Tobii and Gazepoint) with synthetic saliency maps.
        
        This evaluator processes user fixation data and compares it with synthetic saliency maps.
        It calculates three metrics for each image:
        - CC (Pearson Correlation Coefficient): Measures linear correlation between maps
        - KL (Kullback-Leibler divergence): Measures information loss between distributions
        - SIM (Similarity): Measures the intersection of the two distributions
        
        Output:
        - Generates overlay images showing the average fixation patterns
        - Creates CSV files in the 'metrics' directory containing:
          * metrics_tobii.csv: Individual metrics for Tobii data
          * metrics_gazepoint.csv: Individual metrics for Gazepoint data
          * global_metrics.csv: Average metrics and standard deviations for both devices
        
        Args:
            user_root: Root path of user data (default: "users")
            synthetic_salmaps_root: Path of synthetic saliency maps (default: "salmaps")
            original_images_root: Path of original images (default: "images")
            output_dir: Output directory for results (default: "salmaps_vis_avg")
            n_prompts: Number of prompts to evaluate (default: 45)
        """
        self.user_root = user_root
        self.synthetic_salmaps_root = synthetic_salmaps_root
        self.original_images_root = original_images_root
        self.output_dir = output_dir
        self.n_prompts = n_prompts
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Dictionaries to store metrics for each source type
        self.metrics_tobii = self._create_metrics_dict()
        self.metrics_gazepoint = self._create_metrics_dict()
        
        # Automatically start evaluation
        self._evaluate()
    
    @staticmethod
    def _create_metrics_dict():
        """Create an empty dictionary for metrics."""
        return {
            "CC": [], 
            "KL": [], 
            "SIM": [],
            "img_id": []  # Store image ID for each metric
        }
    
    @staticmethod
    def cc(pred, gt):
        """Calculate Pearson correlation coefficient."""
        return pearsonr(pred.flatten(), gt.flatten())[0]

    @staticmethod
    def kl(pred, gt):
        """Calculate Kullback-Leibler divergence."""
        pred = pred / (pred.sum() + 1e-8)
        gt = gt / (gt.sum() + 1e-8)
        return np.sum(gt * np.log((gt + 1e-8) / (pred + 1e-8)))

    @staticmethod
    def sim(pred, gt):
        """Calculate similarity."""
        pred = pred / (pred.sum() + 1e-8)
        gt = gt / (gt.sum() + 1e-8)
        return np.sum(np.minimum(pred, gt))
    
    def _save_overlay(self, gt_raw, original_path, img_id, source_type):
        """
        Save the overlay of the average saliency map on the original image.
        
        Args:
            gt_raw: Raw average saliency map
            original_path: Path of the original image
            img_id: Image ID
            source_type: Source type ('tobii' or 'gazepoint')
        """
        avg_img_255 = cv2.normalize(gt_raw, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        heatmap = cv2.applyColorMap(avg_img_255, cv2.COLORMAP_JET)

        if os.path.isfile(original_path):
            original = cv2.imread(original_path)
            original = cv2.resize(original, (heatmap.shape[1], heatmap.shape[0]))
            overlay = cv2.addWeighted(original, 0.7, heatmap, 0.3, 0)
            output_path = os.path.join(self.output_dir, f'img_prompt_{img_id}_{source_type}.png')
            cv2.imwrite(output_path, overlay)
        else:
            print(f"[Warning] Missing original image for img_prompt_{img_id}")
    
    def _compute_metrics(self, synthetic_norm, gt_norm, img_id, metrics_dict):
        """
        Calculate metrics between synthetic and ground truth maps.
        
        Args:
            synthetic_norm: Normalized synthetic saliency map
            gt_norm: Normalized ground truth saliency map
            img_id: Image ID
            metrics_dict: Dictionary to store metrics
        
        Returns:
            dict: Calculated metrics for this image
        """
        metric_cc = self.cc(synthetic_norm, gt_norm)
        metric_kl = self.kl(synthetic_norm, gt_norm)
        metric_sim = self.sim(synthetic_norm, gt_norm)
        
        metrics_dict["img_id"].append(img_id)
        metrics_dict["CC"].append(metric_cc)
        metrics_dict["KL"].append(metric_kl)
        metrics_dict["SIM"].append(metric_sim)
        
        return {
            "CC": metric_cc,
            "KL": metric_kl,
            "SIM": metric_sim
        }
    
    def _collect_saliency_maps(self, img_id, source_type):
        """
        Collect saliency maps from all users for a given image and source type.
        
        Args:
            img_id: Image ID
            source_type: Source type ('tobii' or 'gazepoint')
        
        Returns:
            list: List of saliency maps
        """
        saliency_stack = []
        sub_dir = 'results_tobii' if source_type == 'tobii' else 'results_gazepoint'
        
        for user in os.listdir(self.user_root):
            user_path = os.path.join(self.user_root, user, sub_dir, f'img_prompt_{img_id}.jpg_saliency_map.png')
            if os.path.isfile(user_path):
                img = cv2.imread(user_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)
                saliency_stack.append(img)
            else:
                print(f"[Warning] Missing user map for img_prompt_{img_id} from {user} in {sub_dir}")
        
        return saliency_stack
    
    def process_image(self, img_id):
        """
        Process an image and calculate metrics for both Tobii and Gazepoint.
        
        Args:
            img_id: Image ID to process
        """
        synthetic_path = os.path.join(self.synthetic_salmaps_root, f'img_prompt_{img_id}_5000.jpg')
        original_path = os.path.join(self.original_images_root, f'img_prompt_{img_id}.jpg')
        
        # Check if synthetic map exists
        if not os.path.isfile(synthetic_path):
            print(f"[Warning] Missing synthetic map for img_prompt_{img_id}")
            return
        
        # Load synthetic map
        synthetic = cv2.imread(synthetic_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)
        
        # Process separately for Tobii and Gazepoint
        for source_type in ['tobii', 'gazepoint']:
            saliency_stack = self._collect_saliency_maps(img_id, source_type)
            
            if not saliency_stack:
                print(f"[Skip] No saliency maps found for img_prompt_{img_id} in {source_type}")
                continue
            
            # Calculate average saliency map (GT)
            gt_raw = np.mean(np.stack(saliency_stack), axis=0)
            gt_norm = gt_raw / (gt_raw.max() + 1e-8)
            
            # Resize synthetic map to GT dimensions
            synthetic_resized = cv2.resize(synthetic, (gt_norm.shape[1], gt_norm.shape[0]), interpolation=cv2.INTER_LINEAR)
            synthetic_norm = synthetic_resized / (synthetic_resized.max() + 1e-8)
            
            # Calculate metrics
            metrics_dict = self.metrics_tobii if source_type == 'tobii' else self.metrics_gazepoint
            metrics = self._compute_metrics(synthetic_norm, gt_norm, img_id, metrics_dict)
            
            print(f"[Metrics img_prompt_{img_id} - {source_type}] " +
                  f"CC: {metrics['CC']:.4f}, KL: {metrics['KL']:.4f}, SIM: {metrics['SIM']:.4f}")
            
            # Save overlay
            self._save_overlay(gt_raw, original_path, img_id, source_type)

    def _evaluate(self):
        """
        Evaluate saliency maps for a set of images and save results in CSV.
        """
        # Create metrics directory if it doesn't exist
        metrics_dir = os.path.join(os.getcwd(), 'metrics')
        os.makedirs(metrics_dir, exist_ok=True)
        
        for i in range(1, self.n_prompts + 1):
            self.process_image(i)
        
        # Create DataFrames for single image results
        df_tobii = pd.DataFrame(self.metrics_tobii)
        df_gazepoint = pd.DataFrame(self.metrics_gazepoint)
        
        # Save DataFrames to CSV files in metrics directory
        df_tobii.to_csv(os.path.join(metrics_dir, 'metrics_tobii.csv'), index=False)
        df_gazepoint.to_csv(os.path.join(metrics_dir, 'metrics_gazepoint.csv'), index=False)
        
        # Calculate and save global metrics
        global_metrics = []
        global_metrics.append(self._calculate_global_metrics("Tobii", self.metrics_tobii))
        global_metrics.append(self._calculate_global_metrics("Gazepoint", self.metrics_gazepoint))
        
        # Create and save DataFrame with global metrics
        df_global_metrics = pd.DataFrame(global_metrics)
        df_global_metrics.to_csv(os.path.join(metrics_dir, 'global_metrics.csv'), index=False)
        
        print(f"\nCSV files have been saved in directory: {metrics_dir}")
    
    def _calculate_global_metrics(self, source_name, metrics_dict):
        """
        Calculate global metrics for a given source.
        
        Args:
            source_name: Source name
            metrics_dict: Metrics dictionary
            
        Returns:
            dict: Dictionary containing global metrics (mean and standard deviation)
        """
        global_metrics = {
            "source": source_name,
        }
        
        print(f"\n=== Final average metrics for {source_name} ===")
        for key in ["CC", "KL", "SIM"]:
            if metrics_dict[key]:
                values = np.array(metrics_dict[key])
                mean_val = values.mean()
                std_val = values.std()
                
                # Save both mean and standard deviation
                global_metrics[f"{key}_mean"] = mean_val
                global_metrics[f"{key}_std"] = std_val
                
                print(f"{key}: {mean_val:.4f} ± {std_val:.4f}")
            else:
                global_metrics[f"{key}_mean"] = None
                global_metrics[f"{key}_std"] = None
                print(f"{key}: No data available")
                
        return global_metrics
