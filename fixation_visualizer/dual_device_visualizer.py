from .visualizer import FixationVisualizer
import os
import pandas as pd

class DualDeviceFixationVisualizer:
    """
    A class for visualizing fixation data from two different eye-tracking devices (Tobii and Gazepoint).
    
    This class processes fixation data from multiple users and devices. It processes the fixation data by filtering based on
    duration, and creates saliency maps with and visual overlays for each device.
    The folder structure should be:
    users/
        1/
            fixations_tobii.csv
            fixations_gazepoint.csv
        2/
            fixations_tobii.csv
            fixations_gazepoint.csv
    
    Args:
        users_folder (str): Path to the folder containing user data. Each user folder should contain
                           two CSV files: one for Tobii data and one for Gazepoint data.
        images_folder (str): Path to the folder containing the image prompts to be analyzed.
        n_prompt (int): Number of image prompts to process (default: 30).
        fixation_time (int): Maximum duration in milliseconds to analyze for each image (default: 5000).
   
    Output:
        For each user and each image prompt, generates two visualization results:
        - results_tobii/: Contains saliency maps with and visual overlays for each image
        - results_gazepoint/: Contains saliency maps with and visual overlays for each image
    """
    def __init__(self, users_folder='users', images_folder='images', n_prompt=30, fixation_time=5000):
        self.users_folder = users_folder
        self.images_folder = images_folder
        self.n_prompt = n_prompt
        self.fixation_time = fixation_time
        self.process_user_data()

    def process_user_data(self):
        for user_folder in os.listdir(self.users_folder):
            if 'vertices' in user_folder:
                continue
            joined_path = os.path.join(self.users_folder, user_folder, 'session_1')
            fixations_tobii, fixations_gazepoint = None, None

            for f in os.listdir(joined_path):
                if f.endswith('.csv') and 'tobii' in f.lower():
                    fixations_tobii = pd.read_csv(os.path.join(joined_path, f))

            if fixations_tobii is None and fixations_gazepoint is None:
                print(f"No Tobii or Gazepoint data in {joined_path}. Skipping.")
                continue

            self._process_images(joined_path, fixations_tobii, fixations_gazepoint)

    def _process_images(self, joined_path, fixations_tobii, fixations_gazepoint):
        for i in range(1, self.n_prompt + 1):
            image_path = os.path.join(self.images_folder, f'img_prompt_{i}.jpg')
            if not os.path.exists(image_path):
                print(f"Image {image_path} does not exist. Skipping.")
                continue

            # Process Tobii if available
            if fixations_tobii is not None:
                if str(i) in fixations_tobii['USER'].values:
                    df_filtered_tobii = self._filter_fixations(fixations_tobii, i)
                    self._visualize(image_path, df_filtered_tobii, joined_path, 'results_tobii', 'tobii')
                else:
                    print(f"User {i} not found in Tobii data. Skipping.")

            # Process Gazepoint if available
            if fixations_gazepoint is not None:
                # Try to infer user column name for Gazepoint, fallback to 'USER' if present
                user_col = 'USER' if 'USER' in fixations_gazepoint.columns else None
                if user_col is None or str(i) in fixations_gazepoint.get(user_col, []):
                    df_filtered_gzp = self._filter_fixations_gzp(fixations_gazepoint, i)
                    self._visualize(image_path, df_filtered_gzp, joined_path, 'results_gazepoint', 'gzp')
                else:
                    print(f"User {i} not found in Gazepoint data. Skipping.")

    def _filter_fixations(self, fixations, user_id):
        df_filtered = fixations[fixations['USER'] == str(user_id)].copy()
        df_filtered['effective_duration'] = df_filtered['recording_timestamp'] - df_filtered['recording_timestamp'].iloc[0]
        df_filtered = df_filtered[df_filtered['effective_duration'] <= self.fixation_time]
        if not df_filtered.empty:
            last_row = df_filtered.iloc[-1]
            if last_row['effective_duration'] + last_row['duration'] > self.fixation_time:
                df_filtered.at[last_row.name, 'duration'] = self.fixation_time - last_row['effective_duration']
        return df_filtered

    def _filter_fixations_gzp(self, fixations, user_id):
        fixations = fixations.rename(
            columns={"FPOGX": "x", "FPOGY": "y", "FPOGD": "duration", "FPOGID": "ID", 'TIMETICK(f=10000000)': 'recording_timestamp'}
        )
        fixations['duration'] = fixations['duration'] * 1000
        fixations['recording_timestamp'] = fixations['recording_timestamp'] / 10000
        return self._filter_fixations(fixations, user_id)

    def _visualize(self, image_path, fixation_df, output_path, results_folder, label):
        try:
            print(f"Result for {image_path} with {label}:")
            visualizer = FixationVisualizer(
                image_path=image_path,
                fixation_df=fixation_df,
                output_path=os.path.join(output_path, results_folder),
                mode='saliency',
                sigma=20,
                alpha=0.3,
                normalized=True
            )
        except Exception as e:
            print(f"Error processing {image_path} with {label}: {str(e)}")




