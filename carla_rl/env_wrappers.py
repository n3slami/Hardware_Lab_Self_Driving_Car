from gym.core import ObservationWrapper, Wrapper, RewardWrapper
from gym.spaces import Box
import cv2
import numpy as np

from image_processing import lane_detection


class PreprocessCARLAObs(ObservationWrapper):
    def __init__(self, env):
        ObservationWrapper.__init__(self, env)

        self.observation_count = 15
        self.observation_space = Box(0.0, 10.0, (self.observation_count, ))

    def _get_curvature(self, lane_samples):
        if len(lane_samples) == 0:
            return 0
        dx_dt = np.gradient(lane_samples[:, 0])
        dy_dt = np.gradient(lane_samples[:, 1])

        d2x_dt2 = np.gradient(dx_dt)
        d2y_dt2 = np.gradient(dy_dt)

        curvature = np.abs(d2x_dt2 * dy_dt - dx_dt * d2y_dt2) / (dx_dt * dx_dt + dy_dt * dy_dt) ** 1.5
        return curvature.mean()
    
    def _get_single_lane_data(self, lane_data, is_left_lane):
        if lane_data is None:
            sqrt_one_half = np.sqrt(0.5)
            return np.array([0.0, -1.0 if is_left_lane else 1.0, 0.0,
                             sqrt_one_half, sqrt_one_half, sqrt_one_half, -sqrt_one_half])
        
        has_lane = 1.0
        avg_curvature = self._get_curvature(lane_data)
        norm_x = (lane_data[-1][0] - self.im_width / 2) / (self.im_width / 2)
        diffs = np.diff(lane_data, axis=0)
        angles = np.arctan2(diffs[:, 0], diffs[:, 1])
        max_angle_sin, max_angle_cos = np.sin(angles.max()), np.cos(angles.max())
        min_angle_sin, min_angle_cos = np.sin(angles.min()), np.cos(angles.min())
        return np.array([has_lane, norm_x, avg_curvature,
                         max_angle_sin, max_angle_cos, min_angle_sin, min_angle_cos])

    def _get_lane_data(self, obs_image):
        all_lane_data = lane_detection.extract_lane_samples(obs_image.copy())
        left_ind, right_ind = -1, -1
        for i, lane in enumerate(all_lane_data):
            current_bottom_x = lane[-1, 0]
            if current_bottom_x < self.im_width / 2:
                if left_ind == -1 or all_lane_data[left_ind][-1][0] < current_bottom_x:
                    left_ind = i
            else:
                if right_ind == -1 or current_bottom_x < all_lane_data[right_ind][-1][0]:
                    right_ind = i
        left_lane_data = self._get_single_lane_data(all_lane_data[left_ind] if left_ind != -1 else None, True)
        right_lane_data = self._get_single_lane_data(all_lane_data[right_ind] if right_ind != -1 else None, False)
        return np.concatenate([left_lane_data, right_lane_data])

    def observation(self, obs):
        """what happens to each observation"""
        image, speed = obs
        lane_data = self._get_lane_data(image)
        return np.concatenate([lane_data, speed])