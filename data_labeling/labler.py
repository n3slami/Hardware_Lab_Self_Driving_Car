import numpy as np
from scipy.spatial.distance import cdist


# Constants
CLASSIFY_DIST_THRESHOLD = 10


def get_labels(ground_truth_description, lanes):
    res = np.zeros(lanes.shape[0], dtype=np.int32)
    for i, lane in enumerate(lanes):
        left_dist = np.min(cdist(ground_truth_description[0], lane))
        right_dist = np.min(cdist(ground_truth_description[1], lane))
        if left_dist < CLASSIFY_DIST_THRESHOLD:
            res[i] = 1      # Left lanes are labeled as 1
        elif right_dist < CLASSIFY_DIST_THRESHOLD:
            res[i] = 2      # Right lanes are labeled as 2
    return res
