import numpy as np
import cv2
import os
import argparse

from image_processing.lane_detection import extract_lane_samples
from image_processing.lane_detection import HORIZON_RATIO as DETECTION_RATIO
from data_labeling.specifier import HORIZON_RATIO as LABELING_RATIO
from data_labeling.labler import get_labels

def dir_path(string):
    if os.path.isdir(string):
        return string
    else:
        raise NotADirectoryError(string)


def setup_and_store_labels(dirname):
    img = cv2.imread(f"{dirname}/1.jpg")
    lane_samples = extract_lane_samples(img)
    for lane_data in lane_samples:
        lane_data[:, 1] += int(img.shape[0] * (DETECTION_RATIO - LABELING_RATIO))
    lane_samples = np.array(lane_samples)

    ground_truth = np.load(f"{dirname}/label.npy")
    lane_labels = get_labels(ground_truth, lane_samples)

    np.save(f"{dirname}/lane_samples.npy", lane_samples)
    np.save(f"{dirname}/lane_labels.npy", lane_labels)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="""Tool to generate stored lanes and labels for a TUSimple test.""")
    parser.add_argument("dirname", type=dir_path, default=None, nargs=1)
    args = parser.parse_args()
    
    setup_and_store_labels(args.dirname[0])