import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse


# Smoothing, feature extraction, and line detection constants
BLUR_KERNEL_SIZE = 15
CANNY_LOW_T = 50
CANNY_HIGH_T = 100
IGNORE_MASK_COLOR = 255
HOUGH_RHO_QUANT = 7
HOUGH_THETA_QUANT = np.pi / 180
HOUGH_THRESHOLD = 100
MIN_LINE_LEN = 80
MAX_LINE_GAP = 250

# Image contants, including cropping and filtering ones
HORIZON_RATIO = 0.58
BOTTOM_RATIO = 0.9
SLOPE_FILTER_THRESHOLD = 0.25
BOTTOM_IMAGE_SLOPE_FILTER_THRESHOLD = 0.4
BOTTOM_IMAGE_RATIO = 0.9


def is_valid_line(line, H, W):
    x1, y1, x2, y2 = line[0]
    slope = (y2 - y1) / (x2 - x1)
    if abs(slope) < SLOPE_FILTER_THRESHOLD:
        return False
    if max(x1, x2) > W // 2 and slope < 0:
        return False
    if min(x1, x2) < W // 2 and slope > 0:
        return False
    if max(y1, y2) > H * BOTTOM_IMAGE_RATIO and abs(slope) < BOTTOM_IMAGE_SLOPE_FILTER_THRESHOLD:
        return False
    return True


def extract_raw_lane_lines(rgb_img, debug=False):
    H, W, _ = rgb_img.shape
    img = rgb_img[int(H * HORIZON_RATIO):int(H * BOTTOM_RATIO), :]
    H = img.shape[0]
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Remove some of the noise
    img = cv2.GaussianBlur(img, (BLUR_KERNEL_SIZE, BLUR_KERNEL_SIZE), 0)

    if debug:
        cv2.imshow('image', img)
        cv2.waitKey(0)
    
    img = cv2.Canny(img, CANNY_LOW_T, CANNY_HIGH_T)

    if debug:
        cv2.imshow('image', img)
        cv2.waitKey(0)

    # Guess we ignored the Sobel filter? =)
    # sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
    # sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)

    lines = cv2.HoughLinesP(img, HOUGH_RHO_QUANT, HOUGH_THETA_QUANT, HOUGH_THRESHOLD, np.array([]), 
                            minLineLength=MIN_LINE_LEN, maxLineGap=MAX_LINE_GAP)
    lines = list(filter(lambda line: is_valid_line(line, H=H, W=W), lines))
    return lines, img




def draw_lines(img, lines, color=[0, 0, 255], thickness=7):
    H, W, _ = img.shape
    for line in lines:
        for x1, y1, x2, y2 in line:
            # print(f"DRAWING LINE: {line}")
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find the street lanes in an image extracted with the microlino car in CARLA.")
    parser.add_argument("filename", type=str, default=None, nargs=1)
    args = parser.parse_args()

    img = cv2.imread(args.filename[0], cv2.IMREAD_COLOR)
    
    cv2.imshow('image', img)
    cv2.waitKey(0)

    lines, img = extract_raw_lane_lines(img, debug=True)
    
    res = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    draw_lines(res, lines)
    
    cv2.imshow('image', res)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
