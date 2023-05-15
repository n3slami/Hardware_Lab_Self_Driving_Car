import cv2
import numpy as np
import argparse
from .utils import *


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

# Image contants, including cropping and filtering constants
HORIZON_RATIO = 0.58
BOTTOM_RATIO = 0.9

# Connected component constants
MERGE_DIST_THRESHOLD = 20

# Lane line interpolation
LANE_LINE_LOW_FRACTION = 0.03
LANE_LINE_HIGH_FRACTION = 0.97
LANE_LINE_SAMPLE_COUNT = 15


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
    for i in range(len(lines)):
        # print(i, lines[i])
        if lines[i][0][1] > lines[i][0][3]:
            lines[i][0] = lines[i][0][np.array((2, 3, 0, 1))]
    lines = list(filter(lambda line: is_valid_line(line, H=H, W=W), lines))
    lines = np.concatenate(lines, axis=0)
    return lines, img


def dsu_find(par, u):
    if par[u] != -1:
        par[u] = dsu_find(par, par[u])
        return par[u]
    return u


def dsu_merge(par, u, v):
    u, v = dsu_find(par, u), dsu_find(par, v)
    if u == v:
        return
    par[v] = u


def get_line_connect_components(lines):
    par = -np.ones(lines.shape[0], dtype=np.int32)
    for i, line_a in enumerate(lines):
        for _j, line_b in enumerate(lines[i + 1:]):
            j = _j + i + 1
            _, _, dist = line_segment_dist(line_a, line_b)
            if dist < MERGE_DIST_THRESHOLD:
                dsu_merge(par, i, j)
    
    res = np.zeros_like(par)
    par_mapping = {}
    component_count = 0
    for i in range(len(par)):
        node = dsu_find(par, i)
        if node not in par_mapping:
            par_mapping[node] = component_count
            component_count += 1
        res[i] = par_mapping[node]
    return res


def interpolate_line_components(lines, components, clip=True):
    res = []
    # print("______________________", lines, components)
    for component_id in range(max(components) + 1):
        component_lines = lines[components == component_id]
        # print("#######################", component_id)
        # print(component_lines)
        component_lines = component_lines[component_lines[:, 1].argsort()]
        if clip:
            sample_y = np.linspace(min(component_lines[:, 1]) * (LANE_LINE_LOW_FRACTION + 1),
                                max(component_lines[:, 3]) * LANE_LINE_HIGH_FRACTION, num=LANE_LINE_SAMPLE_COUNT)
        else:
            sample_y = np.linspace(min(component_lines[:, 1]), max(component_lines[:, 3]), num=LANE_LINE_SAMPLE_COUNT)

        sample_x = np.zeros_like(sample_y)
        for i, y in enumerate(sample_y):
            x_list = []
            for line in component_lines:
                if line[1] <= y and y <= line[3]:
                    alpha = (y - line[1]) / (line[3] - line[1])
                    x_list.append((1 - alpha) * line[0] + alpha * line[2])
            sample_x[i] = np.mean(np.array(x_list))
        res.append(np.stack((sample_x, sample_y), axis=1))
    return res


def add_lines_to_image(img, lines, color=[0, 0, 255], thickness=7):
    H, W, _ = img.shape
    for line in lines:
        x1, y1, x2, y2 = line
        cv2.line(img, (x1, y1), (x2, y2), color, thickness)


def convert_lane_samples_to_lines(interpolations, img_shape):
    interp_lines = []
    interp_components = []
    for component_num, interp in enumerate(interpolations):
        l = []
        for i in range(len(interp) - 1):
            l.append(np.concatenate(interp[i:i + 2], axis=-1))
        MEAN_COUNT = min(5, len(l))

        slope = 0
        for i in range(MEAN_COUNT):
            slope += (l[i][2] - l[i][0]) / (l[i][3] - l[i][1])
        slope /= MEAN_COUNT
        l[0][0], l[0][1] = l[0][0] - l[0][1] * slope, 0

        slope = 0
        for i in range(1, MEAN_COUNT + 1):
            slope += (l[-i][2] - l[-i][0]) / (l[-i][3] - l[-i][1])
        slope /= MEAN_COUNT
        l[-1][2], l[-1][3] = l[-1][2] + (img_shape[0] - l[-1][3]) * slope, img_shape[0]
        interp_lines.append(np.array(l))
        interp_components.append(np.ones(len(l), dtype=np.int32) * component_num)

    interp_lines = np.concatenate(interp_lines, axis=0).astype(np.int32)
    interp_components = np.concatenate(interp_components, axis=0)
    return interp_lines, interp_components


def render_lines(img, lines):
    rgb_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    add_lines_to_image(rgb_img, lines)
    cv2.imshow('image', rgb_img)
    cv2.waitKey(0)


def extract_lane_samples(img, debug=False):
    lines, img = extract_raw_lane_lines(img, debug=True)
    components = get_line_connect_components(lines)
    interpolations = interpolate_line_components(lines, components)
    
    if debug:
        render_lines(img, lines)
    
    interp_lines, interp_components = convert_lane_samples_to_lines(interpolations, img.shape)

    if debug:
        render_lines(img, interp_lines)

    interpolations = interpolate_line_components(interp_lines, interp_components, clip=False)
    interp_lines, interp_components = convert_lane_samples_to_lines(interpolations, img.shape)

    if debug:
        print(interpolations)
        render_lines(img, interp_lines)
    
    return interpolations


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find the street lanes in an image extracted with the microlino car in CARLA.")
    parser.add_argument("filename", type=str, default=None, nargs=1)
    args = parser.parse_args()

    img = cv2.imread(args.filename[0], cv2.IMREAD_COLOR)
    
    cv2.imshow('image', img)
    cv2.waitKey(0)

    extract_lane_samples(img, debug=True)

    cv2.destroyAllWindows()
