import numpy as np
import cv2
import argparse
import os


# Cropping Constants
HORIZON_RATIO = 0.35
BOTTOM_RATIO = 1

# Window and Drawing Constants
WINDOW_NAME = "Data Image"
LINE_INPUT_COLOR = (0, 255, 0)
LINE_THICKNESS = 7
LINE_OUTPUT_COLOR = (255, 0, 0)

# Output Constants
LANE_SAMPLE_COUNT = 25


def record_point_and_draw(event, x, y, flags, param, img, lane_points):
    if event != cv2.EVENT_LBUTTONDOWN:  # Check for the event to be a click
        return
    
    point = (x, y)
    lane_points.append(point)
    
    if len(lane_points) > 1:
        cv2.line(img, lane_points[-2], lane_points[-1], LINE_INPUT_COLOR, LINE_THICKNESS)
        cv2.imshow(WINDOW_NAME, img)


def get_lane(img):
    lane_points = []

    cv2.namedWindow(WINDOW_NAME)
    cv2.imshow(WINDOW_NAME, img)
    cv2.setMouseCallback(WINDOW_NAME, lambda event, x, y, flags, param:
                         record_point_and_draw(event, x, y, flags, param, img, lane_points))
    while cv2.waitKey(0) != ord(' '):
        pass
    cv2.destroyWindow(WINDOW_NAME)

    assert (len(lane_points) > 1), "Each lane specification should have at least two points in it."

    return lane_points


def expand_lane_points(base_lane_points):
    new_y = np.linspace(base_lane_points[0][1], base_lane_points[-1][1], LANE_SAMPLE_COUNT)
    new_x = np.zeros_like(new_y)

    base_ind = 0
    for i, y in enumerate(new_y):
        if y < base_lane_points[base_ind + 1][1]:
            base_ind += 1
        alpha = (y - base_lane_points[base_ind][1]) / (base_lane_points[base_ind + 1][1] - base_lane_points[base_ind][1])
        interpolated_x = (1 - alpha) * base_lane_points[base_ind][0] + alpha * base_lane_points[base_ind + 1][0]
        new_x[i] = interpolated_x

    return np.stack((new_x, new_y), axis=-1).astype(np.int32)


def draw_output_lines(img, lane_points):
    for i in range(len(lane_points) - 1):
        cv2.line(img, lane_points[i], lane_points[i + 1], LINE_OUTPUT_COLOR, LINE_THICKNESS)


def store_result(filename, res):
    dir_path = os.path.dirname(os.path.abspath(filename))
    print(dir_path)
    np.save(os.path.join(dir_path, "label.npy"), res)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="""Tool to label our dataset. You only specify the left and right
                                            lanes of the road by roughly clicking the points from bottom to top, and
                                            then use the checker utility to get a labeling for some output lane description.
                                            First the left lane is expected, then by pressing the space bar, you can
                                            switch to the right lane. Finally, pressing the space bar again will show the
                                            lanes selected, and pressing the space bar one last time will store the lanes
                                            as a numpy array shape (2, NUM_POINTS, 2) the final resulting labeling. The first
                                            array in this results corresponds to the left lane, and the second oen corresponds
                                            to the right lane, respectively.""")
    parser.add_argument("filename", type=str, default=None, nargs=1)
    args = parser.parse_args()

    img = cv2.imread(args.filename[0], cv2.IMREAD_COLOR)
    H, W, _ = img.shape
    img = img[int(H * HORIZON_RATIO):int(H * BOTTOM_RATIO), :]

    left_lane = get_lane(img)
    left_lane = expand_lane_points(left_lane)
    draw_output_lines(img, left_lane)
    # print("RESULT LEFT", left_lane)

    right_lane = get_lane(img)
    right_lane = expand_lane_points(right_lane)
    draw_output_lines(img, right_lane)
    # print("RESULT RIGHT", right_lane)

    cv2.namedWindow(WINDOW_NAME)
    cv2.imshow(WINDOW_NAME, img)
    cv2.waitKey()

    store_result(args.filename[0], np.stack((left_lane, right_lane)))