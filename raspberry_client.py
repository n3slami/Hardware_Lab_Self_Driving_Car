import sched
import time
import numpy as np
from socket import *
import logging

from tflite_runtime.interpreter import Interpreter

from image_processing import lane_detection


logging.basicConfig(level=logging.DEBUG)


MODEL_PATH = "./carla_rl/agent.tflite"
MODEL = Interpreter(MODEL_PATH)
MODEL.allocate_tensors()
MODEL_INPUT_DETAILS= MODEL.get_input_details()
MODEL_OUTPUT_DETAILS = MODEL.get_output_details()

SERVER_NAME = "127.0.0.1"
SERVER_PORT = 20002
BUFFER_SIZE = 4096
WAIT_LENGTH = 2.0
ACT_PERIOD = 0.05


class PreprocessCARLAImage:
    def __init__(self, height, width):
        self.im_width = width
        self.im_height = height

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

    def observation(self, image):
        """what happens to each observation"""
        lane_data = self._get_lane_data(image)
        return lane_data.astype(np.float32)


def get_image_from_socket(s, height, width, channel):
    recieve_buffer = bytearray()
    image_size = height * width * channel * 1
    logging.debug(f"EXPECTED SIZE {image_size}")
    while len(recieve_buffer) < image_size:
        recieve_buffer += s.recv(min(BUFFER_SIZE, image_size - len(recieve_buffer)))
    logging.info(f"Got image with size {len(recieve_buffer)}.")
    res = np.frombuffer(recieve_buffer, dtype=np.ubyte)
    return res.reshape(height, width, channel)


def get_action_from_observation(obs):
    MODEL.set_tensor(MODEL_INPUT_DETAILS[0]["index"], np.expand_dims(obs, axis=0))
    MODEL.invoke()
    qvalue_params = np.squeeze(MODEL.get_tensor(MODEL_OUTPUT_DETAILS[0]["index"]))
    logging.debug(qvalue_params)

    max_score = 1 / (np.sqrt(2 * np.pi) * qvalue_params[..., 1]) * qvalue_params[..., 2] + qvalue_params[..., 3]
    best_throttle_action = max_score.argmax()
    best_steer_action = qvalue_params[best_throttle_action, 0]
    return np.array([best_throttle_action, best_steer_action], dtype=np.float64)


def get_image_and_act(scheduler, client_socket, height, width, channel, extractor):
    scheduler.enter(ACT_PERIOD, 1, get_image_and_act, (scheduler, client_socket, height, width, channel, extractor))
    logging.info("Getting image from server.")
    client_socket.send(bytearray([0x00]))
    image = get_image_from_socket(client_socket, height=height, width=width, channel=channel)
    logging.debug(f"IMAGE SHAPE {image.shape}")
    observation = extractor.observation(image)
    action = get_action_from_observation(observation)
    logging.info(f"Observation: {observation}\nAction: {action}")
    command = bytearray([0x01]) + action.tobytes()
    logging.info(f"Sending action to server. {command}")
    client_socket.send(command)


def main(height=720, width=1280, channel=3, lifetime=60 * 5):
    client_socket = socket(AF_INET, SOCK_STREAM)
    client_socket.connect((SERVER_NAME, SERVER_PORT))

    extractor = PreprocessCARLAImage(height=height, width=width)
    logging.info("Connected to server.")

    scheduler = sched.scheduler(time.time, time.sleep)
    scheduler.enter(ACT_PERIOD, 1, get_image_and_act, (scheduler, client_socket, height, width, channel, extractor))
    scheduler.run()

    time.sleep(lifetime)
    client_socket.send(bytearray([0x02]))
    client_socket.close()
    logging.info("Closing server connection.")


if __name__ == "__main__":
    main()
