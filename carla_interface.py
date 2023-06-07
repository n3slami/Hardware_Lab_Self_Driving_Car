import math
import os
import signal
import atexit
import carla
import time
from queue import Queue
import random
import numpy as np
import logging
from socket import *
import threading

from carla_rl.setup import setup


# logging.basicConfig(level=logging.DEBUG)


LOCAL_IP = ""
LOCAL_PORT = 20002
BUFFER_SIZE = 4096
MAX_THREAD_COUNT = 1

CMD_GET_FRAME = 0
CMD_ACTION = 1
CMD_DONE = 2

ENV_RESET_PERIOD = 6000 * 5


# Carla environment
class CarlaEnv:

    metadata = {'render.modes': ['human']}

    def __init__(self, town=None, fps=20, im_width=1280, im_height=720, start_transform_type="first", timeout=60):

        self.client, self.world, self.frame, self.server = setup(town=town, fps=fps, client_timeout=timeout, evaluation=True)
        self.client.set_timeout(2.0)
        self.map = self.world.get_map()
        blueprint_library = self.world.get_blueprint_library()
        self.microlino = blueprint_library.find('vehicle.micro.microlino')
        self.im_width = im_width
        self.im_height = im_height
        self.start_transform_type = start_transform_type
        self.actor_list = []

        self.tmp_counter = 0


    def set_front_image(self, x):
        self.front_image = x


    # Resets environment
    def reset(self):
        self.step_counter = 0

        self._destroy_agents()
        # logging.debug("Resetting environment")
        # Car, sensors, etc. We create them every episode then destroy
        self.collision_hist = []
        self.lane_invasion_hist = []
        self.actor_list = []

        self.front_image = None

        # When Carla breaks (stopps working) or spawn point is already occupied, spawning a car throws an exception
        # We allow it to try for 3 seconds then forgive
        spawn_start = time.time()
        while True:
            try:
                # Get random spot from a list from predefined spots and try to spawn a car there
                self.start_transform = self._get_start_transform()
                self.curr_loc = self.start_transform.location
                self.vehicle = self.world.spawn_actor(self.microlino, self.start_transform)
                break
            except Exception as e:
                logging.error('Error carla 141 {}'.format(str(e)))
                time.sleep(0.01)

            # If that can't be done in 3 seconds - forgive (and allow main process to handle for this problem)
            if time.time() > spawn_start + 3:
                raise Exception('Can\'t spawn a car')

        # Append actor to a list of spawned actors, we need to remove them later
        self.actor_list.append(self.vehicle)

        # TODO: combine the sensors
        self.rgb_cam = self.world.get_blueprint_library().find('sensor.camera.rgb')
        self.rgb_cam.set_attribute('image_size_x', f'{self.im_width}')
        self.rgb_cam.set_attribute('image_size_y', f'{self.im_height}')
        self.rgb_cam.set_attribute('fov', '110')

        transform_front  = carla.Transform(carla.Location(x=0.3, z=1.3))
        self.sensor_front = self.world.spawn_actor(self.rgb_cam, transform_front, attach_to=self.vehicle)
        self.sensor_front.listen(self.set_front_image)
        self.actor_list.extend([self.sensor_front])

        # Here's some workarounds.
        self.vehicle.apply_control(carla.VehicleControl(throttle=1.0, brake=1.0))
        time.sleep(4)

        # Collision history is a list callback is going to append to (we brake simulation on a collision)
        self.collision_hist = []
        self.lane_invasion_hist = []

        colsensor = self.world.get_blueprint_library().find('sensor.other.collision')
        lanesensor = self.world.get_blueprint_library().find('sensor.other.lane_invasion')
        self.colsensor = self.world.spawn_actor(colsensor, carla.Transform(), attach_to=self.vehicle)
        self.lanesensor = self.world.spawn_actor(lanesensor, carla.Transform(), attach_to=self.vehicle)
        self.colsensor.listen(self._collision_data)
        self.lanesensor.listen(self._lane_invasion_data)
        self.actor_list.append(self.colsensor)
        self.actor_list.append(self.lanesensor)

        # Wait for a camera to send first image (important at the beginning of first episode)
        while self.front_image is None:
            logging.debug("waiting for camera to be ready")
            time.sleep(0.01)

        # Disengage brakes
        self.vehicle.apply_control(carla.VehicleControl(brake=0.0))

    # Steps environment
    def step(self, action):
        # Apply control to the vehicle based on an action
        if action[0] == 0:
            action = carla.VehicleControl(throttle=0, steer=float(action[1]), brake=1)
        else:
            v = self.vehicle.get_velocity()
            kmh = 3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2)
            threshold = 12
            action = carla.VehicleControl(throttle=float((action[0])/3) if kmh < threshold else 0, steer=float(action[1]),
                                          brake=0 if kmh < threshold else 1)
        logging.debug('{}, {}, {}'.format(action.throttle, action.steer, action.brake))
        self.vehicle.apply_control(action)
    

    def close(self):
        logging.info("Closes the CARLA server with process PID {}".format(self.server.pid))
        os.killpg(self.server.pid, signal.SIGKILL)
        atexit.unregister(lambda: os.killpg(self.server.pid, signal.SIGKILL))
    

    def _destroy_agents(self):
        for actor in self.actor_list:
            # If it has a callback attached, remove it first
            if hasattr(actor, 'is_listening') and actor.is_listening:
                actor.stop()

            # If it's still alive - desstroy it
            if actor.is_alive:
                actor.destroy()

        self.actor_list = []

    def _collision_data(self, event):

        # What we collided with and what was the impulse
        collision_actor_id = event.other_actor.type_id
        collision_impulse = math.sqrt(event.normal_impulse.x ** 2 + event.normal_impulse.y ** 2 + event.normal_impulse.z ** 2)

        # # Filter collisions
        # for actor_id, impulse in COLLISION_FILTER:
        #     if actor_id in collision_actor_id and (impulse == -1 or collision_impulse <= impulse):
        #         return

        # Add collision
        self.collision_hist.append(event)
    

    def _lane_invasion_data(self, event):
        # Change this function to filter lane invasions
        self.lane_invasion_hist.append(event)


    def _get_start_transform(self):
        if self.start_transform_type == 'random':
            return random.choice(self.map.get_spawn_points())
        elif self.start_transform_type == 'first':
            self.tmp_counter += 1
            return self.map.get_spawn_points()[self.tmp_counter]
        elif self.start_transform_type == 'highway':
            if self.map.name == "Town04":
                for trial in range(10):
                    start_transform = random.choice(self.map.get_spawn_points())
                    start_waypoint = self.map.get_waypoint(start_transform.location)
                    if start_waypoint.road_id in list(range(35, 50)): # TODO: change this
                        break
                return start_transform
            else:
                raise NotImplementedError


def handle_session(connection_socket:socket, addr, env):
    logging.info(f"Connection with client {addr} accepted.")

    done = False
    env.reset()
    last_reset_time = time.time()
    while not done:
        if time.time() - last_reset_time > ENV_RESET_PERIOD :
            env.reset()
            last_reset_time = time.time()
        
        client_command = connection_socket.recv(1)
        if client_command[0] == CMD_GET_FRAME:
            image = np.array(env.front_image.raw_data)
            image = image.reshape(720, 1280, -1)[:, :, :3]
            logging.info(f"Sending image with size {len(image.tobytes())}")
            connection_socket.send(image.tobytes())
        elif client_command[0] == CMD_ACTION:
            client_command = connection_socket.recv(2 * 8)
            action = np.array(np.frombuffer(client_command), dtype=np.float64)
            logging.info(f"Got action {action}.")
            env.step(action)
        elif client_command[0] == CMD_DONE:
            done = True

    logging.info(f"Closing connection with client {addr}.")
    connection_socket.close()


def main():
    env = CarlaEnv()

    server_socket = socket(AF_INET, SOCK_STREAM)
    server_socket.bind((LOCAL_IP, LOCAL_PORT))
    server_socket.listen(MAX_THREAD_COUNT)
    logging.info("TCP server up and listening")
    while True:
        connection_socket, addr = server_socket.accept()
        service_thread = threading.Thread(target=handle_session, args=(connection_socket, addr, env, ))
        service_thread.start()


if __name__ == '__main__':
    main()
