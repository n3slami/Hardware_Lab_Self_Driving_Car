import os
import atexit
import signal
import carla
import gym
import time
import random
import numpy as np
import math
from queue import Queue
from .misc import dist_to_roadline, exist_intersection
from gym import spaces
from .setup import setup
from absl import logging
from .graphics import setup as setup_graphics
from .graphics import make_dashboard as make_graphics_dashboard
import pygame

logging.set_verbosity(logging.INFO)

# Carla environment
class CarlaEnv:

    metadata = {'render.modes': ['human']}
    MAX_SPEED = 15
    MAX_NORMALIZED_SPEED = 1.5

    def __init__(self, town, fps, im_width, im_height, repeat_action, start_transform_type, sensors,
                 action_type, enable_preview, steps_per_episode, playing=False, timeout=60):

        self.client, self.world, self.frame, self.server = setup(town=town, fps=fps, client_timeout=timeout)
        self.client.set_timeout(2.0)
        self.map = self.world.get_map()
        blueprint_library = self.world.get_blueprint_library()
        self.microlino = blueprint_library.find('vehicle.micro.microlino')
        self.im_width = im_width
        self.im_height = im_height
        self.repeat_action = repeat_action
        self.action_type = action_type
        self.start_transform_type = start_transform_type
        self.sensors = sensors
        self.actor_list = []
        self.preview_camera = None
        self.steps_per_episode = steps_per_episode
        self.playing = playing
        self.preview_camera_enabled = enable_preview
        self.step_counter = 0

    @property
    def observation_space(self, *args, **kwargs):
        """Returns the observation spec of the sensor."""
        return gym.spaces.Tuple((gym.spaces.Box(low=0.0, high=255.0, shape=(self.im_height, self.im_width, 3),
                                                dtype=np.uint8),
                                 gym.spaces.Box(low=0.0, high=CarlaEnv.MAX_NORMALIZED_SPEED, shape=(1, ))))

    @property
    def action_space(self):
        """Returns the expected action passed to the `step` method."""
        if self.action_type == 'continuous':
            return gym.spaces.Box(low=np.array([-1.0, -1.0]), high=np.array([1.0, 1.0]))
        elif self.action_type == 'discrete':
            return gym.spaces.MultiDiscrete([4, 9])
        elif self.action_type == 'mixed':
            throttle = gym.spaces.Discrete(4)
            wheel = gym.spaces.Box(low=-1, high=1, shape=(1, ))
            return gym.spaces.Tuple((throttle, wheel))
        else:
            raise NotImplementedError()
        # TODO: Add discrete actions (here and anywhere else required)


    def seed(self, seed):
        if not seed:
            seed = 7
        random.seed(seed)
        self._np_random = np.random.RandomState(seed) 
        return seed

    # Resets environment for new episode
    def reset(self):
        self.step_counter = 0

        self._destroy_agents()
        # logging.debug("Resetting environment")
        # Car, sensors, etc. We create them every episode then destroy
        self.collision_hist = []
        self.lane_invasion_hist = []
        self.actor_list = []
        self.frame_step = 0
        self.out_of_loop = 0
        self.dist_from_start = 0
        # self.total_reward = 0

        self.front_image_Queue = Queue()
        self.preview_image_Queue = Queue()

        # self.episode += 1

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
        if 'rgb' in self.sensors:
            self.rgb_cam = self.world.get_blueprint_library().find('sensor.camera.rgb')
        elif 'semantic' in self.sensors:
            self.rgb_cam = self.world.get_blueprint_library().find('sensor.camera.semantic_segmentation')
        else:
            raise NotImplementedError('unknown sensor type')

        self.rgb_cam.set_attribute('image_size_x', f'{self.im_width}')
        self.rgb_cam.set_attribute('image_size_y', f'{self.im_height}')
        self.rgb_cam.set_attribute('fov', '110')

        transform_front  = carla.Transform(carla.Location(x=0.3, z=1.3))
        self.sensor_front = self.world.spawn_actor(self.rgb_cam, transform_front, attach_to=self.vehicle)
        self.sensor_front.listen(self.front_image_Queue.put)
        self.actor_list.extend([self.sensor_front])

        # Preview ("above the car") camera
        if self.preview_camera_enabled:
            # TODO: add the configs
            self.preview_cam = self.world.get_blueprint_library().find('sensor.camera.rgb')
            self.preview_cam.set_attribute('image_size_x', '400')
            self.preview_cam.set_attribute('image_size_y', '400')
            self.preview_cam.set_attribute('fov', '100')
            transform = carla.Transform(carla.Location(x=-5.5, z=2.5), carla.Rotation(pitch=8.0))
            self.preview_sensor = self.world.spawn_actor(self.preview_cam, transform, attach_to=self.vehicle, attachment_type=carla.AttachmentType.SpringArm)
            self.preview_image_Queue.queue.clear()
            self.preview_sensor.listen(self.preview_image_Queue.put)
            self.actor_list.append(self.preview_sensor)

        # Here's some workarounds.
        self.vehicle.apply_control(carla.VehicleControl(throttle=1.0, brake=1.0))
        time.sleep(4)
        self.preview_image_Queue.queue.clear()

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

        self.world.tick()

        # Wait for a camera to send first image (important at the beginning of first episode)
        while self.front_image_Queue.empty():
            logging.debug("waiting for camera to be ready")
            time.sleep(0.01)
            self.world.tick()

        # Disengage brakes
        self.vehicle.apply_control(carla.VehicleControl(brake=0.0))

        image = self.front_image_Queue.get()
        image = np.array(image.raw_data)
        image = image.reshape((self.im_height, self.im_width, -1))
        image = image[:, :, :3]

        return (image, np.zeros(1)), None

    def step(self, action):
        total_reward = 0
        for _ in range(self.repeat_action):
            obs, rew, done, info = self._step(action)
            total_reward += rew
            if done:
                break
        return obs, total_reward, done, done, info

    # Steps environment
    def _step(self, action):
        self.world.tick()
        self.render()
            
        self.frame_step += 1

        # Calculate speed in km/h from car's velocity (3D vector)
        v = self.vehicle.get_velocity()
        kmh = 3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2)

        # Apply control to the vehicle based on an action
        if self.action_type == 'continuous':
            if action[0] > 0:
                action = carla.VehicleControl(throttle=float(action[0]), steer=float(action[1]), brake=0)
            else:
                action = carla.VehicleControl(throttle=0, steer=float(action[1]), brake= -float(action[0]))
        elif self.action_type == 'discrete':
            if action[0] == 0:
                action = carla.VehicleControl(throttle=0, steer=float((action[1] - 4)/4), brake=1)
            else:
                action = carla.VehicleControl(throttle=float((action[0])/3), steer=float((action[1] - 4)/4), brake=0)
        elif self.action_type == 'mixed':
            if action[0] == 0:
                action = carla.VehicleControl(throttle=0, steer=float(action[1]), brake=1)
            else:
                action = carla.VehicleControl(throttle=float(action[0]/3) if kmh < CarlaEnv.MAX_SPEED else 0,
                                              steer=float(action[1]), brake=0 if kmh < CarlaEnv.MAX_SPEED else 1)
        else:
            raise NotImplementedError()
        logging.debug('{}, {}, {}'.format(action.throttle, action.steer, action.brake))
        self.vehicle.apply_control(action)

        loc = self.vehicle.get_location()
        new_dist_from_start = loc.distance(self.start_transform.location)
        square_dist_diff = new_dist_from_start ** 2 - self.dist_from_start ** 2
        self.dist_from_start = new_dist_from_start

        image = self.front_image_Queue.get()
        image = np.array(image.raw_data)
        image = image.reshape((self.im_height, self.im_width, -1))

        # TODO: Combine the sensors
        if 'rgb' in self.sensors:
            image = image[:, :, :3]
        if 'semantic' in self.sensors:
            image = image[:, :, 2]
            image = (np.arange(13) == image[..., None])
            image = np.concatenate((image[:, :, 2:3], image[:, :, 6:8]), axis=2)
            image = image * 255
            # logging.debug('{}'.format(image.shape))
            # assert image.shape[0] == self.im_height
            # assert image.shape[1] == self.im_width
            # assert image.shape[2] == 3

        # dis_to_left, dis_to_right, sin_diff, cos_diff = dist_to_roadline(self.map, self.vehicle)

        done = False
        reward = 0

        # # If car collided - end and episode and send back a penalty
        if len(self.collision_hist) != 0:
            done = True
            reward += -200
            self.collision_hist = []
            self.lane_invasion_hist = []

        if len(self.lane_invasion_hist) != 0:
            done = True
            reward += -200
            self.lane_invasion_hist = []

        # # Reward for speed
        # if not self.playing:
        #     reward += 0.1 * kmh * (self.frame_step + 1)
        # else:
        #     reward += 0.1 * kmh

        reward += 0.1 * kmh

        # This should be ignored after some point I think, but I'm leaving it be for now...
        # reward += square_dist_diff

        # # Reward for distance to road lines
        # if not self.playing:
        #     reward -= math.exp(-dis_to_left)
        #     reward -= math.exp(-dis_to_right)
        
        if self.frame_step >= self.steps_per_episode:
            done = True

        # This thing is weird, so I'm removing it...
        # if not self._on_highway():
        #     print("OH...")
        #     self.out_of_loop += 1
        #     if self.out_of_loop >= 20:
        #         done = True
        # else:
        #     self.out_of_loop = 0

        # self.total_reward += reward

        if done:
            # info['episode'] = {}
            # info['episode']['l'] = self.frame_step
            # info['episode']['r'] = reward
            logging.debug("Env lasts {} steps, restarting ... ".format(self.frame_step))
            self._destroy_agents()
        
        self.step_counter += 1
        return (image, np.array([kmh / CarlaEnv.MAX_SPEED])), reward, done, self.step_counter
    
    def close(self):
        logging.info("Closes the CARLA server with process PID {}".format(self.server.pid))
        os.killpg(self.server.pid, signal.SIGKILL)
        atexit.unregister(lambda: os.killpg(self.server.pid, signal.SIGKILL))
    
    def render(self, mode='human'):
        # TODO: clean this
        # TODO: change the width and height to compat with the preview cam config

        if self.preview_camera_enabled:
            self._display, self._clock, self._font = setup_graphics(
                width=400,
                height=400,
                render=(mode=="human"),
            )

            preview_img = self.preview_image_Queue.get()
            # preview_img.save_to_disk(f"frame{self.step_counter}.png")
            preview_img = np.array(preview_img.raw_data)
            preview_img = preview_img.reshape((400, 400, -1))
            preview_img = preview_img[:, :, :3]
            make_graphics_dashboard(
                display=self._display,
                font=self._font,
                clock=self._clock,
                observations={"preview_camera":preview_img},
            )

            if mode == "human":
                # Update window display.
                pygame.display.flip()
            else:
                raise NotImplementedError()

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

    def _on_highway(self):
        goal_abs_lane_id = 4
        vehicle_waypoint_closest_to_road = \
            self.map.get_waypoint(self.vehicle.get_location(), project_to_road=True, lane_type=carla.LaneType.Driving)
        road_id = vehicle_waypoint_closest_to_road.road_id
        lane_id_sign = int(np.sign(vehicle_waypoint_closest_to_road.lane_id))
        assert lane_id_sign in [-1, 1]
        goal_lane_id = goal_abs_lane_id * lane_id_sign
        vehicle_s = vehicle_waypoint_closest_to_road.s
        goal_waypoint = self.map.get_waypoint_xodr(road_id, goal_lane_id, vehicle_s)
        return not (goal_waypoint is None)

    def _get_start_transform(self):
        if self.start_transform_type == 'random':
            return random.choice(self.map.get_spawn_points())
        if self.start_transform_type == 'highway':
            if self.map.name == "Town04":
                for trial in range(10):
                    start_transform = random.choice(self.map.get_spawn_points())
                    start_waypoint = self.map.get_waypoint(start_transform.location)
                    if start_waypoint.road_id in list(range(35, 50)): # TODO: change this
                        break
                return start_transform
            else:
                raise NotImplementedError