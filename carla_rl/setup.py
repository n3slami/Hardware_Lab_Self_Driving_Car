import os
import sys
import atexit
import signal
import subprocess
import time
from typing import Any
from typing import List
from typing import Mapping
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Union

import carla
import numpy as np
from absl import logging

logging.set_verbosity(logging.DEBUG)


def setup(
    town: str,
    fps: int = 20,
    server_timestop: float = 10.0,
    client_timeout: float = 2.0,
    num_max_restarts: int = 10,
    evaluation: bool = False,
):
    """Returns the `CARLA` `server`, `client` and `world`.

    Args:
        town: The `CARLA` town identifier.
        fps: The frequency (in Hz) of the simulation.
        server_timestop: The time interval between spawing the server
        and resuming program.
        client_timeout: The time interval before stopping
        the search for the carla server.
        num_max_restarts: Number of attempts to connect to the server.

    Returns:
        client: The `CARLA` client.
        world: The `CARLA` world.
        frame: The synchronous simulation time step ID.
        server: The `CARLA` server.
    """
    assert town in ("Town01", "Town02", "Town03", "Town04", "Town05", None)

    # The attempts counter.
    attempts = 0

    while attempts < num_max_restarts:
        logging.debug("{} out of {} attempts to setup the CARLA simulator".format(
            attempts + 1, num_max_restarts))

        # Port assignment.
        port = 2000

        # Start CARLA server.
        logging.debug("Inits a CARLA server at port={}".format(port))
        CARLA_SERVER_LOCATION = str(os.path.join(os.environ.get("CARLA_ROOT"), "CarlaUE4.sh"))
        
        server = None
        if evaluation:  # Show main viewport
            server = subprocess.Popen([CARLA_SERVER_LOCATION, f"-carla-port={port}"],
                                    stdout=None, stderr=subprocess.STDOUT, preexec_fn=os.setsid)
        else:   # Don't show any viewport
            server = subprocess.Popen([CARLA_SERVER_LOCATION, "-RenderOffScreen", f"-carla-port={port}"],
                                    stdout=None, stderr=subprocess.STDOUT, preexec_fn=os.setsid)
        
        atexit.register(os.killpg, server.pid, signal.SIGKILL)
        time.sleep(server_timestop)

        # Connect client.
        logging.debug("Connects a CARLA client at port={}".format(port))
        try:
            client = carla.Client("localhost", port)  # pylint: disable=no-member
            client.set_timeout(client_timeout)
            if town is not None:
                client.load_world(map_name=town)
            world = client.get_world()
            world.set_weather(carla.WeatherParameters.CloudySunset)  # pylint: disable=no-member
            if evaluation:
                frame = world.apply_settings(
                    carla.WorldSettings(  # pylint: disable=no-member
                        synchronous_mode=False
                    ))
            else:
                frame = world.apply_settings(
                    carla.WorldSettings(  # pylint: disable=no-member
                        synchronous_mode=True,
                        fixed_delta_seconds=1.0 / fps,
                    ))
            logging.debug("Server version: {}".format(client.get_server_version()))
            logging.debug("Client version: {}".format(client.get_client_version()))
            return client, world, frame, server
        except RuntimeError as msg:
            logging.debug(msg)
            attempts += 1
            logging.debug("Stopping CARLA server at port={}".format(port))
            os.killpg(server.pid, signal.SIGKILL)
            atexit.unregister(lambda: os.killpg(server.pid, signal.SIGKILL))

    logging.debug(
        "Failed to connect to CARLA after {} attempts".format(num_max_restarts))
    sys.exit()