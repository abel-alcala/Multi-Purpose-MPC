import numpy as np
import matplotlib.pyplot as plt
from gymnasium import spaces
from lidar_model import LidarModel, OptimizedLidarModel
from map import Map, Obstacle
from reference_path import ReferencePath
from spatial_bicycle_models import BicycleModel

# The map and reference path are built once in __init__ and shared across all attempt to not rebuild each time
simSettings = {
    'Sim_Track': {
        'mapFile': 'maps/sim_map.png',
        'mapOrigin': [-1, -2],
        'mapResolution': 0.005,
        'wpX': [-0.75, -0.25, -0.25, 0.25, 0.25, 1.25, 1.25, 0.75, 0.75, 1.25, 1.25, -0.75, -0.75, -0.25],
        'wpY': [-1.5, -1.5, -0.5, -0.5, -1.5, -1.5, -1, -1, -0.5, -0.5, 0, 0, -1.5, -1.5],
        'pathResolution': 0.05,
        'smoothingDistance': 5,
        'maxWidth': 0.23,
        'circular': True,
        'carLength': 0.12,
        'carWidth': 0.06,
        'Ts': 0.05,
        'useObstacles': True,
    },
    'Real_Track': {
        'mapFile': 'maps/real_map.png',
        'mapOrigin': (-30.0, -24.0),
        'mapResolution': 0.06,
        'wpX': [-9.169, 11.9, 7.3, -6.95],
        'wpY': [-15.678, 10.9, 14.5, -3.31],
        'pathResolution': 0.20,
        'smoothingDistance': 5,
        'maxWidth': 1.50,
        'circular': False,
        'carLength': 0.30,
        'carWidth': 0.20,
        'Ts': 0.05,
        'useObstacles': False,
    },
}

# car action variables
maxVelocity = 1.0  # car velocity capped to [0, 1] m/s
maxDelta = 0.66  # steering angle clipped to [-0.66, 0.66] rad


# Wraps the bicycle model simulation so a RL agent can interact with it
# The agent calls reset() to start a new lap and step(action) to move the car forward one timestep
class TrackingEnv:
    metadata = {'render_modes': ['human']}

    def __init__(self, simMode='Sim_Track', nWaypointsAhead=5, maxSteps=2000, renderMode=None,
                 useObstacles=None):
        if simMode not in simSettings:
            raise ValueError(f"simMode must be 'Sim_Track' or 'Real_Track'")

        # Tracking Env configuration
        config = simSettings[simMode]
        self._config = config
        self._useObstacles = config.get('useObstacles') if useObstacles is None else useObstacles
        self.simMode = simMode
        self.nWaypointsAhead = nWaypointsAhead
        self.maxSteps = maxSteps
        self.renderMode = renderMode

        # Car configuration
        self.vMax = maxVelocity
        self.deltaMax = maxDelta
        self.carLength = config['carLength']
        self.carWidth = config['carWidth']
        self.Ts = config['Ts']
        self.lidarModel = OptimizedLidarModel(120, 1, 10) # 120 degrees, 0.5 meters range, 10 points

        # Build the map
        self._map = Map(
            file_path=config['mapFile'],
            origin=config['mapOrigin'],
            resolution=config['mapResolution'],
        )

        # Builds reference path before adding obstacles to not corrupt map
        self.referencePath = ReferencePath(
            self._map,
            config['wpX'],
            config['wpY'],
            config['pathResolution'],
            smoothing_distance=config['smoothingDistance'],
            max_width=config['maxWidth'],
            circular=config['circular']
        )
        self.referencePath.compute_speed_profile(
            {'a_min': -0.1, 'a_max': 0.5, 'v_min': 0.0, 'v_max': maxVelocity, 'ay_max': 4.0})

        self._obstacles = []
        if self._useObstacles:
            self._obstacles = [
                # TODO - Make obstacles configurable rather than hardcoded
                Obstacle(cx=0.0, cy=0.0, radius=0.05),
                Obstacle(cx=-0.8, cy=-0.5, radius=0.08),
                Obstacle(cx=-0.7, cy=-1.5, radius=0.05),
                Obstacle(cx=-0.3, cy=-1.0, radius=0.08),
                Obstacle(cx=0.27, cy=-1.0, radius=0.05),
                Obstacle(cx=0.78, cy=-1.47, radius=0.05),
                Obstacle(cx=0.73, cy=-0.9, radius=0.07),
                Obstacle(cx=1.2, cy=0.0, radius=0.08),
                Obstacle(cx=0.67, cy=-0.05, radius=0.06)]
            self._map.add_obstacles(self._obstacles)

        # Define action and observation spaces
        self.action_space = spaces.Box(
            low=np.array([0.0, -self.deltaMax], dtype=np.float32),
            high=np.array([self.vMax, self.deltaMax], dtype=np.float32))
        self.observation_space = spaces.Box(
            low=np.array([-np.inf, -np.pi, -np.inf, 0.0, -np.inf], dtype=np.float32),
            high=np.array([np.inf, np.pi, np.inf, self.vMax, np.inf], dtype=np.float32))

        self.car = None
        self._stepCount = 0
        self._lastVelocity = 0.0

    def reset(self, seed=None):
        if seed is not None:
            self.action_space.seed(seed)

        self.car = BicycleModel(self.referencePath, self.carLength, self.carWidth, self.Ts)
        self._stepCount = 0
        self._lastVelocity = 0.0
        self._lastS = self.car.s
        return self.getObs(), {}

    def step(self, action):
        if self.car is None:
            raise RuntimeError("Call reset() before step()")

        velocity = float(np.clip(action[0], 0.0, self.vMax))
        steeringAngle = float(np.clip(action[1], -self.deltaMax, self.deltaMax))
        self.car.drive(np.array([velocity, steeringAngle]))

        lapComplete = bool(self.car.s >= self.referencePath.length)

        # get_current_waypoint indexes into the waypoint array via s; skip if lap is done
        # to avoid an out-of-bounds access when s has passed the end of the track.
        if not lapComplete:
            self.car.get_current_waypoint()
            self.car.spatial_state = self.car.t2s(self.car.current_waypoint, self.car.temporal_state)
        self._lastVelocity = velocity
        self._stepCount += 1

        lateralError = self.car.spatial_state.e_y
        leftBound = self.car.current_waypoint.lb
        rightBound = self.car.current_waypoint.ub

        # collision is either leaving the track or hitting one of the obstacles
        offTrack = bool(lateralError < leftBound or lateralError > rightBound)
        collision = offTrack or self._hitObstacle()
        obs = self.getObs()
        reward = self.computeReward(velocity, lateralError, self.car.spatial_state.e_psi, collision, self.car.s - self._lastS, self.lidarModel.measurements[1,:], lapComplete)
        terminated = collision or lapComplete
        truncated = self._stepCount >= self.maxSteps

        info = {
            's': self.car.s, # distance traveled along path
            'wp_id': self.car.wp_id, # current waypoint index
            'e_y': lateralError, # lateral deviation from centerline
            'collision': collision,
            'lap_complete': lapComplete
        }

        if self.renderMode == 'human':
            self.render()

        return obs, reward, terminated, truncated, info

    def render(self):
        self.referencePath.show()
        self.car.show()
        plt.title(f'RL Environment  |  s={self.car.s:.2f}/{self.referencePath.length:.2f} m  |  lateralError={self.car.spatial_state.e_y:.3f} m')
        plt.axis('off')
        self.lidarModel.plot_scan(self.car.temporal_state)
        plt.pause(0.001)  # runs faster
        # plt.pause(0.1) # slo mo

    def close(self):
        plt.close('all')

    # True if the car body overlaps any obstacle disk (car treated as a disk of half its width)
    def _hitObstacle(self):
        if not self._obstacles:
            return False
        carX = self.car.temporal_state.x
        carY = self.car.temporal_state.y
        halfCar = self.carWidth / 2.0
        for obs in self._obstacles:
            if np.hypot(carX - obs.cx, carY - obs.cy) < obs.radius + halfCar:
                return True
        return False

    def getObs(self):
        lateralError = self.car.spatial_state.e_y
        headingError = self.car.spatial_state.e_psi
        numWaypoints = self.referencePath.n_waypoints

        # MPC relevant features. Pushes the agent to stay near the centerline and go fast.
        # Average curvature over the next N waypoints so agent can anticipate turns, not just react to them
        kappas = []
        for i in range(1, self.nWaypointsAhead + 1):
            waypointIdx = self.car.wp_id + i
            if waypointIdx >= numWaypoints:
                waypointIdx = waypointIdx % numWaypoints if self._config['circular'] else numWaypoints - 1
            kappas.append(self.referencePath.waypoints[waypointIdx].kappa)
        kappaAhead = float(np.mean(kappas))
        leftBound = self.car.current_waypoint.lb
        rightBound = self.car.current_waypoint.ub
        clearance = min(rightBound - lateralError, lateralError - leftBound)  # distance to the closer boundary

        mpc_obs =  np.array([
            lateralError,   # how far left/right from centerline
            headingError,   # how misaligned with path tangent
            kappaAhead,     # upcoming path curvature
            self._lastVelocity,   # previous commanded speed
            clearance,      # distance to nearest track boundary
        ], dtype=np.float32)

        # LiDAR relevant features. Provides information about the environment around the car.
        self.lidarModel.scan(self.car.temporal_state, self._map)
        
        lidarRanges = self.lidarModel.measurements[1, :]

        return np.concatenate([mpc_obs, lidarRanges]).astype(np.float32)
    

    # Stay near center (-|lateralError|), rewards going fast (+0.1*velocity) and penalizes collisions (-10)
    def computeReward(
        self,
        velocity,
        lateralError,
        headingError,
        collision,
        progress,
        lidarRanges=None,
        lapComplete=False,
    ):
        heading_error_norm = abs(headingError) / np.pi

        # Main objective: move forward along the track.
        progress_reward = 20.0 * max(progress, 0.0)

        # Penalize moving backward or failing to make progress.
        reverse_penalty = -5.0 * max(-progress, 0.0)

        # Keep path tracking as a soft preference, not the main objective.
        lateral_penalty = -0.5 * abs(lateralError)
        heading_penalty = -0.5 * heading_error_norm

        # Encourage speed mildly, but do not let speed dominate safety.
        speed_reward = 0.1 * velocity

        # Obstacle proximity penalty from lidar.
        obstacle_penalty = 0.0
        if lidarRanges is not None and len(lidarRanges) > 0:
            nearest_obstacle = float(np.min(lidarRanges))
            safe_distance = 0.25

            if nearest_obstacle < safe_distance:
                obstacle_penalty = -2.0 * (safe_distance - nearest_obstacle) / safe_distance

        # Large terminal penalties/rewards.
        collision_penalty = -25.0 * float(collision)
        lap_bonus = 50.0 * float(lapComplete)

        reward = (
            progress_reward
            + reverse_penalty
            + speed_reward
            + lateral_penalty
            + heading_penalty
            + obstacle_penalty
            + collision_penalty
            + lap_bonus
        )

        return float(reward)