import numpy as np
import matplotlib.pyplot as plt
from gymnasium import spaces
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

        config = simSettings[simMode]
        self._config = config
        self._useObstacles = config.get('useObstacles') if useObstacles is None else useObstacles
        self.simMode = simMode
        self.nWaypointsAhead = nWaypointsAhead
        self.maxSteps = maxSteps
        self.renderMode = renderMode

        self.vMax = maxVelocity
        self.deltaMax = maxDelta
        self.carLength = config['carLength']
        self.carWidth = config['carWidth']
        self.Ts = config['Ts']

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
                Obstacle(cx=0.0, cy=0.0, radius=0.05),
                Obstacle(cx=-0.8, cy=-0.5, radius=0.08),
                Obstacle(cx=-0.7, cy=-1.5, radius=0.05),
                Obstacle(cx=-0.3, cy=-1.0, radius=0.05),
                Obstacle(cx=0.27, cy=-1.0, radius=0.05),
                Obstacle(cx=0.78, cy=-1.47, radius=0.05),
                Obstacle(cx=0.73, cy=-0.9, radius=0.05),
                Obstacle(cx=1.2, cy=0.0, radius=0.05),
                Obstacle(cx=0.67, cy=-0.05, radius=0.05)]
            self._map.add_obstacles(self._obstacles)

        # Precompute the obstacle-aware corridor for every waypoint, the same way the MPC agent does.
        # _refCenter[i] is the lateral position the car should target at waypoint i to drive around obstacles.
        # _ubDyn and _lbDyn are the left and right corridor walls at each waypoint.
        # When obstacles are off, the corridor matches the static drivable area so the reward
        # and observation code works the same in both the obstacle and no-obstacle training phases.
        self._buildCorridor()

        # Define action and observation spaces
        self.action_space = spaces.Box(
            low=np.array([0.0, -self.deltaMax], dtype=np.float32),
            high=np.array([self.vMax, self.deltaMax], dtype=np.float32))
        # 7D observation: [devFromRef, headingError, kappaAhead, lastVelocity, clearance, upcomingRefShift, upcomingMinWidth]
        self.observation_space = spaces.Box(
            low=np.array([-np.inf, -np.pi, -np.inf, 0.0, -np.inf, -np.inf, 0.0], dtype=np.float32),
            high=np.array([np.inf, np.pi, np.inf, self.vMax, np.inf, np.inf, np.inf], dtype=np.float32))

        self.car = None
        self._stepCount = 0
        self._lastVelocity = 0.0
        self._lastKappaAhead = 0.0

    # Builds the obstacle-aware corridor used by the reward function and observations.
    def _buildCorridor(self):
        nWaypoints = self.referencePath.n_waypoints
        staticUb = np.array([wp.ub for wp in self.referencePath.waypoints])
        staticLb = np.array([wp.lb for wp in self.referencePath.waypoints])

        if not self._useObstacles:
            # Without obstacles the corridor matches the static drivable area and the reference stays at the centerline.
            self._ubDyn = staticUb
            self._lbDyn = staticLb
            self._refCenter = np.zeros(nWaypoints)
            return

        # Use the same widest-gap corridor solver the MPC uses. carWidth/4 as the safety margin
        # has been confirmed to keep all corridors feasible with 0 infeasible waypoints.
        sm = self.carWidth / 4.0
        ub, lb, _ = self.referencePath.update_path_constraints(0, nWaypoints, 2 * sm, sm)
        ub = np.asarray(ub, dtype=float)
        lb = np.asarray(lb, dtype=float)

        # If a waypoint has no valid corridor (ub <= lb), fall back to the static track bounds
        # so the agent is never placed in a position it cannot escape regardless of steering.
        infeasible = ub <= lb
        ub[infeasible] = staticUb[infeasible]
        lb[infeasible] = staticLb[infeasible]

        self._ubDyn = ub
        self._lbDyn = lb
        self._refCenter = (ub + lb) / 2.0
        self._refCenter[infeasible] = 0.0

        # Diagnostic so corridor validity is visible before training.
        widths = self._ubDyn - self._lbDyn
        print(f"[corridor] min width {widths.min():.4f} m (car {self.carWidth} m) | "
              f"infeasible {int(infeasible.sum())}/{nWaypoints} | "
              f"narrower-than-car {int((widths < self.carWidth).sum())} | "
              f"refCenter range [{self._refCenter.min():.4f}, {self._refCenter.max():.4f}] m")

    def reset(self, seed=None):
        if seed is not None:
            self.action_space.seed(seed)

        self.car = BicycleModel(self.referencePath, self.carLength, self.carWidth, self.Ts)
        self._stepCount = 0
        self._lastVelocity = 0.0
        self._lastKappaAhead = 0.0
        return self.getObs(), {}

    def step(self, action):
        if self.car is None:
            raise RuntimeError("Call reset() before step()")

        velocity = float(np.clip(action[0], 0.0, self.vMax))
        steeringAngle = float(np.clip(action[1], -self.deltaMax, self.deltaMax))
        self.car.drive(np.array([velocity, steeringAngle]))

        lapComplete = bool(self.car.s >= self.referencePath.length)

        # Skip get_current_waypoint when the lap is complete to avoid an out-of-bounds
        # array access when s has moved past the last waypoint index.
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
        # reward deviation from the obstacle-aware reference, not the raw centerline
        refCenter = float(self._refCenter[self.car.wp_id])
        reward = self.computeReward(velocity, lateralError, refCenter, collision)
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
        wpId = self.car.wp_id
        K = self.nWaypointsAhead
        circular = self._config['circular']

        # Average curvature over the next N waypoints so agent can anticipate turns, not just react to them
        kappas = []
        for i in range(1, K + 1):
            idx = self.car.wp_id + i
            if idx >= numWaypoints:
                idx = idx % numWaypoints if circular else numWaypoints - 1
            kappas.append(self.referencePath.waypoints[idx].kappa)
        kappaAhead = float(np.mean(kappas))
        self._lastKappaAhead = kappaAhead  # cached for use in computeReward

        # devFromRef measures how far the car is from the obstacle-avoiding target line, not the raw centerline.
        # clearance measures the gap to the nearer corridor wall.
        # When obstacles are off, both values reduce to standard track-centered equivalents.
        refCenter = float(self._refCenter[wpId])
        ubDyn = float(self._ubDyn[wpId])
        lbDyn = float(self._lbDyn[wpId])
        devFromRef = lateralError - refCenter
        clearance = min(ubDyn - lateralError, lateralError - lbDyn)

        # How much the obstacle-avoiding target line will shift K waypoints from now.
        # Positive means the reference moves right, negative means left.
        # This gives the agent advance warning before it needs to steer around an obstacle,
        # similar to how MPC uses a prediction horizon. Returns zero when obstacles are off.
        futureIdx = (wpId + K) % numWaypoints if circular else min(wpId + K, numWaypoints - 1)
        upcomingRefShift = float(self._refCenter[futureIdx] - refCenter)

        # Minimum corridor width over the next K waypoints, warning the agent of a narrow section ahead.
        futureWidths = []
        for i in range(1, K + 1):
            idx = wpId + i
            if idx >= numWaypoints:
                idx = idx % numWaypoints if circular else numWaypoints - 1
            futureWidths.append(self._ubDyn[idx] - self._lbDyn[idx])
        upcomingMinWidth = float(min(futureWidths))

        return np.array([devFromRef, headingError, kappaAhead, self._lastVelocity, clearance,
                         upcomingRefShift, upcomingMinWidth], dtype=np.float32)

    # Reward = -|deviation from obstacle-avoiding reference| + 0.1*velocity - 10*collision - curvature penalty.
    # The curvature penalty kicks in when the agent goes faster than the safe cornering speed
    # v_safe = sqrt(ay_max / |kappa|), which is the same formula MPC uses for its lateral acceleration limit.
    # The penalty is zero at or below v_safe so the agent is never penalized for driving at a safe speed.
    def computeReward(self, velocity, lateralError, refCenter, collision):
        safeSpeed = min(self.vMax, np.sqrt(4.0 / max(abs(self._lastKappaAhead), 0.1)))
        curvaturePenalty = 0.1 * max(0.0, velocity - safeSpeed)
        return -abs(lateralError - refCenter) - 10.0 * float(collision) + 0.1 * velocity - curvaturePenalty
