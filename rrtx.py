# Type hints
from dataclasses import dataclass
from typing import Sequence, Optional, List, Tuple, Dict, Set
from numpy.typing import NDArray
# Required for operation
import shapely.geometry as geom
import shapely.affinity as affi
from shapely.ops import nearest_points
from shapely.strtree import STRtree
import numpy as np
import scipy.spatial
import heapq
# Profiling and debugging
import matplotlib.pyplot as plt
import datetime
import cProfile as profile
import pstats
# For the task. This can be ignored.
from commonroad.scenario.lanelet import LaneletNetwork
from dg_commons import PlayerName, X
from dg_commons.sim.goals import PlanningGoal
from dg_commons.sim import SimObservations, InitSimObservations
from dg_commons.sim.agents import Agent
from dg_commons.sim.models.obstacles import StaticObstacle
from dg_commons.sim.models.vehicle import VehicleCommands
from dg_commons.sim.models.vehicle_structures import VehicleGeometry
from dg_commons.sim.models.vehicle_utils import VehicleParameters
# Warnings can be suppressed
import shapely
import warnings
from shapely.errors import ShapelyDeprecationWarning
warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)

@dataclass(frozen=True)
class Pdm4arAgentParams:
    param1: float = 0.2

ROUND_DP = 2
'''when represented as `kd`, the precision to round to.'''
np.random.seed(42)  # We want repeatable testing!

class Pdm4arAgent(Agent):
    """This is the PDM4AR agent.
    Do *NOT* modify the naming of the existing methods and the input/output types.
    Feel free to add additional methods, objects and functions that help you to solve the task"""

    def __init__(self,
                 sg: VehicleGeometry,
                 sp: VehicleParameters
                 ):
        self.sg = sg
        self.sp = sp
        self.name: PlayerName = None
        self.goal: geom.Polygon = None
        self.lanelet_network: LaneletNetwork = None
        self.static_obstacles: Sequence[StaticObstacle] = None

        # For plotting
        self.x_bounds = (np.inf, -np.inf)
        self.y_bounds = (np.inf, -np.inf)

        # To get the furthest point from the LaneletNetwork (the road) in the goal polygon.
        self.furthest_point = (0.0, 0.0)

        # The RRTx engine/object
        self.RRT: RRT_X = None

        self.prev_delta_err = 0.0
        '''Derivative tracker for steering control'''
        self.prev_vx_err = 0.0
        '''Derivative tracker for speed control'''
        self.prev_des_delta = 0.0
        '''Buffer variable to keep track of past steering angle in case there is currently no trajectory.'''

        ## TUNABLES ##
        self.kp_delta = 1.0
        '''P constant for tracking desired steering angle'''
        self.kd_delta = 1.0
        '''D constant for tracking desired steering angle'''
        self.kp_acc = 3.0
        '''P constant for tracking desired speed'''
        self.kd_acc = 1.0
        '''D constant for tracking desired speed'''
        self.delta_k = 0.5
        '''Constant to adjust how far steering angle shld be tweaked in response to cross-track error.'''
        self.k_soft = 1.0
        '''Softening quantity to prevent steering angle from becoming too large at low speeds.'''

        self.K_LOOKAHEAD = 8/np.pi
        '''Slow down by 1m/s if we see a curve of 45deg 3m away --> K=12/pi'''
        self.K_DEVIATION = (8/np.pi)
        '''Slow down by 2m/s if we have a 45 deg deviation from the heading --> K=8/pi'''
        self.C_MAX = 0.0
        ''' Clearance cost applied at max obstacle distance (min value)'''
        self.C_MIN = 3.0
        ''' Clearance cost applied at min obstacle distance (max value)'''
        self.D_MAX = 10.0
        ''' Max clearance from obstacle for robot for the linear relation to hold '''
        self.D_MIN = 1.5
        ''' Min clearance from obstacle for robot to increase speed '''
        self.V_MAX = 5.0
        '''Highest speed'''
        self.V_MIN = 1.0
        '''Lowest speed'''
        self.D_MULTIROBOT_MIN = 7.5
        '''Two robots will always consider each other if they are closer than this'''
        self.D_MULTIROBOT_MAX = 20.0
        '''Distance for two robots to consider each other in planning.'''
        self.T_MULTIROBOT_BEARING = np.pi/2
        '''Relative bearing of two robots for them not to consider each other'''
        self.T_MULTIROBOT_HEADING = np.pi/4
        '''Relative heading of two robots for them not to consider each other'''
        self.NUM_RRT_NODES = 1250
        '''Number of RRT nodes'''
        self.EDGE_COLL_RAD = self.sg.lf*2.5
        ''' how far around a dynamic obstacle's centroid should we look for nodes to check
            if they will collide?
        '''

        # Heartbeat: Status of the vehicle each iteration
        self.PRINT_HEARTBEAT = False
        # LOG_TELEM > PRINT_PROFILING (print_profiling doesn't work if log_telem if false)
        self.LOG_TELEM = True
        self.PLOT_INTERVAL = 1
        self.PRINT_PROFILING = False and self.LOG_TELEM

        # feel free to remove/modify  the following
        self.params = Pdm4arAgentParams()

        self.traj_hist = []
        '''History of robot states (curr_x, curr_y, curr_theta, curr_vx)'''
        self.traj: Optional[NDArray] = None
        '''Trajectory sampled from RRTx'''
        self.time_step = 0
        '''Current iteration count (`get_commands` call)'''

        if self.LOG_TELEM:
            self.vx_hist = []
            self.x_hist = []
            self.y_hist = []
            self.delta_hist = []
            self.theta_hist = []
            self.acc_hist = []
            self.ddelta_hist = []
            self.des_delta_hist = []
            self.cross_err_hist = []
            self.des_speed_hist = []

            if self.PRINT_PROFILING:
                self.prof = profile.Profile()

    def on_episode_init(self, init_obs: InitSimObservations):
        """This method is called by the simulator at the beginning of each episode."""

        # These are from the simulation. Adapt to your use case.
        self.name = init_obs.my_name
        self.goal = init_obs.goal.goal
        self.lanelet_network = init_obs.dg_scenario.lanelet_network
        self.static_obstacles: List[geom.base.BaseGeometry] = \
            [x.shape for x in list(init_obs.dg_scenario.static_obstacles.values())]

        # Get the bounds of the simulation (Sampling area)
        polygons = []
        for polygon in self.lanelet_network.lanelet_polygons:
            polygons += [polygon._shapely_polygon]
        self.lanelet_network_poly: geom.Polygon = shapely.ops.unary_union(polygons)
        self.x_bounds = (self.lanelet_network_poly.bounds[0], self.lanelet_network_poly.bounds[2])
        self.y_bounds = (self.lanelet_network_poly.bounds[1], self.lanelet_network_poly.bounds[3])

        # Find point on goal furthest from the obstacle as the goal
        max_dist = -np.inf
        for point in self.goal.exterior.coords:
            ext_point = geom.Point(point[0],point[1])
            dist = self.lanelet_network_poly.exterior.distance(ext_point)
            if dist > max_dist and self.lanelet_network_poly.contains(ext_point):
                max_dist = dist
                self.furthest_point = (point[0], point[1])

        self.goal_centroid = self.goal.centroid

        self.RRT = RRT_X(RRT_Node(self.furthest_point[0], self.furthest_point[1]),
            self.static_obstacles, self.lanelet_network_poly.exterior,
            self.x_bounds, self.y_bounds,
            self.sg, self.EDGE_COLL_RAD
            )

        # Plan for a certain number of iterations
        s = datetime.datetime.now()
        while self.RRT.num_nodes < self.NUM_RRT_NODES:
            self.RRT.step(None, None)
        e = datetime.datetime.now()
        print(f'! Pre-planning took {str(e-s)}')

        if self.LOG_TELEM:
            plot_things([geom.Point(self.furthest_point[0], self.furthest_point[1])],
                self.RRT.static_obs+self.RRT.X_obs_d+[self.RRT.lanelet_ext],
                self.RRT.all_nodes,
                self.x_bounds, self.y_bounds, goal=self.goal, name=f"{self.name}_{self.NUM_RRT_NODES}_Graph_{str(e-s)}",)

    def get_commands(self, sim_obs: SimObservations) -> VehicleCommands:
        """ This method is called by the simulator at each time step.
        For instance, this is how you can get your current state from the observations:
        my_current_state: VehicleState = sim_obs.players[self.name].state

        :param sim_obs:
        :return:
        """

        curr_x = sim_obs.players[self.name].state.x
        curr_y = sim_obs.players[self.name].state.y
        ct = sim_obs.players[self.name].state.psi               # heading
        curr_theta = np.arctan2(np.sin(ct), np.cos(ct))
        curr_delta = sim_obs.players[self.name].state.delta     # steering angle
        curr_vx = sim_obs.players[self.name].state.vx

        curr_obs = sim_obs.players[self.name].occupancy         # Polygon representing current car

        other_obs:List[geom.Polygon] = []
        other_players:List[geom.Polygon] = []
        for p in sim_obs.players:
            # Players name for priority
            if self.name == p:
                continue

            # other_players is our 'ground-truth'
            other_players.append(sim_obs.players[p].occupancy)

            d_x = sim_obs.players[p].state.x-curr_x
            d_y = sim_obs.players[p].state.y-curr_y
            d_xy = np.sqrt(d_x**2+d_y**2)
            d_bearing = np.arctan2(d_y,d_x)
            # The bearing from our car to their car
            d_bearing = np.abs( diff_between_angles(curr_theta, d_bearing) )
            d_theta = np.abs(diff_between_angles(curr_theta,
                sim_obs.players[p].state.psi
            ))

            # If cars are super close, we definitely need to consider them!
            if d_xy > self.D_MULTIROBOT_MIN:
                if d_xy > self.D_MULTIROBOT_MAX or \
                    d_bearing > self.T_MULTIROBOT_BEARING:
                    # Don't care about cars super far away.
                    # Don't care about cars 'behind' us.
                    continue

            # Compute the players' projected position in the next timestep (0.2s)
            # dt should be small because we are only considering a small timestep anyway.
            # These are computed using the Kinematic Bicycle model.
            pred_pos = get_predicted_position(
                sim_obs.players[p].state, sim_obs.players[p].occupancy,
                self.sg.lf+self.sg.lr, self.sg.lr, dt=0.2 )
            pred_pos = pred_pos.union(sim_obs.players[p].occupancy)
            other_obs.append(pred_pos)

        fa_x:float = curr_x + np.cos(curr_theta)*self.sg.lf # X pos of front axle
        fa_y:float = curr_y + np.sin(curr_theta)*self.sg.lf # Y pos of front axle

        if self.PRINT_PROFILING:
            self.prof.enable()
        # State update for RRT every few iterations
        if self.time_step % 1 == 0:
            self.RRT.step(v_curr=(curr_x, curr_y, curr_theta), new_obs=other_obs)
            self.traj = self.RRT.get_trajectory()
        if self.PRINT_PROFILING:
            self.prof.disable()

        # no trajectory. Stop robot. Also just step, in case a better way can be found.
        if self.traj is None:
            des_speed = 0.0
            des_delta = self.prev_des_delta

            # These are here for logging.
            cross_err = 0.0
            delta_ang = 0.0
            ang = 0.0
            atan = 0.0

            lookahead_cost = 0.0
            deviation_cost = 0.0
            clearance_cost = 0.0
            cross_pt = geom.Point(fa_x, fa_y)
            closest_parent = geom.Point(fa_x, fa_y)
            self.traj = np.array([(curr_x, curr_y),(self.RRT.v_bot.p_T_out.x, self.RRT.v_bot.p_T_out.y)])

            for ii in range(25):
                # ? print(f"{self.name} has no trajectory. Random step {ii}...")
                self.RRT.step(None,None)
        else:
            # Get desired steering angle
            cross_err, ang, cross_pt, closest_parent = get_crosstrack_err(fa_x, fa_y, self.traj)
            # The angle between the traj. heading and the vehicle heading.
            delta_ang = ang-curr_theta      # using atan2 should fix sudden jumps in sign
            delta_ang = np.arctan2(np.sin(delta_ang), np.cos(delta_ang))
            if delta_ang > 0:     # on the left, must steer left.
                cross_err *= 1
            else:
                cross_err *= -1
            atan = np.arctan((self.delta_k*cross_err) / (curr_vx+self.k_soft))
            des_delta = delta_ang + atan
            des_delta = max(min(des_delta, np.pi), -np.pi)
            self.prev_des_delta = des_delta

            # Get desired speed. Remember these should always be negative!
            lookahead_cost = get_lookahead_cost(self.traj, self.K_LOOKAHEAD)
            deviation_cost = np.abs(delta_ang)*self.K_DEVIATION
            clearance_cost, _ = get_obstacle_cost(self.RRT, curr_theta,
                                self.C_MAX, self.C_MIN, self.D_MAX, self.D_MIN)
            des_speed = self.V_MAX-lookahead_cost-deviation_cost-clearance_cost
            des_speed = max(min(des_speed, self.V_MAX), self.V_MIN)

        # Controller parts are here
        # steering angle rate control
        ddelta = self.kp_delta*(des_delta-curr_delta) + self.kd_delta*((des_delta-curr_delta)-self.prev_delta_err)
        if curr_delta > self.sp.delta_max*0.95 and des_delta > self.sp.delta_max:
            ddelta = min(0.0, ddelta)
        elif curr_delta < -self.sp.delta_max*0.95 and des_delta < -self.sp.delta_max:
            ddelta = max(0.0, ddelta)
        ddelta = np.clip(ddelta, -self.sp.ddelta_max, self.sp.ddelta_max)
        self.prev_delta_err = des_delta-curr_delta

        # acceleration control
        acc = self.kp_acc*(des_speed-curr_vx) + self.kd_acc*((des_speed-curr_vx)-self.prev_vx_err)
        if curr_vx > self.V_MAX and acc > 0:
            acc = 0.0
        acc = np.clip(acc, self.sp.acc_limits[0], self.sp.acc_limits[1])
        self.prev_vx_err = des_speed-curr_vx

        # LOGGING
        if self.PRINT_HEARTBEAT:
            print(f"\nTime {self.time_step:03d} | {self.name}")
            print(f"Robot at ({curr_x:.2f}, {curr_y:.2f}), {np.degrees(curr_theta):.2f}, {curr_vx:.2f}ms | Goal: {self.furthest_point[0]:.1f}, {self.furthest_point[1]:.1f}")
            # print(f"cross_err: {cross_err:.2f}, cross_pt: ({cross_pt.x:.2f}, {cross_pt.y:.2f}), front axle: {fa_x:.2f}, {fa_y:.2f}")
            print(f"Current speed: {curr_vx:.2f}, Desired speed: {des_speed:.2f}, lookahead: {lookahead_cost:.2f}, deviation: {deviation_cost:.2f}, clearance_cost: {clearance_cost:.2f}")
            # print(f"Angle: {np.degrees(ang):.2f}, psi: {np.degrees(delta_ang):.2f}, atan: {np.degrees(atan):.2f}")
            # print(f"Delta: curr:{np.degrees(curr_delta):.2f}, des:{np.degrees(des_delta):.2f}, ddelta:{ddelta:.2f}, accel:{acc:.2f}.")
            # print(f"Detected obs: {len(other_players)}, after filt: {len(other_obs)}, after update: {len(self.RRT.X_obs_d)}")
            print("=================================\n")
        if self.LOG_TELEM:
            self.traj_hist.append((curr_x, curr_y, curr_theta, curr_vx))
            self.vx_hist.append(curr_vx)
            self.x_hist.append(curr_x)
            self.y_hist.append(curr_y)
            self.delta_hist.append(np.degrees(curr_delta))
            self.theta_hist.append(np.degrees(curr_theta))
            self.acc_hist.append(acc)
            self.ddelta_hist.append(ddelta)
            self.des_delta_hist.append(np.degrees(des_delta))
            self.cross_err_hist.append(cross_err)
            self.des_speed_hist.append(des_speed)

            if self.time_step % self.PLOT_INTERVAL == 0 and self.time_step > 0:
                if self.PRINT_PROFILING:
                    stats = pstats.Stats(self.prof).strip_dirs().sort_stats("cumtime")
                    stats.print_stats(15)

                plot_ctr_x = curr_x + (curr_vx/2)*np.cos(curr_theta)
                plot_ctr_y = curr_y + (curr_vx/2)*np.sin(curr_theta)
                other_players_circ = [
                    geom.Point(p.centroid.x, p.centroid.y).buffer(self.EDGE_COLL_RAD) for p in self.RRT.X_obs_d
                ]
                plot_things(
                    [curr_obs, cross_pt, closest_parent]+other_obs,
                    self.RRT.static_obs+self.RRT.X_obs_d+[self.RRT.lanelet_ext]+other_players_circ,
                    self.RRT.all_nodes,
                    # self.x_bounds, self.y_bounds,     # Uncomment to plot globally
                    (plot_ctr_x-25.0, plot_ctr_x+25.0), (plot_ctr_y-25.0, plot_ctr_y+25.0),     # Plot centered on car
                    goal=self.goal, name=f"{self.name} {self.time_step}",
                    write_date=False,
                    trajectory=self.traj.tolist(),
                    history=None,
                    history=self.traj_hist[-self.PLOT_INTERVAL*2:],
                    frontAxle=(fa_x, fa_y)
                    )

                # This plots the behaviour of the speed and steering controllers, if you wish.
                # plot_telem(self)

        self.time_step += 1
        return VehicleCommands(acc, ddelta)

def get_crosstrack_err(fa_x:float, fa_y:float, cs:NDArray) -> Tuple[float, geom.Point, float]:
    '''
    Given:
    - Robot front axle coorindates `fa_x`, `fa_y`
    - The commanded trajectory `cs`
    Returns:
    - crosstrack error (min distance from path)
    - a Point expressing where the closest point on the path is.
    - The heading of the path at the closest point.
    '''
    pt = geom.Point(fa_x, fa_y)
    ls = geom.LineString(cs)
    cross_pt = ls.interpolate(ls.project(pt))

    # Find parent
    closest_parent = cs[1,:]
    for i in range(1,cs.shape[0]):
        hyp = geom.LineString([cs[i-1,:], cs[i,:]])
        if cross_pt.distance(hyp) < 1e-5:
            closest_parent = cs[i,:]
            break

    cross_err = ls.distance(pt)
    heading = np.arctan2(closest_parent[1]-cross_pt.y, closest_parent[0]-cross_pt.x)

    return cross_err, heading, cross_pt, geom.Point(closest_parent[0], closest_parent[1])

def get_lookahead_cost(traj:NDArray, K_LOOKAHEAD)->float:
    '''get the lookahead cost based on the expected curvature of the path
        By default, `K_LOOKAHEAD` sets us to slow down by 1m/s if we see a curve of 45deg 2m away.
        --> K=8/pi
    '''
    cost = 0
    cum_seg = 0     # Cumulative segment length
    for i in range(1,traj.shape[0]-1):
        cum_seg += np.linalg.norm(traj[i-1,:]-traj[i,:])
        ang0 = np.arctan2((traj[i,1]-traj[i-1,1]),(traj[i,0]-traj[i-1,0]))
        ang1 = np.arctan2((traj[i+1,1]-traj[i,1]),(traj[i+1,0]-traj[i,0]))
        ang_diff = np.abs(diff_between_angles(ang0,ang1))

        cost += K_LOOKAHEAD*ang_diff/cum_seg

    return cost

def get_obstacle_cost(rrt:"RRT_X", curr_hdg:float, C_MAX, C_MIN, D_MAX, D_MIN)->float:
    ''' get the cost based on obstacle clearance '''

    # First, query all obstacles in D_MAX radius of V_bot.
    v_bot_pt = geom.Point(rrt.v_bot.x, rrt.v_bot.y)
    obs = rrt.X_obs_s.nearest(v_bot_pt)

    # Filter for static obstacles in the front 180deg arc.
    clearance = D_MAX
    _, nearest = nearest_points(v_bot_pt, obs)
    obs_x = nearest.x
    obs_y = nearest.y
    hdg = np.arctan2(obs_y-v_bot_pt.y, obs_x-v_bot_pt.x)

    intersect = nearest.buffer(0.5).exterior.intersection(obs)  # Multipoint
    angle_pass = True
    if intersect.geom_type=="MultiPoint":
        mp1_x, mp1_y = intersect[0].x, intersect[0].y
        mp2_x, mp2_y = intersect[-1].x, intersect[-1].y
    elif intersect.geom_type=="LineString":
        if intersect.coords:
            mp1_x, mp1_y = intersect.coords[0]
            mp2_x, mp2_y = intersect.coords[-1]
        else:
            angle_pass = False
    else:
        # unable to figure out what this is
        angle_pass = False

    if angle_pass:
        h1 = np.arctan2(mp1_y-mp2_y, mp1_x-mp2_x)
        h2 = np.arctan2(mp2_y-mp1_y, mp2_x-mp1_x)
        h1_d = np.abs( diff_between_angles(curr_hdg, h1) )
        h2_d = np.abs( diff_between_angles(curr_hdg, h2) )
        # print(f"curr_hdg:{np.degrees(curr_hdg):.2f}, h1:{np.degrees(h1):.2f}, h1_d:{np.degrees(h1_d):.2f}, h2:{np.degrees(h2):.2f}, h2_d:{np.degrees(h2_d):.2f}")
        ang_diff = min(h1_d, h2_d)
    else:
        # Assume the worst if we are not able to get a value
        ang_diff = np.radians(90.0)

    if np.abs(diff_between_angles(hdg, curr_hdg)) < np.pi/2 and ang_diff > np.radians(30.0):
        clearance = min(clearance, nearest.distance(v_bot_pt))

    # Dynamic obstacles are already filtered wrt heading
    for obs in rrt.X_obs_d:
        clearance = min(clearance, obs.distance(v_bot_pt))

    # Distance to lanelet_ext should not be discounted
    clearance = min(clearance, v_bot_pt.distance(rrt.lanelet_ext))

    cost = ((C_MAX-C_MIN)/(D_MAX-D_MIN))*(clearance-D_MAX) + C_MAX

    # This may be confusing but look at the formulation above
    return max(min(cost, C_MIN), C_MAX), intersect

def get_predicted_position(state:X, occupancy:geom.Polygon, L:float, Lr:float, dt:float) -> geom.Polygon:
    '''
    Returns a transformed polygon based on how `occupancy` is predicted to move given its
    current state readout. (Given the Kinematic Bicycle Model)

    state, occupancy
    L: wheelbase
    Lr: distance from rear axle to COG
    dt: time interval.
    '''
    # x = state.x
    # y = state.y
    ct = state.psi               # heading
    theta = np.arctan2(np.sin(ct), np.cos(ct))
    delta = state.delta     # steering angle
    v = state.vx

    t_delta = np.tan(delta)
    S = L/t_delta
    beta = np.arctan(Lr/S)

    d_theta = (( v*t_delta*np.cos(beta) )/L) *dt
    d_x = (v*np.cos(theta+beta)) *dt
    d_y = (v*np.sin(theta+beta)) *dt

    # First rotate around the object's center, then shift.
    shifted = affi.rotate(occupancy, d_theta, origin='center', use_radians=True)
    shifted = affi.affine_transform(shifted, [1,0,0,1,d_x,d_y])

    return shifted

KDState = Tuple[float,float]
'''Tuple of x, y'''

class RRT_Node:
    '''
    A node in the tree structure of RRT.

    It exists in the state space of the robot: (x, y, theta) wrt the world frame.
    '''
    def __init__(self, x:float, y:float) -> None:
        self.x = x                  # X pos of COG in world frame
        self.y = y                  # Y pos of COG in world frame

        self.g = np.inf
        '''The e-consistent cost-to-goal of reaching the goal from V'''
        self.lmc = np.inf
        '''The lookahead estimate of cost-to-goal'''

        self.N_o_inc: Dict["RRT_Node",float] = {}
        '''original list of incoming nodes. These map to their calculated distance.'''
        self.N_o_out: Dict["RRT_Node",float] = {}
        '''original list of outgoing nodes. These map to their calculated distance.'''
        self.N_r_inc: Dict["RRT_Node",float] = {}
        '''running list of incoming nodes. These map to their calculated distance.'''
        self.N_r_out: Dict["RRT_Node",float] = {}
        '''running list of outgoing nodes. These map to their calculated distance.'''

        self.p_T_out: Optional["RRT_Node"] = None
        '''Parent node for tree to goal'''
        self.C_T_inc: Set["RRT_Node"] = set()
        '''Child nodes for tree to goal.'''

        self.obst_clearance:Optional[float] = None
        '''The distance to the nearest obstacle for this node.'''

        self.is_in_collision = False
        self.is_v_bot = False

    def __lt__(self, other:"RRT_Node"):
        ''' This lets us use a heap-queue `heapq` of RRT_Node objects. '''
        return (min(self.g, self.lmc), self.g) < (min(other.g, other.lmc), other.g)

    def kd(self) -> KDState:
        '''
        Expression of node state as an array for use in KD_tree.

        `(x,y)` rounded off to ROUND_DP precision
        '''
        return (
            np.round(self.x, ROUND_DP),
            np.round(self.y, ROUND_DP),
                )

    def __repr__(self) -> str:
        return f"RRT_Node(x:{self.x:.2f}, y:{self.y:.2f})"

class RRT_X:
    def __init__(self, v_goal: RRT_Node, static_obs: List[geom.Polygon],
        lanelet_poly_exterior:geom.Polygon,
        x_bounds:Tuple[float,float], y_bounds:Tuple[float,float],
        agent_params:VehicleGeometry, edge_coll_rad:float
    ) -> None:

        ########################
        ## Tunable parameters ##
        ########################

        self.delta = 2.5
        '''Max connection distance between nodes'''
        self.r = self.delta*2.7
        '''Default distance to search for connection'''
        self.max_search_radius = self.r
        '''Max search radius. Used in `shrinking_ball_rad()`'''
        self.search_rad = self.r*4.5
        '''Multiplier for log term. Used in `shrinking_ball_rad()`'''
        self.kd_tree_rebuild_interval = 25
        '''How often to rebuild the KD-trees'''
        self.epsilon = 1.0
        '''threshold for ϵ-consistency'''
        self.curr_it = 0
        self.OBST_CLEARANCE = 0.5
        ''' An acceptable distance away from obstacles for the car to keep.
            This should be added with a representative distance from the car, eg. self.lf.
        '''
        self.EDGE_COLL_RAD = edge_coll_rad
        ''' how far around a dynamic obstacle's centroid should we look for nodes to check
            if they will collide?
        '''

        ## outward_facing status pointers ##
        self.curr_it = 0
        '''Current RRT iteration'''
        self.num_nodes = 0
        '''Current number of nodes in the RRT graph'''

        # Dimensions of robot
        self.hw = agent_params.w_half
        self.lf = agent_params.lf + agent_params.bumpers_length[0]
        self.lr = agent_params.lr + agent_params.bumpers_length[0]

        # initialise goal_node information
        self.goal_node = v_goal
        self.goal_node.g = 0
        self.goal_node.lmc = 0
        self.goal_node.p_T_out = None

        ######################
        ## Class attributes ##
        ######################

        self.V = scipy.spatial.KDTree( [self.goal_node.kd()] )
        '''KD-tree of nodes'''
        self.V_l: List[KDState] = []
        '''List to periodically update the KDTree'''
        self.Vq: Dict[KDState, RRT_Node] = {self.goal_node.kd(): self.goal_node}
        '''Mapping the KD tree coords to an actual node object'''
        self.X_obs_s = STRtree(static_obs)
        '''R-tree of static obstacles'''
        self.static_obs = static_obs    # collision
        self.X_obs_d: List[geom.Polygon] = []
        '''List of dynamic obstacles'''

        self.theta: Optional[float] = None
        '''Current heading of the robot'''
        self.v_bot:Optional[RRT_Node] = None
        self.pos_polygon:geom.Polygon = geom.Point(v_goal.x, v_goal.y).buffer(0.01)   #dummy init
        # self.v_bot:RRT_Node = RRT_Node(init_x,init_y)
        # self.pos_polygon = geom.Point(init_x,init_y).buffer(self.lf)
        '''Robot pose / state'''
        self.all_nodes = [self.goal_node] # for debugging

        self.x_bounds = x_bounds
        self.y_bounds = y_bounds
        self.lanelet_ext = lanelet_poly_exterior.buffer(self.hw).exterior
        assert self.lanelet_ext.geom_type=="LinearRing"

        self.Q: List[RRT_Node] = []
        '''Priority queue to determine the order in which nodes become eps-consistent
        during the rewiring cascades.
        '''

        # For collision checking
        # Check if this is free.
        self.V_free:scipy.spatial.KDTree = None
        ''' Set of points that have been explicitly found to be collision-free from static obstacles '''
        self.V_free_l:List[KDState] = []
        '''List to periodically update the V_free KDTree (points free from static obs)'''
        self.V_obs:scipy.spatial.KDTree = None
        ''' Set of points that have been explicitly found to in collision with static obstacles '''
        self.V_obs_l:List[KDState] = []
        '''List to periodically update the V_obs KDTree (pts colliding w static obs)'''
        self.V_dist: Dict[KDState, float] = {}
        ''' Mapping a KDState to a min distance. (to obstacle, or free space) '''
        self.orphan_nodes: Set[RRT_Node] = set()
        '''List of orphaned nodes from obstacles appearing'''
        self.nodes_in_collision: Set[RRT_Node] = set()
        '''List of nodes for which their `is_in_collision` param is True'''

        self._get_obst_clearance(self.goal_node)
        # initialize goal_node's distance to nearest (static) obstacle

    def step(self, v_curr:Optional[Tuple[float,float,float]]=None,
            new_obs:Optional[List[geom.Polygon]]=None):

        self.curr_it += 1
        self.r = self._shrinking_ball_rad()

        if new_obs is not None:
            self._update_obstacles(new_obs)

        if v_curr is not None:
            self._update_robot(v_curr)

        v_coords = self._sample_node()
        # ? print(f"step(): sampled node at x:{v_coords[0]:.2f}, y:{v_coords[1]:.2f}")

        v_nearest_coords = self._get_nearest_coords(v_coords,self.V,self.V_l)
        if v_nearest_coords not in self.Vq:
            raise KeyError(f"{v_nearest_coords} not in Vq. {self.Vq}")
        v_nearest = self.Vq[v_nearest_coords]
        dist_to_nearest = self.dist(v_coords, v_nearest.kd())
        # ? print(f"step(): Nearest node: {v_nearest}, distance: {dist_to_nearest:.2f}")

        # if dist_to_nearest > self.delta:
        v_coords = self._saturate(v_coords, v_nearest.kd(), dist_to_nearest)
        v = RRT_Node(x=v_coords[0], y=v_coords[1])
        # ? print(f"step(): v after saturation: x:{v}, dist: {self.dist(v_coords, v_nearest.kd()):.2f}")

        #? print("step(): Checking for collision")
        if self._not_colliding(v):
            # ? print("step(): Extending")
            if self._extend(v):
                # ? print("step(): Rewiring neighbors")
                self._rewire_neighbors(v)
                # ? print("step(): Reducing inconsistency.")
                self._reduce_inconsistency()
                self.all_nodes.append(v)
                self.num_nodes += 1

    def get_trajectory(self) -> NDArray:
        ''' Sample trajectory from T (the shortest path subtree) to get the current
            control inputs.

            - hdg: Current heading of robot
            - vx: Current speed of robot
        '''
        if self.v_bot is None or self.theta is None:
            raise AttributeError("Position of v_bot has not been initialized.")

        traj = [self.v_bot]
        p = self.v_bot
        p_it = p.p_T_out
        its = 0

        while p_it is not None:
            path = self._get_path(p, p_it)
            if path is not None:
                traj.append(p_it)
            else:
                break

            assert p_it.p_T_out is not p, f"The parent of p_it:{p_it} is {p}."  # infinite loop

            if p_it.is_in_collision:
                print(f"{p_it} collides with an obstacle. This trajectory is invalid.")
                return None

            # There must be a loop somewhere if this happens
            if its > 100:
                print("Took > 100 iterations to find traj. We are likely to have loops in our graph.")
                return None

            p = p_it
            p_it = p_it.p_T_out
            its += 1

        # Under the full-traversal model, we only know if the subtree we are connected to
        # is invalid if it does not end at the goal.
        if traj[-1] is not self.goal_node:
            print("The last node is not the goal node. Invalid path.")
            print(traj[-5:])
            return None

        # Prune trajectory here
        p_id = 0
        p_prev_id = 0
        p_it_id = 1
        pruned_traj = [traj[0].kd()]
        while True:
            if p_prev_id==p_id:
                col_check = False
            else:
                col_check = True

            path = self._get_path(traj[p_id], traj[p_it_id], col_check)
            if path is not None:
                p_prev_id = p_it_id
                p_it_id += 1
            else:
                assert p_id != p_prev_id, f"{pruned_traj}"
                p_id = p_prev_id
                p_it_id = p_id + 1
                pruned_traj.append(traj[p_prev_id].kd())

                assert traj[p_id] is not self.v_bot, "Was unable to find a trajectory from v_bot"

            if p_it_id >= len(traj):
                pruned_traj.append(traj[-1].kd())
                break

        return np.array(pruned_traj)

    def _shrinking_ball_rad(self) -> float:
        ''' Returns the radius of a ball based on the cardinality of `self.V`
            We have a 4-dimensional space to search so d=4
        '''
        # Don't let this value become too small
        if self.r <= self.delta*1.2:
            return self.r

        c_v = max(self.num_nodes,1)

        # here we have max(c_v,2) because at init, log(1) = 0
        res = min(
            self.search_rad*np.power(np.log(max(c_v,2))/c_v, 1/2),
            self.max_search_radius
        )
        # print(f"Card(V): {c_v}, at iteration {self.curr_it}. new_r: {res:.2f}")
        return res

    def _update_obstacles(self, new_obs_l:List[geom.Polygon]):
        ''' Updates the value of `self.X_obs_d` (dynamic obstacles in environment)
        based on new observations. '''

        while len(self.X_obs_d)>0:
            obs = self.X_obs_d.pop()
            # ? print(f"_update_obstacles(): Removing obstacle at ({obs.centroid.x:.2f},{obs.centroid.y:.2f})")
            self._remove_obstacle(obs)
        # ? print("_update_obstacles(): Reducing inconsistency 1")
        self._reduce_inconsistency()

        # ? print(f"_update_obstacles(): {self.nodes_in_collision}")
        assert len(self.nodes_in_collision)==0, f"Not all collisions flags were cleared. {self.nodes_in_collision}"

        for obs in new_obs_l:
            # ? print(f"_update_obstacles(): Adding obstacle at ({obs.centroid.x:.2f},{obs.centroid.y:.2f})")
            self._add_obstacle(obs)
            self.X_obs_d.append(obs)

        assert len(self.X_obs_d) == len(new_obs_l), f"Not all elems of new_obs_l (len {len(new_obs_l)}) were copied to X_obs_d (len {len(self.X_obs_d)})"

        # ? print("_update_obstacles(): Propagating descendants")
        self._propagate_descendants()

        # ? print("_update_obstacles(): Verifying queue for v_bot")
        if self.v_bot is not None:
            self._verify_queue(self.v_bot)

        # ? print("_update_obstacles(): Reducing inconsistency 2")
        self._reduce_inconsistency()

    def _remove_obstacle(self, obs:geom.Polygon):
        # the edges that have path in collision with O
        E_o, c_l = self._get_all_colliding_edges(obs, coll_rad=self.EDGE_COLL_RAD)
        # ! Remove from E_o the edges that have path in collision with obs_near.
        # ! This is in the original algorithm,
        # ! but we skip it here because we assume all dynamic obstacle are cars,
        # ! which should not intersect with any static obstacles.
        for v in c_l:
            v.is_in_collision = False
            self.nodes_in_collision.discard(v)

        for v in E_o:
            v.is_in_collision = False
            self.nodes_in_collision.discard(v)
            for u, dic_name in E_o[v]:
                # Update the cost of d_pi(v,u)
                if dic_name=="N_o_inc":
                    dic = v.N_o_inc
                elif dic_name=="N_o_out":
                    dic = v.N_o_out
                elif dic_name=="N_r_inc":
                    dic = v.N_r_inc
                elif dic_name=="N_r_out":
                    dic = v.N_r_out
                assert u in dic, f"{u} was not found in {v}.{dic_name}"

                path = self._get_path(v,u)
                if path is not None:
                    dic[u] = path.cost
                else:
                    dic[u] = np.inf

            self._updateLMC(v)
            if not np.allclose(v.lmc, v.g):
                self._verify_queue(v)

    def _add_obstacle(self, obs: geom.Polygon):
        E_o, c_l = self._get_all_colliding_edges(obs, coll_rad=self.EDGE_COLL_RAD)
        for v in c_l:
            v.is_in_collision = True
            self.nodes_in_collision.add(v)

        for v in E_o:
            for u, dic_name in E_o[v]:
                # Update the cost of d_pi(v,u)
                if dic_name=="N_o_inc":
                    dic = v.N_o_inc
                elif dic_name=="N_o_out":
                    dic = v.N_o_out
                elif dic_name=="N_r_inc":
                    dic = v.N_r_inc
                elif dic_name=="N_r_out":
                    dic = v.N_r_out

                assert u in dic, f"{u} was not found in {v}.{dic_name}"
                dic[u] = np.inf
                if v.p_T_out is u:
                    self._verify_orphan(v)

    def _get_all_colliding_edges(self, obs: geom.Polygon, coll_rad) \
        ->Tuple[
            Dict[RRT_Node, List[Tuple[RRT_Node, str]]],
            List[RRT_Node]
        ]:
        ''' Given an obstacle `obs`, find all graph edges between RRT_Nodes that come into
            collision with it.

            Return a dictionary mapping an RRT_Node, v, to a list of (RRT_Node, dict_name).

            This way, we can iterate over edges while knowing which dict in the source the destination comes from.
        '''
        # init a collision set
        coll_set: Dict[Tuple[RRT_Node, RRT_Node], str] = {}
        coll_list: List[RRT_Node] = []

        # Get all nodes that collide with obs
        obs_coords = np.array( (obs.centroid.x, obs.centroid.y) )
        v_ind = self.V.query_ball_point(obs_coords, coll_rad, workers=-1)
        V_near = self.V.data[v_ind,:].tolist()
        for u in self.V_l:
            dist_to_node = self.dist(obs_coords, u)
            if dist_to_node <= coll_rad:
                V_near.append(u)

        # ? print(f"_get_all_colliding_edges(): obs@{obs_coords}. v_near:{V_near}")

        for u_c in V_near:
            u_c = tuple(np.round(u_c, ROUND_DP))
            x = self.Vq[u_c]
            coll_list.append(x)
            for dic_name in ["N_o_inc", "N_o_out", "N_r_inc", "N_r_out"]:
                if dic_name=="N_o_inc":
                    dic = x.N_o_inc
                elif dic_name=="N_o_out":
                    dic = x.N_o_out
                elif dic_name=="N_r_inc":
                    dic = x.N_r_inc
                elif dic_name=="N_r_out":
                    dic = x.N_r_out

                for y in dic:
                    coll_set[x,y] = dic_name

                    # Search through y to see if there is a correspondence in x
                    for y_dic_name in ["N_o_inc", "N_o_out", "N_r_inc", "N_r_out"]:
                        if y_dic_name=="N_o_inc":
                            y_dic = y.N_o_inc
                        elif y_dic_name=="N_o_out":
                            y_dic = y.N_o_out
                        elif y_dic_name=="N_r_inc":
                            y_dic = y.N_r_inc
                        elif y_dic_name=="N_r_out":
                            y_dic = y.N_r_out
                        if x in y_dic:
                            coll_set[y,x] = y_dic_name

        ret: Dict[RRT_Node, List[Tuple[RRT_Node, str]]] = {}
        for x,y in coll_set:
            if x not in ret:
                ret[x] = []
            ret[x].append((y, coll_set[(x,y)]))

        # ? print("_get_all_colliding_edges(): obs@{obs_coords}. Nodes in v_near:")
        # ? for u_c in V_near:
        # ?     u_c = tuple(np.round(u_c, ROUND_DP))
        # ?     x = self.Vq[u_c]
        # ?     print(f"{x}-->")
        # ?     print(ret[x])

        return ret, coll_list

    def _verify_orphan(self, v: RRT_Node):
        # If v is in the Q, remove it
        if v in self.Q:
            self.Q.remove(v)
        heapq.heapify( self.Q ) # Remove from the queue and preserve invariant

        self.orphan_nodes.add(v)

    def _propagate_descendants(self) -> None:
        ''' If a new obstacle has appeared and blocks things, then we remove the affected nodes. '''
        # ? print("_propagate_descendants(): Adding children of orphans to orphan list")
        add_to_orphans = []
        for v in self.orphan_nodes:
            for ch in v.C_T_inc:
                add_to_orphans.append(ch)

        for ch in add_to_orphans:
            self.orphan_nodes.add(ch)

        # ? print("_propagate_descendants(): Setting outgoing nodes from orphan list to infinite cost")
        for v in self.orphan_nodes:
            itx = list(v.N_o_out.keys())+list(v.N_r_out.keys())
            itx += [v.p_T_out] if v.p_T_out is not None else []
            for u in itx:
                if u in self.orphan_nodes:
                    continue
                u.g = np.inf
                self._verify_queue(u)

        while len(self.orphan_nodes)>0:
            # print(self.orphan_nodes)
            p = self.orphan_nodes.pop()
            p.g = np.inf
            p.lmc = np.inf
            if p.p_T_out is not None:
                p.p_T_out.C_T_inc.discard(p)
                p.p_T_out = None

    def _update_robot(self, v_curr: Tuple[float,float,float]):
        ''' Updates the value of `self.v_bot` based on new observations. '''
        x = v_curr[0]
        y = v_curr[1]
        self.theta = v_curr[2]

        self.v_bot = RRT_Node(x,y)
        self.v_bot.is_v_bot = True
        self._get_obst_clearance(self.v_bot)

        while True:
            # ? print("_update_robot(): Extending")
            if self._extend(self.v_bot):
                # ? print("_update_robot(): Rewiring neighbors")
                self._rewire_neighbors(self.v_bot)
                # self._verify_queue(self.v_bot)      # this is needed to ensure v_bot.g is updated
                # ? print("_update_robot(): Reducing inconsistency.")
                self._reduce_inconsistency()
                self.all_nodes.append(self.v_bot)
                self.num_nodes += 1
                break

            else:
                print(f"It {self.curr_it} | No nodes near v_bot {self.v_bot} yet. Sampling more pts.")
                self.step()

    def _sample_node(self) -> KDState:
        ''' Samples a node in the state space.

            The state space is:
            - x,y coordinates (within `self.x_bounds`, `self.y_bounds`)
            - heading (from 0 to 2pi)
        '''
        x = np.random.uniform(self.x_bounds[0], self.x_bounds[1])
        y = np.random.uniform(self.y_bounds[0], self.y_bounds[1])

        return (x, y)

    def _get_nearest_coords(self, coords:KDState,
            query_KD:Optional[scipy.spatial.KDTree],
            query_list:Optional[List[KDState]]
        ) -> Optional[KDState]:

        ''' Queries `self.V` to find the nearest node in the list of vertices,
        if the query KD-tree + query_list are not empty.'''
        closest: KDState = None

        if query_KD is not None:
            # dist, idx = query_KD.query(coords)
            dist, idx = query_KD.query(coords, workers=1)
            closest = tuple( query_KD.data[idx,:] )

        # only iterates if there are elements inside.
        for vx in query_list:
            c_dist = self.dist(vx, coords)
            if closest is None or c_dist < dist:
                closest = vx
                dist = c_dist

        if closest is not None:
            return tuple(np.round(e, ROUND_DP) for e in closest)
            # Round off for accessing the dictionary
        else:
            return None

    def _saturate(self, coords:KDState, n_coords:KDState, dist:float) -> KDState:
        ''' Returns a `KDState` that is `self.delta` away from `n_coords`.

            If `use_L2`, then we just saturate with regards to the L2 norm.
        '''
        x_sat = n_coords[0] + (coords[0]-n_coords[0])*(self.delta/dist)
        y_sat = n_coords[1] + (coords[1]-n_coords[1])*(self.delta/dist)
        return (x_sat, y_sat)

    def _static_collision(self, v: RRT_Node) -> bool:
        ''' Static obstacle checking for the node at `v`.
        Returns `True` if there is no collision and `False` otherwise.'''
        curr_poly = geom.Point(v.x,v.y).buffer((self.lf + self.OBST_CLEARANCE))
        # ? print(f"_static_collision(): Checking highway collision with {v}")
        if self.lanelet_ext.intersects(curr_poly):
            return False    # Extension collides with the highway
        # ? print(f"_static_collision(): Checking static collision with {v}")
        obs = self.X_obs_s.query(curr_poly)
        for x in obs:
            if curr_poly.intersects(x):
                return False
        return True

    def _get_obst_clearance(self, v:RRT_Node)->None:
        '''Populates the "obst_clearance" attribute of RRT_node V, by finding the distance to the closest obstacle.'''
        curr_point = geom.Point(v.x, v.y)

        hw_dist = curr_point.distance(self.lanelet_ext)
        obs = self.X_obs_s.nearest(curr_point)
        obs_dist = curr_point.distance(obs)
        for o in self.X_obs_d:
            obs_dist = min(obs_dist, curr_point.distance(o))
        v.obst_clearance = min(obs_dist, hw_dist)

    def _not_colliding(self, v:RRT_Node) -> bool:
        ''' Checks if v is in collision with `self.X_obs`.
            Returns `True` there is no collision, and `False` otherwise.
        '''
        # Use safety certificates for static obstacles, and iterative collision-checking
        # for dynamic obstacles (we assume that there are not so many)
        ## Static obstacle checking
        # ? print(f"_not_colliding(): Checking static collision with {v}")
        if not self._static_collision(v):
            return False
        # ? print(f"_not_colliding(): Checking dynamic collision with {v}")
        vp = geom.Point(v.x,v.y).buffer(self.EDGE_COLL_RAD)
        for obst in self.X_obs_d:
            if vp.intersects(obst):
                v.is_in_collision = True
                # self.nodes_in_collision.add(v)
                # return False

        return True

    def _extend(self, v: RRT_Node) -> bool:
        ''' Attempts to insert v into
            G (the overall graph)
            T (the shortest-path subtree)

            Returns True if extension succeeded, else False.
        '''
        # Find nodes near to new graph v
        v_ind = self.V.query_ball_point(v.kd(), self.r, workers=-1)
        V_near = self.V.data[v_ind,:].tolist()
        for u in self.V_l:
            dist_to_node = self.dist(v.kd(), u)
            if dist_to_node <= self.r:
                V_near.append(u)

        if len(V_near)==0:
            print(f"_extend(): Node {v} has no neighbours within {self.r:.2f}")
            return False    # Early stop.
        # print("_extend(): V_near, nodes close to v:", V_near)

        # Find the parent of v in V_Near
        for x_coords in V_near:
            x_coords = tuple(np.round(x_coords, ROUND_DP))
            x = self.Vq[x_coords]
            # Compute a trajectory through the state space through points v and x.
            traj = self._get_path(v, x, explicit_col_check=False)
            if traj is not None:    # if traj is valid and obstacle free:
                if traj.path.length < self.r:
                    if v.lmc > traj.cost+x.lmc:
                        assert v is not x, f"Attempting to assign v:{v} as its own parent x:{x}"
                        v.p_T_out = x
                        x.C_T_inc.add(v)
                        v.lmc = traj.cost + x.lmc

        if v.p_T_out is None:
            print(f"_extend(): No feasible way of getting from {v} to other nodes")
            return False

        # Add v to the list of nodes
        if v.kd() not in self.Vq.keys():
            self.V_l.append(v.kd())
            self.Vq[v.kd()] = v
            print(f"It {self.curr_it:04d}, #Nodes: {self.num_nodes:04d} || Inserting {v.kd()} into Vq. r: {self.r:.2f}")
        else:
            print(f"It {self.curr_it:04d}, #Nodes: {self.num_nodes:04d} || {v.kd()} is already in self.Vq. Updating its key.")

        # ! Periodically rebuild the KD-Tree (tune how often this is done for performance)
        if len(self.V_l) > self.kd_tree_rebuild_interval:
            self.V = scipy.spatial.KDTree(
                np.vstack((self.V.data, np.array(self.V_l, ndmin=2))) )
            self.V_l = []

        for x_coords in V_near:
            x_coords = tuple(np.round(x_coords, ROUND_DP))
            x = self.Vq[x_coords]
            p_vx = self._get_path(v, x)
            if p_vx is not None:
                v.N_o_out[x] = p_vx.cost
                x.N_r_inc[v] = p_vx.cost
            p_xv = self._get_path(x, v)
            if p_xv is not None:
                v.N_o_inc[x] = p_xv.cost
                x.N_r_out[v] = p_xv.cost

        return True

    def _cull_neighbors(self, v:RRT_Node)->None:
        '''Removes all neighbors in the running outgoing list if they are more than `self.r` away.'''
        rem_list = []
        for u in v.N_r_out:
            if self.r < self.dist(v.kd(),u.kd()) and v.p_T_out is not u:
                rem_list.append(u)

        for u in rem_list:
            v.N_r_out.pop(u, None)      # Returns None if u is not found
            v.N_r_inc.pop(u, None)      # Returns None if u is not found

    def _rewire_neighbors(self, v: RRT_Node)->None:
        ''' Rewires v's in-neighbors to use v as their parent, if this
            results in a better cost-to-go
        '''
        if v.g-v.lmc > self.epsilon:
            # Cull neighbors
            self._cull_neighbors(v)

            for dic in [v.N_r_inc, v.N_o_inc]:
                for u in dic:
                    if v.p_T_out is u:
                        continue
                    assert u is not v, f"Somehow v:{v} got into its own incoming neighbor set."

                    dist = dic[u]
                    if u.lmc > dist + v.lmc:
                        u.lmc = dist + v.lmc
                        # makeParentOf(v,u)
                        if u.p_T_out is not None:
                            u.p_T_out.C_T_inc.discard(u)
                        u.p_T_out = v
                        v.C_T_inc.add(u)
                        if u.g - u.lmc > self.epsilon:
                            self._verify_queue(u)

    def _reduce_inconsistency(self):
        ''' Manages the rewiring cascade that propagates cost-to-goal info. '''

        if len(self.Q)>0 and self.v_bot is not None:
            while \
                not np.allclose(self.v_bot.lmc, self.v_bot.g) or \
                np.allclose(self.v_bot.g, np.inf) or \
                self.Q[0] < self.v_bot or \
                self.v_bot in self.Q:

                # ? print("Reducing Inconsistency. Q contents:", self.Q[:5])

                v = heapq.heappop(self.Q)
                if v.g-v.lmc > self.epsilon:
                    self._updateLMC(v)
                    if not v.is_in_collision:
                        self._rewire_neighbors(v)

                v.g = v.lmc

                if len(self.Q)==0:
                    break

        if self.v_bot is not None:
            # ? print(f"_reduce_inconsistency(): v_bot lmc:{self.v_bot.lmc:.2f}, g:{self.v_bot.g:.2f}")
            pass

    def _updateLMC(self, v: RRT_Node):
        # Cull neighbors
        self._cull_neighbors(v)

        p_prime = None
        for u in v.N_r_out:
            # Consider using set subtraction if self.orphan nodes is large.
            if u in self.orphan_nodes or u.p_T_out is v:
                continue
            dist = v.N_r_out[u]
            if v.lmc > dist + u.lmc:
                p_prime = u
                v.lmc = dist + u.lmc

        for u in v.N_o_out:
            # Consider using set subtraction if self.orphan nodes is large.
            if u in self.orphan_nodes or u.p_T_out is v:
                continue
            dist = v.N_o_out[u]
            if v.lmc > dist + u.lmc:
                p_prime = u
                v.lmc = dist + u.lmc

        if p_prime is not None:
            # makeParentOf(p_prime,v)
            if v.p_T_out is not None:
                v.p_T_out.C_T_inc.discard(v)
            v.p_T_out = p_prime
            p_prime.C_T_inc.add(v)

    def _verify_queue(self, v: RRT_Node):
        # If v is in Q, update it. If not, add v to Q.
        if v in self.Q:
            self.Q.remove(v)
            heapq.heapify(self.Q)

        heapq.heappush(self.Q, v)

    def dist(self, v1:KDState, v2:KDState):
        '''
        Returns a measure of distance between the two KDStates v1 and v2

        A KDState is a tuple (x, y).
        '''
        return  np.sqrt(
            (v1[0]-v2[0])**2 + \
            (v1[1]-v2[1])**2
        )

    def _get_path(self, v1:RRT_Node, v2:RRT_Node, explicit_col_check=False) -> "Path":
        ''' Compute path from one node to another.

            For now this is a straight-line path.

            If explicit_col_check is true, clearance of (self.lf + self.OBST_CLEARANCE)
            is checked for.
        '''
        line = geom.LineString([[v1.x, v1.y], [v2.x, v2.y]])
        # Populate this for trajectory cost information
        if v1.obst_clearance is None:
            self._get_obst_clearance(v1)
        if v2.obst_clearance is None:
            self._get_obst_clearance(v2)

        if explicit_col_check is False:
            if line.intersects(self.lanelet_ext):
                return None
            # Check static obstacles (and dynamic ones)
            obsts = self.X_obs_s.query(line)
            for obs in obsts+self.X_obs_d:
                if line.intersects(obs):
                    return None

            min_d_obs = min(v1.obst_clearance, v2.obst_clearance)

        else:
            # Checks for clearance
            min_d_obs = line.distance(self.lanelet_ext)
            if min_d_obs < (self.lf + self.OBST_CLEARANCE):
                return None
            # Check static obstacles (and dynamic ones)
            for obs in [self.X_obs_s.nearest(line)]+self.X_obs_d:
                dist = line.distance(obs)
                if dist < (self.lf + self.OBST_CLEARANCE):
                    return None
                if dist < min_d_obs: min_d_obs = dist

        return Path(path=line, obs_clearance=min_d_obs)

@dataclass
class Path:
    path:geom.LineString
    obs_clearance:float
    cost:float=np.inf
    def __post_init__(self):
        self.cost = self.path.length + 5/self.obs_clearance

## Utilities
def diff_between_angles(a1:float, a2:float)->float:
    '''Returns smallest a1-a2, assuming both in radians'''
    hdg_diff_a = (a2-a1) % (2*np.pi)
    hdg_diff_b = (a1-a2) % (2*np.pi)

    return -hdg_diff_a if hdg_diff_a < hdg_diff_b else hdg_diff_b

def plot_things(things:List[geom.base.BaseGeometry], obstacles:List[geom.base.BaseGeometry],
points:List["RRT_Node"],
x_bounds: Tuple[float,float], y_bounds: Tuple[float,float], goal:geom.base.BaseGeometry=None,
name="", write_date=True,
trajectory:List["KDState"]=[],
history:List[Tuple[float,float,float,float]]=[],
frontAxle:Tuple[float,float]=None
):

    def plot_geom(thing, opt='k'):
        if thing.geom_type=="LineString":
            plt.plot(*thing.coords.xy, opt)
        elif thing.geom_type=="Polygon":
            plt.plot(*thing.exterior.xy, opt)
        elif thing.geom_type=="Point":
            # plt.plot( thing.x, thing.y, opt+'x', markersize=4, markeredgewidth=0.5)
            plt.plot( thing.x, thing.y, opt+'x')
        elif thing.geom_type=="LinearRing":
            coords = list(thing.coords.xy)
            coords[0].append(coords[0][0])
            coords[1].append(coords[1][0])
            plt.plot(*coords, opt)
        elif thing.geom_type=="MultiLineString":
            for subthing in thing:
                plot_geom(subthing, opt)
        elif thing.geom_type=="MultiPolygon":
            for subthing in thing:
                plot_geom(subthing, opt)
        elif thing.geom_type=="MultiPoint":
            for subthing in thing:
                plot_geom(subthing, opt)
        elif thing.geom_type=="MultiLinearRing":
            for subthing in thing:
                plot_geom(subthing, opt)
        else:
            raise ValueError(f"Unrecognised {thing.geom_type}")

    plt.figure()
    plt.grid()
    plt.gca().axis('equal')
    plt.xlim(x_bounds)
    plt.ylim(y_bounds)

    if goal is not None:
        plt.fill(*goal.exterior.xy, "y")

    for thing in obstacles:
        plot_geom(thing, opt='r')

    if trajectory:
        plt.plot( *zip(*trajectory), 'bx--' , linewidth=0.75 )

    for thing in things:
        plot_geom(thing, opt='k')

    if history:
        MIN_ALPHA = 0.1
        MAX_ALPHA = 0.5
        ALPHA_SCALE = np.linspace(MIN_ALPHA, MAX_ALPHA, len(history))
        # Each tuple is (x,y,theta,speed)
        for i, tup in enumerate(history):
            plt.arrow(
                x=tup[0], y=tup[1], 
                dx=(0.5+0.5*tup[3])*np.cos(tup[2]),
                dy=(0.5+0.5*tup[3])*np.sin(tup[2]), color='m',
                linewidth=0.25, head_length=0.5, head_width=0.5,
                alpha=ALPHA_SCALE[i]
            )

    if frontAxle:
        # plt.plot( frontAxle[0], frontAxle[1], 'x', markersize=4, markeredgewidth=0.5)
        plt.plot( frontAxle[0], frontAxle[1], 'x')

    for el in points:
        # Don't plot v_bot nodes (too many!)
        if el.is_v_bot:
            continue

        node = el.kd()
        if el.p_T_out is not None:
            conn = el.p_T_out.kd()
            plt.plot(
                (node[0], conn[0]), (node[1], conn[1]), 'k--', linewidth=0.25
            )
            msize = 3
            mw = 0.5
        else:
            msize = 6
            mw = 2.0

        marker_color = 'b' if el.is_in_collision else 'g'

        plt.plot(node[0], node[1], marker_color+'x', markersize=msize, markeredgewidth=mw)
        # plt.annotate(f"{el.lmc:.1f}|{el.g:.1f}", (el.x,el.y), fontsize=2)

    if write_date or name=='':
        figname = f'{datetime.datetime.now().strftime("%H_%M_%S")}_{name}.png'
    else:
        figname = f'{name}.png'
    plt.tight_layout()
    plt.savefig(figname,dpi=250)
    plt.close('all')

def plot_telem(agent):
    # Plot telemetry
    plt.figure()
    _, ((ax00, ax01), (ax10, ax11)) = plt.subplots(2,2)
    ax00.set_title("Longitudinal Acceleration")
    ax00.grid()
    ax00.plot(agent.acc_hist[-15:], 'x-', label="Commanded acceleration")
    ax00.legend()

    ax01.set_title("Longitudinal Velocity")
    ax01.grid()
    ax01.plot(agent.vx_hist[-15:], 'x-', label="Current velocity")
    ax01.plot(agent.des_speed_hist[-15:], '-', label="Desired velocity")
    ax01.legend()

    ax10.set_title("Lateral Velocity")
    ax10.grid()
    ax10.plot(agent.ddelta_hist[-15:], 'x-', label="Desired steering rate")
    ax10.legend()

    ax11.set_title("Lateral Position")
    ax11.grid()
    ax11.plot(agent.delta_hist[-15:], label="Steering angle")
    ax11.plot(agent.des_delta_hist[-15:], label="Desired angle")
    ax11.set_ylim(np.degrees(-np.pi*1.05), np.degrees(np.pi*1.05))
    ax11_ = ax11.twinx()
    ax11_.plot(agent.cross_err_hist[-15:], 'g', label="Crosstrack error")
    ax11.legend()
    ax11_.legend()

    figname = f'{datetime.datetime.now().strftime("%H_%M_%S")}_Telem {agent.time_step}.png'
    plt.tight_layout()
    plt.savefig(figname,dpi=200)
    plt.close('all')
