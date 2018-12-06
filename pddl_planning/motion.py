import numpy as np
import random

from pddl_planning.utils import get_random_positions, get_joint_limits, set_joint_positions, exists_colliding_pair, \
    create_transform, solve_inverse_kinematics, get_unit_vector, get_body_pose
from examples.pybullet.utils.motion.motion_planners.rrt_connect import birrt

DEFAULT_WEIGHT = 1.0
DEFAULT_RESOLUTION = 0.01*np.pi

##################################################

def sample_nearby_positions(positions, variance=0.1):
    cov = variance * np.eye(len(positions)) # Normalize by dim
    return np.random.multivariate_normal(positions, cov)


def get_sample_fn(joints, start_conf=None, end_conf=None, collision_fn=lambda q: False):
    def fn():
        while True:
            variance = 0.05
            r = random.random()
            if (start_conf is not None) and (0 <= r < 0.3):
                q = sample_nearby_positions(start_conf, variance=variance)
            elif (end_conf is not None) and (0.3 <= r < 0.6):
                q = sample_nearby_positions(end_conf, variance=variance)
            else:
                q = get_random_positions(joints)
            if not collision_fn(q):
                return q
    return fn


def get_difference_fn(joints):

    def fn(q2, q1):
        assert len(joints) == len(q2)
        assert len(joints) == len(q1)
        return np.array(q2) - np.array(q1)
    return fn


def get_distance_fn(joints, weights=None):
    if weights is None:
        weights = DEFAULT_WEIGHT*np.ones(len(joints))
    difference_fn = get_difference_fn(joints)

    def fn(q1, q2):
        diff = np.array(difference_fn(q2, q1))
        return np.sqrt(np.dot(weights, diff * diff))
    return fn


def get_extend_fn(joints, resolutions=None):
    if resolutions is None:
        resolutions = DEFAULT_RESOLUTION*np.ones(len(joints))
    assert len(joints) == len(resolutions)
    difference_fn = get_difference_fn(joints)

    def fn(q1, q2):
        num_steps = int(np.max(np.abs(np.divide(difference_fn(q2, q1), resolutions)))) + 1
        q = q1
        for i in range(num_steps):
            q = (1. / (num_steps - i)) * np.array(difference_fn(q2, q)) + q
            yield q
    return fn


def within_limits(joint, position):
    lower, upper = get_joint_limits(joint)
    return lower <= position <= upper


def get_collision_fn(diagram, diagram_context, plant, scene_graph,
                     joints, collision_pairs=set(), attachments=[]):
    plant_context = diagram.GetMutableSubsystemContext(plant, diagram_context)

    def fn(q):
        if any(not within_limits(joint, position) for joint, position in zip(joints, q)):
            return True
        if not collision_pairs:
            return False
        set_joint_positions(joints, plant_context, q)
        for attachment in attachments:
            attachment.assign(plant_context)
        colliding = exists_colliding_pair(diagram, diagram_context, plant, scene_graph, collision_pairs)
        return colliding
    return fn

##################################################


def plan_workspace_motion(plant, joints, frame, frame_path, initial_guess=None,
                          collision_fn=lambda q: False, **kwargs):
    solution = initial_guess
    waypoints = []
    for frame_pose in frame_path:
        solution = solve_inverse_kinematics(plant, frame, frame_pose, initial_guess=solution, **kwargs)
        if solution is None:
            return None
        positions = [solution[j.position_start()] for j in joints]
        if collision_fn(positions):
            return None
        waypoints.append(positions)
    return waypoints


def plan_waypoints_joint_motion(joints, waypoints, resolutions=None,
                                collision_fn=lambda q: False):
    if not waypoints:
        return []
    for waypoint in waypoints:
        assert len(joints) == len(waypoint)
        if collision_fn(waypoint):
            return None

    extend_fn = get_extend_fn(joints, resolutions=resolutions)
    path = [waypoints[0]]
    for waypoint in waypoints[1:]:
        for q in extend_fn(path[-1], waypoint):
            if collision_fn(q):
                return None
            path.append(q)
    return path


def plan_joint_motion(joints, start_positions, end_positions,
                      weights=None, resolutions=None,
                      sample_fn=None, distance_fn=None,
                      collision_fn=lambda q: False,
                      **kwargs):
    assert len(joints) == len(start_positions)
    assert len(joints) == len(end_positions)
    if collision_fn(start_positions):
        print('Warning! Start positions in collision')
        return None
    if collision_fn(end_positions):
        print('Warning! End positions in collision')
        return None
    if distance_fn is None:
        distance_fn = get_distance_fn(joints, weights=weights)
    if sample_fn is None:
        sample_fn = get_sample_fn(joints, collision_fn=collision_fn)
    extend_fn = get_extend_fn(joints, resolutions=resolutions)
    return birrt(start_positions, end_positions, distance=distance_fn, sample=sample_fn,
                 extend=extend_fn, collision=collision_fn, **kwargs)


##################################################


def interpolate_translation(transform, translation, step_size=0.01):
    distance = np.linalg.norm(translation)
    if distance == 0:
        yield transform
        return
    direction = np.array(translation) / distance
    for t in list(np.arange(0, distance, step_size)) + [distance]:
        yield create_transform(translation=t * direction).multiply(transform)


##################################################

from utils import bodies_from_models, get_movable_joints, get_world_pose, get_joint_positions
from itertools import product
from scipy.spatial.kdtree import KDTree
from perception_tools.visualization_utils import MakeMeshcatColorArray, PlotMeshcatPointCloud
import meshcat.geometry as g
import time
import meshcat

def test_roadmap(task, meshcat_vis, num_samples=1000):
    # https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.spatial.KDTree.html
    # https://github.com/rdeits/meshcat-python/blob/master/src/meshcat/geometry.py
    joints = get_movable_joints(task.mbp, task.robot)
    moving = bodies_from_models(task.mbp, [task.robot, task.gripper])
    obstacles = set(task.fixed_bodies())

    collision_pairs = set(product(moving, obstacles))
    collision_fn = get_collision_fn(task.diagram, task.diagram_context, task.mbp, task.scene_graph,
                                    joints, collision_pairs=collision_pairs, attachments=[])

    sample_fn = get_sample_fn(joints, collision_fn=collision_fn)
    samples = []

    initial_conf = get_joint_positions(joints, task.plant_context)
    for _ in range(num_samples):
        mean = initial_conf
        cov = 0.1*np.eye(len(initial_conf)) # Normalize by dim
        sample = np.random.multivariate_normal(mean, cov)
        samples.append(sample)
    print(np.average([np.linalg.norm(sample - initial_conf) for sample in samples]))


    t0 = time.time()
    kt_tree = KDTree(samples)
    pairs = kt_tree.query_pairs(2.5, p=2, eps=0)
    print(time.time() - t0, len(pairs), float(len(pairs))/len(samples))

    points = []
    for i, conf in enumerate(samples):
        set_joint_positions(joints, task.plant_context, conf)
        pose = get_world_pose(task.plant, task.plant_context, task.gripper)
        points.append(pose.translation())
    red = MakeMeshcatColorArray(len(points), 0.5, 0, 0)
    PlotMeshcatPointCloud(meshcat_vis, 'vertices', np.array(points), red)

    edges = []
    for i, j in pairs:
        if (points[i][0] < 0.5) or (points[j][0] < 0.5):
            continue
        difference = points[j] - points[i]
        distance = np.linalg.norm(difference)
        for t in np.arange(0, distance, 0.02):
            point = points[i] + (t/distance) * difference
            edges.append(point)
    edges = np.array(edges).T
    red = MakeMeshcatColorArray(edges.shape[1], 0, 0.5, 0)
    PlotMeshcatPointCloud(meshcat_vis, 'edges', edges, red)

    for conf in samples:
        set_joint_positions(joints, task.plant_context, conf)
        task.publish()
        raw_input('Continue?')
    raw_input('Finish?')
