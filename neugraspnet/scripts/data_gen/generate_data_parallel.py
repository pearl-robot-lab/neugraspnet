import argparse
from pathlib import Path

import numpy as np
import open3d as o3d
import scipy.signal as signal
from tqdm import tqdm
import multiprocessing as mp

from neugraspnet.grasp import Grasp, Label
from neugraspnet.io import *
from neugraspnet.perception import *
from neugraspnet.simulation import ClutterRemovalSim
from neugraspnet.utils.transform import Rotation, Transform
from neugraspnet.utils.implicit import get_mesh_pose_list_from_world
import os


OBJECT_COUNT_LAMBDA = 3
MAX_VIEWPOINT_COUNT = 24


def main(args, rank):
    GRASPS_PER_SCENE = args.grasps_per_scene
    np.random.seed()
    seed = np.random.randint(0, 1000) + rank
    np.random.seed(seed)
    sim = ClutterRemovalSim(args.scene, args.object_set, gui=args.sim_gui)
    finger_depth = sim.gripper.finger_depth
    grasps_per_worker = args.num_grasps // args.num_proc
    pbar = tqdm(total=grasps_per_worker, disable=rank != 0)

    # if rank == 0:
    #     if args.save_scene:
    #         # ("./" + args.root / "mesh_pose_list").mkdir(parents=True)
    #         (args.root / "scan").mkdir(parents=True)

    for scene_id in range(grasps_per_worker // GRASPS_PER_SCENE): # range determines number of scenes
        # generate heap
        object_count = 1 # CHANGED for one object #np.random.poisson(OBJECT_COUNT_LAMBDA) + 1
        sim.reset(object_count)
        sim.save_state()

        # render synthetic depth images
        # (args.root / f"scan_{scene_id}").mkdir(parents=True)
        n = MAX_VIEWPOINT_COUNT
        depth_imgs, extrinsics = render_images(sim, n)
        render_side_images(sim, n, args.random)

        # reconstrct point cloud using a subset of the images
        tsdf = create_tsdf(sim.size, 40, depth_imgs, sim.camera.intrinsic, extrinsics)
        voxels = tsdf.get_grid()
        pc = tsdf.get_cloud()

        # # store the raw data
        if args.save_scene:
            mesh_pose_list = get_mesh_pose_list_from_world(sim.world, args.object_set)
            write_point_cloud(args.root, pc) ## Point cloud save
            write_voxel_grid(args.root,  voxels) ## Point cloud save

    pbar.close()
    print('Process %d finished!' % rank)

def render_images(sim, n):
    height, width = sim.camera.intrinsic.height, sim.camera.intrinsic.width
    origin = Transform(Rotation.identity(), np.r_[sim.size / 2, sim.size / 2, 0.0])

    extrinsics = np.empty((n, 7), np.float32)
    depth_imgs = np.empty((n, height, width), np.float32)

    for i in range(n):
        r = np.random.uniform(1.6, 2.4) * sim.size
        theta = np.random.uniform(0.0, np.pi / 4.0)
        phi = np.random.uniform(0.0, 2.0 * np.pi)

        extrinsic = camera_on_sphere(origin, r, theta, phi)
        depth_img = sim.camera.render(extrinsic)[1]

        extrinsics[i] = extrinsic.to_list()
        depth_imgs[i] = depth_img

    return depth_imgs, extrinsics


def render_side_images(sim, n=1, random=False):
    #height, width = sim.camera.intrinsic.height, sim.camera.intrinsic.width
    origin = Transform(Rotation.identity(), np.r_[sim.size / 2, sim.size / 2, sim.size / 3])
    
    for i in range(n):
        view_path = args.root / f"view_{i}"
        camera = {}
        if random:
            r = np.random.uniform(1.6, 2.4) * sim.size
            theta = np.random.uniform(0, np.pi / 7.0)
            phi = np.random.uniform(- 5.0 * np.pi / 5, - 3.0 * np.pi / 8.0)
        else:
            r = 2 * sim.size
            theta = np.pi / 6.0
            phi = - np.pi / 2.0

        extrinsic = camera_on_sphere(origin, r, theta, phi)
        depth_img = sim.camera.render(extrinsic)[1]
        camera[f'world_mat'] = extrinsic.as_matrix()
        camera[f'camera_mat'] = sim.camera.intrinsic.K_4

        view_path.mkdir(parents=True)
        np.save(view_path / 'depth.npy', depth_img)
        np.savez(view_path / 'camera.npz', **camera)

        # cameras[f'world_mat_{i}'] = extrinsic.to_list()
        # cameras[f'camera_mat_{i}'] = sim.camera.intrinsic.K
        #cameras[f'scale_mat_{i}'] = np.identity(3)

    

    #return depth_imgs, extrinsics


def sample_grasp_point(point_cloud, finger_depth, eps=0.1):
    points = np.asarray(point_cloud.points)
    normals = np.asarray(point_cloud.normals)
    ok = False
    while not ok:
        # TODO this could result in an infinite loop, though very unlikely
        idx = np.random.randint(len(points))
        point, normal = points[idx], normals[idx]
        ok = normal[2] > -0.1  # make sure the normal is poitning upwards
    grasp_depth = np.random.uniform(-eps * finger_depth, (1.0 + eps) * finger_depth)
    point = point + normal * grasp_depth
    return point, normal


def evaluate_grasp_point(sim, pos, normal, num_rotations=6):
    # define initial grasp frame on object surface
    z_axis = -normal
    x_axis = np.r_[1.0, 0.0, 0.0]
    if np.isclose(np.abs(np.dot(x_axis, z_axis)), 1.0, 1e-4):
        x_axis = np.r_[0.0, 1.0, 0.0]
    y_axis = np.cross(z_axis, x_axis)
    x_axis = np.cross(y_axis, z_axis)
    R = Rotation.from_matrix(np.vstack((x_axis, y_axis, z_axis)).T)

    # try to grasp with different yaw angles
    yaws = np.linspace(0.0, np.pi, num_rotations)
    outcomes, widths = [], []
    for yaw in yaws:
        ori = R * Rotation.from_euler("z", yaw)
        sim.restore_state()
        candidate = Grasp(Transform(ori, pos), width=sim.gripper.max_opening_width)
        outcome, width = sim.execute_grasp(candidate, remove=False)
        outcomes.append(outcome)
        widths.append(width)

    # detect mid-point of widest peak of successful yaw angles
    # TODO currently this does not properly handle periodicity
    successes = (np.asarray(outcomes) == Label.SUCCESS).astype(float)
    if np.sum(successes):
        peaks, properties = signal.find_peaks(
            x=np.r_[0, successes, 0], height=1, width=1
        )
        idx_of_widest_peak = peaks[np.argmax(properties["widths"])] - 1
        ori = R * Rotation.from_euler("z", yaws[idx_of_widest_peak])
        width = widths[idx_of_widest_peak]

    return Grasp(Transform(ori, pos), width), int(np.max(outcomes))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--scene", type=str, choices=["pile", "packed"], default="pile")
    parser.add_argument("--object-set", type=str, default="pile/train")
    parser.add_argument("--num-grasps", type=int, default=10000)
    parser.add_argument("--grasps-per-scene", type=int, default=1)
    parser.add_argument("--num-proc", type=int, default=1)
    parser.add_argument("--save-scene", action="store_true")
    parser.add_argument("--random", action="store_true", help="Add distrubation to camera pose")
    parser.add_argument("--sim-gui", action="store_true")
    args = parser.parse_args()
    args.save_scene = True
    if args.num_proc > 1:
        pool = mp.Pool(processes=args.num_proc)
        for i in range(args.num_proc):
            pool.apply_async(func=main, args=(args, i))
        pool.close()
        pool.join()
    else:
        main(args, 0)
