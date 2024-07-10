import argparse
from pathlib import Path

import os
import glob
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
from neugraspnet.utils.implicit import get_mesh_pose_list_from_world, get_scene_from_mesh_pose_list

from neugraspnet.grasp_sampler import GpgGraspSamplerPcl

OBJECT_COUNT_LAMBDA = 4
# MAX_VIEWPOINT_COUNT = 6
RESOLUTION = 64


def main(args, rank):
    GRASPS_PER_SCENE = args.grasps_per_scene
    GRASPS_PER_SCENE_GPG = args.grasps_per_scene_gpg
    np.random.seed()
    seed = np.random.randint(0, 1000) + rank
    np.random.seed(seed)
    sim = ClutterRemovalSim(args.scene, args.object_set, gui=args.sim_gui, gripper_type=args.gripper)
    finger_depth = sim.gripper.finger_depth
    scenes_per_worker = args.num_scenes // args.num_proc
    pbar = tqdm(total=scenes_per_worker, disable=rank != 0)

    if rank == 0:
        (args.root / "scenes").mkdir(parents=True, exist_ok=True)
        write_setup(
            args.root,
            sim.size,
            sim.camera.intrinsic,
            sim.gripper.max_opening_width,
            sim.gripper.finger_depth,
        )
        if args.save_scene:
            (args.root / "mesh_pose_list").mkdir(parents=True, exist_ok=True)

    for _ in range(scenes_per_worker):
        # generate heap
        object_count = np.random.poisson(OBJECT_COUNT_LAMBDA) + 1
        sim.reset(object_count)
        sim.save_state()

        # render synthetic depth images
        # n = MAX_VIEWPOINT_COUNT
        # depth_imgs, extrinsics = render_images(sim, n)
        depth_imgs_side, extrinsics_side = render_side_images(sim, 1, args.random)
        
        # Debug
        # import cv2
        # cv2.imshow("la", depth_imgs_side[0])
        # cv2.waitKey(1000)

        # store the raw data
        scene_id = write_sensor_data(args.root, depth_imgs_side, extrinsics_side)
        mesh_pose_list = get_mesh_pose_list_from_world(sim.world, args.object_set)
        if args.save_scene:
            mesh_pose_list = np.array(mesh_pose_list, dtype=object)
            write_point_cloud(args.root, scene_id, mesh_pose_list, name="mesh_pose_list")

        # reconstrct point cloud using a subset of the images
        # tsdf = create_tsdf(sim.size, 120, depth_imgs, sim.camera.intrinsic, extrinsics)
        # pc = tsdf.get_cloud()

        # Get scene point cloud and normals using ground truth meshes
        scene_mesh = get_scene_from_mesh_pose_list(mesh_pose_list)
        o3d_scene_mesh = scene_mesh.as_open3d
        o3d_scene_mesh.compute_vertex_normals()
        pc = o3d_scene_mesh.sample_points_uniformly(number_of_points=1000)

        # # crop surface and borders from point cloud
        bounding_box = o3d.geometry.AxisAlignedBoundingBox(sim.lower, sim.upper)
        pc = pc.crop(bounding_box)
        # o3d.visualization.draw_geometries([pc], point_show_normal=True)

        if pc.is_empty():
            print("Point cloud empty, skipping scene")
            continue

        # sample grasps with GPG:
        sampler = GpgGraspSamplerPcl(gripper_type=args.gripper) # Franka finger depth is actually a little less than 0.05
        safety_dist_above_table = finger_depth # table is spawned at finger_depth
        grasps, _, _ = sampler.sample_grasps_parallel(pc, num_parallel=24, num_grasps=GRASPS_PER_SCENE_GPG, max_num_samples=180,
                                            safety_dis_above_table=safety_dist_above_table, show_final_grasps=False)
        for grasp in grasps:
            label, width = evaluate_grasp_gpg(sim, grasp) # try grasp and get true width
            grasp.width = width
            # store the sample
            write_grasp(args.root, scene_id, grasp, label)
            pbar.update()

        # Optional: sample remaining grasps with regular sampling
        for _ in range(GRASPS_PER_SCENE-GRASPS_PER_SCENE_GPG):
            # sample and evaluate a grasp point
            point, normal = sample_grasp_point(pc, finger_depth)
            grasp, label = evaluate_grasp_point(sim, point, normal)

            # store the sample
            write_grasp(args.root, scene_id, grasp, label)
            pbar.update()

    pbar.close()
    print('Process %d finished!' % rank)


def generate_from_existing_scene(mesh_pose_list_path, args):
    GRASPS_PER_SCENE = args.grasps_per_scene
    GRASPS_PER_SCENE_GPG = args.grasps_per_scene_gpg

    # Re-create the saved simulation
    sim = ClutterRemovalSim('pile', 'pile/train', gui=args.sim_gui, gripper_type=args.gripper) # parameters 'pile' and 'pile/train' are not used
    mesh_pose_list = np.load(mesh_pose_list_path, allow_pickle=True)['pc']
    scene_id = os.path.basename(mesh_pose_list_path)[:-4] # scene id without .npz extension
    sim.setup_sim_scene_from_mesh_pose_list(mesh_pose_list)

    sim.save_state()

    if args.partial_pc == True:
        sim.world.remove_body(sim.world.bodies[0]) # remove table because we dont want to render it # normally table is the first body
        if args.random == True:
            # Render random scene view: depth image from random view on sphere (elevation: 0 to 75 degrees)
            depth_imgs, extrinsics = render_random_images(sim, 1)
        else:
            depth_imgs, extrinsics = render_images(sim, 1)
        sim.place_table(height=sim.gripper.finger_depth) # Add table back
        if args.save_scene:
            # store the raw data    
            # Save to data_random_raw/scenes
            name = os.path.basename(mesh_pose_list_path) # same as scene_id
            np.savez_compressed(os.path.join(args.save_root,name), depth_imgs=depth_imgs, extrinsics=extrinsics)
            write_point_cloud(args.root, scene_id, mesh_pose_list, name="mesh_pose_list")
        
        # construct point cloud
        tsdf = create_tsdf(sim.size, RESOLUTION, depth_imgs, sim.camera.intrinsic, extrinsics)
        # voxels = tsdf.get_grid()
        pc = tsdf.get_cloud()
        # Downsample PC:
        pc = pc.voxel_down_sample(voxel_size=0.005)
    else:
            
            
        if args.random == True:
            # Render random scene view: depth image from random view on sphere (elevation: 0 to 75 degrees)
            depth_imgs, extrinsics = render_random_images(sim, 1)
        else:
            depth_imgs, extrinsics = render_images(sim, 1)
        
        if args.save_scene:
            # store the raw data    
            # Save to data_random_raw/scenes
            name = os.path.basename(mesh_pose_list_path) # same as scene_id
            np.savez_compressed(os.path.join(args.save_root,name), depth_imgs=depth_imgs, extrinsics=extrinsics)
            write_point_cloud(args.root, scene_id, mesh_pose_list, name="mesh_pose_list")


        # Get scene point cloud and normals using ground truth meshes
        scene_mesh = get_scene_from_mesh_pose_list(mesh_pose_list)
        o3d_scene_mesh = scene_mesh.as_open3d
        o3d_scene_mesh.compute_vertex_normals()
        pc = o3d_scene_mesh.sample_points_uniformly(number_of_points=1000)

        # crop surface and borders from point cloud
        lower = np.array([0.02 , 0.02 , 0.04])
        upper = np.array([0.28, 0.28, 0.3])
        bounding_box = o3d.geometry.AxisAlignedBoundingBox(lower, upper)
        pc = pc.crop(bounding_box)


    # # crop surface and borders from point cloud
    # bounding_box = o3d.geometry.AxisAlignedBoundingBox(sim.lower, sim.upper)
    # pc = pc.crop(bounding_box)
    # o3d.visualization.draw_geometries([pc], point_show_normal=True)

    grasps = []
    if GRASPS_PER_SCENE_GPG > 0:
        # sample grasps with GPG:
        sampler = GpgGraspSamplerPcl(gripper_type=args.gripper) # Franka finger depth is actually a little less than 0.05
        safety_dist_above_table = sim.gripper.finger_depth # table is spawned at finger_depth
        grasps, _, _ = sampler.sample_grasps_parallel(pc, num_parallel=24, num_grasps=GRASPS_PER_SCENE_GPG, max_num_samples=180,
                                            safety_dis_above_table=safety_dist_above_table, show_final_grasps=False)
        for grasp in grasps:
            label, width = evaluate_grasp_gpg(sim, grasp) # try grasp and get true width
            grasp.width = width
            # store the sample
            write_grasp(args.root, scene_id, grasp, label)

    # # Optional: sample remaining grasps with regular sampling
    # for _ in range(GRASPS_PER_SCENE-GRASPS_PER_SCENE_GPG):
    #     # sample and evaluate a grasp point
    #     point, normal = sample_grasp_point(pc, sim.gripper.finger_depth)
    #     grasp, label = evaluate_grasp_point(sim, point, normal)

    #     # store the sample
    #     write_grasp(args.root, scene_id, grasp, label)


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
    height, width = sim.camera.intrinsic.height, sim.camera.intrinsic.width
    origin = Transform(Rotation.identity(), np.r_[sim.size / 2, sim.size / 2, sim.size / 3])

    extrinsics = np.empty((n, 7), np.float32)
    depth_imgs = np.empty((n, height, width), np.float32)

    for i in range(n):
        if random:
            r = np.random.uniform(1.6, 2.4) * sim.size
            theta = np.random.uniform(np.pi / 4.0, 5.0 * np.pi / 12.0)
            phi = np.random.uniform(- 5.0 * np.pi / 5, - 3.0 * np.pi / 8.0)
        else:
            r = 2 * sim.size
            theta = np.pi / 3.0
            phi = - np.pi / 2.0

        extrinsic = camera_on_sphere(origin, r, theta, phi)
        depth_img = sim.camera.render(extrinsic)[1]

        extrinsics[i] = extrinsic.to_list()
        depth_imgs[i] = depth_img

    return depth_imgs, extrinsics

def render_random_images(sim, n): # Adapted from render_images in scripts/generate_data_parallel.py
    height, width = sim.camera.intrinsic.height, sim.camera.intrinsic.width
    origin = Transform(Rotation.identity(), np.r_[sim.size / 2, sim.size / 2, 0.0])

    extrinsics = np.empty((n, 7), np.float32)
    depth_imgs = np.empty((n, height, width), np.float32)

    for i in range(n):
        r = np.random.uniform(1.6, 2.4) * sim.size
        theta = np.random.uniform(0.0, 5* np.pi / 12.0) # elevation: 0 to 75 degrees
        phi = np.random.uniform(0.0, 2.0 * np.pi)
        # # Edge grasp randomizations
        # r = np.random.uniform(2, 2.5) * sim.size
        # theta = np.random.uniform(np.pi/4, np.pi/3)
        # phi = np.random.uniform(0.0, 2.0 * np.pi)


        extrinsic = camera_on_sphere(origin, r, theta, phi)
        depth_img = sim.camera.render(extrinsic)[1]

        extrinsics[i] = extrinsic.to_list()
        depth_imgs[i] = depth_img

    return depth_imgs, extrinsics

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


def evaluate_grasp_gpg(sim, grasp):
    sim.restore_state()
    grasp.width = sim.gripper.max_opening_width
    outcome, width = sim.execute_grasp(grasp, remove=False, allow_contact=True)

    return int(outcome), width


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Generate grasping data with GPG or optionally point normal sampling")
    parser.add_argument("--root", type=Path)
    parser.add_argument("--use_previous_scenes", type=bool, default=False)
    parser.add_argument("--previous_root", type=Path, default="")
    parser.add_argument("--scene", type=str, choices=["pile", "packed", "egad"], default="pile")
    parser.add_argument("--object_set", type=str, default="pile/train")
    parser.add_argument("--gripper", type=str, default='franka')
    parser.add_argument("--num_scenes", type=int, default=33313)
    parser.add_argument("--grasps_per_scene", type=int, default=60)
    parser.add_argument("--grasps_per_scene_gpg", type=int, default=60)
    parser.add_argument("--num_proc", type=int, default=1)
    parser.add_argument("--save_scene", type=bool, default=False)
    parser.add_argument("--partial_pc", type=bool, default=False, help="Use partial point cloud to sample grasps")
    parser.add_argument("--random", type=bool, default=False, help="Add distrubation to camera pose")
    parser.add_argument("--sim_gui", type=bool, default=False)
    args, _ = parser.parse_known_args()

    if args.use_previous_scenes:
        if args.previous_root is None:
            raise ValueError("Previous root directory is not specified")
        
        # Write GPG grasp sampler parameters
        (args.root).mkdir(parents=True, exist_ok=True)
        write_json(GpgGraspSamplerPcl().params, args.root / "gpg_setup.json")

        if args.save_scene:
            args.save_root = os.path.join(args.root,'scenes')
            os.makedirs(args.save_root, exist_ok=True)
            (args.root / "mesh_pose_list").mkdir(parents=True, exist_ok=True)
        
        mesh_list_files = glob.glob(os.path.join(args.previous_root, 'mesh_pose_list', '*.npz'))
        global g_completed_jobs
        global g_num_total_jobs
        global g_starting_time
        g_num_total_jobs = len(mesh_list_files)
        g_completed_jobs = []
        g_starting_time = time.time()

        print('Total jobs: %d, CPU num: %d' % (g_num_total_jobs, args.num_proc))
        from joblib import Parallel, delayed
        results = Parallel(n_jobs=args.num_proc)(delayed(generate_from_existing_scene)(f, args) for f in tqdm(mesh_list_files))
        
        # for result in results:
        #     g_completed_jobs.append(result)
        #     elapsed_time = time.time() - g_starting_time
        #     if len(g_completed_jobs) % 1000 == 0:
        #         msg = "%05d/%05d finished! " % (len(g_completed_jobs), g_num_total_jobs)
        #         msg = msg + 'Elapsed time: ' + \
        #                 time.strftime("%H:%M:%S", time.gmtime(elapsed_time)) + '. '
        #         print(msg)
    else:
        # Regular data generation with some samples from GPG:

        # Using fix from: https://github.com/UT-Austin-RPL/GIGA/issues/1
        from joblib import Parallel, delayed
        Parallel(n_jobs=args.num_proc)(delayed(main)(args, i) for i in range(args.num_proc))
        # if args.num_proc > 1:
        #     pool = mp.Pool(processes=args.num_proc)
        #     for i in range(args.num_proc):
        #         pool.apply_async(func=main, args=(args, i))
        #     pool.close()
        #     pool.join()
        # else:
        #     main(args, 0)
