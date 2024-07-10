import os
import glob
import time
import argparse
import numpy as np

from neugraspnet.simulation import ClutterRemovalSim
from neugraspnet.utils.transform import Transform, Rotation
from neugraspnet.perception import camera_on_sphere

def randomize_scene_view(mesh_pose_list_path, args):
    # Re-create the saved simulation
    sim = ClutterRemovalSim('pile', 'pile/train', gui=args.sim_gui) # parameters 'pile' and 'pile/train' are not used
    mesh_pose_list = np.load(mesh_pose_list_path, allow_pickle=True)['pc']
    sim.setup_sim_scene_from_mesh_pose_list(mesh_pose_list, table=args.add_table)

    # Render random scene view: depth images from random view on sphere
    depth_imgs, extrinsics = render_random_images(sim, 1) # generate single random view image (elevation: 0 to 75 degrees)
    # Save to data_random_raw/scenes_randomized
    save_root = args.save_root
    name = os.path.basename(mesh_pose_list_path) # same as scene_id
    np.savez_compressed(os.path.join(save_root,name), depth_imgs=depth_imgs, extrinsics=extrinsics)

def render_random_images(sim, n): # Adapted from render_images in scripts/generate_data_parallel.py
    height, width = sim.camera.intrinsic.height, sim.camera.intrinsic.width
    origin = Transform(Rotation.identity(), np.r_[sim.size / 2, sim.size / 2, 0.0])

    extrinsics = np.empty((n, 7), np.float32)
    depth_imgs = np.empty((n, height, width), np.float32)

    for i in range(n):
        r = np.random.uniform(1.6, 2.4) * sim.size
        theta = np.random.uniform(0.0, 5* np.pi / 12.0) # elevation: 0 to 75 degrees
        # r = np.random.uniform(2.0, 3.0) * sim.size # increased distance for real world experiments
        # theta = np.random.uniform(0.0, np.pi / 3.0) # elevation: 0 to 60 degrees. Reduced for real world exps
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

def log_result(result):
    g_completed_jobs.append(result)
    elapsed_time = time.time() - g_starting_time

    if len(g_completed_jobs) % 1000 == 0:
        msg = "%05d/%05d %s finished! " % (len(g_completed_jobs), g_num_total_jobs, result)
        msg = msg + 'Elapsed time: ' + \
                time.strftime("%H:%M:%S", time.gmtime(elapsed_time)) + '. '
        print(msg)

def main(args):
    mesh_list_files = glob.glob(os.path.join(args.raw, 'mesh_pose_list', '*.npz'))
    

    global g_completed_jobs
    global g_num_total_jobs
    global g_starting_time

    g_num_total_jobs = len(mesh_list_files)
    g_completed_jobs = []

    g_starting_time = time.time()

    if args.num_proc > 1:
        print('Total jobs: %d, CPU num: %d' % (g_num_total_jobs, args.num_proc))
        # Using fix from: https://github.com/UT-Austin-RPL/GIGA/issues/1
        from joblib import Parallel, delayed
        results = Parallel(n_jobs=args.num_proc)(delayed(randomize_scene_view)(f, args) for f in mesh_list_files)
        
        for result in results:
            log_result(result)
        # import multiprocessing as mp
        # pool = mp.Pool(processes=args.num_proc) 
        # print('Total jobs: %d, CPU num: %d' % (g_num_total_jobs, args.num_proc))
        # for f in mesh_list_files:
        #     pool.apply_async(func=save_occ, args=(f,args), callback=log_result)
        # pool.close()
        # pool.join()
    else:
        for f in mesh_list_files:
            randomize_scene_view(f, args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-proc", type=int, default=1)
    parser.add_argument("raw", type=str)
    parser.add_argument("--sim_gui", action='store_true')
    parser.add_argument("--add_table", action='store_true')
    args = parser.parse_args()
    
    # Save to data_random_raw/scenes_randomized
    if args.add_table == False:
        args.save_root = os.path.join(args.raw, 'scenes_randomized_no_table')
    else:
        args.save_root = os.path.join(args.raw, 'scenes_randomized')
    os.makedirs(args.save_root)

    main(args)