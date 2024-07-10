import argparse
from pathlib import Path

import numpy as np
import open3d as o3d
from tqdm import tqdm
import multiprocessing as mp

from neugraspnet.io import *
from neugraspnet.perception import *
from neugraspnet.utils.misc import apply_noise


RESOLUTION = 64 # High Resolution
# RESOLUTION = 40 # Low Resolution
PCD_SIZE = 2048

def process_one_scene(args, f):
    if f.suffix != ".npz":
        return f.stem
    depth_imgs, extrinsics = read_sensor_data(args.raw, f.stem)
    # add noise
    depth_imgs = np.array([apply_noise(x, args.add_noise) for x in depth_imgs])
    if args.single_view:
        tsdf = create_tsdf(size, RESOLUTION, depth_imgs[[0]], intrinsic, extrinsics[[0]])
    else:
        tsdf = create_tsdf(size, RESOLUTION, depth_imgs, intrinsic, extrinsics)
    grid = tsdf.get_grid()
    write_voxel_grid(args.dataset, f.stem, grid)

    pc = tsdf.get_cloud()
    # crop surface and borders from point cloud
    # o3d.visualization.draw_geometries([pc])
    lower = np.array([0.0 , 0.0 , 0.0])
    upper = np.array([0.3, 0.3, 0.3])
    bounding_box = o3d.geometry.AxisAlignedBoundingBox(lower, upper)
    pc_cropped = pc.crop(bounding_box)

    # If more than max points in point cloud, uniformly sample
    if len(pc_cropped.points) > PCD_SIZE:
        indices = np.random.choice(np.arange(len(pc_cropped.points)), PCD_SIZE, replace=False)
        pc_cropped = pc_cropped.select_by_index(indices)

    pc_cropped = np.asarray(pc_cropped.points)

    pc_final = np.zeros((PCD_SIZE, 3)) # pad zeros to have uniform size
    pc_final[0:pc_cropped.shape[0]] = pc_cropped
    write_point_cloud(args.dataset, f.stem, pc_final)
    return str(f.stem)

def log_result(result):
    g_num_completed_jobs.append(result)
    elapsed_time = time.time() - g_starting_time

    if len(g_num_completed_jobs) % 1000 == 0:
        msg = "%05d/%05d %s finished! " % (len(g_num_completed_jobs), g_num_total_jobs, result)
        msg = msg + 'Elapsed time: ' + \
                time.strftime("%H:%M:%S", time.gmtime(elapsed_time)) + '. '
        print(msg)

def main(args):
    if args.single_view:
        print('Loading first view only!')
    # create directory of new dataset
    (args.dataset / "scenes").mkdir(parents=True)
    (args.dataset / "point_clouds").mkdir(parents=True)

    global g_num_completed_jobs
    global g_num_total_jobs
    global g_starting_time
    global size
    global intrinsic

    # load setup information
    size, intrinsic, _, finger_depth = read_setup(args.raw)
    assert np.isclose(size, 6.0 * finger_depth)
    voxel_size = size / RESOLUTION

    # Optional: create new df
    # df = read_df(args.raw)
    # df["x"] /= voxel_size
    # df["y"] /= voxel_size
    # df["z"] /= voxel_size
    # df["width"] /= voxel_size
    # df = df.rename(columns={"x": "i", "y": "j", "z": "k"})
    # write_df(df, args.dataset)

    g_num_completed_jobs = []
    file_list = list((args.raw / "scenes").iterdir())
    g_num_total_jobs = len(file_list)
    g_starting_time = time.time()

    # create tsdfs and pcs

    if args.num_proc > 1:
        print('Total jobs: %d, CPU num: %d' % (g_num_total_jobs, args.num_proc))
        # Using fix from: https://github.com/UT-Austin-RPL/GIGA/issues/1
        from joblib import Parallel, delayed
        results = Parallel(n_jobs=args.num_proc)(delayed(process_one_scene)(args, f,) for f in file_list)
        
        for result in results:
            log_result(result)
        # pool = mp.Pool(processes=args.num_proc)

        # print('Total jobs: %d, CPU num: %d' % (g_num_total_jobs, args.num_proc))
        # for f in file_list:
        #     pool.apply_async(func=process_one_scene, args=(args, f,), callback=log_result)
        # pool.close()
        # pool.join()
    else:
        for f in tqdm(file_list, total=len(file_list)):
            process_one_scene(args, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("raw", type=Path)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--num-proc", type=int, default=1)
    parser.add_argument("--single-view", action='store_true')
    parser.add_argument("--add-noise", type=str, default='')
    args = parser.parse_args()
    main(args)
