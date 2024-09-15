import os
from pathlib import Path
import argparse
import numpy as np
import pickle as pkl
from tqdm import tqdm

from vgn.io import *
from vgn.perception import *
from vgn.grasp import Grasp
from vgn.simulation import ClutterRemovalSim
from vgn.utils.transform import Rotation, Transform
from vgn.utils.implicit import get_scene_from_mesh_pose_list, as_mesh, get_occ_specific_points, get_occ_and_sem_class_specific_points
from vgn.grasp_renderer import generate_gt_grasp_cloud
from vgn.assign_grasp_affordance import affrdnce_label_dict, get_affordance

from joblib import Parallel, delayed

def generate_from_existing_grasps(grasp_data_entry, args, render_settings):

    # Get mesh pose list
    scene_id = grasp_data_entry['scene_id']
    file_name = scene_id + ".npz"
    mesh_pose_list = np.load(args.raw_root / "mesh_pose_list" / file_name, allow_pickle=True)['pc']

    # Re-create the saved simulation
    sim = ClutterRemovalSim(args.scene, args.object_set , gui=args.sim_gui, data_root=args.data_root) # args 'pile' and 'pile/train' are not used here
    sim.setup_sim_scene_from_mesh_pose_list(mesh_pose_list, table=args.render_table, data_root=args.data_root)

    # Load the grasp
    pos = grasp_data_entry["x":"z"].to_numpy(np.single)
    rotation = Rotation.from_quat(grasp_data_entry["qx":"qw"].to_numpy(np.single))
    grasp = Grasp(Transform(rotation, pos), sim.gripper.max_opening_width)
    if args.debug:
        scene_mesh = get_scene_from_mesh_pose_list(mesh_pose_list, data_root=args.data_root)
    else:
        scene_mesh = None
    result, grasp_pc_local, grasp_pc = generate_gt_grasp_cloud(sim, render_settings, grasp, scene_mesh=scene_mesh, debug=args.debug)


    if result:
        if args.debug:
            return True
        else:
            ## Success. Save grasp & surface point cloud & optionally occupancy and affordance values
            grasp_id = grasp_data_entry['grasp_id']
            
            if args.save_affrdnce_values is True:
                # check if it is a successful grasp
                if grasp_data_entry['label'] == 1:
                    # get affordance of the grasp (scene must contain affnet dataset objects)
                    affrdnce_labels = get_affordance(sim, sim.aff_dataset, grasp, mesh_pose_list) # affdnces are 0/1 labels for each aff class
                else:
                    # unsuccessful grasp. Set all affordances to 0
                    affrdnce_labels = [int(0)] * len(affrdnce_label_dict.keys())
                # save grasp, affordance to another grasps_with_clouds csv
                append_csv(args.csv_path,
                        grasp_id, scene_id, grasp_data_entry["qx"], grasp_data_entry["qy"], grasp_data_entry["qz"], grasp_data_entry["qw"],
                        grasp_data_entry['x'], grasp_data_entry['y'], grasp_data_entry['z'], grasp_data_entry['width'], grasp_data_entry['label'],
                        *affrdnce_labels)
                    
            else:    
                # save grasp to another grasps_with_clouds csv
                append_csv(args.csv_path,
                        grasp_id, scene_id, grasp_data_entry["qx"], grasp_data_entry["qy"], grasp_data_entry["qz"], grasp_data_entry["qw"],
                        grasp_data_entry['x'], grasp_data_entry['y'], grasp_data_entry['z'], grasp_data_entry['width'], grasp_data_entry['label'])
            
            # save surface point cloud
            surface_pc_path = args.raw_root / "grasp_point_clouds" / f"{grasp_id}.npz"
            np.savez_compressed(surface_pc_path, pc=grasp_pc)
            
            if args.save_occ_values:
                # Also get occupancies of the points and save these occupancy values. Optionally also save the semantic class labels

                surface_pc_occ_path = args.raw_root / "occ_grasp_point_clouds" / f"{grasp_id}.npz"
                scene, mesh_list = get_scene_from_mesh_pose_list(mesh_pose_list, return_list=True, data_root=args.data_root)
                
                if args.save_occ_semantics:
                    # get semantic class labels from affnet dataset
                    points, occ, sem = get_occ_and_sem_class_specific_points(sim.aff_dataset, mesh_pose_list, mesh_list, grasp_pc)
                    # save occupancy values and semantic class labels
                    np.savez_compressed(surface_pc_occ_path, points=points, occ=occ, sem=sem)
                else:
                    points, occ = get_occ_specific_points(mesh_list, grasp_pc)
                    # save occupancy values
                    np.savez_compressed(surface_pc_occ_path, points=points, occ=occ)

            if args.sim_gui:
                sim.world.p.disconnect()

            return True
    else:
        # Points are too few! Skipping this grasp...
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Generate surface point clouds for each grasp. Assign affordance to succesful grasps")
    parser.add_argument("--raw_root", type=Path)
    parser.add_argument("--data_root", type=Path, default="", help="Root directory for the dataset obj files")
    parser.add_argument("--debug", type=bool, default=False)
    parser.add_argument("--scene", type=str, choices=["pile", "packed", "egad", "affnet"], default="pile")
    parser.add_argument("--object_set", type=str, default="pile/train")
    parser.add_argument("--num_proc", type=int, default=1)
    parser.add_argument("--three_cameras", type=bool, default=True)
    parser.add_argument("--camera_fov", type=float, default=120.0, help="Camera image angular FOV in degrees")
    parser.add_argument("--camera_image_res", type=int, default=64)
    parser.add_argument("--render_table", type=bool, default='', help="Also render table in depth images")
    parser.add_argument("--voxel_downsample_size", type=float, default=0.002) # 2mm
    parser.add_argument("--max_points", type=int, default=1023)
    parser.add_argument("--min_points", type=int, default=50)
    parser.add_argument("--add_noise", type=bool, default='', help="Add dex noise to point clouds and depth images")
    parser.add_argument("--save_affrdnce_values", type=bool, default='', help="Also save the affordances of the grasps using our 3DGraspAffNet dataset")
    parser.add_argument("--save_occ_values", type=bool, default='', help="Also save the occupancy values of the grasp cloud")
    parser.add_argument("--save_occ_semantics", type=bool, default='', help="Also save the occupancy semantics of the grasp cloud")
    parser.add_argument("--sim_gui", type=bool, default=False)
    args, _ = parser.parse_known_args()

    if args.raw_root is None:
        raise ValueError("Root directory is not specified")

    # Write grasp cloud parameters
    render_settings = dict({"scene": args.scene,
                     "object_set": args.object_set,
                     "three_cameras": args.three_cameras,
                     "camera_fov": args.camera_fov,
                     "camera_image_res": args.camera_image_res,
                     "n_proposal_steps": 15, # Optional settings for neural grasp renderer
                     "n_secant_steps": 8, # Optional settings for neural grasp renderer
                     "render_table": args.render_table,
                     "voxel_downsample_size": args.voxel_downsample_size,
                     "max_points": args.max_points,
                     "min_points": args.min_points,
                     "add_noise": args.add_noise
                     })
    write_json(render_settings, args.raw_root / "grasp_cloud_setup.json")

    # Read all grasp data
    df = read_df(args.raw_root)
    print('Num grasps in raw dataset: %d' % len(df))
    if 'grasp_id' not in df.columns:
        # Add a column for grasp id. Use index values
        df.insert(0,'grasp_id',df.index)
    if not args.debug:
        # Create a directory for storing grasp point clouds
        os.makedirs(args.raw_root / "grasp_point_clouds", exist_ok=True)
        if args.save_occ_values is True:
            os.makedirs(args.raw_root / "occ_grasp_point_clouds", exist_ok=True)
        # Create another csv file for storing grasps that have point clouds
        args.csv_path = args.raw_root / "grasps_balanced_with_clouds.csv"
        if args.csv_path.exists():
            print("[Error]: CSV file with same name already exists. Exiting...")
            exit()
        csv_header = ["grasp_id", "scene_id", "qx", "qy", "qz", "qw", "x", "y", "z", "width", "label"]
        if args.save_affrdnce_values is True:
            # # pre-load the affnet dataset? (slower)
            # affnet_root = Path(args.data_root) / 'data/3DGraspAff/'
            # aff_path = os.path.join(affnet_root, 'filt_scaled_anntd_remapped_full_shape_'+args.object_set+'_data.pkl')
            # with open(aff_path, 'rb') as f:
            #     args.aff_dataset = pkl.load(f)
            # add affordance labels to csv header
            for affdnce_class in affrdnce_label_dict.keys():
                csv_header.append(affdnce_class)
        create_csv(
            args.csv_path,
            csv_header,
        )

    global g_completed_jobs
    global g_num_total_jobs
    global g_starting_time
    g_num_total_jobs = len(df)
    g_completed_jobs = []
    g_starting_time = time.time()
    print('Total jobs: %d, CPU num: %d' % (g_num_total_jobs, args.num_proc))

    if args.debug:
        assert args.num_proc == 1, "Debug mode only allowed with num_proc=1"
        # Test only one scene
        indices = [np.random.randint(len(df))]
        results = Parallel(n_jobs=args.num_proc)(delayed(generate_from_existing_grasps)(df.loc[index], args, render_settings) for index in indices)
    else:
        results = Parallel(n_jobs=args.num_proc)(delayed(generate_from_existing_grasps)(df.loc[index], args, render_settings) for index in tqdm(range(len(df))))
    
    for result in results:
        g_completed_jobs.append(result)
        elapsed_time = time.time() - g_starting_time
        if len(g_completed_jobs) % 1000 == 0:
            msg = "%05d/%05d finished! " % (len(g_completed_jobs), g_num_total_jobs)
            msg = msg + 'Elapsed time: ' + \
                    time.strftime("%H:%M:%S", time.gmtime(elapsed_time)) + '. '
            print(msg)
