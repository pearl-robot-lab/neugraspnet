import collections
import os
import argparse
from datetime import datetime
import uuid
# import wandb
from pathlib import Path

import numpy as np
import pandas as pd
import tqdm
import trimesh
import open3d as o3d

from neugraspnet.src.dataset import io#, vis
from neugraspnet.src.utils.grasp import *
# from neugraspnet.src.utils.perception import camera_on_sphere
from neugraspnet.src.simulation.simulation import ClutterRemovalSim
from neugraspnet.src.utils.transform import Rotation, Transform
from neugraspnet.src.utils.implicit import get_mesh_pose_list_from_world, get_scene_from_mesh_pose_list

MAX_CONSECUTIVE_FAILURES = 2
MAX_SKIPS = 3

State = collections.namedtuple("State", ["tsdf", "pc"])
# wandb.init(project="6dgrasp", entity="irosa-ias")

def run(
    grasp_plan_fn,
    logdir,
    description,
    scene,
    object_set,
    num_objects=5,
    n=6,
    N=None,
    num_rounds=40,
    seed=1,
    sim_gui=False,
    result_path=None,
    add_noise=False,
    randomize_view=False,
    tight_view=False,
    see_table=False,
    sideview=False,
    resolution=40,
    silence=False,
    visualize=False,
    save_dir=None,
    use_nvisii=False,
):
    """Run several rounds of simulated clutter removal experiments.

    Each round, m objects are randomly placed in a tray. Then, the grasping pipeline is
    run until (a) no objects remain, (b) the planner failed to find a grasp hypothesis,
    or (c) maximum number of consecutive failed grasp attempts.
    """
    #sideview=False
    #n = 6
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
    sim = ClutterRemovalSim(scene, object_set, gui=sim_gui, seed=seed, add_noise=add_noise, sideview=sideview, save_dir=save_dir, use_nvisii=use_nvisii)
    logger = Logger(logdir, description)
    if visualize:
        # Running viz of the scene point clouds and meshes
        o3d_vis = o3d.visualization.Visualizer()
        o3d_vis.create_window()
        first_call = True
        # import pdb; pdb.set_trace()
        # vc = o3d_vis.get_view_control()
        # cam_params = vc.convert_to_pinhole_camera_parameters()
        # cam_params.intrinsic.intrinsic_matrix
        # cam_params.extrinsic = np.array([[0.0, 0.0, 1.0, -0.155], [0.0, 1.0, 0.0, 0.148], [-1.0, 0.0, 0.0, 0.411], [0.0, 0.0, 0.0, 1.0]])
        # # origin = Transform(Rotation.identity(), np.r_[0.15, 0.50, -0.3])
        # origin = Transform(Rotation.identity(), np.r_[0.0, 0.0, 0.0])
        # from neugraspnet.src.utils.perception import camera_on_sphere
        # old_ext = np.array(cam_params.extrinsic)
        # cam_params.extrinsic = camera_on_sphere(origin, radius=1.0, theta=np.pi/4, phi=0.0).as_matrix()
        # vc.convert_from_pinhole_camera_parameters(cam_params, allow_arbitrary=True)

    cnt = 0
    seen_cnt = 1 # to avoid division by zero
    unseen_cnt = 1 # to avoid division by zero
    success = 0
    seen_success = 0
    unseen_success = 0
    left_objs = 0
    total_objs = 0
    cons_fail = 0
    no_grasp = 0
    planning_times = []
    total_times = []

    for _ in tqdm.tqdm(range(num_rounds), disable=silence):

        if visualize and o3d_vis is not None:
            o3d_vis.clear_geometries()
            o3d_vis.poll_events()
        sim.reset(num_objects)

        round_id = logger.last_round_id() + 1
        logger.log_round(round_id, sim.num_objects)
        total_objs += sim.num_objects
        consecutive_failures = 1
        skip_time = 0 # number of times we skip a round because of no grasp or no good quality grasp
        last_label = None
        trial_id = -1

        while sim.num_objects > 0 and consecutive_failures < MAX_CONSECUTIVE_FAILURES and skip_time<MAX_SKIPS:
            trial_id += 1
            timings = {}

            # scan the scene: with optionally, RANDOMIZED view
            if see_table == False:
                sim.world.remove_body(sim.world.bodies[0]) # remove table because we dont want to render it # normally table is the first body
            tsdf, pc, timings["integration"] = sim.acquire_tsdf(n=n, N=N, resolution=resolution, randomize_view=randomize_view, tight_view=tight_view)
            if see_table == False:
                sim.place_table(height=sim.gripper.finger_depth) # Add table back            
            
            # FOR DEBUG: sample extended scene PC for grasp queries on GT cloud
            # while True:
            # _, pc_extended, _ = sim.acquire_tsdf(n=6, N=N, resolution=resolution)
                # if len(pc_extended.points) >= 1000:
                #     break
            
            state = argparse.Namespace(tsdf=tsdf, pc=pc)#, pc_extended=pc_extended)
            # if resolution != 40:
            #     extra_tsdf, _, _ = sim.acquire_tsdf(n=n, N=N, resolution=resolution)
            #     state.tsdf_process = extra_tsdf

            if pc.is_empty():
                print("[WARNING] Empty point cloud, aborting this round")
                total_objs -= sim.num_objects
                break  # empty point cloud, abort this round TODO this should not happen

            # plan grasps
            if visualize:
                # Running viz of the scene point clouds and meshes
                o3d_vis.clear_geometries()
                # Also return the scene mesh with or without grasps
                mesh_pose_list = get_mesh_pose_list_from_world(sim.world, object_set)
                scene_mesh = get_scene_from_mesh_pose_list(mesh_pose_list)
                grasps, scores, unseen_flags, timings["planning"], visual_mesh = grasp_plan_fn(state, scene_mesh, sim=sim, debug_data=None, seed=seed, o3d_vis=o3d_vis, first_call=first_call)
                first_call = False
                # assert not visual_mesh.is_empty
                # o3d.visualization.draw_geometries([visual_mesh.as_open3d])
                # logger.log_mesh(scene_mesh, visual_mesh, f'round_{round_id:03d}_trial_{trial_id:03d}')
                
            else:
                grasps, scores, unseen_flags, timings["planning"] = grasp_plan_fn(state, sim=sim, debug_data=None, seed=seed)
            planning_times.append(timings["planning"])
            total_times.append(timings["planning"] + timings["integration"])

            if len(grasps) == 0:
                # not enough candidate grasps were sampled OR no grasps above quality threshold were found
                no_grasp += 1
                skip_time += 1
                continue  # no good grasps found, skip

            # execute grasp
            grasp, score, unseen_flag = grasps[0], scores[0], unseen_flags[0]
            # print(f"[Unseen grasp?: {unseen_flag}]")
            print("[BEST Score: %.2f]" % score)
            label, _ = sim.execute_grasp(grasp, allow_contact=True)
            print("[RESULT: %s]" % label)
            cnt += 1
            if unseen_flag:
                unseen_cnt += 1
            else:
                seen_cnt += 1
            if label != Label.FAILURE:
                success += 1
                if unseen_flag:
                    unseen_success += 1
                else:
                    seen_success += 1

            # log the grasp
            logger.log_grasp(round_id, state, timings, grasp, score, label)

            if last_label == Label.FAILURE and label == Label.FAILURE:
                consecutive_failures += 1
            else:
                consecutive_failures = 1
            if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                cons_fail += 1
            last_label = label

        left_objs += sim.num_objects
    success_rate = 100.0 * success / cnt
    seen_success_rate = 100.0 * seen_success / seen_cnt
    unseen_success_rate = 100.0 * unseen_success / unseen_cnt
    declutter_rate = 100.0 * success / total_objs
    print('Grasp success rate: %.2f %%, Declutter rate: %.2f %%' % (success_rate, declutter_rate))
    print('Seen success rate: %.2f %%, Unseen success rate: %.2f %%' % (seen_success_rate, unseen_success_rate))
    print('Seen grasp count: %d, Unseen grasp count: %d' % (seen_cnt, unseen_cnt))
    print(f'Average planning time: {np.mean(planning_times)}, total time: {np.mean(total_times)}')
    print('Consecutive failures and no detections: %d, %d' % (cons_fail, no_grasp))
    if result_path is not None:
        with open(result_path, 'w') as f:
            f.write('%.2f%%, %.2f%%, %.2f%%, %.2f%%; %d, %d, %d, %d\n' % (success_rate, declutter_rate, seen_success_rate, unseen_success_rate,
                                                   seen_cnt, unseen_cnt, cons_fail, no_grasp))
    return success_rate, declutter_rate, seen_success_rate, unseen_success_rate, seen_cnt, unseen_cnt
    


class Logger(object):
    def __init__(self, root, description):
        time_stamp = datetime.now().strftime("%y-%m-%d-%H-%M-%S")
        description = "{}_{}".format(time_stamp, description).strip("_")

        self.logdir = root / description
        self.scenes_dir = self.logdir / "scenes"
        self.scenes_dir.mkdir(parents=True, exist_ok=True)

        self.mesh_dir = self.logdir / "meshes"
        self.mesh_dir.mkdir(parents=True, exist_ok=True)

        self.rounds_csv_path = self.logdir / "rounds.csv"
        self.grasps_csv_path = self.logdir / "grasps.csv"
        self._create_csv_files_if_needed()

    def _create_csv_files_if_needed(self):
        if not self.rounds_csv_path.exists():
            io.create_csv(self.rounds_csv_path, ["round_id", "object_count"])

        if not self.grasps_csv_path.exists():
            columns = [
                "round_id",
                "scene_id",
                "qx",
                "qy",
                "qz",
                "qw",
                "x",
                "y",
                "z",
                "width",
                "score",
                "label",
                "integration_time",
                "planning_time",
            ]
            io.create_csv(self.grasps_csv_path, columns)

    def last_round_id(self):
        df = pd.read_csv(self.rounds_csv_path)
        return -1 if df.empty else df["round_id"].max()

    def log_round(self, round_id, object_count):
        io.append_csv(self.rounds_csv_path, round_id, object_count)

    def log_mesh(self, scene_mesh, aff_mesh, name):
        scene_mesh.export(self.mesh_dir / (name + "_scene.obj"), 'obj')
        #aff_mesh = aff_mesh.scaled(4)
        aff_mesh.export(str(self.mesh_dir / (name + "_aff.stl")), 'stl')
        #trimesh.exchange.export.export_scene(aff_mesh, 'abc1.obj', file_type='obj')
        assert not aff_mesh.is_empty
        # wandb.log({'Grasps (Scene vs Grasp)' : [wandb.Object3D(open(self.mesh_dir / (name + "_scene.obj"))),
        #                                        wandb.Object3D(open(self.mesh_dir / (name + "_aff.obj")))]})

    def log_grasp(self, round_id, state, timings, grasp, score, label):
        # log scene
        tsdf, points = state.tsdf, np.asarray(state.pc.points)
        scene_id = uuid.uuid4().hex
        scene_path = self.scenes_dir / (scene_id + ".npz")
        np.savez_compressed(scene_path, grid=tsdf.get_grid(), points=points)

        # log grasp
        qx, qy, qz, qw = grasp.pose.rotation.as_quat()
        x, y, z = grasp.pose.translation
        width = grasp.width
        label = int(label)
        io.append_csv(
            self.grasps_csv_path,
            round_id,
            scene_id,
            qx,
            qy,
            qz,
            qw,
            x,
            y,
            z,
            width,
            score,
            label,
            timings["integration"],
            timings["planning"],
        )


class Data(object):
    """Object for loading and analyzing experimental data."""

    def __init__(self, logdir):
        self.logdir = logdir
        self.rounds = pd.read_csv(logdir / "rounds.csv")
        self.grasps = pd.read_csv(logdir / "grasps.csv")

    def num_rounds(self):
        return len(self.rounds.index)

    def num_grasps(self):
        return len(self.grasps.index)

    def success_rate(self):
        return self.grasps["label"].mean() * 100

    def percent_cleared(self):
        df = (
            self.grasps[["round_id", "label"]]
            .groupby("round_id")
            .sum()
            .rename(columns={"label": "cleared_count"})
            .merge(self.rounds, on="round_id")
        )
        return df["cleared_count"].sum() / df["object_count"].sum() * 100

    def avg_planning_time(self):
        return self.grasps["planning_time"].mean()

    def read_grasp(self, i):
        scene_id, grasp, label = io.read_grasp(self.grasps, i)
        score = self.grasps.loc[i, "score"]
        scene_data = np.load(self.logdir / "scenes" / (scene_id + ".npz"))

        return scene_data["points"], grasp, score, label
