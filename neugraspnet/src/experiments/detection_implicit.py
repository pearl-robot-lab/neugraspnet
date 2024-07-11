from pathlib import Path
import time

import numpy as np
from scipy import ndimage
import torch
import open3d as o3d

#from vgn import vis
from neugraspnet.src.utils.grasp import *
from neugraspnet.src.dataset.io import read_json
from neugraspnet.src.utils.transform import Transform, Rotation
from neugraspnet.src.network.networks import load_network
from neugraspnet.src.utils import visual
import trimesh
from neugraspnet.src.experiments.grasp_sampler import GpgGraspSamplerPcl
from neugraspnet.src.experiments.grasp_renderer import generate_gt_grasp_cloud, generate_neur_grasp_clouds
from neugraspnet.src.experiments.scene_renderer import get_scene_surf_render

axes_cond = lambda x,z: np.isclose(np.abs(np.dot(x, z)), 1.0, 1e-4)


class NeuGraspImplicit(object):
    def __init__(self, model_path, model_type, best=False, force_detection=False, seen_pc_only=False, qual_th=0.5, out_th=0.5, visualize=False, resolution=40, max_grasp_queries_at_once=40, **kwargs):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.device = "cpu"
        self.net = load_network(model_path, self.device, model_type=model_type)
        self.net.eval() # Set to eval mode
        self.model_type = model_type
        self.qual_th = qual_th
        self.best = best
        self.force_detection = force_detection
        self.seen_pc_only = seen_pc_only
        self.out_th = out_th
        self.visualize = visualize
        
        self.resolution = resolution
        self.finger_depth = 0.05
        self.max_grasp_queries_at_once = max_grasp_queries_at_once
        #x, y, z = torch.meshgrid(torch.linspace(start=-0.5, end=0.5 - 1.0 / self.resolution, steps=self.resolution), torch.linspace(start=-0.5, end=0.5 - 1.0 / self.resolution, steps=self.resolution), torch.linspace(start=-0.5, end=0.5 - 1.0 / self.resolution, steps=self.resolution))
        # 1, self.resolution, self.resolution, self.resolution, 3
        #pos = torch.stack((x, y, z), dim=-1).float().unsqueeze(0).to(self.device)
        #self.pos = pos.view(1, self.resolution * self.resolution * self.resolution, 3)
        
    @staticmethod
    def sample_grasp_points(pcd, finger_depth=0.05, eps=0.1):
        # Use masks instead of while loop
        # points Shape (N, 3)
        # normals Shape (N, 3)
        points = np.asarray(pcd.points)        
        normals = np.asarray(pcd.normals)
        mask  = normals[:,-1] > -0.1
        grasp_depth = np.random.uniform(-eps * finger_depth, (1.0 + eps) * finger_depth)
        points = points[mask] + normals[mask] * grasp_depth
        return points, normals[mask]
    
    def get_grasp_queries(self, points, normals):
        # Initializing axes (PyB cam)
        z_axes = -normals
        x_axes = []
        y_axes = []
        num_rotations = int(self.resolution**3//len(points)) + 1

        # Possible x axis
        x1 = np.r_[1.0, 0.0, 0.0]
        x2 = np.r_[0.0, 1.0, 0.0]

        # Defining x_axis and y_axis for each point based on dot product condition (overlapping x and z)
        y_axes = [np.cross(z, x2) if axes_cond(x1,z) else np.cross(z, x1) for z in z_axes]
        x_axes = [np.cross(y, z) for y, z in zip(y_axes, z_axes)]
        
        # Defining rotation matrix with fixed (roll, pitch)
        R = [Rotation.from_matrix(np.vstack((x, y, z)).T) for x,y,z in zip(x_axes, y_axes, z_axes)]

        # Varying yaws from 0, PI with evenly spaced steps
        yaws = np.linspace(0, np.pi, num_rotations)
        pos_queries = []
        rot_queries = []

        # Loop for saving queries
        for i, p in enumerate(points):
            for yaw in yaws:
                ori = R[i] * Rotation.from_euler("z", yaw)
                pos_queries.append(p)
                rot_queries.append(ori.as_quat())

        return pos_queries, rot_queries
            
    
    def __call__(self, state, scene_mesh=None, sim=None, debug_data=None, seed=None, o3d_vis=None, first_call=False, aff_kwargs={}):
        if hasattr(state, 'tsdf_process'):
            tsdf_process = state.tsdf_process
        else:
            tsdf_process = state.tsdf

        if isinstance(state.tsdf, np.ndarray):
            tsdf_vol = state.tsdf
            voxel_size = 0.3 / self.resolution
            size = 0.3
            # pc_extended = state.pc_extended # If debug: Using extended PC for grasp sampling

        else:
            tsdf_vol = state.tsdf.get_grid()
            voxel_size = tsdf_process.voxel_size
            tsdf_process = tsdf_process.get_grid()
            size = state.tsdf.size
            # pc_extended = state.pc_extended # If debug: Using extended PC for grasp sampling

        tic = time.time()

        encoded_tsdf = None
        ## Test with point input?
        # point_input = False
        if 'vn' in self.model_type:
            # Use point cloud input instead of TSDF
            lower = np.array([0.0 , 0.0 , 0.0])
            upper = np.array([size, size, size])
            bounding_box = o3d.geometry.AxisAlignedBoundingBox(lower, upper)
            pc_cropped = state.pc.crop(bounding_box)
            # If more than max points in point cloud, uniformly sample
            if len(pc_cropped.points) > 2048:
                indices = np.random.choice(np.arange(len(pc_cropped.points)), 2048, replace=False)
                pc_cropped = pc_cropped.select_by_index(indices)
            pc_cropped = np.asarray(pc_cropped.points)
            pc_final = np.zeros((2048, 3), dtype=np.float32) # pad zeros to have uniform size
            pc_final[0:pc_cropped.shape[0]] = pc_cropped
            pc_final = pc_final / size - 0.5 # Norm and shift AFTER adding zeros
            tsdf_vol = np.expand_dims(pc_final,0)
        
        ## Get scene render
        torch.cuda.empty_cache()
        if o3d_vis is not None:
            # Running simulation of the scene, grasps generated and grasp clouds
            state.pc.colors = o3d.utility.Vector3dVector(np.tile(np.array([0, 0, 0]), (np.asarray(state.pc.points).shape[0], 1)))
            if first_call:
                o3d_vis.add_geometry(state.pc, reset_bounding_box=True) # HACK TO SET O3D CAMERA VIEW for seed 100!!
                o3d_vis.poll_events()
                o3d_vis.update_renderer()

                # Optional:
                # Now set saved camera view
                ctr = o3d_vis.get_view_control()
                parameters = o3d.io.read_pinhole_camera_parameters("scripts/rendering/ScreenCamera_2023-05-23-22-26-03.json")
                ctr.convert_from_pinhole_camera_parameters(parameters)
                for i in range(50):
                    o3d_vis.poll_events()
                    o3d_vis.update_renderer()

                return [], [], [], 0.0, scene_mesh
            else:
                # Viz input point cloud
                o3d_vis.add_geometry(state.pc, reset_bounding_box=False) # point cloud
                o3d_vis.poll_events()
                if (self.seen_pc_only == True):
                    pc_extended = o3d.geometry.PointCloud(state.pc) # Use only seen areas for grasp sampling
                else:
                    # Use full rendered cloud for grasp sampling
                    with torch.no_grad():
                        encoded_tsdf = self.net.encode_inputs(torch.from_numpy(tsdf_vol).to(self.device))
                    pc_extended = get_scene_surf_render(sim, size, self.resolution, self.net, encoded_tsdf, device=self.device)
                    o3d_vis.add_geometry(pc_extended, reset_bounding_box=False) # point cloud
                o3d_vis.poll_events()
                # num_grasps_gpg = 20
                num_grasps_gpg = 60
        else:
            o3d_vis = None
            # num_grasps_gpg = 40
            num_grasps_gpg = 60
            if (self.seen_pc_only == True):
                pc_extended = o3d.geometry.PointCloud(state.pc) # Use only seen areas for grasp sampling
            else:
                # Use full rendered cloud for grasp sampling
                with torch.no_grad():
                    encoded_tsdf = self.net.encode_inputs(torch.from_numpy(tsdf_vol).to(self.device))
                pc_extended = get_scene_surf_render(sim, size, self.resolution, self.net, encoded_tsdf, device=self.device)

        if debug_data is not None:
            # debug validation
            # get grasps from dataset
            grasps, pos_queries, rot_queries, debug_labels = [], [], [], []
            # Optional: Use only successful grasps for debug
            # debug_data = debug_data[debug_data["label"] == 1]
            for i in debug_data.index:
                rota = debug_data.loc[i, "qx":"qw"].to_numpy(np.single)
                posi = debug_data.loc[i, "x":"z"].to_numpy(np.single)
                grasps.append(Grasp(Transform(Rotation.from_quat(rota), posi), sim.gripper.max_opening_width))
                pos_queries.append(posi)
                rot_queries.append(rota)
                debug_labels.append(debug_data.loc[i, "label"])
                # if len(grasps) >= 16:
                #     break
            # best_only = False # Show all grasps and not just the best one
        else:
            # Use GPG grasp sampling:
            # Optional: Get GT point cloud from scene mesh:
            # Get scene point cloud and normals using ground truth meshes
            # o3d_scene_mesh = scene_mesh.as_open3d
            # o3d_scene_mesh.compute_vertex_normals()
            # pc_extended = o3d_scene_mesh.sample_points_uniformly(number_of_points=1000)
            # Optional: Crop pc and downsample point cloud if too large
            lower = np.array([0.0 , 0.0 , 0.0])
            upper = np.array([size, size, size])
            bounding_box = o3d.geometry.AxisAlignedBoundingBox(lower, upper)
            pc_extended = pc_extended.crop(bounding_box)
            voxel_size = 0.005
            pc_extended_down = pc_extended.voxel_down_sample(voxel_size)

            if pc_extended_down.is_empty():
                print("[Warning]: Empty point cloud input to gpg")
                if self.visualize:
                    return [], [], [], 0.0, scene_mesh
                else:
                    return [], [], [], 0.0
            
            sampler = GpgGraspSamplerPcl(0.05-0.0075) # Franka finger depth is actually a little less than 0.05
            safety_dist_above_table = 0.05 # table is spawned at finger_depth
            num_parallel = 1 # no parallel sampling for now
            grasps, pos_queries, rot_queries, gpg_origin_points = sampler.sample_grasps_parallel(pc_extended_down, num_parallel=num_parallel, num_grasps=num_grasps_gpg, max_num_samples=180,#320
                                                safety_dis_above_table=safety_dist_above_table, show_final_grasps=False, verbose=False,
                                                return_origin_point=True)
            # Optional: Find out if the point comes from a seen or unseen area
            if (self.seen_pc_only == True):
                unseen_flags = None # will be set to 0 for all grasps in select()
            else:
                unseen_flags = []
                distance_threshold = np.float32(voxel_size - 0.0001) # 4.9 mm
                current_pc = o3d.geometry.PointCloud() # Point cloud with one point
                for grasp, origin_point in zip(grasps, gpg_origin_points):
                    current_pc.points = o3d.utility.Vector3dVector(np.array([origin_point]))
                    dist = current_pc.compute_point_cloud_distance(pc_extended_down) 
                    if dist < distance_threshold: # If close to a seen point
                        unseen_flags.append(0) # Seen
                    else:
                        unseen_flags.append(1) # Unseen

            # Use standard GIGA grasp sampling:
            # points, normals = self.sample_grasp_points(pc_extended, self.finger_depth)
            # pos_queries, rot_queries = self.get_grasp_queries(points, normals) # Grasp queries :: (pos ;xyz, rot ;as quat)
            # best_only = True

            if (len(pos_queries) < 2):
                print("[Warning]: No grasps found by GPG")
                if self.visualize:
                    return [], [], [], 0.0, scene_mesh
                else:
                    return [], [], [], 0.0

        # Convert to torch tensor
        pos_queries = torch.Tensor(np.array(pos_queries)).view(1, -1, 3).to(self.device)
        rot_queries = torch.Tensor(np.array(rot_queries)).view(1, -1, 4).to(self.device)
        # Choose queries upto resolution^3 (TODO: Check if needed)
        # chosen_indices = np.random.choice(len(pos_queries),size=(self.resolution**3))
        # pos_queries = pos_queries[chosen_indices].view(1, -1, 3)
        # rot_queries = rot_queries[chosen_indices].view(1, -1, 4)
        # Normalize 3D pos queries
        pos_queries = pos_queries/size - 0.5

        # Variable rot_vol replaced with ==> rot
        # assert tsdf_vol.shape == (1, self.resolution, self.resolution, self.resolution)

        # Query network
        if 'neu' in self.model_type:
            # Also generate grasp point clouds
            # render_settings = read_json(Path("data/grasp_cloud_setup.json"))
            render_settings = read_json(Path("data/grasp_cloud_setup_efficient.json"))
            gt_render = False
            if gt_render:
                # remove table because we dont want to render it
                sim.world.remove_body(sim.world.bodies[0]) # normally table is the first body
                grasps_pc_local = torch.zeros((len(grasps),render_settings['max_points'],3), device=self.device)
                grasps_pc = grasps_pc_local.clone()
                bad_indices = []
                for ind, grasp in enumerate(grasps):
                    result, grasp_pc_local, grasp_pc = generate_gt_grasp_cloud(sim, render_settings, grasp, scene_mesh, debug=False)
                    grasp_pc_local = grasp_pc_local / size # - 0.5 DONT SUBTRACT HERE!
                    grasp_pc = grasp_pc / size - 0.5
                    if result:
                        grasps_pc_local[ind, :grasp_pc_local.shape[0],:] = torch.tensor(grasp_pc_local, device=self.device)
                        grasps_pc[ind, :grasp_pc.shape[0], :] = torch.tensor(grasp_pc, device=self.device)
                    else:
                        # no surface points found for these grasps
                        bad_indices.append(ind)
                # Add table back
                sim.place_table(height=sim.gripper.finger_depth)
            else:
                # Use neural rendered grasp point clouds
                if encoded_tsdf is None:
                    with torch.no_grad():
                        encoded_tsdf = self.net.encode_inputs(torch.from_numpy(tsdf_vol).to(self.device))
                torch.cuda.empty_cache()
                # split into parts to avoid CUDA memory issues
                n = len(grasps)
                remaining_grasps = n
                # max_grasp_queries_at_once = 15
                grasp_idx = 0
                bad_indices, grasps_pc_local, grasps_pc, grasps_viz_list = [], torch.zeros((len(grasps),render_settings['max_points'],3), device=self.device), torch.zeros((len(grasps),render_settings['max_points'],3), device=self.device), []
                while remaining_grasps > 1: # Because we can't use just 1 grasp
                    if remaining_grasps > self.max_grasp_queries_at_once:
                        grasps_batch = grasps[grasp_idx:grasp_idx+self.max_grasp_queries_at_once]
                        
                        curr_bad_indices, curr_grasps_pc_local, curr_grasps_pc, curr_grasps_viz_list = generate_neur_grasp_clouds(sim, render_settings, grasps_batch, size, encoded_tsdf, 
                                                                            self.net, self.device, scene_mesh, debug=False, o3d_vis=o3d_vis)
                        bad_indices += curr_bad_indices
                        grasps_pc_local[grasp_idx:grasp_idx+self.max_grasp_queries_at_once] = curr_grasps_pc_local
                        grasps_pc[grasp_idx:grasp_idx+self.max_grasp_queries_at_once] = curr_grasps_pc
                        grasps_viz_list += curr_grasps_viz_list

                        remaining_grasps -= self.max_grasp_queries_at_once
                        grasp_idx += self.max_grasp_queries_at_once
                    else:
                        grasps_batch = grasps[grasp_idx:]

                        curr_bad_indices, curr_grasps_pc_local, curr_grasps_pc, curr_grasps_viz_list = generate_neur_grasp_clouds(sim, render_settings, grasps_batch, size, encoded_tsdf, 
                                                                        self.net, self.device, scene_mesh, debug=False, o3d_vis=o3d_vis)
                        bad_indices += curr_bad_indices
                        grasps_pc_local[grasp_idx:] = curr_grasps_pc_local
                        grasps_pc[grasp_idx:] = curr_grasps_pc
                        grasps_viz_list += curr_grasps_viz_list
                        
                        remaining_grasps = 0
            
            # Make separate queries for each grasp
            pos_queries = pos_queries.transpose(0,1)
            rot_queries = rot_queries.transpose(0,1)
            # if tsdf_vol.shape[0] == 1:
            #     tsdf_vol = np.repeat(tsdf_vol,len(grasps), axis=0)
            if encoded_tsdf is None:
                with torch.no_grad():
                    encoded_tsdf = self.net.encode_inputs(torch.from_numpy(tsdf_vol).to(self.device))
            qual_vol, width_vol = predict(encoded_tsdf, (pos_queries.to(self.device), rot_queries.to(self.device), grasps_pc_local.to(self.device), grasps_pc.to(self.device)),
                                          self.net, self.device, seed=seed, encoded_input=True)
            # import pdb; pdb.set_trace()
            qual_vol[bad_indices] = 0.0 # set bad grasp scores to zero
            
            # put back into original shape so that we can use the same code as before
            pos_queries = pos_queries.transpose(0,1)
            rot_queries = rot_queries.transpose(0,1)

            if o3d_vis is not None:
                # Visualize the grasp point clouds
                for ind, grasp_viz_mesh in enumerate(grasps_viz_list):
                    if qual_vol[ind] > self.qual_th:
                        grasp_viz_mesh.paint_uniform_color([125/255, 202/255, 92/255])
                        # grasp_viz_mesh.paint_uniform_color([0,1,0])
                        o3d_vis.update_geometry(grasp_viz_mesh)
                    else:
                        # Either remove the bad grasps from viz or color them red
                        o3d_vis.remove_geometry(grasp_viz_mesh, reset_bounding_box=False)
                        # grasp_viz_mesh.paint_uniform_color([1,0,0])
                        # o3d_vis.update_geometry(grasp_viz_mesh)
                    o3d_vis.poll_events()
                    # o3d_vis.update_renderer()
        elif self.model_type == 'pointnetgpd':
            grasps_pc_local = torch.zeros((len(grasps),1000,3), device=self.device)
            grasps_pc = grasps_pc_local.clone()
            bad_indices = []

            # loop over grasps and get local grasp point clouds
            for ind, grasp in enumerate(grasps):
                # grasp_pc_local = 
                # grasp_vgn_inv_tfs[idx] = torch.from_numpy(grasp.pose.inverse().as_matrix()) # This still needs to be in the original grasp frame
                grasp_center = grasp.pose.translation
                # Unfortunately VGN/GIGA grasps are not in the grasp frame we want (frame similar to PointNetGPD), so we need to transform them
                grasp_frame_rot =  grasp.pose.rotation * Rotation.from_euler('Y', np.pi/2) * Rotation.from_euler('Z', np.pi)
                grasp_tf = Transform(grasp_frame_rot, grasp_center)
                p_local = grasp_tf.inverse().transform_point(np.array(state.pc.points))
                # Find local point cloud around grasp
                # X limit (depth)
                p_local = p_local[p_local[:,0] <  sim.gripper.finger_depth]
                p_local = p_local[p_local[:,0] >  0]
                # Y limit (max opening width)
                p_local = p_local[p_local[:,1] <  sim.gripper.max_opening_width/2]
                p_local = p_local[p_local[:,1] > -sim.gripper.max_opening_width/2]
                # Z limit (height)
                sim.gripper.height = 0.03
                p_local = p_local[p_local[:,2] <  sim.gripper.height/2]
                p_local = p_local[p_local[:,2] > -sim.gripper.height/2]
                # Get 50 to 1000 points
                # If more than max points, uniformly sample
                if len(p_local) > 1000:
                    indices = np.random.choice(np.arange(len(p_local)), 1000, replace=False)
                    p_local = p_local[indices]
                # If less than min points, skip this grasp
                if len(p_local) < 50:
                    bad_indices.append(ind)
                else:
                    # Convert local pointcloud back to world frame
                    p_world = grasp_tf.transform_point(p_local)
                    # convert to VGN frame
                    grasp_pc_local = grasp.pose.inverse().transform_point(p_world)

                    grasp_pc_local = grasp_pc_local / size # - 0.5 DONT SUBTRACT HERE!
                    # grasp_pc = grasp_pc / size - 0.5 # Not needed
                    grasps_pc_local[ind, :grasp_pc_local.shape[0],:] = torch.tensor(grasp_pc_local, device=self.device)
                    # grasps_pc[ind, :grasp_pc.shape[0], :] = torch.tensor(grasp_pc, device=self.device) # Not needed

            # Make separate queries for each grasp
            pos_queries = pos_queries.transpose(0,1)
            rot_queries = rot_queries.transpose(0,1)

            qual_vol, width_vol = predict_pngpd((pos_queries.to(self.device), rot_queries.to(self.device), grasps_pc_local.to(self.device), grasps_pc.to(self.device)),
                                self.net, self.device)

            qual_vol[bad_indices] = 0.0 # set bad grasp scores to zero
            
            # put back into original shape so that we can use the same code as before
            pos_queries = pos_queries.transpose(0,1)
            rot_queries = rot_queries.transpose(0,1)
        else:
            qual_vol, width_vol = predict(tsdf_vol, (pos_queries.to(self.device), rot_queries.to(self.device)), self.net, self.device, seed=seed)
        
        # reject voxels with predicted widths that are too small or too large
        # qual_vol, width_vol = process(tsdf_process, qual_vol, rot, width_vol, out_th=self.out_th)
        min_width=0.033
        max_width=0.233
        # qual_vol[np.logical_or(width_vol < min_width, width_vol > max_width)] = 0.0

        # qual_vol = bound(qual_vol, voxel_size) # TODO: Check if needed

        # Reshape to 3D grid. TODO: Check if needed. Update: Not needed
        # qual_vol = qual_vol.reshape((self.resolution, self.resolution, self.resolution))
        # rot_queries = rot_queries.reshape((self.resolution, self.resolution, self.resolution, 4))
        # width_vol = width_vol.reshape((self.resolution, self.resolution, self.resolution))
        
        if self.visualize:
            ### TODO - check affordance - visual (WEIRD)
            colored_scene_mesh = scene_mesh
            # colored_scene_mesh = visual.affordance_visual(qual_vol, rot, scene_mesh, size, self.resolution, **aff_kwargs)
        grasps, scores, unseen_flags, bad_grasps, bad_scores = select(qual_vol.copy(), pos_queries[0].squeeze().cpu(), rot_queries[0].squeeze().cpu(), width_vol,
                                                        threshold=self.qual_th, force_detection=self.force_detection, unseen_flags=unseen_flags)
        toc = time.time() - tic

        bad_grasps, bad_scores = np.asarray(bad_grasps), np.asarray(bad_scores)
        grasps, scores = np.asarray(grasps), np.asarray(scores)

        new_bad_grasps = []
        new_grasps = []
        if len(grasps) > 0:
            p = np.arange(len(grasps))
            p_bad = np.arange(len(bad_grasps))
            for bad_g in bad_grasps[p_bad]:
                pose = bad_g.pose
                # Un-normalize
                pose.translation = (pose.translation + 0.5) * size
                width = bad_g.width * size
                new_bad_grasps.append(Grasp(pose, width))
            for g in grasps[p]:
                pose = g.pose
                # Un-normalize
                pose.translation = (pose.translation + 0.5) * size
                width = g.width * size
                new_grasps.append(Grasp(pose, width))
            bad_scores = bad_scores[p_bad]
            scores = scores[p]
        bad_grasps = new_bad_grasps
        grasps = new_grasps

        if self.visualize:
            composed_scene = colored_scene_mesh
            # # Need coloured mesh from affordance_visual()
            # grasp_mesh_list = [visual.grasp2mesh(g, s) for g, s in zip(grasps, scores)]
            # composed_scene = trimesh.Scene(colored_scene_mesh)
            # for i, g_mesh in enumerate(grasp_mesh_list):
            #     composed_scene.add_geometry(g_mesh, node_name=f'grasp_{i}')
            # bad_grasp_mesh_list = [visual.grasp2mesh(g, s, color='red') for g, s in zip(bad_grasps, bad_scores)]
            # for i, g_mesh in enumerate(bad_grasp_mesh_list):
            #     composed_scene.add_geometry(g_mesh, node_name=f'bad_grasp_{i}')
            # # Optional: Show grasps (for debugging)
            # composed_scene.show()
            return grasps, scores, unseen_flags, toc, composed_scene
        else:
            return grasps, scores, unseen_flags, toc


def bound(qual_vol, voxel_size, limit=[0.02, 0.02, 0.055]):
    # avoid grasp out of bound [0.02  0.02  0.055]
    x_lim = int(limit[0] / voxel_size)
    y_lim = int(limit[1] / voxel_size)
    z_lim = int(limit[2] / voxel_size)
    qual_vol[:x_lim] = 0.0
    qual_vol[-x_lim:] = 0.0
    qual_vol[:, :y_lim] = 0.0
    qual_vol[:, -y_lim:] = 0.0
    qual_vol[:, :, :z_lim] = 0.0
    return qual_vol

def predict(tsdf_vol, pos, net, device, seed=0, encoded_input=False):
    #_, rot = pos
    if encoded_input == False:
        # move input to the GPU
        tsdf_vol = torch.from_numpy(tsdf_vol).to(device)
    else:
        batched_encoded_tsdf = {}
        for keyz, encoded_feats in tsdf_vol.items():
            batched_encoded_tsdf[keyz] = encoded_feats.repeat(pos[0].shape[0], 1, 1, 1)

    # forward pass
    # if seed != 100:
    torch.cuda.empty_cache()
    with torch.no_grad():
        if encoded_input == False:
            qual_vol, width_vol = net(tsdf_vol, pos)
        else:
            qual_vol, width_vol = net.decode(pos, batched_encoded_tsdf) # careful of the order!

    # move output back to the CPU
    qual_vol = qual_vol.cpu().squeeze().numpy()
    #rot = rot.cpu().squeeze().numpy()
    width_vol = width_vol.cpu().squeeze().numpy()
    return qual_vol, width_vol

def predict_pngpd(query, net, device):

    # forward pass
    with torch.no_grad():
        qual_vol = net(query)

    # move output back to the CPU
    qual_vol = qual_vol.cpu().squeeze().numpy()
    #rot = rot.cpu().squeeze().numpy()
    # width_vol = width_vol.cpu().squeeze().numpy()
    width_vol = np.zeros(qual_vol.shape)# zeros
    return qual_vol, width_vol

def process(
    tsdf_vol,
    qual_vol,
    rot_vol,
    width_vol,
    gaussian_filter_sigma=1,
    min_width=0.033,
    max_width=0.233,
    out_th=0.5
):
    tsdf_vol = tsdf_vol.squeeze()

    # smooth quality volume with a Gaussian
    qual_vol = ndimage.gaussian_filter(
        qual_vol, sigma=int(gaussian_filter_sigma), mode="nearest"
    )

    # mask out voxels too far away from the surface
    outside_voxels = tsdf_vol > out_th
    inside_voxels = np.logical_and(1e-3 < tsdf_vol, tsdf_vol < out_th)
    valid_voxels = ndimage.morphology.binary_dilation(
        outside_voxels, iterations=2, mask=np.logical_not(inside_voxels)
    )
    qual_vol[valid_voxels == False] = 0.0

    # reject voxels with predicted widths that are too small or too large
    qual_vol[np.logical_or(width_vol < min_width, width_vol > max_width)] = 0.0

    #return qual_vol, rot_vol, width_vol
    return qual_vol, width_vol


def select(qual_vol, center_vol, rot_vol, width_vol, threshold=0.50, force_detection=False, unseen_flags=None):

    bad_mask = np.where(qual_vol<threshold, 1.0, 0.0)

    if force_detection and (qual_vol >= threshold).sum() == 0:
        pass
    else:
        # zero if lower than threshold
        qual_vol[qual_vol < threshold] = 0.0
    
    # Remaining values are above threshold
    mask = np.where(qual_vol, 1.0, 0.0)

    # construct grasps
    bad_grasps, bad_scores = [], []
    for index in np.argwhere(bad_mask):
        index = index.squeeze()
        ori = Rotation.from_quat(rot_vol[index])
        pos = center_vol[index].numpy()
        bad_grasp, bad_score = Grasp(Transform(ori, pos), width_vol[index]), qual_vol[index]
        # bad_grasp, bad_score = select_index(qual_vol, center_vol, rot_vol, width_vol, index)
        bad_grasps.append(bad_grasp)
        bad_scores.append(bad_score)
    
    grasps, scores, unseen_vals = [], [], []
    for index in np.argwhere(mask):
        index = index.squeeze()
        ori = Rotation.from_quat(rot_vol[index])
        pos = center_vol[index].numpy()
        grasp, score = Grasp(Transform(ori, pos), width_vol[index]), qual_vol[index]
        # grasp, score = select_index(qual_vol, center_vol, rot_vol, width_vol, index)
        grasps.append(grasp)
        scores.append(score)
        if unseen_flags is not None:
            unseen_vals.append(unseen_flags[index])
        else:
            unseen_vals.append(0)
    
    sorted_bad_grasps = [bad_grasps[i] for i in np.argsort(bad_scores)]
    sorted_bad_scores = [bad_scores[i] for i in np.argsort(bad_scores)]
    sorted_grasps = [grasps[i] for i in reversed(np.argsort(scores))]
    sorted_scores = [scores[i] for i in reversed(np.argsort(scores))]
    sorted_unseen_vals = [unseen_vals[i] for i in reversed(np.argsort(scores))]
        
    return sorted_grasps, sorted_scores, sorted_unseen_vals, sorted_bad_grasps, sorted_bad_scores


def select_index(qual_vol, center_vol, rot_vol, width_vol, index):
    i, j, k = index
    score = qual_vol[i, j, k]
    ori = Rotation.from_quat(rot_vol[i, j, k])
    #pos = np.array([i, j, k], dtype=np.float64)
    pos = center_vol[i, j, k].numpy()
    width = width_vol[i, j, k]
    return Grasp(Transform(ori, pos), width), score