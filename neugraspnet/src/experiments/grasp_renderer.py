import numpy as np
import torch
import open3d as o3d
import matplotlib.pyplot as plt
import trimesh
from neugraspnet.src.utils.implicit import as_mesh
from neugraspnet.src.utils.perception import CameraIntrinsic
from neugraspnet.src.utils.misc import apply_noise
from neugraspnet.src.utils.grasp import *
from neugraspnet.src.utils.transform import Transform, Rotation


def get_simple_proposal(ray0, ray_direction, n_steps=25, depth_range=[0.001, 0.18]):
    
    d_proposal = torch.linspace(0, 1, steps=n_steps).view(1, 1, n_steps, 1)
    d_proposal = depth_range[0] * (1. - d_proposal) + depth_range[1]* d_proposal

    p_proposal = ray0.unsqueeze(2).repeat(1, 1, n_steps, 1) + ray_direction.unsqueeze(2).repeat(1, 1, n_steps, 1) * d_proposal

    return p_proposal, d_proposal


def secant(net, encoded_tsdf, f_low, f_high, d_low, d_high, n_secant_steps,
                          ray0_masked, ray_direction_masked, tau, scale_size):
    ''' Runs the secant method for interval [d_low, d_high].

    Args:
        d_low (tensor): start values for the interval
        d_high (tensor): end values for the interval
        n_secant_steps (int): number of steps
        ray0_masked (tensor): masked ray start points
        ray_direction_masked (tensor): masked ray direction vectors
        model (nn.Module): model model to evaluate point occupancies
        c (tensor): latent conditioned code c
        tau (float): threshold value in logits
        scale_size (float): optional scaling factor for the input points
    '''
    d_pred = - f_low * (d_high - d_low) / (f_high - f_low) + d_low
    for i in range(n_secant_steps):
        p_mid = ray0_masked + d_pred.unsqueeze(-1) * ray_direction_masked
        p_scaled_mid = (p_mid/scale_size - 0.5).float()
        with torch.no_grad():
            f_mid = (net.infer_occ(p_scaled_mid.view(1, -1, 3), encoded_tsdf, encoded_inputs=True)- tau).squeeze()
            torch.cuda.empty_cache()
        ind_low = f_mid < 0
        ind_low = ind_low
        if ind_low.sum() > 0:
            d_low[ind_low] = d_pred[ind_low]
            f_low[ind_low] = f_mid[ind_low]
        if (ind_low == 0).sum() > 0:
            d_high[ind_low == 0] = d_pred[ind_low == 0]
            f_high[ind_low == 0] = f_mid[ind_low == 0]

        d_pred = - f_low * (d_high - d_low) / (f_high - f_low) + d_low
    return d_pred


def generate_neur_grasp_clouds(sim, render_settings, grasps, size, encoded_tsdf, net, device=torch.device("cuda"), scene_mesh=None, debug=False, o3d_vis=None, viz_rays=False):
    max_points = render_settings['max_points']
    voxel_downsample_size = render_settings['voxel_downsample_size']
    width = height = render_settings['camera_image_res']

    batch_size = len(grasps)
    n_steps = render_settings['n_proposal_steps']
    n_pts = width*height*3
    n_secant_steps = 8 # For surface refinement (crucial)
    # if tsdf.shape[0] == 1:
    #     tsdf = np.repeat(tsdf,len(grasps), axis=0)
    # tsdf_t = torch.tensor(tsdf, device=device, dtype=torch.float32)
    # if encoded_tsdf.shape[0] == 1: # We assume this is always the case!
    # add batch dimension
    batched_encoded_tsdf = {}
    for keyz, encoded_feats in encoded_tsdf.items():
        batched_encoded_tsdf[keyz] = encoded_feats.repeat(len(grasps), 1, 1, 1)
    
    # Camera settings
    width_fov  = np.deg2rad(render_settings['camera_fov']) # angular FOV (120 by default)
    height_fov = np.deg2rad(render_settings['camera_fov']) # angular FOV (120 by default)
    f_x = width  / (np.tan(width_fov / 2.0))
    f_y = height / (np.tan(height_fov / 2.0))
    intrinsic = CameraIntrinsic(width, height, f_x, f_y, width/2, height/2)
    # To capture 5cms on both sides of the gripper, using a 120 deg FOV, we need to be atleast 0.05/tan(60) = 2.8 cms away
    height_max_dist = sim.gripper.max_opening_width/2.5
    width_max_dist  = sim.gripper.max_opening_width/2.0 + 0.015 # 1.5 cm extra
    dist_from_gripper = width_max_dist/np.tan(width_fov/2.0)
    min_measured_dist = 0.001
    max_measured_dist = dist_from_gripper + sim.gripper.finger_depth + 0.005 # 0.5 cm extra
    max_measured_dist_ray = max_measured_dist*1.5 # Adding 50% extra for the ray marching
    # if render_settings['three_cameras']:
    # Use one camera for wrist and two cameras for the fingers
    finger_width_max_dist = sim.gripper.finger_depth/2.0 + 0.005 # 0.5 cm extra
    dist_from_finger = finger_width_max_dist/np.tan(width_fov/2.0)
    finger_max_measured_dist = (dist_from_finger + 0.95*sim.gripper.max_opening_width)*1.1 # Adding 10% extra

    if debug:
        # DEBUG: Viz scene point cloud and normals using ground truth meshes
        o3d_scene_mesh = scene_mesh.as_open3d
        o3d_scene_mesh.compute_vertex_normals()
        pc = o3d_scene_mesh.sample_points_uniformly(number_of_points=1000)
        pc.colors = o3d.utility.Vector3dVector(np.tile(np.array([0, 0, 0]), (np.asarray(pc.points).shape[0], 1)))
        o3d.visualization.draw_geometries([pc])

    # get grasp tf and extrinsic matrices
    grasp_inv_tfs = torch.ones((batch_size, 4, 4), dtype=torch.float32)
    grasp_vgn_inv_tfs = torch.ones((batch_size, 4, 4), dtype=torch.float32)
    grasp_cam_inv_extrinsics = torch.ones((batch_size, 4, 4), dtype=torch.float32)
    left_grasp_cam_inv_extrinsics = torch.ones((batch_size, 4, 4), dtype=torch.float32)
    right_grasp_cam_inv_extrinsics = torch.ones((batch_size, 4, 4), dtype=torch.float32)
    camera_world = torch.zeros((batch_size, width*height, 3))
    left_camera_world = torch.zeros((batch_size, width*height, 3))
    right_camera_world = torch.zeros((batch_size, width*height, 3))
    grasps_viz_list = []
    for idx, grasp in enumerate(grasps):
        ## Move camera to grasp offset frame
        grasp_center = grasp.pose.translation
        # Unfortunately VGN/GIGA grasps are not in the grasp frame we want (frame similar to PointNetGPD), so we need to transform them
        grasp_frame_rot =  grasp.pose.rotation * Rotation.from_euler('Y', np.pi/2) * Rotation.from_euler('Z', np.pi)
        grasp_tf = Transform(grasp_frame_rot, grasp_center).as_matrix()
        offset_pos =  (grasp_tf @ np.array([[-dist_from_gripper],[0],[0],[1.]]))[:3].squeeze() # Move to offset frame
        # Unfortunately the bullet renderer uses the OpenGL format so we have to use yet another extrinsic
        grasp_up_axis = grasp_tf.T[2,:3] # np.array([0.0, 0.0, 1.0]) # grasp_tf z-axis
        inv_extrinsic_bullet = Transform.look_at(eye=offset_pos, center=grasp_center, up=grasp_up_axis).inverse()
        # Store
        grasp_inv_tfs[idx] = torch.from_numpy(np.linalg.inv(grasp_tf)) # This still needs to be in the original grasp frame
        grasp_vgn_inv_tfs[idx] = torch.from_numpy(grasp.pose.inverse().as_matrix()) # This still needs to be in the original grasp frame
        grasp_cam_inv_extrinsics[idx] = torch.from_numpy(inv_extrinsic_bullet.as_matrix()).float()
        camera_world[idx, :] = torch.from_numpy(inv_extrinsic_bullet.translation).float()
        ## Do the same for the other cameras
        # if render_settings['three_cameras']:
        ## Move camera to finger offset frame
        fingers_center =  (grasp_tf @ np.array([[sim.gripper.finger_depth/2.0],[0],[0],[1.]]))[:3].squeeze()
        left_finger_offset_pos  =  (grasp_tf @ np.array([[sim.gripper.finger_depth/2.0],[ (dist_from_finger + sim.gripper.max_opening_width/2.0)],[0],[1.]]))[:3].squeeze()
        right_finger_offset_pos =  (grasp_tf @ np.array([[sim.gripper.finger_depth/2.0],[-(dist_from_finger + sim.gripper.max_opening_width/2.0)],[0],[1.]]))[:3].squeeze()
        # Unfortunately the bullet renderer uses the OpenGL format so we have to use yet another extrinsic
        left_finger_inv_extrinsic_bullet  = Transform.look_at(eye=left_finger_offset_pos,  center=fingers_center, up=grasp_up_axis).inverse()
        right_finger_inv_extrinsic_bullet = Transform.look_at(eye=right_finger_offset_pos, center=fingers_center, up=grasp_up_axis).inverse()
        # Store
        left_grasp_cam_inv_extrinsics[idx]  = torch.from_numpy(left_finger_inv_extrinsic_bullet.as_matrix()).float()
        right_grasp_cam_inv_extrinsics[idx] = torch.from_numpy(right_finger_inv_extrinsic_bullet.as_matrix()).float()
        left_camera_world[idx, :]  = torch.from_numpy(left_finger_inv_extrinsic_bullet.translation).float().unsqueeze(0)
        right_camera_world[idx, :] = torch.from_numpy(right_finger_inv_extrinsic_bullet.translation).float().unsqueeze(0)

        if o3d_vis is not None:
            grasps_scene = trimesh.Scene()
            from neugraspnet.src.utils import visual
            grasps_scene.add_geometry(visual.grasp2mesh(grasp), node_name=f'grasp_{idx}')
            o3d_gripper_mesh = as_mesh(grasps_scene).as_open3d
            o3d_gripper_mesh.paint_uniform_color([1.0, 0.85, 0.0]) # yellow
            o3d_vis.add_geometry(o3d_gripper_mesh, reset_bounding_box=False)
            o3d_vis.poll_events()
            # o3d_vis.update_renderer()
            grasps_viz_list.append(o3d_gripper_mesh)
    
    ## Make proposal points for rendering
    # Make pixel points
    pixel_grid = torch.meshgrid(torch.arange(width), torch.arange(height))
    pixels = torch.dstack((pixel_grid[0],pixel_grid[1])).reshape(-1, 2)
    pixels_hom = torch.hstack((pixels, torch.ones((pixels.shape[0], 2)))) # Homogenous co-ordinates
    intrinsic_hom = torch.eye(4)
    intrinsic_hom[:3,:3] = torch.tensor(intrinsic.K)
    image_plane_depth = 0.001 # 1mm
    pixels_hom[:,:3] *= image_plane_depth # Multiply by depth
    pixels_local_hom = (torch.inverse(intrinsic_hom) @ pixels_hom.T).unsqueeze(0).repeat(batch_size,1,1)

    pixels_world = torch.bmm(grasp_cam_inv_extrinsics, pixels_local_hom)[:,:3,:].transpose(1,2)
    left_pixels_world = torch.bmm(left_grasp_cam_inv_extrinsics, pixels_local_hom)[:,:3,:].transpose(1,2)
    right_pixels_world = torch.bmm(right_grasp_cam_inv_extrinsics, pixels_local_hom)[:,:3,:].transpose(1,2)
    ray_vector_world = (pixels_world - camera_world)
    ray_vector_world = ray_vector_world/ray_vector_world.norm(2,2).unsqueeze(-1)
    left_ray_vector_world = (left_pixels_world - left_camera_world)
    left_ray_vector_world = left_ray_vector_world/left_ray_vector_world.norm(2,2).unsqueeze(-1)
    right_ray_vector_world = (right_pixels_world - right_camera_world)
    right_ray_vector_world = right_ray_vector_world/right_ray_vector_world.norm(2,2).unsqueeze(-1)

    p_proposal_world, d_proposal = get_simple_proposal(camera_world,
                                                ray_vector_world,
                                                n_steps=n_steps,
                                                depth_range=[min_measured_dist, max_measured_dist_ray])
    left_p_proposal_world, left_d_proposal = get_simple_proposal(left_camera_world,
                                                left_ray_vector_world,
                                                n_steps=n_steps,
                                                depth_range=[min_measured_dist, finger_max_measured_dist])
    right_p_proposal_world, right_d_proposal = get_simple_proposal(right_camera_world,
                                                right_ray_vector_world,
                                                n_steps=n_steps,
                                                depth_range=[min_measured_dist, finger_max_measured_dist])

    p_proposal_world_combined = torch.cat((p_proposal_world, left_p_proposal_world, right_p_proposal_world), dim=1)
    # Normalize and convert to query for network
    p_scaled_proposal_query = ((p_proposal_world_combined)/size - 0.5).view(batch_size,-1, 3)
    # Query network
    torch.cuda.empty_cache()
    with torch.no_grad():
        val = net.infer_occ(p_scaled_proposal_query.clone().to(device), batched_encoded_tsdf, encoded_inputs=True)
    val = val.cpu()
    torch.cuda.empty_cache()
    val = (val - 0.5) # Center occupancies around 0
    # points_occ = p_proposal_world_combined.view(batch_size,-1, 3)[val>0]

    ## Surface rendering
    val = val.reshape(batch_size, -1, n_steps)
    mask_0_not_occupied = val[:, :, 0] < 0
    sign_matrix = torch.cat([torch.sign(val[:, :, :-1] * val[:, :, 1:]),
                                    torch.ones(batch_size, n_pts, 1)],
                                    dim=-1)
    cost_matrix = sign_matrix * torch.arange(n_steps, 0, -1).float()
    values, indices = torch.min(cost_matrix, -1)
    mask_sign_change = values < 0
    mask_neg_to_pos = val[torch.arange(batch_size).unsqueeze(-1),torch.arange(n_pts).unsqueeze(-0), indices] < 0
    mask = mask_sign_change & mask_neg_to_pos & mask_0_not_occupied

    n = batch_size * n_pts
    d_proposal = d_proposal.repeat(batch_size,int(n_pts/3),1,1)
    left_d_proposal = left_d_proposal.repeat(batch_size,int(n_pts/3),1,1)
    right_d_proposal = right_d_proposal.repeat(batch_size,int(n_pts/3),1,1)
    d_proposal = torch.cat((d_proposal, left_d_proposal, right_d_proposal), dim=1) # combine for all cameras
    d_low = d_proposal.view(n, n_steps, 1)[torch.arange(n), indices.view(n)].view(batch_size, n_pts)[mask]
    f_low = val.view(n, n_steps, 1)[torch.arange(n), indices.view(n)].view(
                batch_size, n_pts)[mask]
    indices = torch.clamp(indices + 1, max=n_steps-1)
    d_high = d_proposal.view(n, n_steps, 1)[torch.arange(n), indices.view(n)].view(batch_size, n_pts)[mask]
    f_high = val.view(n, n_steps, 1)[torch.arange(n), indices.view(n)].view(batch_size, n_pts)[mask]

    ray0_masked = torch.cat((camera_world, left_camera_world, right_camera_world), dim=1)[mask]
    ray_direction_masked = torch.cat((ray_vector_world, left_ray_vector_world, right_ray_vector_world), dim=1)[mask]

    # Apply surface depth refinement step (e.g. Secant method)
    if mask.sum() > 0:
        torch.cuda.empty_cache()
        with torch.no_grad():
            d_pred = secant(net, encoded_tsdf, # we don't care about batch size here
                f_low.to(device), f_high.to(device), d_low.to(device), d_high.to(device), n_secant_steps, ray0_masked.to(device),
                ray_direction_masked.to(device), tau=torch.tensor(0.5, device=device, dtype=torch.float), scale_size=torch.tensor(size, device=device, dtype=torch.float))
        d_pred = d_pred.cpu()
        torch.cuda.empty_cache()
    else:
        d_pred = torch.zeros(0, dtype=torch.float32)

    ## Project depth to 3d points
    # t_nan = torch.tensor(float('nan'))
    points_out = torch.zeros(batch_size, n_pts, 3, dtype=torch.float32)#*t_nan # set default (invalid) values
    points_out[mask] = ray0_masked + ray_direction_masked * d_pred.unsqueeze(1)

    # Filter points too far away from the gripper
    # transform world points to grasp frame
    points_out_hom = torch.cat((points_out, torch.ones(batch_size, n_pts, 1)), dim=-1)
    points_out_local = torch.bmm(grasp_inv_tfs, points_out_hom.transpose(1,2))
    inval_mask = points_out_local[:,0,:] > (max_measured_dist - dist_from_gripper) # too far X
    inval_mask = inval_mask | (points_out_local[:,0,:] < -dist_from_gripper)       # too close X
    inval_mask = inval_mask | (points_out_local[:,2,:] >  height_max_dist-0.005)         # too far Z
    inval_mask = inval_mask | (points_out_local[:,2,:] < -height_max_dist+0.005)         # too close Z

    surf_mask = mask & ~inval_mask
    
    # Surface Rendering & filtering complete
    if viz_rays:
        # For visualization, we need to only return the surface points and the ray points
        return points_out[mask], p_proposal_world_combined[mask]
    ## Now loop over grasp clouds, downsample and reject grasps with too few points
    grasps_pc_local = torch.zeros((batch_size,max_points,3))
    grasps_pc = grasps_pc_local.clone()
    bad_indices = []
    for ind, grasp in enumerate(grasps):        
        surface_points_out = points_out[ind, surf_mask[ind]]
        surf_pc = o3d.geometry.PointCloud()
        surf_pc.points = o3d.utility.Vector3dVector(surface_points_out)
        down_surf_pc = surf_pc.voxel_down_sample(voxel_size=voxel_downsample_size)
        
        if len(down_surf_pc.points) < render_settings['min_points']:
            # If less than min points, skip this grasp
            print("[Warning]: Points are too few! Skipping this grasp...")
            bad_indices.append(ind)
        else:
            if len(down_surf_pc.points) > max_points:
                # If more than max points, uniformly sample
                indices = np.random.choice(np.arange(len(down_surf_pc.points)), max_points, replace=False)
                down_surf_pc = down_surf_pc.select_by_index(indices)
        
            grasp_pc = torch.tensor(np.array(down_surf_pc.points), dtype=torch.float32)
            grasp_pc_local = grasp_vgn_inv_tfs[ind] @ torch.hstack((grasp_pc, torch.ones(grasp_pc.shape[0],1))).T
            grasp_pc_local = grasp_pc_local[:3,:].T
            grasp_pc_local = grasp_pc_local / size # - 0.5 DONT SUBTRACT HERE!
            grasp_pc = grasp_pc/size - 0.5

            grasps_pc_local[ind, :grasp_pc_local.shape[0],:] = grasp_pc_local
            grasps_pc[ind, :grasp_pc.shape[0], :] = grasp_pc

        if debug:
            # Viz
            down_surf_pc.colors = o3d.utility.Vector3dVector(np.tile(np.array([0.6, 0.0, 1]), (np.asarray(down_surf_pc.points).shape[0], 1)))
            # viz grasp pc, original pc and gripper
            grasps_scene = trimesh.Scene()
            from neugraspnet.src.utils import visual
            grasp_mesh_list = [visual.grasp2mesh(grasp)]
            for i, g_mesh in enumerate(grasp_mesh_list):
                grasps_scene.add_geometry(g_mesh, node_name=f'grasp_{i}')
            o3d_gripper_mesh = as_mesh(grasps_scene).as_open3d
            gripper_pc = o3d_gripper_mesh.sample_points_uniformly(number_of_points=3000)
            gripper_pc.colors = o3d.utility.Vector3dVector(np.tile(np.array([1, 1, 0]), (np.asarray(gripper_pc.points).shape[0], 1)))
            o3d.visualization.draw_geometries([down_surf_pc, pc, gripper_pc])
            # origin_pc = o3d.geometry.PointCloud()
            # origin_pc.points = o3d.utility.Vector3dVector(np.array([[0,0,0]]))
            # origin_pc.colors = o3d.utility.Vector3dVector(np.tile(np.array([1, 0, 0]), (np.asarray(origin_pc.points).shape[0], 1)))
            # down_surf_pc_local = o3d.geometry.PointCloud()
            # down_surf_pc_local.points = o3d.utility.Vector3dVector(grasp_pc_local*size)
            # o3d.visualization.draw_geometries([down_surf_pc_local, down_surf_pc, origin_pc, pc, gripper_pc])

        if o3d_vis is not None:
            down_surf_pc.colors = o3d.utility.Vector3dVector(np.tile(np.array([0.9, 0.9, 0.0]), (np.asarray(down_surf_pc.points).shape[0], 1)))
            # o3d_vis.add_geometry(down_surf_pc, reset_bounding_box=False)
            o3d_vis.poll_events()
            # o3d_vis.update_renderer()

    return bad_indices, grasps_pc_local, grasps_pc, grasps_viz_list


def generate_gt_grasp_cloud(sim, render_settings, grasp, scene_mesh=None, debug=False):
    if debug:
        # DEBUG: Viz scene point cloud and normals using ground truth meshes
        o3d_scene_mesh = scene_mesh.as_open3d
        o3d_scene_mesh.compute_vertex_normals()
        pc = o3d_scene_mesh.sample_points_uniformly(number_of_points=1000)
        pc.colors = o3d.utility.Vector3dVector(np.tile(np.array([0, 0, 0]), (np.asarray(pc.points).shape[0], 1)))
        o3d.visualization.draw_geometries([pc])

    # Create our own camera(s)
    width, height = render_settings['camera_image_res'], render_settings['camera_image_res'] # relatively low resolution (128 by default)
    width_fov  = np.deg2rad(render_settings['camera_fov']) # angular FOV (120 by default)
    height_fov = np.deg2rad(render_settings['camera_fov']) # angular FOV (120 by default)
    f_x = width  / (np.tan(width_fov / 2.0))
    f_y = height / (np.tan(height_fov / 2.0))
    intrinsic = CameraIntrinsic(width, height, f_x, f_y, width/2, height/2)
    # To capture 5cms on both sides of the gripper, using a 120 deg FOV, we need to be atleast 0.05/tan(60) = 2.8 cms away
    height_max_dist = sim.gripper.max_opening_width/2.5
    width_max_dist  = sim.gripper.max_opening_width/2.0 + 0.015 # 1.5 cm extra
    dist_from_gripper = width_max_dist/np.tan(width_fov/2.0)
    min_measured_dist = 0.001
    max_measured_dist = dist_from_gripper + sim.gripper.finger_depth + 0.005 # 0.5 cm extra
    camera = sim.world.add_camera(intrinsic, min_measured_dist, max_measured_dist+0.05) # adding 5cm extra for now but will filter it below
    if render_settings['three_cameras']:
        # Use one camera for wrist and two cameras for the fingers
        # finger_height_max_dist = sim.gripper.max_opening_width/2.5 # Not required if filtering combined cloud
        finger_width_max_dist = sim.gripper.finger_depth/2.0 + 0.005 # 0.5 cm extra
        dist_from_finger = finger_width_max_dist/np.tan(width_fov/2.0)
        finger_max_measured_dist = dist_from_finger + 0.95*sim.gripper.max_opening_width
        finger_camera  = sim.world.add_camera(intrinsic, min_measured_dist, finger_max_measured_dist+0.05) # adding 5cm extra for now but will filter it below
    
    # Load the grasp
    pos = grasp.pose.translation
    rotation = grasp.pose.rotation
    grasp = Grasp(Transform(rotation, pos), sim.gripper.max_opening_width)
    if debug:
        # DEBUG: Viz grasp
        grasps_scene = trimesh.Scene()
        from neugraspnet.src.utils import visual
        grasp_mesh_list = [visual.grasp2mesh(grasp)]
        for i, g_mesh in enumerate(grasp_mesh_list):
            grasps_scene.add_geometry(g_mesh, node_name=f'grasp_{i}')
        # grasps_scene.show()
        composed_scene = trimesh.Scene([scene_mesh, grasps_scene])
        composed_scene.show()

    ## Move camera to grasp offset frame
    grasp_center = grasp.pose.translation
    # Unfortunately VGN/GIGA grasps are not in the grasp frame we want (frame similar to PointNetGPD), so we need to transform them
    grasp_frame_rot =  grasp.pose.rotation * Rotation.from_euler('Y', np.pi/2) * Rotation.from_euler('Z', np.pi)
    grasp_tf = Transform(grasp_frame_rot, grasp_center).as_matrix()
    offset_pos =  (grasp_tf @ np.array([[-dist_from_gripper],[0],[0],[1.]]))[:3].squeeze() # Move to offset frame
    # Unfortunately the bullet renderer uses the OpenGL format so we have to use yet another extrinsic
    grasp_up_axis = grasp_tf.T[2,:3] # np.array([0.0, 0.0, 1.0]) # grasp_tf z-axis
    extrinsic_bullet = Transform.look_at(eye=offset_pos, center=grasp_center, up=grasp_up_axis)
    ## render image
    depth_img = camera.render(extrinsic_bullet)[1]
    # Optional: Add some dex noise like GIGA
    if render_settings['add_noise']:
        depth_img = apply_noise(depth_img, noise_type='mod_dex')
    if debug:
        # DEBUG: Viz
        plt.imshow(depth_img)
        plt.show()
    
    ## Do the same for the other cameras
    if render_settings['three_cameras']:
        ## Move camera to finger offset frame
        fingers_center =  (grasp_tf @ np.array([[sim.gripper.finger_depth/2.0],[0],[0],[1.]]))[:3].squeeze()
        left_finger_offset_pos  =  (grasp_tf @ np.array([[sim.gripper.finger_depth/2.0],[ (dist_from_finger + sim.gripper.max_opening_width/2.0)],[0],[1.]]))[:3].squeeze()
        right_finger_offset_pos =  (grasp_tf @ np.array([[sim.gripper.finger_depth/2.0],[-(dist_from_finger + sim.gripper.max_opening_width/2.0)],[0],[1.]]))[:3].squeeze()
        
        # Unfortunately the bullet renderer uses the OpenGL format so we have to use yet another extrinsic
        left_finger_extrinsic_bullet  = Transform.look_at(eye=left_finger_offset_pos,  center=fingers_center, up=grasp_up_axis)
        right_finger_extrinsic_bullet = Transform.look_at(eye=right_finger_offset_pos, center=fingers_center, up=grasp_up_axis)

        ## render image
        left_finger_depth_img  = finger_camera.render(left_finger_extrinsic_bullet )[1]
        right_finger_depth_img = finger_camera.render(right_finger_extrinsic_bullet)[1]
        # Optional: Add some dex noise like GIGA
        if render_settings['add_noise']:
            left_finger_depth_img = apply_noise(left_finger_depth_img, noise_type='mod_dex')
            right_finger_depth_img = apply_noise(right_finger_depth_img, noise_type='mod_dex')
    
    ## Convert to point cloud
    pixel_grid = np.meshgrid(np.arange(width), np.arange(height))
    pixels = np.dstack((pixel_grid[0],pixel_grid[1])).reshape(-1, 2)

    # depth_eps = 0.0001
    depth_array = depth_img.reshape(-1)
    relevant_mask = depth_array < (max_measured_dist) #- depth_eps) # only depth values in range
    filt_pixels = np.array(pixels[relevant_mask]) # only consider pixels with depth values in range
    filt_pixels = np.hstack((filt_pixels, np.ones((filt_pixels.shape[0], 2)))) # Homogenous co-ordinates
    # Project pixels into camera space
    filt_pixels[:,:3] *= depth_array[relevant_mask].reshape(-1, 1) # Multiply by depth
    intrinsic_hom = np.eye(4)
    intrinsic_hom[:3,:3] = intrinsic.K
    p_local = np.linalg.inv(intrinsic_hom) @ filt_pixels.T
    # Also filter out points that are more than max dist height # Not required if filtering combined cloud
    # p_local = p_local[:, p_local[1,:] <  height_max_dist]
    # p_local = p_local[:, p_local[1,:] > -height_max_dist]
    p_world = np.linalg.inv(extrinsic_bullet.as_matrix()) @ p_local
    surface_pc = o3d.geometry.PointCloud()
    surface_pc.points = o3d.utility.Vector3dVector(p_world[:3,:].T)

    if debug:
        ## DEBUG: Viz point cloud and grasp
        grasp_cam_world_depth_pc = o3d.geometry.PointCloud()
        grasp_cam_world_depth_pc.points = o3d.utility.Vector3dVector(p_world[:3,:].T)
        grasp_cam_world_depth_pc.colors = o3d.utility.Vector3dVector(np.tile(np.array([0, 0, 1]), (np.asarray(grasp_cam_world_depth_pc.points).shape[0], 1)))
        # viz original pc and gripper
        o3d_gripper_mesh = as_mesh(grasps_scene).as_open3d
        gripper_pc = o3d_gripper_mesh.sample_points_uniformly(number_of_points=3000)
        gripper_pc.colors = o3d.utility.Vector3dVector(np.tile(np.array([1, 1, 0]), (np.asarray(gripper_pc.points).shape[0], 1)))
        o3d.visualization.draw_geometries([gripper_pc, grasp_cam_world_depth_pc, pc])

    if render_settings['three_cameras']:
        left_finger_depth_array = left_finger_depth_img.reshape(-1)
        left_relevant_mask = left_finger_depth_array < (finger_max_measured_dist)# - depth_eps) # only depth values in range
        left_filt_pixels = np.array(pixels[left_relevant_mask]) # only consider pixels with depth values in range
        
        left_filt_pixels = np.hstack((left_filt_pixels, np.ones((left_filt_pixels.shape[0], 2)))) # Homogenous co-ordinates
        # Project pixels into camera space
        left_filt_pixels[:,:3] *= left_finger_depth_array[left_relevant_mask].reshape(-1, 1) # Multiply by depth
        left_p_local = np.linalg.inv(intrinsic_hom) @ left_filt_pixels.T
        # Also filter out points that are more than max dist height and width # Not required if filtering combined cloud
        # left_p_local = left_p_local[:, left_p_local[0,:] <  finger_width_max_dist]
        # left_p_local = left_p_local[:, left_p_local[0,:] > -finger_width_max_dist]
        # left_p_local = left_p_local[:, left_p_local[1,:] <  finger_height_max_dist]
        # left_p_local = left_p_local[:, left_p_local[1,:] > -finger_height_max_dist]
        left_p_world = np.linalg.inv(left_finger_extrinsic_bullet.as_matrix()) @ left_p_local

        right_finger_depth_array = right_finger_depth_img.reshape(-1)
        right_relevant_mask = right_finger_depth_array < (finger_max_measured_dist)# - depth_eps) # only depth values in range
        right_filt_pixels = np.array(pixels[right_relevant_mask]) # only consider pixels with depth values in range
        
        right_filt_pixels = np.hstack((right_filt_pixels, np.ones((right_filt_pixels.shape[0], 2)))) # Homogenous co-ordinates
        # Project pixels into camera space
        right_filt_pixels[:,:3] *= right_finger_depth_array[right_relevant_mask].reshape(-1, 1) # Multiply by depth
        right_p_local = np.linalg.inv(intrinsic_hom) @ right_filt_pixels.T
        # Also filter out points that are more than max dist height and width # Not required if filtering combined cloud
        # right_p_local = right_p_local[:, right_p_local[0,:] <  finger_width_max_dist]
        # right_p_local = right_p_local[:, right_p_local[0,:] > -finger_width_max_dist]
        # right_p_local = right_p_local[:, right_p_local[1,:] <  finger_height_max_dist]
        # right_p_local = right_p_local[:, right_p_local[1,:] > -finger_height_max_dist]
        right_p_world = np.linalg.inv(right_finger_extrinsic_bullet.as_matrix()) @ right_p_local    

        if debug:
            # Viz
            left_cam_world_depth_pc = o3d.geometry.PointCloud()
            left_cam_world_depth_pc.points = o3d.utility.Vector3dVector(left_p_world[:3,:].T)
            right_cam_world_depth_pc = o3d.geometry.PointCloud()
            right_cam_world_depth_pc.points = o3d.utility.Vector3dVector(right_p_world[:3,:].T)

            left_cam_world_depth_pc.colors = o3d.utility.Vector3dVector(np.tile(np.array([1, 0, 0]), (np.asarray(left_cam_world_depth_pc.points).shape[0], 1)))
            right_cam_world_depth_pc.colors = o3d.utility.Vector3dVector(np.tile(np.array([0, 1, 0]), (np.asarray(right_cam_world_depth_pc.points).shape[0], 1)))
            o3d.visualization.draw_geometries([left_cam_world_depth_pc, right_cam_world_depth_pc, gripper_pc, grasp_cam_world_depth_pc, pc])

        # Combine surface point cloud
        combined_world_points = np.hstack((p_world, left_p_world, right_p_world))
        # filter points that are too far away
        combined_world_points_local = np.linalg.inv(grasp_tf) @ combined_world_points
        inval_mask = combined_world_points_local[0,:] > (max_measured_dist - dist_from_gripper) # too far X
        inval_mask = inval_mask | (combined_world_points_local[0,:] < -dist_from_gripper)       # too close X
        inval_mask = inval_mask | (combined_world_points_local[2,:] >  height_max_dist)         # too far Z
        inval_mask = inval_mask | (combined_world_points_local[2,:] < -height_max_dist)         # too close Z
        combined_world_points_filt = combined_world_points[:, ~inval_mask]

        surface_pc.points = o3d.utility.Vector3dVector(combined_world_points_filt[:3,:].T)

    down_surface_pc = surface_pc.voxel_down_sample(voxel_size=render_settings['voxel_downsample_size'])
    # If more than max points, uniformly sample
    if len(down_surface_pc.points) > render_settings['max_points']:
        indices = np.random.choice(np.arange(len(down_surface_pc.points)), render_settings['max_points'], replace=False)
        down_surface_pc = down_surface_pc.select_by_index(indices)
    # If less than min points, skip this grasp
    if len(down_surface_pc.points) < render_settings['min_points']:
        # Points are too few! skip this grasp
        print("[Warning]: Points are too few! Skipping this grasp...")
        # import pdb; pdb.set_trace()
        return False, 0, 0
    if debug:
        # viz original pc and gripper
        down_surface_pc.colors = o3d.utility.Vector3dVector(np.tile(np.array([1.0, 0.45, 0.]), (np.asarray(down_surface_pc.points).shape[0], 1)))
        o3d.visualization.draw_geometries([down_surface_pc, gripper_pc, pc])
    
    grasp_pc = np.asarray(down_surface_pc.points)
    T_inv = Transform(rotation, pos).inverse()
    grasp_pc_local = T_inv.transform_point(grasp_pc)

    # pc_trimesh = trimesh.points.PointCloud(down_surface_pc.points)
    # pc_colors = np.array([[255, 255, 0] for i in down_surface_pc.points])
    # # increase sphere size of trimesh points
    # pc_trimesh.colors = np.array([255, 125, 0])
    # box = trimesh.creation.box(extents=[0.5, 0.5, 0.1])
    # box.visual.face_colors = [0.9, 0.9, 0.9, 1.0]
    # translation = [0.15, 0.15, -0.05+0.05]
    # box.apply_translation(translation)
    # trimesh.Scene([composed_scene, pc_trimesh, box]).show(line_settings= {'point_size': 20})
    # import pdb; pdb.set_trace()

    if debug:
        # viz local and global and original pc and gripper
        gripper_pc_local = T_inv.transform_point(np.asarray(gripper_pc.points))
        gripper_pc_local_o3d = o3d.geometry.PointCloud()
        gripper_pc_local_o3d.points = o3d.utility.Vector3dVector(gripper_pc_local)
        gripper_pc_local_o3d.colors = o3d.utility.Vector3dVector(np.tile(np.array([1, 1, 0.]), (np.asarray(gripper_pc_local_o3d.points).shape[0], 1)))
        grasp_pc_local_o3d = o3d.geometry.PointCloud()
        grasp_pc_local_o3d.points = o3d.utility.Vector3dVector(grasp_pc_local)
        grasp_pc_local_o3d.colors = o3d.utility.Vector3dVector(np.tile(np.array([1.0, 0.45, 0.]), (np.asarray(grasp_pc_local_o3d.points).shape[0], 1)))
        o3d.visualization.draw_geometries([grasp_pc_local_o3d, gripper_pc_local_o3d, down_surface_pc, gripper_pc, pc])

    return True, grasp_pc_local, grasp_pc