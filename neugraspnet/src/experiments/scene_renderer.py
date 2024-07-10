import numpy as np
import torch
import open3d as o3d
import matplotlib.pyplot as plt
import trimesh
from neugraspnet.src.utils.implicit import as_mesh
from neugraspnet.src.utils.perception import CameraIntrinsic
from neugraspnet.src.utils.misc import apply_noise
from neugraspnet.src.utils.perception import *
from neugraspnet.src.utils.grasp import *
from neugraspnet.src.utils.transform import Transform, Rotation


def render_n_images(sim, n=1, random=False, noise_type='', size=0.3):
    origin = Transform(Rotation.identity(), np.r_[size / 2, size / 2, 0.0])
    if random:
        # theta = np.random.uniform(0.0, 5* np.pi / 12.0) # elevation: 0 to 75 degrees
        theta = np.random.uniform(5*np.pi/12.0)
        # 75 degree reconstruction is unrealistic, try 60
        # theta = np.random.uniform(np.pi/3)
        # theta = np.random.uniform(np.pi/6, np.pi/4) # elevation: 30 to 45 degrees
        r = np.random.uniform(2, 2.4) * size
    else:
        # theta = np.pi / 4.0 # 45 degrees from top view
        # SJ EDIT! Packed scene prefer scene renders from the top!
        theta = np.random.uniform(np.pi / 9.0, np.pi / 4.5) # SJ EDIT!!! random 18 to 40 degrees from top view
        r = 2.0 * size
    
    phi_list = 2.0 * np.pi * np.arange(n) / n # circle around the scene
    extrinsics = [camera_on_sphere(origin, r, theta, phi) for phi in phi_list]
    depth_imgs = []

    for extrinsic in extrinsics:
        # Multiple views -> for getting other sides of pc
        depth_img = sim.camera.render(extrinsic)[1]
        # add noise
        depth_img = apply_noise(depth_img, noise_type)
        
        depth_imgs.append(depth_img)

    return depth_imgs, extrinsics

def get_simple_proposal(ray0, ray_direction, n_steps=256, depth_range=[0.001, 0.18]):
    
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

def render_occ(intrinsic, extrinsics, net, encoded_tsdf, min_measured_depth, max_measured_depth, size, device, n_steps=256):

    width, height = (intrinsic.K[:2, 2]*2).astype(int)
    batch_size = n_images = 1 # Use 1 for now because GPU memory len(extrinsics)
    n_pts = width*height

    ## Generate proposal points

    # Make pixel points
    pixel_grid = torch.meshgrid(torch.arange(width), torch.arange(height))
    pixels = torch.dstack((pixel_grid[0],pixel_grid[1])).reshape(-1, 2)
    pixels_hom = torch.hstack((pixels, torch.ones((pixels.shape[0], 2)))) # Homogenous co-ordinates

    cam_inv_extrinsics = torch.ones((batch_size, 4, 4), dtype=torch.float32)
    camera_world = torch.zeros((batch_size, n_pts, 3))
    # Using only first camera for now (GPU memory)
    # for idx, cam_extrinsic in enumerate(extrinsics):
    camera_world[0, :] = torch.from_numpy(extrinsics[0].inverse().translation).float()
    cam_inv_extrinsics[0] = torch.from_numpy(extrinsics[0].inverse().as_matrix()).float()

    # Convert pixel points to world co-ords:
    intrinsic_hom = torch.eye(4)
    intrinsic_hom[:3,:3] = torch.tensor(intrinsic.K)
    image_plane_depth = 0.01 # 10mm
    pixels_hom[:,:3] *= image_plane_depth # Multiply by depth
    pixels_local_hom = (torch.inverse(intrinsic_hom) @ pixels_hom.T).unsqueeze(0).repeat(batch_size,1,1)
    pixels_world = torch.bmm(cam_inv_extrinsics, pixels_local_hom)[:,:3,:].transpose(1,2)

    ray_vector_world = (pixels_world - camera_world)
    ray_vector_world = ray_vector_world/ray_vector_world.norm(2,2).unsqueeze(-1)

    p_proposal_world, d_proposal = get_simple_proposal(camera_world,
                                                    ray_vector_world,
                                                    n_steps=n_steps,
                                                    depth_range=[min_measured_depth, max_measured_depth])
    # Debug: Viz proposal points
    # prop_o3d = o3d.geometry.PointCloud()
    # max_points = 2000
    # points = p_proposal_world[0].view(-1,3)
    # indices = np.random.randint(points.shape[0], size=max_points)
    # prop_o3d.points = o3d.utility.Vector3dVector(points[indices].numpy())
    # prop_o3d.colors = o3d.utility.Vector3dVector(np.random.uniform(0,1,size=(max_points,3)))
    # visualizer3 = JVisualizer()
    # visualizer3.add_geometry(prop_o3d)
    # visualizer3.add_geometry(pc_full)
    # visualizer3.show()

    # Normalize and convert to query for network
    p_scaled_proposal_query = ((p_proposal_world)/size - 0.5).view(batch_size,-1, 3)
    # Query network
    torch.cuda.empty_cache()
    with torch.no_grad():
        val = net.infer_occ(p_scaled_proposal_query.clone().to(device), encoded_tsdf, encoded_inputs=True)
    val = val.cpu()
    torch.cuda.empty_cache()
    val = (val - 0.5) # Center occupancies around 0

    # Debug: Viz occupancied points
    # points_occ = p_proposal_world.view(-1, 3)[val.squeeze()>0]
    # occ_pc = o3d.geometry.PointCloud()
    # occ_pc.points = o3d.utility.Vector3dVector(points_occ)
    # occ_pc.colors = o3d.utility.Vector3dVector(np.tile(np.array([0.6, 0.0, 1]), (np.asarray(occ_pc.points).shape[0], 1)))
    # visualizer4 = JVisualizer()
    # visualizer4.add_geometry(occ_pc)
    # visualizer4.add_geometry(pc_full)
    # visualizer4.show()

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
    d_proposal = d_proposal.repeat(1,n_pts,1,1)
    d_low = d_proposal.view(n, n_steps, 1)[torch.arange(n), indices.view(n)].view(batch_size, n_pts)[mask]
    f_low = val.view(n, n_steps, 1)[torch.arange(n), indices.view(n)].view(
                batch_size, n_pts)[mask]
    indices = torch.clamp(indices + 1, max=n_steps-1)
    d_high = d_proposal.view(n, n_steps, 1)[torch.arange(n), indices.view(n)].view(batch_size, n_pts)[mask]
    f_high = val.view(n, n_steps, 1)[torch.arange(n), indices.view(n)].view(batch_size, n_pts)[mask]

    ray0_masked = camera_world[mask]
    ray_direction_masked = ray_vector_world[mask]

    # Apply surface depth refinement step (e.g. Secant method)
    if mask.sum() == 0:
        # No surface points found
        return torch.zeros(batch_size, n_pts, 3, dtype=torch.float), mask
    else:
        n_secant_steps = 8
        torch.cuda.empty_cache()
        with torch.no_grad():
            d_pred = secant(net, encoded_tsdf, # we don't care about batch size here
                f_low.to(device), f_high.to(device), d_low.to(device), d_high.to(device), n_secant_steps, ray0_masked.to(device),
                ray_direction_masked.to(device), tau=torch.tensor(0.5, device=device, dtype=torch.float), scale_size=torch.tensor(size, device=device, dtype=torch.float))
        d_pred = d_pred.cpu()
        torch.cuda.empty_cache()
        points_out = torch.zeros(batch_size, n_pts, 3, dtype=torch.float)#*t_nan # set default (invalid) values
        points_out[mask] = ray0_masked + ray_direction_masked * d_pred.unsqueeze(1)

    return points_out, mask


def get_scene_surf_render(sim, size, resolution, net, encoded_tsdf, device, args=None):

    # Get views from a circular path around the scene
    depth_imgs_full, extrinsics_full = render_n_images(sim, n=6, random=False, noise_type='', size=size)

    # For Debug: Make tsdf and pc from the images
    # tsdf_full = TSDFVolume(size, resolution)
    # for depth_img, extrinsic in zip(depth_imgs_full, extrinsics_full):
    #     tsdf_full.integrate(depth_img, sim.camera.intrinsic, extrinsic)
    # pc_full = tsdf_full.get_cloud()
    # pc_full.colors = o3d.utility.Vector3dVector(np.tile(np.array([0, 0.0, 0.0]), (np.asarray(pc_full.points).shape[0], 1)))

    ## Render the scene using the occupancy network with the same extrinsics

    # Neural render camera settings
    width = 32
    height = 32
    width_fov  = np.deg2rad(60) # angular FOV (120 by default)
    height_fov = np.deg2rad(60) # angular FOV (120 by default)
    f_x = width  / (np.tan(width_fov / 2.0))
    f_y = height / (np.tan(height_fov / 2.0))
    intrinsic = CameraIntrinsic(width, height, f_x, f_y, width/2, height/2)

    min_measured_depth = size
    max_measured_depth = 2.4*size + size/2 # max distance from the origin
    n_steps = 64

    # tsdf_t = torch.tensor(tsdf, device=device, dtype=torch.float32)

    surface_points_combined = None
    for cam_extrinsic in extrinsics_full:
        surf_points_world, surf_mask = render_occ(intrinsic, [cam_extrinsic], net, encoded_tsdf, min_measured_depth, max_measured_depth, size=size, device=device, n_steps=n_steps)
        if surface_points_combined is None:
            surface_points_combined = surf_points_world[0, surf_mask[0]]
        else:
            surface_points_combined = torch.cat([surface_points_combined, surf_points_world[0, surf_mask[0]]], dim=0)
        # break
    
    surf_pc = o3d.geometry.PointCloud()
    surf_pc.points = o3d.utility.Vector3dVector(surface_points_combined)
    down_surf_pc = surf_pc
    down_surf_pc = surf_pc.voxel_down_sample(voxel_size=0.005) # 5mm
    # np.array([194, 30, 86]) # Rose Red
    down_surf_pc.colors = o3d.utility.Vector3dVector(np.tile(np.array([194/255, 30/255, 86/255]), (np.asarray(down_surf_pc.points).shape[0], 1)))
    # down_surf_pc.colors = o3d.utility.Vector3dVector(np.tile(np.array([0.0, 0.2, 1]), (np.asarray(down_surf_pc.points).shape[0], 1)))
    # o3d.visualization.draw_geometries([down_surf_pc, pc_full])

    # Crop to within scene bounds
    down_surf_pc_cropped = down_surf_pc.crop(o3d.geometry.AxisAlignedBoundingBox(np.array([0.0, 0.0, 0.0]), np.array([size, size, size])))

    return down_surf_pc_cropped
