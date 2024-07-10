import os
import glob
from pathlib import Path
import numpy as np
import torch
# import cv2
import open3d as o3d
# from open3d import JVisualizer
import trimesh
import matplotlib.pyplot as plt
# import scipy.signal as signal
# from tqdm import tqdm
# import multiprocessing as mp

# from neugraspnet.grasp import Grasp, Label
from neugraspnet.io import *
from neugraspnet.perception import *
from neugraspnet.simulation import ClutterRemovalSim
from neugraspnet.utils.transform import Rotation, Transform
from neugraspnet.utils.implicit import get_scene_from_mesh_pose_list, as_mesh, get_occ_specific_points
from neugraspnet.grasp_renderer import generate_gt_grasp_cloud, generate_neur_grasp_clouds
from neugraspnet.utils.misc import apply_noise
from neugraspnet.grasp_sampler import GpgGraspSamplerPcl
from neugraspnet.networks import get_network, load_network

# seed = np.random.randint(2**32 - 1)
nice_seeds = [2744579596, 3736952697, 1570237463, 3573398670, 3090687052]
great_scene_forscene_recon = [2837027833, 1075486976]
seed = nice_seeds[4]
np.random.seed(seed)

constructed_root = Path("/home/sjauhri/IAS_WS/potato-net/GIGA-TSDF/GIGA-6DoF/data/pile/data_pile_train_constructed_4M_HighRes_radomized_views_GPG_only")
model_type = "neu_grasp_pn_deeper"
# model_path = "/home/sjauhri/IAS_WS/potato-net/GIGA-TSDF/GIGA-6DoF/data/runs_relevant/23-05-07-10-46-43_dataset=data_packed_train_constructed_4M_GPG_60_randomized_view_no_tab_packked,augment=False,net=6d_neu_grasp_pn_deeper,batch_size=32,lr=5e-05,PN_deeper_no_tab_WITH_occ_PACKED/best_neural_grasp_neu_grasp_pn_deeper_val_acc=0.9317.pt"
# model_path="/home/sjauhri/IAS_WS/potato-net/GIGA-TSDF/GIGA-6DoF/data/runs_relevant/23-05-03-01-42-37_dataset=data_pile_train_constructed_4M_HighRes_radomized_views_no_table,augment=False,net=6d_neu_grasp_pn_deeper,batch_size=32,lr=5e-05,PN_no_tab_deeper_DIMS_WITH_occ/best_neural_grasp_neu_grasp_pn_deeper_val_acc=0.9120.pt"
model_path = "/home/sjauhri/IAS_WS/potato-net/GIGA-TSDF/GIGA-6DoF/data/runs_relevant/23-05-01-08-11-39_dataset=data_pile_train_constructed_4M_HighRes_radomized_views,augment=False,net=6d_neu_grasp_pn_deeper,batch_size=32,lr=5e-05,PN_deeper_DIMS_CONT/best_neural_grasp_neu_grasp_pn_deeper_val_acc=0.9097.pt"
see_table = True
device = "cuda"
net = load_network(model_path, device, model_type)

# previous_root = "/home/sjauhri/IAS_WS/potato-net/GIGA-TSDF/GIGA-6DoF/data/packed/packed_data_for_pngpd"
previous_root="/home/sjauhri/IAS_WS/potato-net/GIGA-TSDF/GIGA-6DoF/data/pile/data_pile_train_random_raw_4M_radomized_views/"
data_root = "/home/sjauhri/IAS_WS/potato-net/GIGA-TSDF/GIGA-6DoF/"

sim_gui = False
three_cameras = True # Use one camera for wrist and two cameras for the fingers
add_noise = False # Add dex noise to the rendered images like GIGA
noise_type = 'mod_dex'
gp_rate = 0.5 # Rate of applying Gaussian process noise
voxel_downsample_size = 0.002 # 2mm
scene_voxel_downsample_size = 0.005 # 5mm
max_points = 1023
resolution = 64
scene='pile'
object_set='pile/train'
size=0.3

## Re-create the saved simulation
# Get random scene
index = np.random.randint(32000) # index 20 is a good example.
mesh_list_files = glob.glob(os.path.join(previous_root, 'mesh_pose_list', '*.npz'))
mesh_pose_list = np.load(mesh_list_files[index], allow_pickle=True)['pc']
scene_id = os.path.basename(mesh_list_files[index])[:-4] # scene id without .npz extension
# "d7e7d6e296ec4abfaad79acd252ac9b3"

## Get specific scene
# scene_id = 'f614e39ed9df4e1094d569cddc20979b'
# mesh_list_file = os.path.join(previous_root, 'mesh_pose_list', scene_id + '.npz')
# mesh_pose_list = np.load(mesh_list_file, allow_pickle=True)['pc']

sim = ClutterRemovalSim(scene, object_set, gui=sim_gui, data_root=data_root) # parameters scene and object_set are not used
sim.setup_sim_scene_from_mesh_pose_list(mesh_pose_list, table=see_table, data_root=data_root) # Setting table to False because we don't want to render it
# sim.save_state()

# Get scene point cloud and normals using ground truth meshes
scene_mesh = get_scene_from_mesh_pose_list(mesh_pose_list, data_root=data_root)
o3d_scene_mesh = scene_mesh.as_open3d
o3d_scene_mesh.compute_vertex_normals()
pc = o3d_scene_mesh.sample_points_uniformly(number_of_points=1000)
points = np.asarray(pc.points)
# pc_trimesh = trimesh.points.PointCloud(points)
# pc_colors = np.array([trimesh.visual.random_color() for i in points])
# pc_trimesh.vertices_color = pc_colors
# trimesh.Scene([scene_mesh, pc_trimesh]).show()
# o3d.visualization.draw_geometries([pc])
# visualizer = JVisualizer()
# pc.colors = o3d.utility.Vector3dVector(np.tile(np.array([0, 0, 0]), (np.asarray(pc.points).shape[0], 1)))
# visualizer.add_geometry(pc)
# visualizer.show()

# sample grasps with GPG:
seed = 216866#np.random.randint(100000)
np.random.seed(seed)

sampler = GpgGraspSamplerPcl(sim.gripper.finger_depth-0.0075) # Franka finger depth is actually a little less than 0.05
safety_dist_above_table = sim.gripper.finger_depth # table is spawned at finger_depth
grasps, _, _ = sampler.sample_grasps(pc, num_grasps=1, max_num_samples=180,
                                    safety_dis_above_table=safety_dist_above_table, show_final_grasps=False)
grasps[0].pose.translation += np.array([0, 0, 0.01]) # Move grasp up by 1cm
# Viz grasps
grasps_scene = trimesh.Scene()
from neugraspnet.utils import visual
grasp_mesh_list = [visual.grasp2mesh(grasps[0],color='yellow')]# for g in grasps]
for i, g_mesh in enumerate(grasp_mesh_list):
    grasps_scene.add_geometry(g_mesh, node_name=f'grasp_{i}')
    break
# grasps_scene.show()
composed_scene = trimesh.Scene([scene_mesh, grasps_scene])
camera = composed_scene.camera
cam_resolution = [1920, 1080]
rot_by_x_degrees = 55
# TODO: euler to rotation matrix
pitch_rot = np.array([   [  1.0000000,  0.0000000,  0.0000000, 0.0],
                            [  0.0000000,  0.5735765, -0.8191521, 0.0],
                            [  0.0000000,  0.8191521,  0.5735765, 0.0],
                            [  0.0000000,  0.0000000,  0.0000000, 1.0]   ])
distance = 0.55
camera_tf = camera.look_at(points=[[size/2,size/2,0.06]], rotation=pitch_rot, distance=distance)
box = trimesh.creation.box(extents=[0.5, 0.5, 0.1])
box.visual.face_colors = [0.9, 0.9, 0.9, 1.0]
translation = [0.15, 0.15, 0.05-0.05]
box.apply_translation(translation)
composed_scene.add_geometry(box)
composed_scene.camera.resolution = cam_resolution
composed_scene.camera_transform = camera_tf
composed_scene.show()


# Render depth image from random viewpoint or hard viewpoint
def render_n_images(sim, n=1, random=False, tight=False, noise_type=''):
    origin = Transform(Rotation.identity(), np.r_[size / 2, size / 2, 0.0])
    if random:
        if tight:
            theta = np.random.uniform(5.5*np.pi/12.0)
        theta = np.random.uniform(np.pi / 12.0, 5* np.pi / 12.0) # elevation: 15 to 75 degrees

        # theta = np.random.uniform(5*np.pi/12.0)
        # 75 degree reconstruction is unrealistic, try 60
        # theta = np.random.uniform(np.pi/3)
        # theta = np.random.uniform(np.pi/6, np.pi/4) # elevation: 30 to 45 degrees
        r = np.random.uniform(2.0) * size
    else:
        # theta = np.random.uniform(np.pi / 18.0, np.pi / 4.5) # SJ EDIT!!! random 10 to 40 degrees from top view
        theta = np.pi / 4.0 # 45 degrees from top view
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

# Get random scene image:
depth_imgs, extrinsics = render_n_images(sim, n=1, random=True, tight=False, noise_type='')
plt.imshow(depth_imgs[0]) # Normally too far away to see anything

# Make tsdf and pc from the image
tsdf = TSDFVolume(size, resolution)
for depth_img, extrinsic in zip(depth_imgs, extrinsics):
    tsdf.integrate(depth_img, sim.camera.intrinsic, extrinsic)
# seen_pc = tsdf.get_cloud()
# # Optional: Crop out table
# lower = np.array([0.0 , 0.0 , 0.055])
# upper = np.array([size, size, size])
# bounding_box = o3d.geometry.AxisAlignedBoundingBox(lower, upper)
# seen_pc = seen_pc.crop(bounding_box)
# # Optional: Downsample
# seen_pc = seen_pc.voxel_down_sample(scene_voxel_downsample_size)

# Viz seen point cloud and camera position
# seen_pc.colors = o3d.utility.Vector3dVector(np.tile(np.array([0, 0.64, 0.93]), (np.asarray(seen_pc.points).shape[0], 1)))
# cam_pos_pc = o3d.geometry.PointCloud()
# cam_pos_pc.points = o3d.utility.Vector3dVector(np.array([extrinsics[0].inverse().translation]))
# cam_pos_pc.colors = o3d.utility.Vector3dVector(np.tile(np.array([0, 0.64, 0.93]), (np.asarray(cam_pos_pc.points).shape[0], 1)))
# visualizer.add_geometry(seen_pc)
# visualizer.add_geometry(cam_pos_pc)
# visualizer.show()

# Create our own camera
width, height = 64, 64 # relatively low resolution
width_fov = np.deg2rad(120.0) # angular FOV
height_fov = np.deg2rad(120.0) # angular FOV
f_x = width / (np.tan(width_fov / 2.0))
f_y = height / (np.tan(height_fov / 2.0))
intrinsic = CameraIntrinsic(width, height, f_x, f_y, width/2, height/2)

# To capture 5cms on both sides of the gripper, using a 120 deg FOV, we need to be atleast 0.05/tan(60) = 2.8 cms away
height_max_dist = sim.gripper.max_opening_width/2.5
width_max_dist  = sim.gripper.max_opening_width/2.0 + 0.015 # 1.5 cm extra
dist_from_gripper = width_max_dist/np.tan(width_fov/2.0) 
min_measured_dist = 0.001
max_measured_dist = dist_from_gripper + sim.gripper.finger_depth + 0.005 # 0.5 cm extra
camera = sim.world.add_camera(intrinsic, min_measured_dist, max_measured_dist+0.05) # adding 5cm extra for now but will filter it below
# depth_imgs_side, extrinsics_side = render_side_images(sim, 1, camera=camera)
# plt.imshow(depth_imgs_side[0]) # Normally too far away to see anything

if three_cameras:
    # Use one camera for wrist and two cameras for the fingers
    # finger_height_max_dist = sim.gripper.max_opening_width/2.5 # Not required if filtering combined cloud
    finger_width_max_dist = sim.gripper.finger_depth/2.0 + 0.005 # 0.5 cm extra
    dist_from_finger = finger_width_max_dist/np.tan(width_fov/2.0)
    finger_max_measured_dist = dist_from_finger + 0.95*sim.gripper.max_opening_width
    finger_camera  = sim.world.add_camera(intrinsic, min_measured_dist, finger_max_measured_dist+0.05) # adding 5cm extra for now but will filter it below
## Move camera to grasp offset frame
grasp_center = grasps[0].pose.translation
# Unfortunately VGN/GIGA grasps are not in the grasp frame we want (frame similar to PointNetGPD), so we need to transform them
grasp_frame_rot =  grasps[0].pose.rotation * Rotation.from_euler('Y', np.pi/2) * Rotation.from_euler('Z', np.pi)
grasp_tf = Transform(grasp_frame_rot, grasp_center).as_matrix()
offset_pos =  (grasp_tf @ np.array([[-dist_from_gripper],[0],[0],[1.]]))[:3].squeeze() # Move to offset frame
# (Debug) viz the grasp center and offset_pos:
# visualizer.add_geometry(o3d.geometry.PointCloud(o3d.utility.Vector3dVector(grasp_center.reshape(1,3))))
# visualizer.add_geometry(o3d.geometry.PointCloud(o3d.utility.Vector3dVector(offset_pos.reshape(1,3))))
# visualizer.show()

# Unfortunately the bullet renderer uses the OpenGL format so we have to use yet another extrinsic
grasp_up_axis = grasp_tf.T[2,:3] # np.array([0.0, 0.0, 1.0]) # grasp_tf z-axis
extrinsic_bullet = Transform.look_at(eye=offset_pos, center=grasp_center, up=grasp_up_axis)
## Do the same for the other cameras
if three_cameras:
    ## Move camera to finger offset frame
    fingers_center =  (grasp_tf @ np.array([[sim.gripper.finger_depth/2.0],[0],[0],[1.]]))[:3].squeeze()
    left_finger_offset_pos  =  (grasp_tf @ np.array([[sim.gripper.finger_depth/2.0],[ (dist_from_finger + sim.gripper.max_opening_width/2.0)],[0],[1.]]))[:3].squeeze()
    right_finger_offset_pos =  (grasp_tf @ np.array([[sim.gripper.finger_depth/2.0],[-(dist_from_finger + sim.gripper.max_opening_width/2.0)],[0],[1.]]))[:3].squeeze()
    
    # Unfortunately the bullet renderer uses the OpenGL format so we have to use yet another extrinsic
    left_finger_extrinsic_bullet  = Transform.look_at(eye=left_finger_offset_pos,  center=fingers_center, up=grasp_up_axis)
    right_finger_extrinsic_bullet = Transform.look_at(eye=right_finger_offset_pos, center=fingers_center, up=grasp_up_axis)


# get renders
with torch.no_grad():
    encoded_tsdf = net.encode_inputs(torch.from_numpy(tsdf.get_grid()).to(device))
render_settings = read_json(Path("/home/sjauhri/IAS_WS/potato-net/GIGA-TSDF/GIGA-6DoF/data/pile/grasp_cloud_setup.json"))
render_settings["add_noise"] = True
sim.world.remove_body(sim.world.bodies[0]) # remove table from sim fir gt render

result, _, grasp_pc = generate_gt_grasp_cloud(sim, render_settings, grasps[0], scene_mesh, debug=False)

sim.place_table(height=sim.gripper.finger_depth) # Add table back

render_settings["add_noise"] = False
_, _, neur_grasps_pc, _ = generate_neur_grasp_clouds(sim, render_settings, [grasps[0]], size, encoded_tsdf, 
                                                                            net, device, scene_mesh, debug=False)

grasp_pc_trimesh = trimesh.points.PointCloud(grasp_pc)
grasp_pc_trimesh.colors = np.array([0, 0, 0]) # Black
# grasp_pc_trimesh.colors = np.array([194, 30, 86]) # Rose Red
gt_render_scene = trimesh.Scene([grasps_scene, grasp_pc_trimesh])
# gt_render_scene.show()

neur_grasps_pc = neur_grasps_pc[0]
# un-normalize
neur_grasps_pc = (neur_grasps_pc + 0.5)*size
# exclude points with only zeros in every row
neur_grasps_pc = neur_grasps_pc[~torch.all(neur_grasps_pc == 0, axis=1)]
neur_grasps_pc_trimesh = trimesh.points.PointCloud(neur_grasps_pc)
neur_grasps_pc_trimesh.colors = np.array([0, 0, 0]) # Black
neur_grasps_pc_trimesh.colors = np.array([194, 30, 86]) # Rose Red
neur_render_scene = trimesh.Scene([grasps_scene, neur_grasps_pc_trimesh])
neur_render_scene.camera_transform = np.array([[ 0.92534726,  0.37798069,  0.02937752,  0.20271586],
       [-0.25004428,  0.66671217, -0.70212018,  0.08510838],
       [-0.28497422,  0.6423593 ,  0.71145219,  0.27398286],
       [ 0.        ,  0.        ,  0.        ,  1.        ]])
# np.array([ [ 0.99804824, -0.05855287,  0.02170875,  0.19948973],
#         [ 0.06178892,  0.87557126, -0.47912117,  0.13691797],
#         [ 0.00904636,  0.4795274 ,  0.87748028,  0.3114854 ],
#         [ 0.        ,  0.        ,  0.        ,  1.        ]])
neur_render_scene.show()


# Also get occupancies of the points and save these occupancy values
scene, mesh_list = get_scene_from_mesh_pose_list(mesh_pose_list, return_list=True, data_root=data_root)
points, occ = get_occ_specific_points(mesh_list, grasp_pc)

grasp_pc_local_occ = trimesh.points.PointCloud(grasp_pc[occ])
grasp_pc_local_occ.colors = np.array([119, 7, 55]) # Mulberry
grasp_pc_local_occ.colors = np.array([74, 4, 4]) # Oxblood
grasp_pc_local_not_occ = trimesh.points.PointCloud(grasp_pc[~occ])
grasp_pc_local_not_occ.colors = np.array([255, 192, 203]) # Pink
local_occ_scene = trimesh.Scene([grasps_scene, grasp_pc_local_occ, grasp_pc_local_not_occ])
local_occ_scene.camera.resolution = cam_resolution
local_occ_scene.camera_transform = neur_render_scene.camera_transform#camera_tf
local_occ_scene.show()



# Get rays and use cameras to make local rendering viz
seed = 309068705
np.random.seed(seed)

viz_camera_mesh = trimesh.load('./SLR_CameraModel/10124_SLR_Camera_SG_V1_Iteration2.obj',process=False)
# Rotate about x axis
rot1 = trimesh.transformations.rotation_matrix(np.pi, [0, 1, 0])
viz_camera_mesh.apply_transform(rot1)
rot2 = trimesh.transformations.rotation_matrix(-np.pi/2, [1, 0, 0])
viz_camera_mesh.apply_transform(rot2)
# Scale
# translate up by 0.0215
viz_camera_mesh.apply_translation([0, 0.05, 0.0])

cam_size = 0.013
viz_camera_mesh.apply_scale(cam_size / viz_camera_mesh.extents)
# texture doesnt work right now, neither does color...........
# texture_img = Image.open("./SLR_CameraModel/10124_SLR_Camera_SG_V1_Iteration2.jpg")
# material = texture.SimpleMaterial(image=texture_img)
#tex = trimesh.visual.TextureVisuals(image=texture_img)
# uv = viz_camera_mesh.visual.uv
# color_visuals = trimesh.visual.TextureVisuals(uv=uv, image=texture_img, material=material)
#viz_camera_mesh.visual.texture = tex
# cam_color = np.array([255, 0, 0, 255]).astype(np.uint8)
# cam_colors = np.repeat(cam_color[np.newaxis, :], len(viz_camera_mesh.faces), axis=0)
# viz_camera_mesh.visual.face_colors = cam_colors

# viz_camera_mesh.show()


# _, extrinsics_cams = render_n_images_close(sim, n=6, random=False, noise_type='')
# Viz grasps
grasps_scene = trimesh.Scene()
from neugraspnet.utils import visual
grasp_mesh_list = [visual.grasp2mesh(grasps[0],color='yellow_light')]# for g in grasps]
for i, g_mesh in enumerate(grasp_mesh_list):
    grasps_scene.add_geometry(g_mesh, node_name=f'grasp_{i}')
    break
# grasps_scene.show()
viz_camera_scene = trimesh.Scene()
viz_camera_scene.add_geometry(grasps_scene)
grasp_cam_extrinsics = [extrinsic_bullet, left_finger_extrinsic_bullet, right_finger_extrinsic_bullet]
for i, extrinsic in enumerate(grasp_cam_extrinsics):
    viz_camera_mesh_copy = viz_camera_mesh.copy()
    viz_camera_mesh_copy.apply_transform(extrinsic.inverse().as_matrix())
    viz_camera_scene.add_geometry(viz_camera_mesh_copy, node_name=f'camera_{i}')

viz_camera_scene.camera_transform = neur_render_scene.camera_transform#camera_tf
viz_camera_scene.camera.resolution = cam_resolution
# viz_camera_scene.show(line_settings= {'point_size': 10})


# Make rays emanating from cameras: Maybe two/three each?
surface_points_combined = None
p_proposal_world_combined = None
viz_rays = True
# min_measured_depth = 0.0
# max_measured_depth = 0.5
n_proposal_steps = 10
render_settings['n_proposal_steps'] = n_proposal_steps
surface_points_combined, p_proposal_world_combined = generate_neur_grasp_clouds(sim, render_settings, [grasps[0]], size, encoded_tsdf, 
                                                    net, device, scene_mesh, debug=False, viz_rays=viz_rays)

# Choose a subset of the rays to show
max_rays = 20
ray_indices = np.random.randint(p_proposal_world_combined.shape[0], size=max_rays)
# points = p_proposal_world[0].view(-1,3)
# indices = np.random.randint(points.shape[0], size=max_points)
# prop_o3d.points = o3d.utility.Vector3dVector(points[indices].numpy())
# prop_o3d.colors = o3d.utility.Vector3dVector(np.random.uniform(0,1,size=(max_points,3)))
ray_points = p_proposal_world_combined[ray_indices,:,:].view(-1,3)
# filter ray points below 0.075 m
ray_points = ray_points[ray_points[:,2] > 0.075]
ray_trim = trimesh.points.PointCloud(ray_points.numpy())
ray_trim.colors = np.array([0, 0, 0]) # Black
ray_trim.colors = np.array([47, 122, 229])  # Bleu De France
ray_trim.colors = np.array([173, 216, 230])  # light blue
surf_ray_points = surface_points_combined[ray_indices,:].view(-1,3)
surf_ray_trim = trimesh.points.PointCloud(surf_ray_points.numpy())
surf_ray_trim.colors = np.array([47, 122, 229])  # Bleu De France
surf_ray_trim.colors = np.array([221, 5, 37]) # Cadmium Red
surf_ray_trim.colors = np.array([194, 30, 86]) # Rose Red
viz_camera_scene.add_geometry(ray_trim)
viz_camera_scene.add_geometry(surf_ray_trim)
viz_camera_scene.show(line_settings= {'point_size': 10})

import pdb; pdb.set_trace()