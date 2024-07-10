import os
import numpy as np
from neugraspnet.networks import load_network
import torch
import plotly.graph_objects as go
from neugraspnet.dataset_voxel_grasp_pc import DatasetVoxelGraspPCOcc
from neugraspnet.dataset_voxel import DatasetVoxelOccFile
from pathlib import Path
import random
import argparse
from neugraspnet.src.network.utils import libmcubes as mcubes
# import mcubes
from neugraspnet.utils.misc import set_random_seed

from voxel_graph import VoxelData#, render_n_images
from neugraspnet.simulation import ClutterRemovalSim
from neugraspnet.perception import *
from neugraspnet.utils.misc import apply_noise
from neugraspnet.utils.transform import Rotation, Transform

RESOLUTION = 64
size=0.3
unsq = lambda x: torch.as_tensor(x).unsqueeze(0).float()

def load_data(data):
    # pc, y, grasp_query, occ_points, occ =  data[index]
    pc, (label, width), (pos, rotations, grasps_pc_local, grasps_pc), pos_occ, occ_value = data
    pc, label, width, pos, rotations, grasps_pc_local, grasps_pc, pos_occ, occ_value = unsq(pc), unsq(label), unsq(width), unsq(pos), unsq(rotations), unsq(grasps_pc_local), unsq(grasps_pc), unsq(pos_occ), unsq(occ_value)
    return pc, (label, width, occ_value), (pos, rotations, grasps_pc_local, grasps_pc), pos_occ

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
        r = np.random.uniform(2, 2.4) * size
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

def make_tsdf(sim, device):
    depth_imgs, extrinsics = render_n_images(sim, n=1, random=True, noise_type='')

    # Make tsdf and pc from the image
    tsdf = TSDFVolume(sim.size, resolution=RESOLUTION)
    for depth_img, extrinsic in zip(depth_imgs, extrinsics):
        tsdf.integrate(depth_img, sim.camera.intrinsic, extrinsic)

    seen_pc = tsdf.get_cloud()
    # Optional: Crop out table
    lower = np.array([0.0 , 0.0 , 0.055])
    upper = np.array([sim.size, sim.size, sim.size])
    bounding_box = o3d.geometry.AxisAlignedBoundingBox(lower, upper)
    seen_pc = seen_pc.crop(bounding_box)
    # convert to torch tensor
    tsdf = torch.tensor(tsdf.get_grid(), device=device, dtype=torch.float32)
    return tsdf

def make_occ_grid(resolution):
    x, y, z = torch.meshgrid(torch.linspace(start=-0.5, end=0.5 - 1.0 / resolution, steps= resolution), torch.linspace(start=-0.5, end=0.5 - 1.0 / resolution, steps=resolution), torch.linspace(start=-0.5, end=0.5 - 1.0 / resolution, steps=resolution))
    # 1, self.resolution, self.resolution, self.resolution, 3
    pos = torch.stack((x, y, z), dim=-1).float().unsqueeze(0)
    pos = pos.view(1, resolution*resolution*resolution, 3)
    return pos

def create_plot(geometry, camera_eye):
    fig = go.Figure(geometry)
    centre_table = np.array([0.15, 0.15, 0.06]) * 0.9
    camera = dict(
        center=dict(x=centre_table[0], y=centre_table[1], z=centre_table[2]),
        # eye=dict(x=camera_eye[0], y=camera_eye[1], z=camera_eye[2]),
    )
    # camera = dict(eye=dict(x=camera_eye[0], y=camera_eye[1], z=camera_eye[2]))
    fig.update_layout(scene_camera=camera)
    fig.update_xaxes(tickmode='linear', visible=False)
    fig.update_yaxes(tickmode='linear', visible=False)
    fig.update_layout(autosize=False,
                      width  = 1600,
                      height = 1600)
    return fig

def make_save_fig(geometry, camera_eye, file_name, type="mesh"):
    if type=="voxel":
        Voxels = VoxelData(geometry.numpy().squeeze())
        voxels = go.Mesh3d(
                x=Voxels.vertices[0],
                y=Voxels.vertices[1],
                z=Voxels.vertices[2],
                i=Voxels.triangles[0],
                j=Voxels.triangles[1],
                k=Voxels.triangles[2],
                color='cornflowerblue',
                opacity=1)
        create_plot(voxels, camera_eye).write_image(file = file_name)

    else:
        vertices, triangles = geometry
        x, y, z = vertices.T
        i, j, k = triangles.T
        mesh = go.Mesh3d(x=x, y=y, z=z,
                         i=i, j=j, k=k,
                         color='rgb(194, 30, 86)')
        create_plot(mesh, camera_eye).write_image(file = file_name)


def main(args):

    device = "cpu" if not args.cuda else "cuda"
    save_path = args.save_path / f"{args.model_path.split('/')[1]}" # TODO: Path by net type
    # print(save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    if not args.random:
        camera_eye = np.load(args.cam_eye_path)
        camera_eye[2, 3] += 0.015
        cam_eye = np.linalg.inv(camera_eye.squeeze())[:3, 3]
        

    net = load_network(args.model_path, device=device, model_type=args.net)
    net = net.eval()

    # NOTE: Renamed "grasps_with_clouds_gt".csv to "grasps_with_clouds.csv"
    # data = DatasetVoxelGraspPCOcc(args.root, args.raw_root, use_grasp_occ=False, num_point_occ=8000)
    data = DatasetVoxelOccFile(args.root, args.raw_root, num_point_occ=8000)
    # Packed scene_id's
    scenes = ["3e096fb3f3ad407bb655a87a16c06efc"] if not args.random \
            else data.df.sample(args.num_scenes, random_state=args.seed)["scene_id"]
    
    for scene in scenes:
        print("Processing scene: ", scene)
        mesh_list_file = os.path.join(args.raw_root, 'mesh_pose_list', scene + '.npz')
        mesh_pose_list = np.load(mesh_list_file, allow_pickle=True)['pc']
        sim = ClutterRemovalSim('pile', 'pile/train', gui=False, data_root=args.data_root) # parameters scene and object_set are not used
        sim.setup_sim_scene_from_mesh_pose_list(mesh_pose_list, table=args.see_table, data_root=args.data_root)

        # index = random.choice(data.df[data.df.scene_id==scene].index)
        # tsdf, _, _, _= load_data(data[index])
        tsdf = make_tsdf(sim, device)

        
        pos = make_occ_grid(args.resolution)
        c = net.encode_inputs(tsdf)

        # Reconstruction
        occupancies = net.decoder_tsdf(pos, c,)
        vertices, triangles = mcubes.marching_cubes(occupancies.view(64, 64, 64).detach().numpy(), 0.5)
                
        if random:
            cam_eye = np.random.uniform(0.9, 1.3, 3)
        # Plotting
        (save_path / f"scene_{scene}").mkdir(parents=True, exist_ok=True)
        make_save_fig((vertices, triangles), cam_eye, file_name= save_path / f"scene_{scene}/reconstruction.png")
        make_save_fig(tsdf, cam_eye, file_name= save_path / f"scene_{scene}/tsdf.png", type="voxel")




if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--raw_root", type=Path, required=True)
    parser.add_argument("--save_path", type=Path, default="figures/")
    parser.add_argument("--net", default="neu_grasp_pn_deeper")
    parser.add_argument("--model_path", type=str, default='best_models/23-05-01-08-11-39_dataset=data_pile_train_constructed_4M_HighRes_radomized_views,augment=False,net=6d_neu_grasp_pn_deeper,batch_size=32,lr=5e-05,PN_deeper_DIMS_CONT/best_neural_grasp_neu_grasp_pn_deeper_val_acc=0.9097.pt')
    parser.add_argument("--see_table", action="store_true")
    parser.add_argument("--data_root", type=Path, required="True")
    parser.add_argument("--size", type=float, default=0.3)
    parser.add_argument("--num_scenes", type=int, default=5)
    parser.add_argument("--resolution", type=int, default=64)
    parser.add_argument("--random", action="store_true")
    parser.add_argument("--cuda", type=bool, default=False)
    parser.add_argument("--camera_shift", type=float, default=0.015)
    parser.add_argument("--cam_eye_path", type=Path, default="")
    parser.add_argument("--seed", type=int, default=7)
    args, _ = parser.parse_known_args()
    # print(args)
    set_random_seed(args.seed)
    main(args)