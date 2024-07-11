# # -*- coding: utf-8 -*-
from abc import ABCMeta, abstractmethod
# import copy
# import logging
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import open3d as o3d
import trimesh
from scipy.spatial.transform import Rotation

import matplotlib.pyplot as plt
from neugraspnet.src.utils.transform import Rotation, Transform
from neugraspnet.src.utils.grasp import Grasp
from neugraspnet.src.utils.implicit import as_mesh

from joblib import Parallel, delayed

# import os, IPython, sys
# import random
# import time
# import scipy.stats as stats
# try:
#     import pcl
# except ImportError as e:
#     print("[grasp_sampler] {}".format(e))

# from autolab_core import RigidTransform
# import scipy
# create logger
# logger = logging.getLogger(__name__)
# logger.setLevel(logging.INFO)

# USE_OPENRAVE = True
# try:
#     import openravepy as rave
# except ImportError:
#     # logger.warning('Failed to import OpenRAVE')
#     USE_OPENRAVE = False

# try:
#     import rospy
#     import moveit_commander
#     ROS_ENABLED = True
# except ImportError:
#     ROS_ENABLED = False

# try:
#     from mayavi import mlab
# except ImportError:
#     mlab = []

"""
Classes for sampling grasps.
Author: Jeff Mahler
"""

class GpgGraspSamplerPcl():
    """
    Sample grasps by GPG with pcl directly.
    http://journals.sagepub.com/doi/10.1177/0278364917735594
    """

    __metaclass__ = ABCMeta

    def __init__(self, gripper_hand_depth=None, debug_vis=False, gripper_type='franka'):
        self.params = {
            # 'num_rball_points': 27,  # FIXME: the same as meshpy..surface_normal()
            'num_dy': 10,  # number
            'dtheta': 10,  # unit degree
            'range_dtheta': 90,
            'range_dy': 0.04, # try to keep this to half of max gripper width
            'debug_vis': False,
            # 'r_ball': self.gripper.hand_height,
            'approach_step': 0.005,
            # 'max_trail_for_r_ball': 1000,
            # 'voxel_grid_ratio': 5,  # voxel_grid/sdf.resolution
            'safety_dis_above_table': 0.005,  # 0.5cm

            # Franka gripper
            "gripper_max_width": 0.08,
            "gripper_min_width": 0.0,
            "gripper_finger_width": 0.02, # Approx
            "gripper_hand_height": 0.030,
            "gripper_hand_outer_diameter": 0.12, # 0.218, the diameter of the robot hand (= maximum aperture plus 2 * finger width)
            "gripper_hand_depth": 0.0425, # 0.125,  # Franka finger depth is actually a little less than 0.05
            "gripper_init_bite": 0.005

            # # Robotiq 2F-85
            # "gripper_max_width": 0.085,
            # "gripper_min_width": 0.0,
            # "gripper_finger_width": 0.02, # Approx
            # "gripper_hand_height": 0.030,
            # "gripper_hand_outer_diameter": 0.125, # the diameter of the robot hand (= maximum aperture plus 2 * finger width)
            # "gripper_hand_depth": 0.04425,
            # "gripper_init_bite": 0.005

            # from gpd code:
            # "gripper_force_limit": 235.0,
            # "gripper_max_width": 0.085,
            # "gripper_finger_radius": 0.01,
            # "gripper_max_depth": 0.03,
            # "gripper_finger_width": 0.0255,
            # "gripper_real_finger_width": 0.0255,
            # "gripper_hand_height": 0.030,
            # "gripper_hand_height_two_finger_side": 0.105,
            # "gripper_hand_outer_diameter": 0.218, # This value seems dodgy
            # "gripper_hand_depth": 0.125,
            # "gripper_real_hand_depth": 0.120,
            # "gripper_init_bite": 0.01
        }
        if gripper_type != 'franka':
            # robotiq 2f-85
            self.params['gripper_hand_depth'] = 0.04025 # 0.03425 # 0.04425
            self.params['gripper_hand_outer_diameter'] = 0.12
            self.params['gripper_max_width'] = 0.08
            self.params['gripper_hand_height'] = 0.030
            self.params['gripper_finger_width'] = 0.02
            self.params['gripper_init_bite'] = 0.005
        
        if gripper_hand_depth is not None:
            # use custom gripper hand depth
            self.params['gripper_hand_depth'] = gripper_hand_depth
        self.params['debug_vis'] = debug_vis

    def sample_grasps_for_point(self, index, num_points_r_ball, kd_indices, distances, all_points, all_normals):

        processed_potential_grasps = []
        processed_potential_grasps_vgn = []
        potential_grasps_vgn_pos = []
        potential_grasps_vgn_rot_quat = []
        origin_points = []

        # calculate major principal curvature
        M = np.zeros((3, 3))
        for _ in range(num_points_r_ball):
            if distances[_] != 0:
                normal = all_normals[kd_indices[_]]
                normal = normal.reshape(-1, 1)
                if np.linalg.norm(normal) != 0:
                    normal /= np.linalg.norm(normal)
                M += np.matmul(normal, normal.T)
        if np.sum(np.sum(M)) == 0:
            print("M matrix is empty as there are no points in the neighbourhood")
            print("This could be problematic. If the number of points are too little we can get stuck in a loop")
            if self.return_origin_point:
                return [], [], [], []
            else:
                return [], [], []

        eigval, eigvec = np.linalg.eig(M)  # compared computed normal
        minor_pc = eigvec[:, np.argmin(eigval)].reshape(3)  # minor principal curvature !!! Here should use column!
        minor_pc /= np.linalg.norm(minor_pc)
        new_normal = eigvec[:, np.argmax(eigval)].reshape(3)  # estimated surface normal !!! Here should use column!
        new_normal /= np.linalg.norm(new_normal)
        major_pc = np.cross(minor_pc, new_normal)  # major principal curvature
        if np.linalg.norm(major_pc) != 0:
            major_pc = major_pc / np.linalg.norm(major_pc)

        # Judge if the new_normal has the same direction with old_normal, here the correct
        # direction in modified meshpy is point outward.
        if np.dot(all_normals[index], new_normal) < 0:
            new_normal = -new_normal
            minor_pc = -minor_pc

        for normal_dir in [1]: # use only one direction
            # Grid search for potential grasps
            potential_grasp = []
            for dtheta in np.arange(-self.params['range_dtheta'], # some magic numbers from original paper
                                    self.params['range_dtheta'] + 1,
                                    self.params['dtheta']):
                dy_potentials = []
                x, y, z = minor_pc
                dtheta = np.float64(dtheta)
                quat = np.array([x, y, z, dtheta / 180 * np.pi])
                rotation = Rotation.from_quat(quat).as_matrix() # NOTE this also rotates about the z axis (minor_pc) to get the correct grasping frame
                for dy in np.arange(-self.params['range_dy'],
                                    (self.params['range_dy'] + self.params['range_dy']/self.params['num_dy']),
                                    self.params['range_dy']/self.params['num_dy']):
                    # compute centers and axes
                    tmp_major_pc = np.dot(rotation, major_pc * normal_dir)
                    tmp_grasp_normal = np.dot(rotation, new_normal * normal_dir)
                    tmp_grasp_bottom_center = all_points[index, :].squeeze() + tmp_major_pc * dy
                    # go back a bite after rotation dtheta and translation dy!
                    tmp_grasp_bottom_center = self.params['gripper_init_bite'] * (
                            -tmp_grasp_normal * normal_dir) + tmp_grasp_bottom_center

                    open_points, _ = self.check_collision_square(tmp_grasp_bottom_center, tmp_grasp_normal,
                                                                    tmp_major_pc, minor_pc, all_points,
                                                                    self.hand_points, "p_open")
                    bottom_points, _ = self.check_collision_square(tmp_grasp_bottom_center, tmp_grasp_normal,
                                                                    tmp_major_pc, minor_pc, all_points,
                                                                    self.hand_points,
                                                                    "p_bottom")
                    if open_points is True and bottom_points is False:
                        left_points, _ = self.check_collision_square(tmp_grasp_bottom_center, tmp_grasp_normal,
                                                                        tmp_major_pc, minor_pc, all_points,
                                                                        self.hand_points,
                                                                        "p_left")
                        right_points, _ = self.check_collision_square(tmp_grasp_bottom_center, tmp_grasp_normal,
                                                                        tmp_major_pc, minor_pc, all_points,
                                                                        self.hand_points,
                                                                        "p_right")
                        if left_points is False and right_points is False:
                            dy_potentials.append([tmp_grasp_bottom_center, tmp_grasp_normal,
                                                    tmp_major_pc, minor_pc])

                if len(dy_potentials) != 0:
                    # we only take the middle grasp from dy direction.
                    center_dy = dy_potentials[int(np.ceil(len(dy_potentials) / 2) - 1)]
                    # we check if the gripper has a potential to collide with the table
                    # by check if the gripper is grasp from a down to top direction
                    finger_top_pos = center_dy[0] + center_dy[1] * self.params['gripper_hand_depth']
                    # [- self.params['gripper_hand_depth'] * 0.5] means we grasp objects as a angel larger than 30 degree
                    if finger_top_pos[2] < center_dy[0][2] - self.params['gripper_hand_depth'] * 0.5:
                        potential_grasp.append(center_dy)

            approach_dist = self.params['gripper_hand_depth']  # use gripper depth
            num_approaches = int(approach_dist / self.params['approach_step'])

            for ptg in potential_grasp:
                for approach_s in range(num_approaches):
                    tmp_grasp_bottom_center = ptg[1] * approach_s * self.params['approach_step'] + ptg[0]
                    tmp_grasp_normal = ptg[1]
                    tmp_major_pc = ptg[2]
                    minor_pc = ptg[3]
                    is_collide = self.check_collide(tmp_grasp_bottom_center, tmp_grasp_normal,
                                                    tmp_major_pc, minor_pc, all_points, self.hand_points)

                    if is_collide:
                        # if collide, go back one step to get a collision free hand position
                        tmp_grasp_bottom_center += (-tmp_grasp_normal) * self.params['approach_step'] #* 2

                        # here we check if the gripper collides with the table.
                        hand_points_ = self.get_hand_points(tmp_grasp_bottom_center,
                                                            tmp_grasp_normal,
                                                            tmp_major_pc)[1:]
                        min_finger_end = hand_points_[:, 2].min()
                        min_finger_end_pos_ind = np.where(hand_points_[:, 2] == min_finger_end)[0][0]

                        # safety_dis_above_table = 0.005 # 0.5cm
                        if min_finger_end < self.params['safety_dis_above_table']:
                            min_finger_pos = hand_points_[min_finger_end_pos_ind]  # the lowest point in a gripper
                            x = -min_finger_pos[2]*tmp_grasp_normal[0]/tmp_grasp_normal[2]+min_finger_pos[0]
                            y = -min_finger_pos[2]*tmp_grasp_normal[1]/tmp_grasp_normal[2]+min_finger_pos[1]
                            p_table = np.array([x, y, 0])  # the point that on the table
                            dis_go_back = np.linalg.norm([min_finger_pos-p_table]) + self.params['safety_dis_above_table']
                            tmp_grasp_bottom_center_modify = tmp_grasp_bottom_center-tmp_grasp_normal*dis_go_back
                        else:
                            # if the grasp does not collide with the table, do not change the grasp
                            tmp_grasp_bottom_center_modify = tmp_grasp_bottom_center

                        # final check
                        _, open_points = self.check_collision_square(tmp_grasp_bottom_center_modify,
                                                                        tmp_grasp_normal,
                                                                        tmp_major_pc, minor_pc, all_points,
                                                                        self.hand_points, "p_open")
                        is_collide = self.check_collide(tmp_grasp_bottom_center_modify, tmp_grasp_normal,
                                                        tmp_major_pc, minor_pc, all_points, self.hand_points)
                        if (len(open_points) > 10) and not is_collide:
                            # here 10 set the minimal points in a grasp, we can set a parameter later
                            processed_potential_grasps.append([tmp_grasp_bottom_center, tmp_grasp_normal,
                                                                tmp_major_pc, minor_pc,
                                                                tmp_grasp_bottom_center_modify])
                            # Convert grasp to VGN style grasp:
                            z_axis = tmp_grasp_normal
                            y_axis = tmp_major_pc
                            x_axis = -minor_pc
                            # x_axis = np.cross(y_axis, z_axis)
                            R = Rotation.from_matrix(np.vstack((x_axis, y_axis, z_axis)).T)
                            curr_grasp_pos = tmp_grasp_bottom_center_modify
                            curr_grasp_rot = R
                            curr_grasp = Grasp(Transform(curr_grasp_rot, curr_grasp_pos), self.params['gripper_max_width']) # make grasp
                            
                            potential_grasps_vgn_pos.append(curr_grasp_pos)
                            potential_grasps_vgn_rot_quat.append(curr_grasp_rot.as_quat())  # quaternion
                            processed_potential_grasps_vgn.append(curr_grasp)

                            if self.return_origin_point:
                                origin_points.append(all_points[index, :].squeeze())
        if self.return_origin_point:
            return processed_potential_grasps_vgn, potential_grasps_vgn_pos, potential_grasps_vgn_rot_quat, origin_points
        else:
            return processed_potential_grasps_vgn, potential_grasps_vgn_pos, potential_grasps_vgn_rot_quat


    def sample_grasps_parallel(self, point_cloud, num_parallel, num_grasps=20, max_num_samples=180, safety_dis_above_table=0.005,
                       show_final_grasps=False, verbose=False, return_origin_point=False, sample_constraints=None):
        """
        (Paralellized with joblib) Returns a list of candidate grasps for the given point cloud.

        Parameters
        ----------
        point_cloud : Open3d point cloud with normals
        num_grasps : int
            the number of grasps to generate

        show_final_grasps :
        max_num_samples :

        Returns
        -------
        potential_grasps: list of generated grasps
        """
        num_parallel_jobs = num_parallel
        # get all surface points
        # NOTE: For a dense point cloud, use voxel grid downsampling to reduce the number of points beforehand
        all_points = np.asarray(point_cloud.points)
        if point_cloud.normals:
            all_normals = np.asarray(point_cloud.normals)
        else:
            point_cloud.estimate_normals()
            all_normals = np.asarray(point_cloud.normals)
        # make sure the normal is pointing upwards
        ok_normal_mask = all_normals[:,2] > 0.1
        # Optional: Use sample constraints eg. only sample in a region of interest
        if sample_constraints is not None:
            for key, value in sample_constraints.items():
                if key == 'x':
                    ok_normal_mask = np.logical_and(ok_normal_mask, all_points[:,0] > value[0])
                    ok_normal_mask = np.logical_and(ok_normal_mask, all_points[:,0] < value[1])
        num_parallel_jobs = min(ok_normal_mask.sum(), num_parallel_jobs) # Handle edge case where ok_points are too few

        if num_parallel_jobs < 1:
            if return_origin_point:
                return [], [], [], []
            else:
                return [], [], []

        kd = o3d.geometry.KDTreeFlann(point_cloud)
        
        self.hand_points = self.get_hand_points(np.array([0, 0, 0]), np.array([1, 0, 0]), np.array([0, 1, 0]))
        self.params['r_ball'] = max(self.params['gripper_hand_outer_diameter'] - self.params['gripper_finger_width'], self.params['gripper_hand_depth'],
                self.params['gripper_hand_height'] / 2.0)
        # r_ball = params['r_ball']  # FIXME: for some relative small obj, we might need to use a pre-defined radius
        self.params['safety_dis_above_table'] = safety_dis_above_table
        
        sampled_surface_amount = 0
        potential_grasps_vgn_pos = []
        potential_grasps_vgn_rot_quat = []
        processed_potential_grasps_vgn = []
        origin_points = []
        self.return_origin_point = return_origin_point

        while len(processed_potential_grasps_vgn) < num_grasps and sampled_surface_amount < max_num_samples:
            # Choose n random points on surface
            indices = np.random.choice(np.where(ok_normal_mask)[0], size=num_parallel_jobs, replace=False)
            sampled_surface_amount += num_parallel_jobs
            if verbose:
                print("No. of sampled surface points:", sampled_surface_amount)
            
            # Get r_ball points for *each* sampled surface point
            all_num_points_r_ball = []
            all_kd_indices = []
            all_sqr_distances = []
            for j in indices:
                num_points_r_ball, kd_indices, sqr_distances =  kd.search_hybrid_vector_3d(point_cloud.points[j], self.params['r_ball'], 100) # TODO: Check if 100 neighbours is enough
                all_num_points_r_ball.append(np.asarray(num_points_r_ball))
                all_kd_indices.append(np.asarray(kd_indices))
                all_sqr_distances.append(np.asarray(sqr_distances))
            # num_points_r_ball, kd_indices, sqr_distances = self.kd.search_hybrid_vector_3d(point_cloud.points[index], self.params['r_ball'], 100) # TODO: Check if 100 neighbours is enough
            # num_points, kd_indices, sqr_distances = kd.search_radius_vector_3d(point_cloud.points[ind], r_ball)
            # kd_indices, sqr_distances = kd.radius_search_for_cloud(point_cloud.points[ind], r_ball, 100)

            # sample grasps for n points in parallel
            results = Parallel(n_jobs=num_parallel_jobs)(delayed(self.sample_grasps_for_point)(index, all_num_points_r_ball[i], all_kd_indices[i], all_sqr_distances[i], all_points, all_normals) for i, index in enumerate(indices))

            # accumulate results
            for result in results:
                if return_origin_point:
                    ret_processed_potential_grasps_vgn, ret_potential_grasps_vgn_pos, ret_potential_grasps_vgn_rot_quat, ret_origin_points = result
                else:
                    ret_processed_potential_grasps_vgn, ret_potential_grasps_vgn_pos, ret_potential_grasps_vgn_rot_quat = result
                processed_potential_grasps_vgn.extend(ret_processed_potential_grasps_vgn)
                potential_grasps_vgn_pos.extend(ret_potential_grasps_vgn_pos)
                potential_grasps_vgn_rot_quat.extend(ret_potential_grasps_vgn_rot_quat)
                if return_origin_point:
                    origin_points.extend(ret_origin_points)

        if show_final_grasps:
            # Show all grasps and the surface point cloud with open3d
            self.show_grasps_and_pcl_open3d(processed_potential_grasps_vgn, point_cloud)
        
        if return_origin_point:
            return processed_potential_grasps_vgn, potential_grasps_vgn_pos, potential_grasps_vgn_rot_quat, origin_points
        else:
            return processed_potential_grasps_vgn, potential_grasps_vgn_pos, potential_grasps_vgn_rot_quat


    def sample_grasps(self, point_cloud, num_grasps=20, max_num_samples=180, safety_dis_above_table=0.005,
                       show_final_grasps=False, verbose=False, return_origin_point=False, sample_constraints=None,
                      **kwargs):
        """
        Returns a list of candidate grasps for the given point cloud.

        Parameters
        ----------
        point_cloud : Open3d point cloud with normals
        num_grasps : int
            the number of grasps to generate

        show_final_grasps :
        max_num_samples :

        Returns
        -------
        potential_grasps: list of generated grasps
        """
        # np.random.seed() # Random seed
        # get all surface points
        # TODO: For dense point cloud, we can use voxel grid downsampling to reduce the number of points
        all_points = np.asarray(point_cloud.points)
        if point_cloud.normals:
            all_normals = np.asarray(point_cloud.normals)
        else:
            point_cloud.estimate_normals()
            all_normals = np.asarray(point_cloud.normals)
        kd = o3d.geometry.KDTreeFlann(point_cloud)
        sampled_surface_amount = 0
        grasps = []
        processed_potential_grasps = []
        potential_grasps_vgn_pos = []
        potential_grasps_vgn_rot_quat = []
        processed_potential_grasps_vgn = []
        if return_origin_point:
            origin_points = []

        hand_points = self.get_hand_points(np.array([0, 0, 0]), np.array([1, 0, 0]), np.array([0, 1, 0]))

        # get all grasps
        while len(grasps) < num_grasps and sampled_surface_amount < max_num_samples:
            ind = np.random.choice(all_points.shape[0], size=1, replace=False)
            sampled_surface_amount += 1
            # print("No. of sampled surface points:", sampled_surface_amount)
            ok = all_normals[ind, 2] > -0.1  # make sure the normal is pointing upwards
            # Optional: Use sample constraints eg. only sample in a region of interest
            if sample_constraints is not None:
                for key, value in sample_constraints.items():
                    if key == 'x':
                        ok = np.logical_and(ok, all_points[ind,0] > value[0])
                        ok = np.logical_and(ok, all_points[ind,0] < value[1])
            if not ok:
                continue
            selected_surface = all_points[ind, :].squeeze()

            # calculate major principal curvature
            # r_ball = params['r_ball']  # FIXME: for some relative small obj, we need to use pre-defined radius
            r_ball = max(self.params['gripper_hand_outer_diameter'] - self.params['gripper_finger_width'], self.params['gripper_hand_depth'],
                         self.params['gripper_hand_height'] / 2.0)
            # point_amount = params['num_rball_points']
            # max_trial = params['max_trail_for_r_ball']
            M = np.zeros((3, 3))
            num_points, kd_indices, sqr_distances = kd.search_hybrid_vector_3d(point_cloud.points[ind], r_ball, 100) # TODO: Check if 100 is enough
            # num_points, kd_indices, sqr_distances = kd.search_radius_vector_3d(point_cloud.points[ind], r_ball)
            # kd_indices, sqr_distances = kd.radius_search_for_cloud(point_cloud.points[ind], r_ball, 100)
            for _ in range(num_points):
                if sqr_distances[_] != 0:
                    normal = all_normals[kd_indices[_]]
                    normal = normal.reshape(-1, 1)
                    if np.linalg.norm(normal) != 0:
                        normal /= np.linalg.norm(normal)
                    M += np.matmul(normal, normal.T)
            if np.sum(np.sum(M)) == 0:
                print("M matrix is empty as there are no points in the neighbourhood")
                print("This could be problematic. If the number of points are too little we can get stuck in a loop")
                continue

            eigval, eigvec = np.linalg.eig(M)  # compared computed normal
            minor_pc = eigvec[:, np.argmin(eigval)].reshape(3)  # minor principal curvature !!! Here should use column!
            minor_pc /= np.linalg.norm(minor_pc)
            new_normal = eigvec[:, np.argmax(eigval)].reshape(3)  # estimated surface normal !!! Here should use column!
            new_normal /= np.linalg.norm(new_normal)
            major_pc = np.cross(minor_pc, new_normal)  # major principal curvature
            if np.linalg.norm(major_pc) != 0:
                major_pc = major_pc / np.linalg.norm(major_pc)

            # Judge if the new_normal has the same direction with old_normal, here the correct
            # direction in modified meshpy is point outward.
            if np.dot(all_normals[ind], new_normal) < 0:
                new_normal = -new_normal
                minor_pc = -minor_pc

            for normal_dir in [1]:
                if self.params['debug_vis']:
                    # show grasping frame
                    self.mpl_fig = plt.figure().add_subplot(projection='3d')
                    plt.xlim(0.0, 0.3)
                    plt.ylim(0.0, 0.3)
                    self.mpl_fig.set_zlim(0.0, 0.3)
                    plt.autoscale(False)
                    self.show_grasp_norm_oneside(selected_surface, new_normal * normal_dir, major_pc * normal_dir,
                                                 minor_pc, scale_factor=0.001)
                    self.show_points(selected_surface, color='g', scale_factor=.002)
                    self.show_points(all_points)
                    # show real norm direction: if new_norm has very diff than pcl cal norm, then maybe a bug.
                    self.show_line(selected_surface, (selected_surface + all_normals[ind]*0.09).reshape(3))
                    plt.show()

                # Grid search for potential grasps
                
                potential_grasp = []
                for dtheta in np.arange(-self.params['range_dtheta'], # some magic numbers from original paper
                                        self.params['range_dtheta'] + 1,
                                        self.params['dtheta']):
                    dy_potentials = []
                    x, y, z = minor_pc
                    dtheta = np.float64(dtheta)
                    quat = np.array([x, y, z, dtheta / 180 * np.pi])
                    rotation = Rotation.from_quat(quat).as_matrix() # NOTE this also rotates about the z axis (minor_pc) to get the correct grasping frame
                    for dy in np.arange(-self.params['range_dy'],
                                        (self.params['range_dy'] + self.params['range_dy']/self.params['num_dy']),
                                        self.params['range_dy']/self.params['num_dy']):
                        # compute centers and axes
                        tmp_major_pc = np.dot(rotation, major_pc * normal_dir)
                        tmp_grasp_normal = np.dot(rotation, new_normal * normal_dir)
                        tmp_grasp_bottom_center = selected_surface + tmp_major_pc * dy
                        # go back a bite after rotation dtheta and translation dy!
                        tmp_grasp_bottom_center = self.params['gripper_init_bite'] * (
                                -tmp_grasp_normal * normal_dir) + tmp_grasp_bottom_center

                        open_points, _ = self.check_collision_square(tmp_grasp_bottom_center, tmp_grasp_normal,
                                                                     tmp_major_pc, minor_pc, all_points,
                                                                     hand_points, "p_open")
                        bottom_points, _ = self.check_collision_square(tmp_grasp_bottom_center, tmp_grasp_normal,
                                                                       tmp_major_pc, minor_pc, all_points,
                                                                       hand_points,
                                                                       "p_bottom")
                        if open_points is True and bottom_points is False:
                            left_points, _ = self.check_collision_square(tmp_grasp_bottom_center, tmp_grasp_normal,
                                                                         tmp_major_pc, minor_pc, all_points,
                                                                         hand_points,
                                                                         "p_left")
                            right_points, _ = self.check_collision_square(tmp_grasp_bottom_center, tmp_grasp_normal,
                                                                          tmp_major_pc, minor_pc, all_points,
                                                                          hand_points,
                                                                          "p_right")

                            if left_points is False and right_points is False:
                                dy_potentials.append([tmp_grasp_bottom_center, tmp_grasp_normal,
                                                      tmp_major_pc, minor_pc])

                    if len(dy_potentials) != 0:
                        # we only take the middle grasp from dy direction.
                        center_dy = dy_potentials[int(np.ceil(len(dy_potentials) / 2) - 1)]
                        # we check if the gripper has a potential to collide with the table
                        # by check if the gripper is grasp from a down to top direction
                        finger_top_pos = center_dy[0] + center_dy[1] * self.params['gripper_hand_depth']
                        # [- self.params['gripper_hand_depth'] * 0.5] means we grasp objects as a angel larger than 30 degree
                        if finger_top_pos[2] < center_dy[0][2] - self.params['gripper_hand_depth'] * 0.5:
                            potential_grasp.append(center_dy)

                approach_dist = self.params['gripper_hand_depth']  # use gripper depth
                num_approaches = int(approach_dist / self.params['approach_step'])

                for ptg in potential_grasp:
                    for approach_s in range(num_approaches):
                        tmp_grasp_bottom_center = ptg[1] * approach_s * self.params['approach_step'] + ptg[0]
                        tmp_grasp_normal = ptg[1]
                        tmp_major_pc = ptg[2]
                        minor_pc = ptg[3]
                        is_collide = self.check_collide(tmp_grasp_bottom_center, tmp_grasp_normal,
                                                        tmp_major_pc, minor_pc, all_points, hand_points)

                        if is_collide:
                            # if collide, go back one step to get a collision free hand position
                            tmp_grasp_bottom_center += (-tmp_grasp_normal) * self.params['approach_step'] #* 2

                            # here we check if the gripper collides with the table.
                            hand_points_ = self.get_hand_points(tmp_grasp_bottom_center,
                                                                tmp_grasp_normal,
                                                                tmp_major_pc)[1:]
                            min_finger_end = hand_points_[:, 2].min()
                            min_finger_end_pos_ind = np.where(hand_points_[:, 2] == min_finger_end)[0][0]

                            # safety_dis_above_table = 0.005 # 0.5cm
                            if min_finger_end < safety_dis_above_table:
                                min_finger_pos = hand_points_[min_finger_end_pos_ind]  # the lowest point in a gripper
                                x = -min_finger_pos[2]*tmp_grasp_normal[0]/tmp_grasp_normal[2]+min_finger_pos[0]
                                y = -min_finger_pos[2]*tmp_grasp_normal[1]/tmp_grasp_normal[2]+min_finger_pos[1]
                                p_table = np.array([x, y, 0])  # the point that on the table
                                dis_go_back = np.linalg.norm([min_finger_pos-p_table]) + safety_dis_above_table
                                tmp_grasp_bottom_center_modify = tmp_grasp_bottom_center-tmp_grasp_normal*dis_go_back
                            else:
                                # if the grasp does not collide with the table, do not change the grasp
                                tmp_grasp_bottom_center_modify = tmp_grasp_bottom_center

                            # final check
                            _, open_points = self.check_collision_square(tmp_grasp_bottom_center_modify,
                                                                         tmp_grasp_normal,
                                                                         tmp_major_pc, minor_pc, all_points,
                                                                         hand_points, "p_open")
                            is_collide = self.check_collide(tmp_grasp_bottom_center_modify, tmp_grasp_normal,
                                                            tmp_major_pc, minor_pc, all_points, hand_points)
                            if (len(open_points) > 10) and not is_collide:
                                # here 10 set the minimal points in a grasp, we can set a parameter later
                                processed_potential_grasps.append([tmp_grasp_bottom_center, tmp_grasp_normal,
                                                                  tmp_major_pc, minor_pc,
                                                                  tmp_grasp_bottom_center_modify])
                                # Convert grasp to VGN style grasp:
                                z_axis = tmp_grasp_normal
                                y_axis = tmp_major_pc
                                x_axis = -minor_pc
                                # x_axis = np.cross(y_axis, z_axis)
                                R = Rotation.from_matrix(np.vstack((x_axis, y_axis, z_axis)).T)
                                curr_grasp_pos = tmp_grasp_bottom_center_modify
                                curr_grasp_rot = R
                                curr_grasp = Grasp(Transform(curr_grasp_rot, curr_grasp_pos), self.params['gripper_max_width']) # make grasp
                                
                                potential_grasps_vgn_pos.append(curr_grasp_pos)
                                potential_grasps_vgn_rot_quat.append(curr_grasp_rot.as_quat())  # quaternion
                                processed_potential_grasps_vgn.append(curr_grasp)

                                if return_origin_point:
                                    origin_points.append(selected_surface)

                                if self.params['debug_vis']:
                                    # Show grasp and the surface point with open3d
                                    self.show_grasps_and_pcl_open3d([curr_grasp], point_cloud)
                                    # self.show_points(selected_surface, color='r', scale_factor=.005)
                                    # logger.info('usefull grasp sample point original: %s', selected_surface)
                                    # self.check_collision_square(tmp_grasp_bottom_center_modify, tmp_grasp_normal,
                                    #                             tmp_major_pc, minor_pc, all_points, hand_points,
                                    #                             "p_open", vis=True)
                                break
                # logger.info("processed_potential_grasp %d", len(processed_potential_grasps))

            # logger.info("current amount of sampled surface %d", sampled_surface_amount)
            if verbose:
                print("No. of sampled surface points:", sampled_surface_amount)
                print("No. of grasp candidates sampled using GPG:", len(processed_potential_grasps_vgn))
            if len(processed_potential_grasps_vgn) >= num_grasps or sampled_surface_amount >= max_num_samples:
                if show_final_grasps:
                    # Show all grasps and the surface point cloud with open3d
                    self.show_grasps_and_pcl_open3d(processed_potential_grasps_vgn, point_cloud)
                if return_origin_point:
                    return processed_potential_grasps_vgn, potential_grasps_vgn_pos, potential_grasps_vgn_rot_quat, origin_points
                else:
                    return processed_potential_grasps_vgn, potential_grasps_vgn_pos, potential_grasps_vgn_rot_quat

        if show_final_grasps:
            # Show all grasps and the surface point cloud with open3d
            self.show_grasps_and_pcl_open3d(processed_potential_grasps_vgn, point_cloud)
        if return_origin_point:
            return processed_potential_grasps_vgn, potential_grasps_vgn_pos, potential_grasps_vgn_rot_quat, origin_points
        else:
            return processed_potential_grasps_vgn, potential_grasps_vgn_pos, potential_grasps_vgn_rot_quat

    def show_grasps_and_pcl_open3d(self, grasps, point_cloud):
        grasps_scene = trimesh.Scene()
        from neugraspnet.utils import visual
        grasp_mesh_list = [visual.grasp2mesh(g) for g in grasps]
        for i, g_mesh in enumerate(grasp_mesh_list):
            grasps_scene.add_geometry(g_mesh, node_name=f'grasp_{i}')

        o3d.visualization.draw_geometries([point_cloud, as_mesh(grasps_scene).as_open3d])

    def show_points(self, point, color='lb', scale_factor=.0005):
        if color == 'b':
            color_f = (0, 0, 1)
        elif color == 'r':
            color_f = (1, 0, 0)
        elif color == 'g':
            color_f = (0, 1, 0)
        elif color == 'lb':  # light blue
            color_f = (0.22, 1, 1)
        else:
            color_f = (1, 1, 1)
        if point.size == 3:  # vis for only one point, shape must be (3,), for shape (1, 3) is not work
            point = point.reshape(3, )
            #mlab.points3d(point[0], point[1], point[2], color=color_f, scale_factor=scale_factor)
            self.mpl_fig.scatter(point[0], point[1], point[2], color=color_f)
        else:  # vis for multiple points
            # mlab.points3d(point[:, 0], point[:, 1], point[:, 2], color=color_f, scale_factor=scale_factor)
            self.mpl_fig.scatter(point[:,0], point[:,1], point[:,2], color=color_f)

    def show_line(self, un1, un2, color='g', scale_factor=0.0005):
        if color == 'b':
            color_f = (0, 0, 1)
        elif color == 'r':
            color_f = (1, 0, 0)
        elif color == 'g':
            color_f = (0, 1, 0)
        else:
            color_f = (1, 1, 1)
        self.mpl_fig.plot([un1[0], un2[0]], [un1[1], un2[1]], [un1[2], un2[2]], color=color_f)
        # mlab.plot3d([un1[0], un2[0]], [un1[1], un2[1]], [un1[2], un2[2]], color=color_f, tube_radius=scale_factor)

    def show_grasp_norm_oneside(self, grasp_bottom_center,
                                grasp_normal, grasp_axis, minor_pc, scale_factor=0.001):

        # un1 = grasp_bottom_center + 0.5 * grasp_axis * self.gripper.max_width
        un2 = grasp_bottom_center
        # un3 = grasp_bottom_center + 0.5 * minor_pc * self.gripper.max_width
        # un4 = grasp_bottom_center
        # un5 = grasp_bottom_center + 0.5 * grasp_normal * self.gripper.max_width
        # un6 = grasp_bottom_center
        self.show_points(grasp_bottom_center, color='g', scale_factor=scale_factor * 4)
        # self.show_points(un1, scale_factor=scale_factor * 4)
        # self.show_points(un3, scale_factor=scale_factor * 4)
        # self.show_points(un5, scale_factor=scale_factor * 4)
        # self.show_line(un1, un2, color='g', scale_factor=scale_factor)  # binormal/ major pc
        # self.show_line(un3, un4, color='b', scale_factor=scale_factor)  # minor pc
        # self.show_line(un5, un6, color='r', scale_factor=scale_factor)  # approach normal
        self.mpl_fig.quiver(un2[0], un2[1], un2[2], grasp_axis[0], grasp_axis[1], grasp_axis[2], length=0.03, normalize=False)
        self.mpl_fig.quiver(un2[0], un2[1], un2[2], minor_pc[0], minor_pc[1], minor_pc[2], length=0.03, normalize=False)
        self.mpl_fig.quiver(un2[0], un2[1], un2[2], grasp_normal[0], grasp_normal[1], grasp_normal[2], length=0.09, normalize=False)
        # mlab.quiver3d(un2[0], un2[1], un2[2], grasp_axis[0], grasp_axis[1], grasp_axis[2],
        #               scale_factor=.03, line_width=0.25, color=(0, 1, 0), mode='arrow')
        # mlab.quiver3d(un2[0], un2[1], un2[2], minor_pc[0], minor_pc[1], minor_pc[2],
        #               scale_factor=.03, line_width=0.1, color=(0, 0, 1), mode='arrow')
        # mlab.quiver3d(un2[0], un2[1], un2[2], grasp_normal[0], grasp_normal[1], grasp_normal[2],
        #               scale_factor=.03, line_width=0.05, color=(1, 0, 0), mode='arrow')

    def get_hand_points(self, grasp_bottom_center, approach_normal, binormal):
        hh = self.params['gripper_hand_height']
        fw = self.params['gripper_finger_width']
        hod = self.params['gripper_hand_outer_diameter']
        hd = self.params['gripper_hand_depth']
        open_w = hod - fw * 2
        minor_pc = np.cross(approach_normal, binormal)
        minor_pc = minor_pc / np.linalg.norm(minor_pc)
        p5_p6 = minor_pc * hh * 0.5 + grasp_bottom_center
        p7_p8 = -minor_pc * hh * 0.5 + grasp_bottom_center
        p5 = -binormal * open_w * 0.5 + p5_p6
        p6 = binormal * open_w * 0.5 + p5_p6
        p7 = binormal * open_w * 0.5 + p7_p8
        p8 = -binormal * open_w * 0.5 + p7_p8
        p1 = approach_normal * hd + p5
        p2 = approach_normal * hd + p6
        p3 = approach_normal * hd + p7
        p4 = approach_normal * hd + p8

        p9 = -binormal * fw + p1
        p10 = -binormal * fw + p4
        p11 = -binormal * fw + p5
        p12 = -binormal * fw + p8
        p13 = binormal * fw + p2
        p14 = binormal * fw + p3
        p15 = binormal * fw + p6
        p16 = binormal * fw + p7

        p17 = -approach_normal * hh + p11
        p18 = -approach_normal * hh + p15
        p19 = -approach_normal * hh + p16
        p20 = -approach_normal * hh + p12
        p = np.vstack([np.array([0, 0, 0]), p1, p2, p3, p4, p5, p6, p7, p8, p9, p10,
                       p11, p12, p13, p14, p15, p16, p17, p18, p19, p20])
        return p

    def show_grasp_3d(self, hand_points, color=(0.003, 0.50196, 0.50196)):
        # for i in range(1, 21):
        #     self.show_points(p[i])
        if color == 'd':
            color = (0.003, 0.50196, 0.50196)
        triangles = [(9, 1, 4), (4, 9, 10), (4, 10, 8), (8, 10, 12), (1, 4, 8), (1, 5, 8),
                     (1, 5, 9), (5, 9, 11), (9, 10, 20), (9, 20, 17), (20, 17, 19), (17, 19, 18),
                     (14, 19, 18), (14, 18, 13), (3, 2, 13), (3, 13, 14), (3, 6, 7), (3, 6, 2),
                     (3, 14, 7), (14, 7, 16), (2, 13, 15), (2, 15, 6), (12, 20, 19), (12, 19, 16),
                     (15, 11, 17), (15, 17, 18), (6, 7, 8), (6, 8, 5)]
        # mlab.triangular_mesh(hand_points[:, 0], hand_points[:, 1], hand_points[:, 2],
        #                      triangles, color=color, opacity=0.5)
        self.mpl_fig.plot_trisurf(hand_points[:, 0], hand_points[:, 1], hand_points[:, 2],
                             triangles=triangles, color=color, antialiased=True)

    def check_collision_square(self, grasp_bottom_center, approach_normal, binormal,
                               minor_pc, graspable, p, way, vis=False):
        approach_normal = approach_normal.reshape(1, 3)
        approach_normal = approach_normal / np.linalg.norm(approach_normal)
        binormal = binormal.reshape(1, 3)
        binormal = binormal / np.linalg.norm(binormal)
        minor_pc = minor_pc.reshape(1, 3)
        minor_pc = minor_pc / np.linalg.norm(minor_pc)
        matrix = np.hstack([approach_normal.T, binormal.T, minor_pc.T])
        grasp_matrix = matrix.T  # same as cal the inverse
        # if isinstance(graspable, dexnet.grasping.graspable_object.GraspableObject3D):
        #     points = graspable.sdf.surface_points(grid_basis=False)[0]
        # else:
        points = graspable
        points = points - grasp_bottom_center.reshape(1, 3)
        # points_g = points @ grasp_matrix
        tmp = np.dot(grasp_matrix, points.T)
        points_g = tmp.T
        if way == "p_open":
            s1, s2, s4, s8 = p[1], p[2], p[4], p[8]
        elif way == "p_left":
            s1, s2, s4, s8 = p[9], p[1], p[10], p[12]
        elif way == "p_right":
            s1, s2, s4, s8 = p[2], p[13], p[3], p[7]
        elif way == "p_bottom":
            s1, s2, s4, s8 = p[11], p[15], p[12], p[20]
        else:
            raise ValueError('No way!')
        a1 = s1[1] < points_g[:, 1]
        a2 = s2[1] > points_g[:, 1]
        a3 = s1[2] > points_g[:, 2]
        a4 = s4[2] < points_g[:, 2]
        a5 = s4[0] > points_g[:, 0]
        a6 = s8[0] < points_g[:, 0]

        a = np.vstack([a1, a2, a3, a4, a5, a6])
        points_in_area = np.where(np.sum(a, axis=0) == len(a))[0]
        if len(points_in_area) == 0:
            has_p = False
        else:
            has_p = True

        if vis:
            print("points_in_area", way, len(points_in_area))
            # mlab.clf()
            plt.cla()
            self.mpl_fig = plt.figure().add_subplot(projection='3d')
            # self.show_one_point(np.array([0, 0, 0]))
            self.show_grasp_3d(p)
            self.show_points(points_g)
            if len(points_in_area) != 0:
                self.show_points(points_g[points_in_area], color='r')
            # mlab.show()
            plt.show()
        # print("points_in_area", way, len(points_in_area))
        return has_p, points_in_area

    def show_all_grasps(self, all_points, grasps_for_show):

        for grasp_ in grasps_for_show:
            grasp_bottom_center = grasp_[4]  # new feature: ues the modified grasp bottom center
            approach_normal = grasp_[1]
            binormal = grasp_[2]
            hand_points = self.get_hand_points(grasp_bottom_center, approach_normal, binormal)
            self.show_grasp_3d(hand_points)
        # self.show_points(all_points)
        # mlab.show()

    def check_collide(self, grasp_bottom_center, approach_normal, binormal, minor_pc, graspable, hand_points):
        bottom_points = self.check_collision_square(grasp_bottom_center, approach_normal,
                                                    binormal, minor_pc, graspable, hand_points, "p_bottom")
        if bottom_points[0]:
            return True

        left_points = self.check_collision_square(grasp_bottom_center, approach_normal,
                                                  binormal, minor_pc, graspable, hand_points, "p_left")
        if left_points[0]:
            return True

        right_points = self.check_collision_square(grasp_bottom_center, approach_normal,
                                                   binormal, minor_pc, graspable, hand_points, "p_right")
        if right_points[0]:
            return True

        return False