import os
from pathlib import Path
import argparse
import numpy as np
import rospy
# import geometry_msgs
from sensor_msgs.msg import PointCloud2, Image
from geometry_msgs.msg import PoseArray, Pose
from std_srvs.srv import Empty, EmptyResponse
# import sensor_msgs.point_cloud2 as pc2
# from std_msgs.msg import Header
# from tf2_sensor_msgs.tf2_sensor_msgs import do_transform_cloud # Deprecated
from tf.transformations import euler_from_quaternion, quaternion_from_euler, quaternion_from_matrix
np.float = np.float64  # temp fix for following import of old ros_numpy
import ros_numpy
# import tf
import tf2_ros
import pickle
import open3d as o3d

# neugraspnet repo imports
from neugraspnet.perception import CameraIntrinsic, TSDFVolume
from neugraspnet.utils.transform import Rotation, Transform
from neugraspnet.grasp import *
from neugraspnet.detection_implicit import NeuGraspImplicit
from neugraspnet.simulation import ClutterRemovalSim

################################################ 
# Perception pipeline
################################################

# SCENE_SIZE = 0.3 # size of the scene in meters

class GraspGenerator:
	def __init__(self, grasp_frame_name='grasp_origin', grasp_srv_name='get_grasps', grasp_topic_name="/generated_grasps", camera_type='zed',
	      		 net_type='neu_grasp_pn_deeper', net_path=None, tsdf_res=64, scene_size=0.3, downsampl_size=0.0075, use_reachability=False):
		
		# Publish grasps to topic
		self.grasp_topic_name = grasp_topic_name
		self.grasp_pub = rospy.Publisher(self.grasp_topic_name, PoseArray, queue_size=1, latch=True)

		# trigger grasp generation via service call
		self.grasp_srv = rospy.Service(grasp_srv_name, Empty, self.grasps_srv_callback)

		# grasps fill be in this frame
		self.grasp_frame_name = grasp_frame_name

		# Setup camera
		self.camera_type = camera_type
		if self.camera_type == 'zed':
			# get the pointcloud from the topic
			self.pcl_topic_name = "/zed2/zed_node/point_cloud/cloud_registered"
			self.depth_topic_name = "/zed2/zed_node/depth/depth_registered"
			# camera intrinsic parameters (TODO: Get this from /zed2/zed_node/depth/camera_info topic)
			# 720p
			self.camera_intrinsic = CameraIntrinsic(width=640, height=360,
													fx=262.1390075683594, fy=262.1390075683594,
													cx=311.67730712890625, cy=185.56422424316406)
			# VGA
			# self.camera_intrinsic = CameraIntrinsic(width=336, height=188,
			# 										fx=132.06214904785156, fy=132.06214904785156,
			# 										cx=161.06512451171875, cy=97.46347045898438)								
										   
		elif self.camera_type == 'xtion':
			self.pcl_topic_name = "/xtion/depth_registered/points"
			self.depth_topic_name = "/xtion/depth/image_rect"
			# camera intrinsic parameters (TODO: Get this from /xtion/depth/camera_info topic)
			self.camera_intrinsic = CameraIntrinsic(width=640, height=480,
													fx=520.7559204101562, fy=524.538330078125,
													cx=331.348811989792, cy=239.89580009973724)
		else:
			raise ValueError('Invalid camera type. Choose either zed or xtion.')
		
		# Optional: point cloud subscriber
		# self.cloud_sub = rospy.Subscriber(pcl_topic_name, PointCloud2, self.process_pcl, queue_size=1, buff_size=52428800)
		# # depth image subscriber
		# self.depth_sub = rospy.Subscriber(depth_topic_name, Image, self.process_depth_img, queue_size=1, buff_size=52428800)

		# tf listener to get the point cloud in the correct frame
		self.tf_buffer = tf2_ros.Buffer()
		self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

		if net_path is None:
			# net_path = Path('/neugraspnet/neugraspnet_repo/data/best_real_robot_runs/PILE_neural_grasp_neu_grasp_pn_deeper_468244.pt')
			# net_path = Path('/neugraspnet/neugraspnet_repo/data/best_real_robot_runs/PACKED_best_neural_grasp_neu_grasp_pn_deeper_val_acc=0.9040.pt')
			net_path = Path(os.path.dirname(os.path.abspath(__file__))+'/../neugraspnet_repo/data/runs_relevant_affnet/23-11-06-04-55-36_dataset=data_affnet_train_constructed_GPG_60,augment=False,net=6d_neu_grasp_pn_affnet,batch_size=64,lr=2e-04,AFFNET_v4_no_hand_balanced/best_neural_grasp_neu_grasp_pn_affnet_val_acc=0.9179.pt')

		self.size = scene_size
		self.tsdf_res = tsdf_res
		self.downsampl_size = downsampl_size
		self.use_reachability = use_reachability
		if use_reachability:
			# Load reachability maps
			# Temp: only loading right arm reachability map
			with open('reach_maps/smaller_full_reach_map_gripper_right_grasping_frame_torso_False_0.05.pkl', 'rb') as f:
				self.right_reachability_map = pickle.load(f)

		self.setup_grasp_planner(model=net_path, type=net_type, qual_th=0.5, aff_thresh=0.5, force=False, seen_pc_only=False)
		
		self.o3d_vis = None # open3d visualizer

	def setup_grasp_planner(self, model, type, qual_th=0.5, aff_thresh=0.5, force=False, seen_pc_only=False, vis_mesh=False):

		self.grasp_planner = NeuGraspImplicit(model,
                                    type,
                                    qual_th=qual_th,
									aff_thresh=aff_thresh,
                                    force_detection=force,
                                    seen_pc_only=seen_pc_only,
                                    resolution=self.tsdf_res,
                                    visualize=vis_mesh)
		
	def process_pcl(self, ros_point_cloud=None):
		# transform the point cloud to the grasp_origin frame
		while not rospy.is_shutdown():
			try:
				if self.camera_type == 'zed':
					# table_to_pcl = self.tf_buffer.lookup_transform(self.grasp_frame_name, 'zed2_left_camera_frame', rospy.Time(0))
					table_to_pcl = self.tf_buffer.lookup_transform(self.grasp_frame_name, 'zed2_left_camera_frame', rospy.Time(0)).transform
				elif self.camera_type == 'xtion':
					# table_to_pcl = self.tf_buffer.lookup_transform(self.grasp_frame_name, 'xtion_rgb_optical_frame', rospy.Time(0))				
					table_to_pcl = self.tf_buffer.lookup_transform(self.grasp_frame_name, 'xtion_rgb_optical_frame', rospy.Time(0)).transform
			except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
				print("[WARNING: Could not find some TFs. Check TF and obj names]")
				# print(e)
				# rate.sleep()
				continue
			break
	
		# Transform point cloud
		# cloud_transformed = do_transform_cloud(ros_point_cloud, table_to_pcl)
		# pc = ros_numpy.numpify(cloud_transformed)
		# convert to numpy array
		pc = ros_numpy.numpify(ros_point_cloud)
		points=np.zeros((pc.shape[0],pc.shape[1],3))
		points[:,:,0]=pc['x']
		points[:,:,1]=pc['y']
		points[:,:,2]=pc['z']
		# flatten points
		points = points.reshape(-1, 3)
		# remove nans
		points = points[~np.isnan(points).any(axis=1)]
		# transform points to tsdf_origin frame
		# transform to translation and rotation
		trans = np.array([table_to_pcl.translation.x, table_to_pcl.translation.y, table_to_pcl.translation.z])
		rot = np.array([table_to_pcl.rotation.x, table_to_pcl.rotation.y, table_to_pcl.rotation.z, table_to_pcl.rotation.w])
		table_to_pcl_tf = Transform(rotation=Rotation.from_quat(rot), translation=trans)
		points = table_to_pcl_tf.transform_point(points)


		# crop point cloud
		# only z axis
		points = points[(points[:,2] > 0) & (points[:,2] < self.size)]
		# x and y axis
		points = points[(points[:,0] > -self.size) & (points[:,0] < 2*self.size) & (points[:,1] > -self.size) & (points[:,1] < 2*self.size) & (points[:,2] > -self.size) & (points[:,2] < 2*self.size)]
		# points = points[(points[:,0] > 0) & (points[:,0] < self.size) & (points[:,1] > 0) & (points[:,1] < self.size) & (points[:,2] > 0) & (points[:,2] < self.size)]
		
		# convert to open3d point cloud
		pcd = o3d.geometry.PointCloud()
		pcd.points = o3d.utility.Vector3dVector(points)
		self.pcd = pcd

		# # Debug: viz open3d pcl
		# origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
		# o3d.visualization.draw_geometries([pcd, origin])

	def process_depth_img(self, ros_depth_image=None):
		# get the camera extrinsic matrix for depth image
		while not rospy.is_shutdown():
			try:
				if self.camera_type == 'zed':
					cam_to_table_tf = self.tf_buffer.lookup_transform('zed2_left_camera_optical_frame', self.grasp_frame_name, rospy.Time(0))
				elif self.camera_type == 'xtion':
					cam_to_table_tf = self.tf_buffer.lookup_transform('xtion_rgb_optical_frame', self.grasp_frame_name, rospy.Time(0))
			except:
				print("[WARNING: Could not find some TFs. Check TF and obj names]")
				# rate.sleep()
				continue
			break
		camera_extrinsic_mat = ros_numpy.numpify(cam_to_table_tf.transform)
		self.camera_extrinsic = Transform.from_matrix(camera_extrinsic_mat)
		
		# convert ros depth image rect to numpy array
		depth_img = ros_numpy.numpify(ros_depth_image)
		# remove nans?
		# depth_img = depth_img[~np.isnan(depth_img).any(axis=1)]
		self.depth_img = depth_img.astype(np.float32)

		# # Debug: viz depth img
		# import matplotlib.pyplot as plt
		# plt.imshow(depth_img)
		# plt.show()

	def get_tsdf_and_down_pcd(self):
		# make tsdf
		tsdf = TSDFVolume(self.size, self.tsdf_res)
		# integrate
		tsdf.integrate(self.depth_img, self.camera_intrinsic, self.camera_extrinsic)

		# Debug: viz tsdf
		# pcl_tsdf = tsdf.get_cloud()
		# pcl_tsdf.colors = o3d.utility.Vector3dVector(np.tile(np.array([0, 0, 0]), (np.asarray(pcl_tsdf.points).shape[0], 1)))
		# o3d.visualization.draw_geometries([pcl_tsdf, self.pcd])

		pcd_down = self.pcd.voxel_down_sample(voxel_size=self.downsampl_size)
		# Optional: crop to scene size
		# pcd_down = pcd_down.crop(o3d.geometry.AxisAlignedBoundingBox(min_bound=(0, 0, 0), max_bound=(self.size, self.size, self.size)))

		# Debug: viz pcd and down pcd
		# pcd_down.colors = o3d.utility.Vector3dVector(np.tile(np.array([0, 0, 0]), (np.asarray(pcd_down.points).shape[0], 1)))
		# o3d.visualization.draw_geometries([pcd_down, self.pcd])

		return tsdf, pcd_down

	def get_grasps(self):
		
		# Get the point cloud and depth image from their respective topics
		ros_point_cloud = rospy.wait_for_message(self.pcl_topic_name, PointCloud2, timeout=10)
		ros_depth_image = rospy.wait_for_message(self.depth_topic_name, Image, timeout=10)
		self.process_pcl(ros_point_cloud)
		self.process_depth_img(ros_depth_image)
		tsdf, pc = self.get_tsdf_and_down_pcd()
		
		# build state
		state = argparse.Namespace(tsdf=tsdf, pc=pc)
		# TEMP: Explicit running viz of the scene point clouds and meshes
		self.o3d_vis = o3d.visualization.Visualizer()
		self.o3d_vis.create_window(width=1920, height=1016)
		if self.o3d_vis is not None:
			# Running viz of the scene point clouds and meshes
			self.o3d_vis.clear_geometries() # clear previous geometries
			state.pc.colors = o3d.utility.Vector3dVector(np.tile(np.array([0, 0, 0]), (np.asarray(state.pc.points).shape[0], 1)))
			self.o3d_vis.add_geometry(state.pc, reset_bounding_box=True)			
			self.o3d_vis.poll_events()
			self.o3d_vis.update_renderer()			
			# Reset view
			ctr = grasp_gen.o3d_vis.get_view_control()
			parameters = o3d.io.read_pinhole_camera_parameters(os.path.dirname(os.path.abspath(__file__))+"/../neugraspnet_repo/ScreenCamera_2023-11-30-11-20-31.json")
			ctr.convert_from_pinhole_camera_parameters(parameters)
			for i in range(50):
				grasp_gen.o3d_vis.poll_events()
				grasp_gen.o3d_vis.update_renderer()
			
		# dummy sim just for parameters. TODO: clean this up
		sim = ClutterRemovalSim('pile', 'pile/test', gripper_type='robotiq', gui=False, data_root=os.path.dirname(os.path.abspath(__file__))+'/../neugraspnet_repo/')

		grasps, scores, _, _, _ = self.grasp_planner(state, sim=sim, o3d_vis=self.o3d_vis)

		# convert grasps to standard XYZ co-ordinate frame (neugraspnet and vgn use a different grasp convention)
		grasps_final = []
		for grasp in grasps:
			grasp_center = grasp.pose.translation
			grasp_frame_rot =  grasp.pose.rotation * Rotation.from_euler('Y', np.pi/2) * Rotation.from_euler('Z', np.pi)
			grasp_tf = Transform(grasp_frame_rot, grasp_center).as_matrix()
			grasps_final.append(grasp_tf)
		
		# Optional: Score grasps using a reachability metric
		if self.use_reachability:
			scores = self.get_reachability_scores(grasps_final, arm='right')
			# Sort grasps by score
			sorted_grasps = [grasps_final[i] for i in reversed(np.argsort(scores))]
			sorted_scores = [k for k in reversed(np.argsort(scores))]
			grasps_final = sorted_grasps
			scores = sorted_scores
	
		return grasps_final, scores
	
	def get_reachability_scores(self, grasp_tfs, arm='right'):
		# Get reachability scores for grasps
		# (based on https://github.com/iROSA-lab/sampled_reachability_maps)
		
		scores = np.zeros((len(grasp_tfs)))
		
		if arm == 'left':
			raise NotImplementedError
		elif arm == 'right':
			reach_map = self.right_reachability_map
		grasp_poses = np.zeros((len(grasp_tfs), 6))
		for id, grasp_tf in enumerate(grasp_tfs):
			# Get the 6D euler grasp pose	
			grasp_poses[id] = np.hstack(( grasp_tf[:3,-1], Rotation.from_matrix(grasp_tf[:3,:3]).as_euler('XYZ') )) # INTRINSIC XYZ

		# if arm == 'left':
			# min_y, max_y, = (-0.6, 1.35)
		# else:
		min_y, max_y, = (-1.35, 0.6)
		min_x, max_x, = (-1.2, 1.2)
		min_z, max_z, = (-0.35, 2.1)
		min_roll, max_roll, = (-np.pi, np.pi)
		min_pitch, max_pitch, = (-np.pi / 2, np.pi / 2)
		min_yaw, max_yaw, = (-np.pi, np.pi)
		cartesian_res = 0.05
		angular_res = np.pi / 8

		# Mask valid grasp_poses that are inside the min-max xyz bounds of the reachability map
		mask = np.logical_and.reduce((grasp_poses[:,0] > min_x, grasp_poses[:,0] < max_x,
									grasp_poses[:,1] > min_y, grasp_poses[:,1] < max_y,
									grasp_poses[:,2] > min_z, grasp_poses[:,2] < max_z))

		x_bins = np.ceil((max_x - min_x) / cartesian_res)
		y_bins = np.ceil((max_y - min_y) / cartesian_res)
		z_bins = np.ceil((max_z - min_z) / cartesian_res)
		roll_bins = np.ceil((max_roll - min_roll) / angular_res)
		pitch_bins = np.ceil((max_pitch - min_pitch) / angular_res)
		yaw_bins = np.ceil((max_yaw - min_yaw) / angular_res)

		# Define the offset values for indexing the map
		x_ind_offset = y_bins * z_bins * roll_bins * pitch_bins * yaw_bins
		y_ind_offset = z_bins * roll_bins * pitch_bins * yaw_bins
		z_ind_offset = roll_bins * pitch_bins * yaw_bins
		roll_ind_offset = pitch_bins * yaw_bins
		pitch_ind_offset = yaw_bins
		yaw_ind_offset = 1

		# Convert the input pose to voxel coordinates
		x_idx = (np.floor((grasp_poses[mask, 0] - min_x) / cartesian_res)).astype(int)
		y_idx = (np.floor((grasp_poses[mask, 1] - min_y) / cartesian_res)).astype(int)
		z_idx = (np.floor((grasp_poses[mask, 2] - min_z) / cartesian_res)).astype(int)
		roll_idx = (np.floor((grasp_poses[mask, 3] - min_roll) / angular_res)).astype(int)
		pitch_idx = (np.floor((grasp_poses[mask, 4] - min_pitch) / angular_res)).astype(int)
		yaw_idx = (np.floor((grasp_poses[mask, 5] - min_yaw) / angular_res)).astype(int)
		# Handle edge cases of discretization (angles can especially cause issues if values contain both ends [-pi, pi] which we don't want
		roll_idx = np.clip(roll_idx, 0, roll_bins-1)
		pitch_idx = np.clip(pitch_idx, 0, pitch_bins-1)
		yaw_idx = np.clip(yaw_idx, 0, yaw_bins-1)

		# Compute the index in the reachability map array
		map_idx = x_idx * x_ind_offset + y_idx * y_ind_offset + z_idx * z_ind_offset + roll_idx  \
		* roll_ind_offset + pitch_idx * pitch_ind_offset + yaw_idx * yaw_ind_offset

		# Get the score from the score map array
		scores[mask] = reach_map[map_idx.astype(int),-1] # -1 is the score index

		return scores

	def grasps_srv_callback(self, req):

		grasps, scores = self.get_grasps()
		# Make grasp pose array message
		grasp_pose_array = PoseArray()
		grasp_pose_array.header.frame_id = grasp_gen.grasp_frame_name
		grasp_pose_array.header.stamp = rospy.Time.now()

		if len(grasps) > 0:
			print("Found {} grasps with scores: {}".format(len(grasps), scores))
			# We also publish the grasps to the appropriate topic
			rospy.loginfo("Publishing grasps to topic: {}".format(grasp_gen.grasp_topic_name))
			grasp_pose_array.poses = [Pose() for _ in range(len(grasps))]
			for i, grasp in enumerate(grasps):
				grasp_pose_array.poses[i].position.x = grasp[0,3]
				grasp_pose_array.poses[i].position.y = grasp[1,3]
				grasp_pose_array.poses[i].position.z = grasp[2,3]
				rot_quatt = quaternion_from_matrix(grasp)
				grasp_pose_array.poses[i].orientation.x = rot_quatt[0]
				grasp_pose_array.poses[i].orientation.y = rot_quatt[1]
				grasp_pose_array.poses[i].orientation.z = rot_quatt[2]
				grasp_pose_array.poses[i].orientation.w = rot_quatt[3]
			
			# Publish
			grasp_gen.grasp_pub.publish(grasp_pose_array)
		else:
			print("[No grasps found. Publishing empty pose array]")
			# Publish empty pose array
			grasp_gen.grasp_pub.publish(grasp_pose_array)
		
		return EmptyResponse()


# Run this as a script
rospy.init_node('grasp_generator')
grasp_gen = GraspGenerator(net_type='neu_grasp_pn_affnet', camera_type='zed')

# Optional visualization
visualize = False
if visualize:
	# Running viz of the scene point clouds and meshes
	grasp_gen.o3d_vis = o3d.visualization.Visualizer()
	grasp_gen.o3d_vis.create_window(width=1920, height=1016)
	
	# # DEBUG: run grasp generator
	# while(1):
	# 	grasps, scores = grasp_gen.get_grasps()
	# 	print("Found {} grasps with scores: {}".format(len(grasps), scores))
	# 	import pdb; pdb.set_trace()

	# # Optional: Keep running viz (Slows down everything else)
	# while not rospy.is_shutdown():
	# 	grasp_gen.o3d_vis.poll_events()
	# 	grasp_gen.o3d_vis.update_renderer()
	# rate.sleep()
	rospy.spin()
else:
	grasp_gen.o3d_vis = None
	rospy.spin()