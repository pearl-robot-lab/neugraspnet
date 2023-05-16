from pathlib import Path
import argparse
import rospy
# import geometry_msgs
from sensor_msgs.msg import PointCloud2, Image
from geometry_msgs.msg import PoseArray, Pose
# import sensor_msgs.point_cloud2 as pc2
# from std_msgs.msg import Header
from tf2_sensor_msgs.tf2_sensor_msgs import do_transform_cloud
from tf.transformations import euler_from_quaternion, quaternion_from_euler, quaternion_from_matrix
import ros_numpy
# import tf
import tf2_ros
import numpy as np
import open3d as o3d

# neugraspnet repo imports
from vgn.perception import CameraIntrinsic, TSDFVolume
from vgn.utils.transform import Rotation, Transform
from vgn.detection_implicit import VGNImplicit
from vgn.simulation import ClutterRemovalSim

################################################ 
# Perception pipeline
################################################

# SCENE_SIZE = 0.3 # size of the scene in meters

class GraspGenerator:
	def __init__(self, grasp_frame_name='grasp_origin', grasp_topic_name="/generated_grasps", camera_type='zed',
	      		 net_type='neu_grasp_pn_deeper', net_path=None, tsdf_res=64, scene_size=0.3, downsampl_size=0.005):
		
		# Publish grasps to topic
		self.grasp_topic_name = grasp_topic_name
		self.grasp_pub = rospy.Publisher(self.grasp_topic_name, PoseArray, queue_size=1, latch=False)

		# grasps fill be in this frame
		self.grasp_frame_name = grasp_frame_name

		# Setup camera
		self.camera_type = camera_type
		if self.camera_type == 'zed':
			# get the pointcloud from the topic
			self.pcl_topic_name = "/zed2/zed_node/point_cloud/cloud_registered"
			self.depth_topic_name = "/zed2/zed_node/depth/depth_registered"
			# camera intrinsic parameters (TODO: Get this from /zed2/zed_node/depth/camera_info topic)
			self.camera_intrinsic = CameraIntrinsic(width=640, height=360,
													fx=262.1390075683594, fy=262.1390075683594,
													cx=311.67730712890625, cy=185.56422424316406)
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
			net_path = Path('/neugraspnet/neugraspnet_repo/data/best_real_robot_runs/PACKED_best_neural_grasp_neu_grasp_pn_deeper_val_acc=0.9040.pt')

		self.size = scene_size
		self.tsdf_res = tsdf_res
		self.downsampl_size = downsampl_size

		self.setup_grasp_planner(model=net_path, type=net_type, qual_th=0.5, force=False, seen_pc_only=False)

	def setup_grasp_planner(self, model, type, qual_th=0.5, force=False, seen_pc_only=False, vis_mesh=False):

		self.grasp_planner = VGNImplicit(model,
                                    type,
                                    qual_th=qual_th,
                                    force_detection=force,
                                    seen_pc_only=seen_pc_only,
                                    resolution=self.tsdf_res,
                                    visualize=vis_mesh)
		
	def process_pcl(self, ros_point_cloud=None):
		# transform the point cloud to the grasp_origin frame
		while not rospy.is_shutdown():
			try:
				if self.camera_type == 'zed':
					table_to_pcl_tf = self.tf_buffer.lookup_transform(self.grasp_frame_name, 'zed2_left_camera_frame', rospy.Time(0))
				elif self.camera_type == 'xtion':
					table_to_pcl_tf = self.tf_buffer.lookup_transform(self.grasp_frame_name, 'xtion_rgb_optical_frame', rospy.Time(0))				
			except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
				print("[WARNING: Could not find some TFs. Check TF and obj names]")
				# print(e)
				# rate.sleep()
				continue
			break
	
		cloud_transformed = do_transform_cloud(ros_point_cloud, table_to_pcl_tf)
		# convert to numpy array
		pc = ros_numpy.numpify(cloud_transformed)
		points=np.zeros((pc.shape[0],3))
		points[:,0]=pc['x']
		points[:,1]=pc['y']
		points[:,2]=pc['z']
		# remove nans
		points = points[~np.isnan(points).any(axis=1)]
		# crop point cloud
		points = points[(points[:,0] > 0) & (points[:,0] < self.size) & (points[:,1] > 0) & (points[:,1] < self.size) & (points[:,2] > 0) & (points[:,2] < self.size)]
		
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

		# Debug: viz pcd and down pcd
		# pcd_down.colors = o3d.utility.Vector3dVector(np.tile(np.array([0, 0, 0]), (np.asarray(pcd_down.points).shape[0], 1)))
		# o3d.visualization.draw_geometries([pcd_down, self.pcd])

		return tsdf, pcd_down

	def get_grasps(self, visualize=True):
		
		# Get the point cloud and depth image from their respective topics
		ros_point_cloud = rospy.wait_for_message(self.pcl_topic_name, PointCloud2, timeout=10)
		ros_depth_image = rospy.wait_for_message(self.depth_topic_name, Image, timeout=10)
		self.process_pcl(ros_point_cloud)
		self.process_depth_img(ros_depth_image)
		tsdf, pc = self.get_tsdf_and_down_pcd()
		
		# build state
		state = argparse.Namespace(tsdf=tsdf, pc=pc)

		if visualize:
			# Running viz of the scene point clouds and meshes
			o3d_vis = o3d.visualization.Visualizer()
			o3d_vis.create_window(width=1863, height=1052)
			state.pc.colors = o3d.utility.Vector3dVector(np.tile(np.array([0, 0, 0]), (np.asarray(state.pc.points).shape[0], 1)))
			o3d_vis.add_geometry(state.pc, reset_bounding_box=True)
			
			# Reset view
			ctr = o3d_vis.get_view_control()
			parameters = o3d.io.read_pinhole_camera_parameters("/neugraspnet/neugraspnet_repo/o3d_vis_screen_camera_params2.json")
			ctr.convert_from_pinhole_camera_parameters(parameters)
			for i in range(50):
				o3d_vis.poll_events()
				o3d_vis.update_renderer()
		else:
			o3d_vis = None
        
		# dummy sim just for parameters. TODO: clean this up
		sim = ClutterRemovalSim('pile', 'pile/test', gripper_type='robotiq', gui=False, data_root='/neugraspnet/neugraspnet_repo/')

		grasps, scores, _, _ = self.grasp_planner(state, sim=sim, o3d_vis=o3d_vis)

		# convert grasps to standard XYZ co-ordinate frame (neugraspnet and vgn use a different grasp convention)
		grasps_final = []
		for grasp in grasps:
			grasp_center = grasp.pose.translation
			grasp_frame_rot =  grasp.pose.rotation * Rotation.from_euler('Y', np.pi/2) * Rotation.from_euler('Z', np.pi)
			grasp_tf = Transform(grasp_frame_rot, grasp_center).as_matrix()
			grasps_final.append(grasp_tf)
	
		return grasps_final, scores, o3d_vis

# # Run this as a script
rospy.init_node('grasp_generator')
grasp_gen = GraspGenerator(camera_type='zed')
grasps, scores, o3d_vis = grasp_gen.get_grasps(visualize=True)
if len(grasps) > 0:
	print("Found {} grasps with scores: {}".format(len(grasps), scores))
	# We publish the grasps to the appropriate topic
	rospy.loginfo("Publishing grasps to topic: {}".format(grasp_gen.grasp_topic_name))
	# Make grasp pose array message
	grasp_pose_array = PoseArray()
	grasp_pose_array.header.frame_id = grasp_gen.grasp_frame_name
	grasp_pose_array.header.stamp = rospy.Time.now()
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

	if o3d_vis:
		# Run viz
		while not rospy.is_shutdown():
			o3d_vis.poll_events()
			o3d_vis.update_renderer()
	rospy.spin()
else:
	print("[No grasps found]")