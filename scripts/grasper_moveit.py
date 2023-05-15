# from pathlib import Path
# import argparse
import rospy
import geometry_msgs
from sensor_msgs.msg import PointCloud2
from std_srvs.srv import Empty, EmptyRequest
# import sensor_msgs.point_cloud2 as pc2
# from std_msgs.msg import Header
# from tf2_sensor_msgs.tf2_sensor_msgs import do_transform_cloud
# import ros_numpy
# import tf
# import tf2_ros
import numpy as np
# import open3d as o3d

from grasp_generator import GraspGenerator

# tiago dual pick place
# from tiago_dual_pick_place.srv import PickPlaceSimple
from moveit_msgs.msg import MoveItErrorCodes
from tf.transformations import euler_from_quaternion, quaternion_from_euler, quaternion_from_matrix

class GrasperMoveit:
	def __init__(self, camera_type='zed', octomap_topic_name="/filtered_points_for_mapping", arm="left"):
		
		# To control when moveit should and should not use the camera depth, 
		# we set up a subscriber and publisher that will forward information to the moveit octomap server
		self.camera_type = camera_type
		if self.camera_type == 'zed':
			# get the pointcloud from the topic
			self.pcl_topic_name = "/zed2/zed_node/point_cloud/cloud_registered"
		elif self.camera_type == 'xtion':
			self.pcl_topic_name = "/xtion/depth_registered/points"
		else:
			raise ValueError('Invalid camera type. Choose either zed or xtion.')
		# start the subscriber with an explicit call
		# self.pcl_sub = rospy.Subscriber(self.pcl_topic_name, PointCloud2, self.forward_pcl, queue_size=1)
		
		self.octomap_topic_name = octomap_topic_name
		self.octomap_pub = rospy.Publisher(self.octomap_topic_name, PointCloud2, queue_size=1, latch=True)

		self.clear_octomap_srv = rospy.ServiceProxy('/clear_octomap', Empty)
		self.clear_octomap_srv.wait_for_service()
		rospy.loginfo("[Connected to octomap clear service]")
		
		self.current_arm = arm # 'left' or 'right'

		# set up the grasp generator
		self.grasp_gen = GraspGenerator(camera_type=self.camera_type)

	def forward_pcl(self, ros_point_cloud):
		# forward the point cloud to the moveit octomap server topic (defined in the sensor yaml file)
		self.octomap_pub.publish(ros_point_cloud)
	
	def stop_mapping(self):
		# stop forwarding the point cloud to the moveit octomap server topic
		self.pcl_sub.unregister()
	
	def start_mapping(self):
		# Clear existing octomap
		rospy.loginfo("Clearing octomap")
		self.clear_octomap_srv.call(EmptyRequest())
		# start forwarding the point cloud to the moveit octomap server topic
		self.pcl_sub = rospy.Subscriber(self.pcl_topic_name, PointCloud2, self.forward_pcl, queue_size=1)


# Run as a script
rospy.init_node('grasper_moveit')

grasper = GrasperMoveit()
grasper.start_mapping()
rospy.sleep(1)
# grasper.stop_mapping()
grasps, scores = grasper.grasp_gen.get_grasps(visualize=True)
# Send grasp command to moveit pick and place pipeline
grasp_pose_pub = rospy.Publisher('/grasp/pose', geometry_msgs.msg.PoseStamped, queue_size=1, latch=True)
grasp_pose_msg = geometry_msgs.msg.PoseStamped()
grasp_pose_msg.header.stamp = rospy.Time.now()
grasp_pose_msg.header.frame_id = 'base_footprint'
grasp_pose_msg.pose.position.x = 0.5
grasp_pose_msg.pose.position.y = -0.5
grasp_pose_msg.pose.position.z = 0.75
grasp_pose_msg.pose.orientation.x = 0.0
grasp_pose_msg.pose.orientation.y = 0.0
grasp_pose_msg.pose.orientation.z = 0.0
grasp_pose_msg.pose.orientation.w = 1.0
grasp_pose_pub.publish(grasp_pose_msg)
# rospy.loginfo("[Sending /pick command...]")
# # call /pick rosservice
# grasper.current_arm = 'left'
# pick_result = rospy.ServiceProxy('/pick', PickPlaceSimple)(grasper.current_arm)
rospy.spin()
