# from pathlib import Path
# import argparse
import rospy
from geometry_msgs.msg import PoseArray, Pose, PoseStamped
from sensor_msgs.msg import PointCloud2
from std_srvs.srv import Empty, EmptyRequest
from sensor_msgs import point_cloud2
from tf2_sensor_msgs.tf2_sensor_msgs import do_transform_cloud
import ros_numpy
# import tf
import tf2_ros
import numpy as np
# from std_msgs.msg import Header

# tiago dual pick place
from tiago_dual_pick_place.srv import PickPlaceSimple
from moveit_msgs.msg import MoveItErrorCodes
from tf.transformations import euler_from_quaternion, quaternion_from_euler, quaternion_from_matrix

from play_motion_msgs.msg import PlayMotionAction, PlayMotionGoal
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from actionlib import SimpleActionClient
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal, MoveBaseResult, MoveBaseActionResult

# Reads the generated grasps and sends them to the pick place service. Also activates and deactivates mapping

class Grasper:
	def __init__(self, grasps_sub_topic="/generated_grasps", grasp_srv_name='get_grasps', grasp_pub_topic="/grasp/pose", grasp_frame_name='grasp_origin',
	      		camera_type='zed', octomap_topic_name="/filtered_points_for_mapping", filter_pcl=True, arm="right"):
		
		self.grasps_sub_topic = grasps_sub_topic

		self.grasp_pub_topic = grasp_pub_topic
		self.grasp_pub = rospy.Publisher(self.grasp_pub_topic, PoseStamped, queue_size=1, latch=True)
		self.grasp_srv_name = grasp_srv_name # when using the grasp service, we don't need to subscribe to the grasps topic

		self.grasp_frame_name = grasp_frame_name
		# tf listener to get the point cloud in the correct frame
		self.tf_buffer = tf2_ros.Buffer()
		self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

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

		# We can optionally filter the point cloud to remove points very close to the grasp
		self.filter_pcl = filter_pcl
		self.filter_radius = 0.06 # 6cm
		
		self.octomap_topic_name = octomap_topic_name
		self.octomap_pub = rospy.Publisher(self.octomap_topic_name, PointCloud2, queue_size=1, latch=True)

		self.clear_octomap_srv = rospy.ServiceProxy('/clear_octomap', Empty)
		self.clear_octomap_srv.wait_for_service()
		rospy.loginfo("[Connected to octomap clear service]")
		
		self.current_arm = arm # 'left' or 'right'

		self.grasp_pose_array = None
		self.current_grasp_pose = None

	def forward_pcl(self, ros_point_cloud):
		# forward the point cloud to the moveit octomap server topic (defined in the sensor yaml file)

		# Optional: filter the point cloud to remove points very close to the grasp
		if self.filter_pcl:
			assert self.current_grasp_pose is not None, "No current grasp pose set. Cannot filter point cloud."

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

			# transform the point cloud to the grasp_origin frame
			cloud_transformed = do_transform_cloud(ros_point_cloud, table_to_pcl_tf)
			# convert to numpy array
			pc = ros_numpy.numpify(cloud_transformed)
			points=np.zeros((pc.shape[0],3))
			points[:,0]=pc['x']
			points[:,1]=pc['y']
			points[:,2]=pc['z']
			# remove nans
			points = points[~np.isnan(points).any(axis=1)]
			# crop point cloud?
			# points = points[(points[:,0] > 0) & (points[:,0] < self.size) & (points[:,1] > 0) & (points[:,1] < self.size) & (points[:,2] > 0) & (points[:,2] < self.size)]
			# remove points very close to the grasp
			curr_grasp_center = np.array([self.current_grasp_pose.position.x, self.current_grasp_pose.position.y, self.current_grasp_pose.position.z])
			# print("Points before filtering: ", points.shape)
			points = points[np.linalg.norm(points - curr_grasp_center, axis=1) > self.filter_radius]
			# print("Points after filtering: ", points.shape)
			
			# convert back to ros point cloud		
			ros_point_cloud.header.stamp = rospy.Time.now()
			ros_point_cloud.header.frame_id = self.grasp_frame_name
			ros_point_cloud = point_cloud2.create_cloud(ros_point_cloud.header, ros_point_cloud.fields[0:3], points) # Using only xyz (no rgb)

		self.octomap_pub.publish(ros_point_cloud)
	
	def stop_mapping(self):
		# stop forwarding the point cloud to the moveit octomap server topic
		self.pcl_sub.unregister()
	
	def start_mapping(self):
		# start forwarding the point cloud to the moveit octomap server topic
		self.pcl_sub = rospy.Subscriber(self.pcl_topic_name, PointCloud2, self.forward_pcl, queue_size=1)
		# # Clear existing octomap here?
		# rospy.sleep(1.0)
		# rospy.loginfo("Clearing octomap")
		# self.clear_octomap_srv.call(EmptyRequest())
		


def lift_torso(torso_cmd):
	rospy.loginfo("Moving torso up")
	jt = JointTrajectory()
	jt.joint_names = ['torso_lift_joint']
	jtp = JointTrajectoryPoint()
	jtp.positions = [0.35]
	jtp.time_from_start = rospy.Duration(1.0)
	jt.points.append(jtp)
	torso_cmd.publish(jt)

def lower_torso(torso_cmd):
	rospy.loginfo("Moving torso down")
	jt = JointTrajectory()
	jt.joint_names = ['torso_lift_joint']
	jtp = JointTrajectoryPoint()
	jtp.positions = [0.0]
	jtp.time_from_start = rospy.Duration(1.0)
	jt.points.append(jtp)
	torso_cmd.publish(jt)

def move_torso(torso_cmd, torso_pos):
	rospy.loginfo("Moving torso to custom position")
	jt = JointTrajectory()
	jt.joint_names = ['torso_lift_joint']
	jtp = JointTrajectoryPoint()
	jtp.positions = [torso_pos]
	jtp.time_from_start = rospy.Duration(1.0)
	jt.points.append(jtp)
	torso_cmd.publish(jt)

def move_head(head_cmd, join_1_pos, joint_2_pos=None):
	rospy.loginfo("Moving head...")
	jt = JointTrajectory()
	jt.joint_names = ['head_1_joint', 'head_2_joint']
	jtp = JointTrajectoryPoint()
	if joint_2_pos is None:
		jtp.positions = [join_1_pos, -0.98]
	else:
		jtp.positions = [join_1_pos, joint_2_pos]
	jtp.time_from_start = rospy.Duration(1.0)
	jt.points.append(jtp)
	head_cmd.publish(jt)

def move_base_in_tab_frame(move_base_ac, xyz_quat):
	# Move base to xyz, quat
	rospy.loginfo("Moving base to xyz, quat")
	move_goal = MoveBaseGoal()
	move_goal.target_pose.header.stamp = rospy.Time.now()
	move_goal.target_pose.header.frame_id = "table"
	move_goal.target_pose.pose.position.x = xyz_quat[0]
	move_goal.target_pose.pose.position.y = xyz_quat[1]
	move_goal.target_pose.pose.position.z = xyz_quat[2]
	move_goal.target_pose.pose.orientation.x = xyz_quat[3]
	move_goal.target_pose.pose.orientation.y = xyz_quat[4]
	move_goal.target_pose.pose.orientation.z = xyz_quat[5]
	move_goal.target_pose.pose.orientation.w = xyz_quat[6]
	move_base_ac.send_goal_and_wait(move_goal)

	return move_base_ac.get_state() # result


# Run as a script
rospy.init_node('grasper_node')

# Optional: Use base placement and head randomization
randomize_views = True
base_place_final_right = [-0.645, -0.152, -0.463, -0.000, -0.005,  0.525, 0.851] # x,y,z,quat
base_place_final_left  = [-0.653, -0.029, -0.460,  0.005,  0.001, -0.600, 0.800] # x,y,z,quat

base_placements_right = [[-0.645, -0.152, -0.463, -0.000, -0.005,  0.525, 0.851], # x,y,z,quat
						 [-0.659, -0.163, -0.461,  0.001, -0.003,  0.352, 0.936]] # x,y,z,quat
base_placements_left  = [[-0.707,  0.131, -0.459,  0.001,  0.000, -0.275, 0.961], # x,y,z,quat
						 [-0.689,  0.169, -0.460,  0.003, -0.000, -0.498, 0.867]] # x,y,z,quat

# Move head accordingly (head_joint_1 only)
head_placements_right = [-0.61, -0.60] # joint 1 only
head_placements_left  = [0.36, 0.35] # joint 1 only
head_joint2_range     = [-0.98, -0.7] # joint 2 only # small range to ensure we look down

# Create move base action client
rospy.loginfo("Waiting for /move_base AS")
move_base_ac = SimpleActionClient('/move_base', MoveBaseAction)
if not move_base_ac.wait_for_server(rospy.Duration(15)):
	rospy.logerr("Could not connect to /move_base AS")
	exit()


# Use gripper grasp service:
gripper_right_grasp_srv = rospy.ServiceProxy('/gripper_right_controller/grasp', Empty)
gripper_left_grasp_srv = rospy.ServiceProxy('/gripper_left_controller/grasp', Empty)
# Set low pressure
left_pressure = 0.05
right_pressure = 0.05
rospy.set_param('/gripper_left_grasping/gripper_left_grasp_service/pressure', left_pressure)
rospy.set_param('/gripper_right_grasping/gripper_right_grasp_service/pressure', right_pressure)

# Optional: Use playmotion
rospy.loginfo("[Waiting for '/play_motion' ActionServer...]")
play_m_ac = SimpleActionClient('/play_motion', PlayMotionAction)
# Optional: torso and head commanders
torso_cmd = rospy.Publisher('/torso_controller/command', JointTrajectory, queue_size=1, latch=True)
head_cmd = rospy.Publisher('/head_controller/command', JointTrajectory, queue_size=1, latch=True)


grasper = Grasper()

if grasper.current_arm == 'right':
	grasper.gripper_grasp_srv = gripper_right_grasp_srv
elif grasper.current_arm == 'left':
	grasper.gripper_grasp_srv = gripper_left_grasp_srv

# Setup grasp service to trigger grasp generation using service call:
rospy.loginfo("[Waiting for grasp service: %s...]", grasper.grasp_srv_name)
rospy.wait_for_service(grasper.grasp_srv_name)
grasp_srv = rospy.ServiceProxy(grasper.grasp_srv_name, Empty)


# Optional: tuck in arms to pregrasp position
# tuck in arm
rospy.loginfo("Preparing arm")
pmg = PlayMotionGoal()
pmg.motion_name = 'prep_l'
pmg.skip_planning = False
import pdb; pdb.set_trace()
play_m_ac.send_goal_and_wait(pmg)
# rospy.sleep(2.0)
# tuck in arm
rospy.loginfo("Preparing arm")
pmg = PlayMotionGoal()
pmg.motion_name = 'prep_r'
pmg.skip_planning = False
play_m_ac.send_goal_and_wait(pmg)
# rospy.sleep(2.0)

# while scene is not emptied:
while not rospy.is_shutdown():
	success = False

	if randomize_views:
		
		# Choose left or right arm
		# grasper.current_arm = np.random.choice(['right', 'left'])
		if grasper.current_arm == 'right':
			grasper.current_arm = 'left'
		else:
			grasper.current_arm = 'right'
		rospy.loginfo("Using [%s] arm", grasper.current_arm)
		
		if grasper.current_arm == 'right':
			# Move base to a random position
			rospy.loginfo("Moving base to a random position")
			place_index = np.random.randint(0, len(base_placements_right))
			move_base_in_tab_frame(move_base_ac, base_placements_right[place_index])
			# Move head
			random_head_2 = np.random.uniform(head_joint2_range[0], head_joint2_range[1])
			move_head(head_cmd, head_placements_right[place_index], random_head_2)
			# randomize torso
			random_torso_pos = np.random.uniform(0.0, 0.15)
			move_torso(torso_cmd, random_torso_pos)
			rospy.sleep(2.0)
			
			# Use correct gripper
			grasper.gripper_grasp_srv = gripper_right_grasp_srv
		elif grasper.current_arm == 'left':
			# Move base to a random position
			rospy.loginfo("Moving base to a random position")
			place_index = np.random.randint(0, len(base_placements_left))
			move_base_in_tab_frame(move_base_ac, base_placements_left[place_index])
			# Move head
			random_head_2 = np.random.uniform(head_joint2_range[0], head_joint2_range[1])
			move_head(head_cmd, head_placements_left[place_index], random_head_2)
			# randomize torso
			random_torso_pos = np.random.uniform(0.0, 0.15)
			move_torso(torso_cmd, random_torso_pos)
			rospy.sleep(2.0)
			# Use correct gripper
			grasper.gripper_grasp_srv = gripper_left_grasp_srv
	else:
		pass
	# Go to home position?
	# Raise arm
	# rospy.loginfo("Moving arm to a safe pose")
	# pmg = PlayMotionGoal()
	# pmg.motion_name = 'pick_final_pose_' + grasper.current_arm[0]  # take first char
	# pmg.skip_planning = False
	# rospy.loginfo("Sending final arm command...")
	# play_m_ac.send_goal_and_wait(pmg)
	# rospy.sleep(1.0)
	# Move torso down (TODO: move to randomized torso and head position)
	# move_torso(torso_cmd, 0.1)
	# rospy.sleep(3.0)

	# Get grasps from the grasp generator
	rospy.loginfo("[Triggering grasp generation...]")
	grasp_srv.call(EmptyRequest())
	# Get from topic:
	rospy.loginfo("[Waiting for grasps published on topic: %s...]", grasper.grasps_sub_topic)
	grasper.grasp_pose_array = rospy.wait_for_message(grasper.grasps_sub_topic, PoseArray, timeout=3)
	rospy.loginfo("[Received %d grasps]", len(grasper.grasp_pose_array.poses))

	# clean and setup the octomap:
	# grasper.current_grasp_pose = grasper.grasp_pose_array.poses[0]
	# grasper.start_mapping()
	# rospy.sleep(1.0)
	# grasper.stop_mapping()

	# We assume the grasp poses are sorted by score, so we start from the topic
	for i, grasp_pose in enumerate(grasper.grasp_pose_array.poses):
		rospy.loginfo("[Trying grasp %d/%d]", i+1, len(grasper.grasp_pose_array.poses))
		# Set current in the grasper object
		grasper.current_grasp_pose = grasp_pose
		# Start mapping with filtering enabled. This should filter out the points very close to the CURRENT grasp
		grasper.start_mapping()
		rospy.loginfo("Clearing octomap")
		grasper.clear_octomap_srv.call(EmptyRequest())
		rospy.sleep(1.0)
		grasper.stop_mapping()
		
		# publish this grasp pose to the topic that pick-place pipeline expects
		grasp_pub_msg = PoseStamped()
		grasp_pub_msg.header.stamp = rospy.Time.now()
		grasp_pub_msg.header.frame_id = grasper.grasp_frame_name
		grasp_pub_msg.pose = grasp_pose
		grasper.grasp_pub.publish(grasp_pub_msg)

		# Now call the pick service and wait for the result
		rospy.loginfo("[Calling pick service...]")
		pick_result = rospy.ServiceProxy('/pick', PickPlaceSimple)(grasper.current_arm)

		# Check if the pick was successful
		if (pick_result.error_code == MoveItErrorCodes.SUCCESS or pick_result.error_code == MoveItErrorCodes.CONTROL_FAILED):
			success = True
			rospy.loginfo("[Pick successful!]")
			# If so, we can stop here
			break
		else:
			debug_try_once_more = True
			if debug_try_once_more:
				grasper.start_mapping()
				rospy.loginfo("Clearing octomap")
				grasper.clear_octomap_srv.call(EmptyRequest())
				rospy.sleep(1.0)
				grasper.stop_mapping()
				# Now call the pick service and wait for the result
				rospy.loginfo("[Calling pick service...]")
				pick_result = rospy.ServiceProxy('/pick', PickPlaceSimple)(grasper.current_arm)

				# Check if the pick was successful
				if (pick_result.error_code == MoveItErrorCodes.SUCCESS or pick_result.error_code == MoveItErrorCodes.CONTROL_FAILED):
					success = True
					rospy.loginfo("[Pick successful!]")
					# If so, we can stop here
					break
		
			# If not, we try the next grasp pose
			rospy.loginfo("[Pick failed. Trying next grasp pose...]")
			continue

	# If we didn't succeed, we can stop here
	if not success:
		rospy.loginfo("[No more grasp poses to try. Exiting...]")
		break
	else:
		# execute grasp
		grasper.gripper_grasp_srv()

		# Move torso up
		move_torso(torso_cmd, 0.3)
		rospy.sleep(3.0)

		# Raise arm
		rospy.loginfo("Moving arm to a safe pose")
		pmg = PlayMotionGoal()
		pmg.motion_name = 'pick_final_pose_' + grasper.current_arm[0]  # take first char
		pmg.skip_planning = False
		rospy.loginfo("Sending final arm command...")
		play_m_ac.send_goal_and_wait(pmg)
		rospy.sleep(1.0)

		if randomize_views:
			# Move base to final pose
			rospy.loginfo("Moving base to final pose")
			if grasper.current_arm == 'right':
				move_base_in_tab_frame(move_base_ac, base_place_final_right)
			elif grasper.current_arm == 'left':
				move_base_in_tab_frame(move_base_ac, base_place_final_left)
			rospy.sleep(2.0)

		# Move torso down
		move_torso(torso_cmd, 0.1)
		rospy.sleep(4.5)

		# Open grippers
		rospy.loginfo("Opening grippers")
		pmg = PlayMotionGoal()
		pmg.motion_name = 'open_gripper_' + grasper.current_arm[0]  # take first char
		pmg.skip_planning = True
		play_m_ac.send_goal_and_wait(pmg)

		if grasper.current_arm == 'left':
			# Also go back to pick final pose
			pmg = PlayMotionGoal()
			pmg.motion_name = 'pick_final_pose_l'
			pmg.skip_planning = False
			play_m_ac.send_goal_and_wait(pmg)
			
		# tuck in arm
		rospy.loginfo("Preparing arm")
		pmg = PlayMotionGoal()
		pmg.motion_name = 'prep_' + grasper.current_arm[0] # take first char
		pmg.skip_planning = False
		play_m_ac.send_goal_and_wait(pmg)
		rospy.sleep(2.0)


# # Testtttt
# grasper.start_mapping()
# rospy.sleep(1)
# grasper.stop_mapping()
# grasps, scores = grasper.grasp_gen.get_grasps(visualize=True)
# # Send grasp command to moveit pick and place pipeline
# grasp_pose_pub = rospy.Publisher('/grasp/pose', geometry_msgs.msg.PoseStamped, queue_size=1, latch=True)
# grasp_pose_msg = geometry_msgs.msg.PoseStamped()
# grasp_pose_msg.header.stamp = rospy.Time.now()
# grasp_pose_msg.header.frame_id = 'base_footprint'
# grasp_pose_msg.pose.position.x = 0.5
# grasp_pose_msg.pose.position.y = -0.5
# grasp_pose_msg.pose.position.z = 0.75
# grasp_pose_msg.pose.orientation.x = 0.0
# grasp_pose_msg.pose.orientation.y = 0.0
# grasp_pose_msg.pose.orientation.z = 0.0
# grasp_pose_msg.pose.orientation.w = 1.0
# grasp_pose_pub.publish(grasp_pose_msg)
# rospy.loginfo("[Sending /pick command...]")
# # call /pick rosservice
# grasper.current_arm = 'left'
# pick_result = rospy.ServiceProxy('/pick', PickPlaceSimple)(grasper.current_arm)
# rospy.spin()
