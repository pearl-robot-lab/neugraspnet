#!/usr/bin/env python
import os
import numpy as np
from copy import deepcopy

import rospy
import geometry_msgs
import tf
import tf2_ros
from visualization_msgs.msg import Marker, MarkerArray

import pdb

SCENE_SIZE = 0.3 # size of the scene in meters
rospy.init_node('optitrack_tf_sync')
listener = tf.TransformListener()
br = tf2_ros.TransformBroadcaster()
rate = rospy.Rate(100.0) # run at 100Hz

# publish a table object marker
obj_markers_pub = rospy.Publisher('/obj_markers', MarkerArray, queue_size=1)
                    
while not rospy.is_shutdown():
	try:
		(trans_torso_map,rot_torso_map) = listener.lookupTransform('torso_lift_link', 'map', rospy.Time(0))
	except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
		print("[WARNING: Could not find some TFs. Check TF and obj names]")
		print(e)
		continue

	# publish this to the tiago159 (real robot) to map tf
	# create the message
	t = geometry_msgs.msg.TransformStamped()
	t.header.stamp = rospy.Time.now()
	t.header.frame_id = "tiago159"
	t.child_frame_id = "map"
	# add the translation and rotation to pose
	t.transform.translation.x = trans_torso_map[0]
	t.transform.translation.y = trans_torso_map[1]
	t.transform.translation.z = trans_torso_map[2]
	t.transform.rotation.x = rot_torso_map[0]
	t.transform.rotation.y = rot_torso_map[1]
	t.transform.rotation.z = rot_torso_map[2]
	t.transform.rotation.w = rot_torso_map[3]
	# publish the message
	br.sendTransform(t)

	# publish a table to grasp origin tf
	# create the message
	grasp_origin_t = geometry_msgs.msg.TransformStamped()
	grasp_origin_t.header.stamp = rospy.Time.now()
	grasp_origin_t.header.frame_id = "table"
	grasp_origin_t.child_frame_id = "grasp_origin"
	# add the translation and rotation to pose
	grasp_origin_t.transform.translation.x = -SCENE_SIZE/2.0
	grasp_origin_t.transform.translation.y = -SCENE_SIZE/2.0
	grasp_origin_t.transform.translation.z = -0.04
	grasp_origin_t.transform.rotation.x = 0.0
	grasp_origin_t.transform.rotation.y = 0.0
	grasp_origin_t.transform.rotation.z = 0.0
	grasp_origin_t.transform.rotation.w = 1.0
	# publish the message
	br.sendTransform(grasp_origin_t)

	# publish table object marker (mammut)
	obj_markers_msg = MarkerArray()

	table_marker_msg = Marker()
	table_marker_msg.id = 10
	table_marker_msg.type = table_marker_msg.CUBE
	table_marker_msg.action = table_marker_msg.ADD
	table_marker_msg.scale.x = 0.57 # 0.6 # 0.55
	table_marker_msg.scale.y = 0.78 # 0.8 # 0.77
	table_marker_msg.scale.z = 0.48
	table_marker_msg.color.r = 1.0
	table_marker_msg.color.g = 1.0
	table_marker_msg.color.b = 1.0
	table_marker_msg.color.a = 1.0
	# Dynamic table marker
	table_marker_msg.header.frame_id = 'table'
	table_marker_msg.pose.orientation.x = 0.0
	table_marker_msg.pose.orientation.y = 0.0
	table_marker_msg.pose.orientation.z = 0.0
	table_marker_msg.pose.orientation.w = 1.0
	table_marker_msg.pose.position.x = 0.0
	table_marker_msg.pose.position.y = 0.0
	table_marker_msg.pose.position.z = -table_marker_msg.scale.z/2.0
	obj_markers_msg.markers.append(table_marker_msg)
	# Publish
	obj_markers_pub.publish(obj_markers_msg)
	

	rate.sleep()
