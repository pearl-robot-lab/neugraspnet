import rospy
import geometry_msgs
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
from std_msgs.msg import Header
from tf2_sensor_msgs.tf2_sensor_msgs import do_transform_cloud
import ros_numpy
import ctypes
import struct
import tf
import tf2_ros
import numpy as np
import open3d as o3d

rospy.init_node('grasp_objs')


# get the pointcloud from the topic
pcl_topic_name = "/zed2/zed_node/point_cloud/cloud_registered"

class GraspObjs:
	def __init__(self):
		# point cloud subscriber
		self.cloud_sub = rospy.Subscriber(pcl_topic_name, PointCloud2, self.callback, queue_size=1, buff_size=52428800)
		# tf listener to get the point cloud in the correct frame
		self.tf_buffer = tf2_ros.Buffer()
		self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

	def callback(self, ros_point_cloud):
		# xyz = np.array([[0,0,0]])
		# rgb = np.array([[0,0,0]])
		# gen = pc2.read_points(ros_point_cloud, skip_nans=True)
		# int_data = list(gen)

		# for x in int_data:
		# 	test = x[3] 
		# 	# cast float32 to int so that bitwise operations are possible
		# 	s = struct.pack('>f' ,test)
		# 	i = struct.unpack('>l',s)[0]
		# 	# you can get back the float value by the inverse operations
		# 	pack = ctypes.c_uint32(i).value
		# 	r = (pack & 0x00FF0000)>> 16
		# 	g = (pack & 0x0000FF00)>> 8
		# 	b = (pack & 0x000000FF)
		# 	# prints r,g,b values in the 0-255 range
		# 	# x,y,z can be retrieved from the x[0],x[1],x[2]
		# 	xyz = np.append(xyz,[[x[0],x[1],x[2]]], axis = 0)
		# 	rgb = np.append(rgb,[[r,g,b]], axis = 0)

		cam_to_table_tf = self.tf_buffer.lookup_transform('table', 'zed2_left_camera_frame', rospy.Time(0))
		cloud_transformed = do_transform_cloud(ros_point_cloud, cam_to_table_tf)
		# convert to numpy array
		pc = ros_numpy.numpify(cloud_transformed)
		points=np.zeros((pc.shape[0],3))
		points[:,0]=pc['x']
		points[:,1]=pc['y']
		points[:,2]=pc['z']
		# remove nans
		points = points[~np.isnan(points).any(axis=1)]
		
		# convert to open3d point cloud
		pcd = o3d.geometry.PointCloud()
		pcd.points = o3d.utility.Vector3dVector(points)
		self.pcd = pcd

		# Debug: viz open3d
		# origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
		# o3d.visualization.draw_geometries([pcd, origin])


grasper = GraspObjs()
rospy.spin()
# - Put it in the correct frame
# - Make tsdf and cleanup point cloud (open3d)
# - Then call the clutter detection implicit function to get grasps and scores. Use the viz
# - We get back grasps and scores and can then call the appropriate tiago dual pick place function that can handle multiple grasps