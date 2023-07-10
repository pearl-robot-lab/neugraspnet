# Clone this ROS repo

# catkin build this ROS repo

# Clone neugraspnet python repo

# Build docker image
docker build ./docker/ -t neugraspnet

# run docker image
./scripts/run_docker_img.sh

# Setup conda env in docker image

# run neugraspnet clutter test

# modify files on robot for occupancy grid:
/opt/pal/ferrum/share/tiago_dual_moveit_config/config/sensors_3d.yaml
sensor_manager.launch.xml
# modify files for zed wrapper:
/home/zedsonalpha/zed_ws/src/zed-ros-wrapper/zed_wrapper/params/zed2.yaml

## Launch files:

roslaunch neugraspnet launch_tiago159_with_zed2.launch
roslaunch neugraspnet launch_moveit_with_zed2_octomap.launch
roslaunch tiago_dual_pick_place pick_place.launch

~/tiago_core/catkin_ws/src/tiago_dual_pick_place
cd ~/tiago_core/catkin_ws/src/neugraspnet && ./scripts/run_docker_img.sh
python scripts/grasper.py
<!-- From inside container -->
python /neugraspnet/scripts/grasp_generator.py
<!-- debug: -->
rosrun rviz rviz -d `rospack find neugraspnet`/rviz/rviz.rviz
rosrun rviz rviz -d `rospack find tiago_dual_2dnav`/config/rviz/advanced_navigation.rviz
rqt
