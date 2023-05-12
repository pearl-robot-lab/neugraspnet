# Clone this ROS repo

# catkin build this ROS repo

# Clone neugraspnet python repo

# Build docker image
docker build ./docker/ -t neugraspnet

# run docker image
./scripts/run_docker_img.sh

# Setup conda env in docker image

# run neugraspnet clutter test