# Pipeline:

Run a `setup_grasp_exp.py` script. It will:
- roslaunch the file for tiago and zed
- setup optitrack, moveit etc.
- setup tfs for map to tiago and for table to 'correct' tsdf frame
- roslaunch tiago dual pick place

Setup tiago dual pick place to:
- Take a set of grasp poses as input (in that specific message that it uses)
- The arm to use will be given
- Delete the parts of the octomap close to the virtual cube it generates
- Plan and execute the grasp without deleting the whoooole octomap

Run a `grasp_objs.py` script. It will:
- get the pointcloud from the topic
- Put it in the correct frame
- Make tsdf and cleanup point cloud (open3d)
- Then call the clutter detection implicit function to get grasps and scores. Use the viz
- We get back grasps and scores and can then call the appropriate tiago dual pick place function that can handle multiple grasps

Later:
- Call the appropriate function in `grasp_objs.py` in a main `run_grasp_exp.py`.
- Use a state machine to decide on left/right approach and to try a grasp, retry a grasp, randomize camera position etc. Go to a fixed position and rotate the gripper and drop the object