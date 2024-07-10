# based on https://github.com/ros-planning/moveit_tutorials/blob/master/doc/move_group_python_interface/scripts/move_group_python_interface_tutorial.py
import copy
import rospy
import moveit_commander
# import moveit_msgs.msg
import geometry_msgs.msg

class MiscMoveItPlanner(object):

    def __init__(self, move_group_name="arm_right"):
        super(MiscMoveItPlanner, self).__init__()

        ## Instantiate a `RobotCommander`_ object. Provides information such as the robot's
        ## kinematic model and the robot's current joint states
        self.robot = moveit_commander.RobotCommander()

        ## Instantiate a `PlanningSceneInterface`_ object.  This provides a remote interface
        ## for getting, setting, and updating the robot's internal understanding of the
        ## surrounding world:
        self.scene = moveit_commander.PlanningSceneInterface()

        ## This interface can be used to plan and execute motions:
        self.move_group = moveit_commander.MoveGroupCommander(move_group_name)

        # We can get the name of the reference frame for this robot:
        self.planning_frame = self.move_group.get_planning_frame()
        print("============ Planning frame: %s" % self.planning_frame)

        # We can also print the name of the end-effector link for this group:
        self.eef_link = self.move_group.get_end_effector_link()
        print("============ End effector link: %s" % self.eef_link)

    def go_to_pose_goal(self, move_group_name="arm_right", pose_goal=None):
        # Copy class variables to local variables to make the web tutorials more clear.
        # In practice, you should use the class variables directly unless you have a good
        # reason not to.
        self.move_group = moveit_commander.MoveGroupCommander(move_group_name)
        move_group = self.move_group

        ## BEGIN_SUB_TUTORIAL plan_to_pose
        ##
        ## Planning to a Pose Goal
        ## ^^^^^^^^^^^^^^^^^^^^^^^
        ## We can plan a motion for this group to a desired pose for the
        ## end-effector:
        if pose_goal is None:
            print("No pose goal provided. Using default pose goal.")
            pose_goal = geometry_msgs.msg.Pose()
            pose_goal.orientation.w = 1.0
            pose_goal.position.x = 0.4
            pose_goal.position.y = 0.1
            pose_goal.position.z = 0.4

        move_group.set_pose_target(pose_goal)

        ## Now, we call the planner to compute the plan and execute it.
        # `go()` returns a boolean indicating whether the planning and execution was successful.
        success = move_group.go(wait=True)
        # Calling `stop()` ensures that there is no residual movement
        move_group.stop()
        # It is always good to clear your targets after planning with poses.
        # Note: there is no equivalent function for clear_joint_value_targets().
        move_group.clear_pose_targets()

        # # For testing:
        # # Note that since this section of code will not be included in the tutorials
        # # we use the class variable rather than the copied state variable
        # current_pose = self.move_group.get_current_pose().pose
        # return all_close(pose_goal, current_pose, 0.01)

    def plan_cartesian_path(self, move_group_name="arm_right", scale=1, waypoints=[]):
        # Copy class variables to local variables to make the web tutorials more clear.
        # In practice, you should use the class variables directly unless you have a good
        # reason not to.
        self.move_group = moveit_commander.MoveGroupCommander(move_group_name)
        move_group = self.move_group

        ## BEGIN_SUB_TUTORIAL plan_cartesian_path
        ##
        ## Cartesian Paths
        ## ^^^^^^^^^^^^^^^
        ## You can plan a Cartesian path directly by specifying a list of waypoints
        ## for the end-effector to go through. If executing  interactively in a
        ## Python shell, set scale = 1.0.
        ##
        if len(waypoints) == 0:
            print("No waypoints provided. Using default waypoints.")
            waypoints = []

            wpose = move_group.get_current_pose().pose
            wpose.position.z -= scale * 0.1  # First move up (z)
            wpose.position.y += scale * 0.2  # and sideways (y)
            waypoints.append(copy.deepcopy(wpose))

            wpose.position.x += scale * 0.1  # Second move forward/backwards in (x)
            waypoints.append(copy.deepcopy(wpose))

            wpose.position.y -= scale * 0.1  # Third move sideways (y)
            waypoints.append(copy.deepcopy(wpose))

        # We want the Cartesian path to be interpolated at a resolution of 1 cm
        # which is why we will specify 0.01 as the eef_step in Cartesian
        # translation.  We will disable the jump threshold by setting it to 0.0,
        # ignoring the check for infeasible jumps in joint space, which is sufficient
        # for this tutorial.
        (plan, fraction) = move_group.compute_cartesian_path(
            waypoints, 0.01, 0.0  # waypoints to follow  # eef_step
        )  # jump_threshold

        # Note: We are just planning, not asking move_group to actually move the robot yet:
        return plan, fraction

    def execute_plan(self, plan):
        # Copy class variables to local variables to make the web tutorials more clear.
        # In practice, you should use the class variables directly unless you have a good
        # reason not to.
        move_group = self.move_group

        ## BEGIN_SUB_TUTORIAL execute_plan
        ##
        ## Executing a Plan
        ## ^^^^^^^^^^^^^^^^
        ## Use execute if you would like the robot to follow
        ## the plan that has already been computed:
        move_group.execute(plan, wait=True)

        ## **Note:** The robot's current joint state must be within some tolerance of the
        ## first waypoint in the `RobotTrajectory`_ or ``execute()`` will fail
        ## END_SUB_TUTORIAL
