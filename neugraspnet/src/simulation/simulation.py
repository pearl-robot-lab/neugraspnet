from pathlib import Path
import time
import os
import numpy as np
import pybullet

from neugraspnet.src.utils.grasp import Label
from neugraspnet.src.utils.perception import *
from neugraspnet.src.simulation import btsim, workspace_lines
from neugraspnet.src.utils.transform import Rotation, Transform
from neugraspnet.src.utils.misc import apply_noise, apply_translational_noise


class ClutterRemovalSim(object):
    def __init__(self, scene, object_set, gui=True, seed=None, add_noise=False, sideview=False, save_dir=None, use_nvisii=False, save_freq=8, data_root=None, gripper_type='franka'):
        assert scene in ["pile", "packed", "egad"]

        self.urdf_root = Path("data/urdfs")
        self.egad_root = Path("data/egad_eval_set")
        if data_root:
            self.urdf_root = Path(data_root) / self.urdf_root
            self.egad_root = Path(data_root) / self.egad_root
        self.scene = scene
        self.object_set = object_set
        if 'pile' in object_set or 'packed' in object_set:
            self.discover_objects()
        if object_set == 'egad':
            self.discover_egad_files()

        self.global_scaling = {
            "blocks": 1.67,
            "google": 0.7,
            'google_pile': 0.7,
            'google_packed': 0.7,
            
        }.get(object_set, 1.0)
        self.gui = gui
        self.add_noise = add_noise
        self.sideview = sideview

        self.rng = np.random.RandomState(seed) if seed else np.random
        self.world = btsim.BtWorld(self.gui, save_dir, save_freq, use_nvisii)
        if gripper_type == 'franka':
            self.gripper = Gripper(self.world, data_root)
        else:
            # robotiq 2f 85
            self.gripper = RobotiqGripper(self.world, data_root)
        self.size = 6 * self.gripper.finger_depth
        intrinsic = CameraIntrinsic(640, 480, 540.0, 540.0, 320.0, 240.0)
        self.camera = self.world.add_camera(intrinsic, 0.1, 2.0)

    @property
    def num_objects(self):
        return max(0, self.world.p.getNumBodies() - 1)  # remove table from body count

    def discover_objects(self):
        root = self.urdf_root / self.object_set
        self.object_urdfs = [f for f in root.iterdir() if f.suffix == ".urdf"]

    def discover_egad_files(self):
        self.obj_egads = [f for f in self.egad_root.iterdir() if f.suffix=='.obj']

    def save_state(self):
        self._snapshot_id = self.world.save_state()

    def restore_state(self):
        self.world.restore_state(self._snapshot_id)

    def setup_sim_scene_from_mesh_pose_list(self, mesh_pose_list, table=True, data_root=None):
        self.world.reset()
        self.world.set_gravity([0.0, 0.0, -9.81])
        self.draw_workspace()

        if self.gui:
            self.world.p.resetDebugVisualizerCamera(
                cameraDistance=1.0,
                cameraYaw=0.0,
                cameraPitch=-45,
                cameraTargetPosition=[0.15, 0.50, -0.3],
            )
        if table:
            table_height = self.gripper.finger_depth
            self.place_table(table_height)

        for mesh_path, scale, pose in mesh_pose_list:
            body_pose = Transform.from_matrix(pose)
            if data_root is not None:
                mesh_path = os.path.join(data_root, mesh_path)

            if 'egad' in str(mesh_path):
                # if it isnt a posixpath, make it one
                if not isinstance(mesh_path, Path):
                    mesh_path = Path(mesh_path)
                self.world.load_obj_fixed_scale(mesh_path, body_pose, scale)
            else:
                if os.path.splitext(mesh_path)[1] == '.urdf':
                    urdf_path = mesh_path            
                else:
                    # path is to the _visual.obj file. Change to urdf
                    urdf_path = mesh_path[:-11] + '.urdf'
                body = self.world.load_urdf(urdf_path, body_pose, scale)
                # assert len(body.links) == 1
                # assert len(body.links[0].visuals) == 1
                # assert len(body.links[0].visuals[0].geometry.meshes) == 1
    
    def reset(self, object_count):
        self.world.reset()
        self.world.set_gravity([0.0, 0.0, -9.81])
        self.draw_workspace()

        if self.gui:
            self.world.p.resetDebugVisualizerCamera(
                cameraDistance=1.0,
                cameraYaw=0.0,
                cameraPitch=-45,
                cameraTargetPosition=[0.15, 0.50, -0.3],
            )

        table_height = self.gripper.finger_depth
        self.place_table(table_height)

        if self.scene == "pile":
            self.generate_pile_scene(object_count, table_height)
        elif self.scene == "packed":
            self.generate_packed_scene(object_count, table_height)
        elif self.scene == 'egad':
            self.generate_egad_obj(object_count, table_height)

        else:
            raise ValueError("Invalid scene argument")

    def draw_workspace(self):
        points = workspace_lines(self.size)
        color = [0.5, 0.5, 0.5]
        for i in range(0, len(points), 2):
            self.world.p.addUserDebugLine(
                lineFromXYZ=points[i], lineToXYZ=points[i + 1], lineColorRGB=color
            )

    def place_table(self, height):
        urdf = self.urdf_root / "setup" / "plane.urdf"
        pose = Transform(Rotation.identity(), [0.15, 0.15, height])
        self.world.load_urdf(urdf, pose, scale=0.6)

        # define valid volume for sampling grasps
        lx, ux = 0.02, self.size - 0.02
        ly, uy = 0.02, self.size - 0.02
        lz, uz = height + 0.005, self.size
        self.lower = np.r_[lx, ly, lz]
        self.upper = np.r_[ux, uy, uz]

    def generate_pile_scene(self, object_count, table_height):
        # place box
        urdf = self.urdf_root / "setup" / "box.urdf"
        pose = Transform(Rotation.identity(), np.r_[0.02, 0.02, table_height])
        box = self.world.load_urdf(urdf, pose, scale=1.3)

        # drop objects
        urdfs = self.rng.choice(self.object_urdfs, size=object_count)
        for urdf in urdfs:
            rotation = Rotation.random(random_state=self.rng)
            xy = self.rng.uniform(1.0 / 3.0 * self.size, 2.0 / 3.0 * self.size, 2)
            pose = Transform(rotation, np.r_[xy, table_height + 0.2])
            scale = self.rng.uniform(0.8, 1.0)
            self.world.load_urdf(urdf, pose, scale=self.global_scaling * scale)
            self.wait_for_objects_to_rest(timeout=1.0)

        # remove box
        self.world.remove_body(box)
        self.remove_and_wait()

    def generate_packed_scene(self, object_count, table_height):
        attempts = 0
        max_attempts = 12

        while self.num_objects < object_count and attempts < max_attempts:
            self.save_state()
            urdf = self.rng.choice(self.object_urdfs)
            x = self.rng.uniform(0.08, 0.22)
            y = self.rng.uniform(0.08, 0.22)
            z = 1.0
            angle = self.rng.uniform(0.0, 2.0 * np.pi)
            rotation = Rotation.from_rotvec(angle * np.r_[0.0, 0.0, 1.0])
            pose = Transform(rotation, np.r_[x, y, z])
            scale = self.rng.uniform(0.7, 0.9)
            body = self.world.load_urdf(urdf, pose, scale=self.global_scaling * scale)
            lower, upper = self.world.p.getAABB(body.uid)
            z = table_height + 0.5 * (upper[2] - lower[2]) + 0.002
            body.set_pose(pose=Transform(rotation, np.r_[x, y, z]))
            self.world.step()

            if self.world.get_contacts(body):
                self.world.remove_body(body)
                self.restore_state()
            else:
                self.remove_and_wait()
            attempts += 1

    def generate_egad_obj(self, object_count, table_height):
        # place box
        urdf = self.urdf_root / "setup" / "box.urdf"
        pose = Transform(Rotation.identity(), np.r_[0.02, 0.02, table_height])
        box = self.world.load_urdf(urdf, pose, scale=1.3)
        # drop objects
        urdfs = self.rng.choice(self.obj_egads, size=object_count)
        for urdf in urdfs:
            rotation = Rotation.random(random_state=self.rng)
            xy = self.rng.uniform(1.0 / 3.0 * self.size, 2.0 / 3.0 * self.size, 2)
            pose = Transform(rotation, np.r_[xy, table_height + 0.2])
            scale = self.rng.uniform(0.8, 1.4)#self.rng.uniform(0.8, 0.9)
            self.world.load_obj(urdf, pose, scale=self.global_scaling*scale)
            self.wait_for_objects_to_rest(timeout=1.0)
        # remove box
        self.world.remove_body(box)
        self.remove_and_wait()
        return urdf, pose

    def acquire_tsdf(self, n, N=None, resolution=40, randomize_view=False, tight_view=False):
        """Render synthetic depth images from n viewpoints and integrate into a TSDF.

        If N is None, the n viewpoints are equally distributed on circular trajectory.

        If N is given, the first n viewpoints on a circular trajectory consisting of N points are rendered.
        """
        tsdf = TSDFVolume(self.size, resolution)
        high_res_tsdf = TSDFVolume(self.size, 120)

        if self.sideview:
            # origin = Transform(Rotation.identity(), np.r_[self.size / 2, self.size / 2, self.size / 3])
            theta = np.pi / 3.0
            r = 2.0 * self.size
        else:
            origin = Transform(Rotation.identity(), np.r_[self.size / 2, self.size / 2, 0])
            if randomize_view:
                if tight_view == True:
                    theta = 5* np.pi / 12.0 # Fixed 75 degrees
                else:
                    theta = np.random.uniform(np.pi / 12.0, 5* np.pi / 12.0) # elevation: 15 to 75 degrees
                # theta = np.random.uniform(0.0, 5* np.pi / 12.0) # elevation: 0 to 75 degrees
                # theta = np.random.uniform(np.pi/4, np.pi/3) # elevation: 45 to 60 degrees
                r = np.random.uniform(1.6, 2.4) * self.size
            else:
                theta = np.pi / 6.0
                r = 2.0 * self.size
        # # randomizations of edge grasp network:
        # r = np.random.uniform(2, 2.5) * sim.size
        # theta = np.random.uniform(np.pi/4, np.pi/3)
        # phi = np.random.uniform(0.0, 2*np.pi)

        

        N = N if N else n
        if self.sideview:
            assert n == 1
            phi_list = [- np.pi / 2.0]
        else:
            phi_list = 2.0 * np.pi * np.arange(n) / N
        extrinsics = [camera_on_sphere(origin, r, theta, phi) for phi in phi_list]

        timing = 0.0
        for extrinsic in extrinsics:
            # Multiple views -> for getting other sides of pc
            depth_img = self.camera.render(extrinsic)[1]
            # add noise
            depth_img = apply_noise(depth_img, self.add_noise)
            
            tic = time.time()
            tsdf.integrate(depth_img, self.camera.intrinsic, extrinsic)
            timing += time.time() - tic
            high_res_tsdf.integrate(depth_img, self.camera.intrinsic, extrinsic)
        bounding_box = o3d.geometry.AxisAlignedBoundingBox(self.lower, self.upper)
        pc = high_res_tsdf.get_cloud()
        pc = pc.crop(bounding_box)

        return tsdf, pc, timing

    def execute_grasp(self, grasp, remove=True, allow_contact=False):
        T_world_grasp = grasp.pose
        T_grasp_pregrasp = Transform(Rotation.identity(), [0.0, 0.0, -0.05])
        T_world_pregrasp = T_world_grasp * T_grasp_pregrasp

        approach = T_world_grasp.rotation.as_matrix()[:, 2]
        angle = np.arccos(np.dot(approach, np.r_[0.0, 0.0, -1.0]))
        if angle > np.pi / 3.0:
            # side grasp, lift the object after establishing a grasp
            T_grasp_pregrasp_world = Transform(Rotation.identity(), [0.0, 0.0, 0.1])
            T_world_retreat = T_grasp_pregrasp_world * T_world_grasp
        else:
            T_grasp_retreat = Transform(Rotation.identity(), [0.0, 0.0, -0.1])
            T_world_retreat = T_world_grasp * T_grasp_retreat

        self.gripper.reset(T_world_pregrasp)

        if self.gripper.detect_contact():
            result = Label.FAILURE, self.gripper.max_opening_width
        else:
            self.gripper.move_tcp_xyz(T_world_grasp, abort_on_contact=True)
            if self.gripper.detect_contact() and not allow_contact:
                result = Label.FAILURE, self.gripper.max_opening_width
            else:
                self.gripper.move(0.0)
                self.gripper.move_tcp_xyz(T_world_retreat, abort_on_contact=False)
                if self.check_success(self.gripper):
                    result = Label.SUCCESS, self.gripper.read()
                    if remove:
                        contacts = self.world.get_contacts(self.gripper.body)
                        self.world.remove_body(contacts[0].bodyB)
                else:
                    result = Label.FAILURE, self.gripper.max_opening_width

        self.world.remove_body(self.gripper.body)

        if remove:
            self.remove_and_wait()

        return result

    def remove_and_wait(self):
        # wait for objects to rest while removing bodies that fell outside the workspace
        removed_object = True
        while removed_object:
            self.wait_for_objects_to_rest()
            removed_object = self.remove_objects_outside_workspace()

    def wait_for_objects_to_rest(self, timeout=2.0, tol=0.01):
        timeout = self.world.sim_time + timeout
        objects_resting = False
        while not objects_resting and self.world.sim_time < timeout:
            # simulate a quarter of a second
            for _ in range(60):
                self.world.step()
            # check whether all objects are resting
            objects_resting = True
            for _, body in self.world.bodies.items():
                if np.linalg.norm(body.get_velocity()) > tol:
                    objects_resting = False
                    break

    def remove_objects_outside_workspace(self):
        removed_object = False
        for body in list(self.world.bodies.values()):
            xyz = body.get_pose().translation
            if np.any(xyz < 0.0) or np.any(xyz > self.size):
                self.world.remove_body(body)
                removed_object = True
        return removed_object

    def check_success(self, gripper):
        # check that the fingers are in contact with some object and not fully closed
        contacts = self.world.get_contacts(gripper.body)
        res = len(contacts) > 0 and gripper.read() > gripper.open_success_thresh
        # DEBUG: print(f"[Label: {res}, width: {gripper.read()}]")
        return res

class RobotiqGripper(object):
    """Simulated Robotiq 2F 85 gripper."""

    def __init__(self, world, data_root=None):
        self.world = world
        if data_root is not None:
            self.urdf_path = Path(data_root) / Path("data/urdfs/robotiq_85/pal_robotiq_85_gripper.urdf")
        else:
            self.urdf_path = Path("data/urdfs/robotiq_85/pal_robotiq_85_gripper.urdf")
        self.main_joint_name = "gripper_finger_joint"
        self.mimic_joints = {'gripper_right_outer_knuckle_joint': 1,
                                'gripper_left_inner_knuckle_joint': 1,
                                'gripper_right_inner_knuckle_joint': 1,
                                'gripper_left_inner_finger_joint': -1,
                                'gripper_right_inner_finger_joint': -1}
        
        self.max_opening_width = 0.085
        self.closed_joint_limit = 0.8
        self.open_joint_limit = 0.0
        self.finger_depth = 0.05
        self.open_success_thresh = 0.0001 # Very small because we are not using the 'true' width of the robotiq gripper
        self.T_body_tcp = Transform(Rotation.identity(), [0.0, 0.0, 0.11+0.02754]) # from URDF link to TCP/grasp frame
        self.T_tcp_body = self.T_body_tcp.inverse()

    def reset(self, T_world_tcp):
        T_world_body = T_world_tcp * self.T_tcp_body
        self.body = self.world.load_urdf(self.urdf_path, T_world_body)
        self.body.set_pose(T_world_body)  # sets the position of the COM, not URDF link
        self.constraint = self.world.add_constraint(
            self.body,
            None,
            None,
            None,
            pybullet.JOINT_FIXED,
            [0.0, 0.0, 0.0],
            Transform.identity(),
            T_world_body,
        )
        self.update_tcp_constraint(T_world_tcp)
        # setup constraints for the robotiq to close as expected and to keep fingers centered
        self.joint = self.body.joints[self.main_joint_name] # we move this joint and other joints should mimic it
        self.joint.set_position(self.open_joint_limit, kinematics=True)
        for joint_name, multiplier in self.mimic_joints.items():
            self.body.joints[joint_name].set_position(self.open_joint_limit * multiplier, kinematics=True)

        # Tried using constraints but it doesn't work....
        # def __setup_mimic_joints__(self):
        #     self.mimic_parent_index = self.body.joints[self.mimic_parent_name].joint_index

        #     for joint_name, multiplier in self.mimic_children_names.items():
        #         mimic_child_joint_index = self.body.joints[joint_name].joint_index
        #         c = self.world.p.createConstraint(self.body.uid, self.mimic_parent_index,
        #                                self.body.uid, mimic_child_joint_index,
        #                                jointType=pybullet.JOINT_GEAR,
        #                                jointAxis=[0, 1, 0], # CHECK
        #                                parentFramePosition=[0, 0, 0],
        #                                childFramePosition=[0, 0, 0])
        #         self.world.p.changeConstraint(c, gearRatio=-multiplier, maxForce=100, erp=0.1)  # Note: the mysterious `erp` is of EXTREME importance

        # # def move_gripper(self, open_length):
        # #     # open_length = np.clip(open_length, *self.gripper_range)
        # #     # open_angle = 0.715 - math.asin((open_length - 0.010) / 0.1143)  # angle calculation # TODO: Check this and use true width
        # #     # Control the mimic gripper joint(s)
        # #     p.setJointMotorControl2(self.id, self.mimic_parent_id, p.POSITION_CONTROL, targetPosition=open_angle,
        # #                             force=self.joints[self.mimic_parent_id].maxForce, maxVelocity=self.joints[self.mimic_parent_id].maxVelocity)

    def update_tcp_constraint(self, T_world_tcp):
        T_world_body = T_world_tcp * self.T_tcp_body
        self.constraint.change(
            jointChildPivot=T_world_body.translation,
            jointChildFrameOrientation=T_world_body.rotation.as_quat(),
            maxForce=300,
        )

    def set_tcp(self, T_world_tcp):
        T_word_body = T_world_tcp * self.T_tcp_body
        self.body.set_pose(T_word_body)
        self.update_tcp_constraint(T_world_tcp)

    def move_tcp_xyz(self, target, eef_step=0.002, vel=0.10, abort_on_contact=True):
        T_world_body = self.body.get_pose()
        T_world_tcp = T_world_body * self.T_body_tcp

        diff = target.translation - T_world_tcp.translation
        n_steps = int(np.linalg.norm(diff) / eef_step)
        dist_step = diff / n_steps
        dur_step = np.linalg.norm(dist_step) / vel

        for _ in range(n_steps):
            T_world_tcp.translation += dist_step
            self.update_tcp_constraint(T_world_tcp)
            for _ in range(int(dur_step / self.world.dt)):
                self.world.step()
            if abort_on_contact and self.detect_contact():
                return

    def detect_contact(self, threshold=5):
        if self.world.get_contacts(self.body):
            return True
        else:
            return False

    def move(self, width):
        if width > self.max_opening_width:
            width = self.max_opening_width
        norm_move = width/self.max_opening_width # normalize to [0,1]
        pos_action = self.closed_joint_limit - norm_move * (self.closed_joint_limit - self.open_joint_limit)
        self.joint.set_position(pos_action)
        # Also set the mimic joints
        for joint_name, multiplier in self.mimic_joints.items():
            self.body.joints[joint_name].set_position(pos_action * multiplier)

        for _ in range(int(0.5 / self.world.dt)):
            self.world.step()

    def read(self):
        width = self.max_opening_width * (1 - self.joint.get_position()/self.closed_joint_limit)
        return width
    

class Gripper(object):
    """Simulated Panda hand."""

    def __init__(self, world, data_root=None):
        self.world = world
        if data_root is not None:
            self.urdf_path = Path(data_root) / Path("data/urdfs/panda/hand.urdf")
        else:
            self.urdf_path = Path("data/urdfs/panda/hand.urdf")

        self.max_opening_width = 0.08
        self.finger_depth = 0.05
        self.open_success_thresh = 0.1 * self.max_opening_width
        self.T_body_tcp = Transform(Rotation.identity(), [0.0, 0.0, 0.022])
        self.T_tcp_body = self.T_body_tcp.inverse()

    def reset(self, T_world_tcp):
        T_world_body = T_world_tcp * self.T_tcp_body
        self.body = self.world.load_urdf(self.urdf_path, T_world_body)
        self.body.set_pose(T_world_body)  # sets the position of the COM, not URDF link
        self.constraint = self.world.add_constraint(
            self.body,
            None,
            None,
            None,
            pybullet.JOINT_FIXED,
            [0.0, 0.0, 0.0],
            Transform.identity(),
            T_world_body,
        )
        self.update_tcp_constraint(T_world_tcp)
        # constraint to keep fingers centered
        self.world.add_constraint(
            self.body,
            self.body.links["panda_leftfinger"],
            self.body,
            self.body.links["panda_rightfinger"],
            pybullet.JOINT_GEAR,
            [1.0, 0.0, 0.0],
            Transform.identity(),
            Transform.identity(),
        ).change(gearRatio=-1, erp=0.1, maxForce=50)
        self.joint1 = self.body.joints["panda_finger_joint1"]
        self.joint1.set_position(0.5 * self.max_opening_width, kinematics=True)
        self.joint2 = self.body.joints["panda_finger_joint2"]
        self.joint2.set_position(0.5 * self.max_opening_width, kinematics=True)

    def update_tcp_constraint(self, T_world_tcp):
        T_world_body = T_world_tcp * self.T_tcp_body
        self.constraint.change(
            jointChildPivot=T_world_body.translation,
            jointChildFrameOrientation=T_world_body.rotation.as_quat(),
            maxForce=300,
        )

    def set_tcp(self, T_world_tcp):
        T_word_body = T_world_tcp * self.T_tcp_body
        self.body.set_pose(T_word_body)
        self.update_tcp_constraint(T_world_tcp)

    def move_tcp_xyz(self, target, eef_step=0.002, vel=0.10, abort_on_contact=True):
        T_world_body = self.body.get_pose()
        T_world_tcp = T_world_body * self.T_body_tcp

        diff = target.translation - T_world_tcp.translation
        n_steps = int(np.linalg.norm(diff) / eef_step)
        dist_step = diff / n_steps
        dur_step = np.linalg.norm(dist_step) / vel

        for _ in range(n_steps):
            T_world_tcp.translation += dist_step
            self.update_tcp_constraint(T_world_tcp)
            for _ in range(int(dur_step / self.world.dt)):
                self.world.step()
            if abort_on_contact and self.detect_contact():
                return

    def detect_contact(self, threshold=5):
        if self.world.get_contacts(self.body):
            return True
        else:
            return False

    def move(self, width):
        self.joint1.set_position(0.5 * width)
        self.joint2.set_position(0.5 * width)
        for _ in range(int(0.5 / self.world.dt)):
            self.world.step()

    def read(self):
        width = self.joint1.get_position() + self.joint2.get_position()
        return width
