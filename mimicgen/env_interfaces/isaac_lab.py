# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the NVIDIA Source Code License [see LICENSE for details].

"""
MimicGen environment interface classes for basic isaac_lab environments.
"""
import numpy as np

import mimicgen.utils.pose_utils as PoseUtils
from mimicgen.env_interfaces.base import MG_EnvInterface

import torch


class IsaacLabInterface(MG_EnvInterface):
    """
    MimicGen environment interface base class for basic isaac_lab environments.
    """

    # Note: base simulator interface class must fill out interface type as a class property
    INTERFACE_TYPE = "isaac_lab"

    def get_robot_eef_pose(self):
        """
        Get current robot end effector pose. Should be the same frame as used by the robot end-effector controller.

        Returns:
            pose (np.array): 4x4 eef pose matrix
        """

        # Just retrieve its current pose
        return self.get_object_pose(
            obj_name="ee_frame", #TODO: generalize and read the e_frame name
            obj_type="frame_transformer",
        )

    def target_pose_to_action(self, target_pose, relative=True):
        """
        Takes a target pose for the end effector controller and returns an action 
        (usually a normalized delta pose action) to try and achieve that target pose. 

        Args:
            target_pose (np.array): 4x4 target eef pose
            relative (bool): if True, use relative pose actions, else absolute pose actions

        Returns:
            action (np.array): action compatible with env.step (minus gripper actuation)
        """


        # target position and rotation
        target_pos, target_rot = PoseUtils.unmake_pose(target_pose)

        # current position and rotation
        curr_pose = self.get_robot_eef_pose()
        curr_pos, curr_rot = PoseUtils.unmake_pose(curr_pose)

        if relative:
            # normalized delta position action
            delta_position = target_pos - curr_pos
            # delta_position = np.clip(delta_position / max_dpos, -1., 1.)

            # normalized delta rotation action
            delta_rot_mat = target_rot.dot(curr_rot.T)
            delta_quat = PoseUtils.mat2quat(delta_rot_mat)
            delta_rotation_axis, delta_rotation_angle = PoseUtils.quat2axisangle(delta_quat)

            # quat2axisangle returns both axis and angle, not axis * angle
            # We want delta_rotation to represent axis * angle so we multiple the two together
            delta_rotation = delta_rotation_axis *delta_rotation_angle

            # delta_rotation = np.clip(delta_rotation / max_drot, -1., 1.)
            return np.concatenate([delta_position, delta_rotation])

    def action_to_target_pose(self, action, relative=True):
        """
        Converts action (compatible with env.step) to a target pose for the end effector controller.
        Inverse of @target_pose_to_action. Usually used to infer a sequence of target controller poses
        from a demonstration trajectory using the recorded actions.

        Args:
            action (np.array): environment action
            relative (bool): if True, use relative pose actions, else absolute pose actions

        Returns:
            target_pose (np.array): 4x4 target eef pose that @action corresponds to
        """

        delta_position = action[:3]
        delta_rotation = action[3:6]

        # current position and rotation
        curr_pose = self.get_robot_eef_pose()
        curr_pos, curr_rot = PoseUtils.unmake_pose(curr_pose)

        # get pose target
        target_pos = curr_pos + delta_position

        # Convert delta_rotation to axis angle form
        delta_rotation_angle = np.linalg.norm(delta_rotation)
        # make sure that axis is a unit vector
        delta_rotation_axis = delta_rotation / delta_rotation_angle
        delta_quat = PoseUtils.axisangle2quat(delta_rotation_axis, delta_rotation_angle)
        delta_rot_mat = PoseUtils.quat2mat(delta_quat)

        target_rot = delta_rot_mat.dot(curr_rot)

        target_pose = PoseUtils.make_pose(target_pos, target_rot)
        return target_pose

    def action_to_gripper_action(self, action):
        """
        Extracts the gripper actuation part of an action (compatible with env.step).

        Args:
            action (np.array): environment action

        Returns:
            gripper_action (np.array): subset of environment action for gripper actuation
        """

        # last dimension is gripper action
        return action[-1:]

    # isaac_lab-specific helper method for getting object poses
    def get_object_pose(self, obj_name, obj_type):
        """
        Returns 4x4 object pose given the name of the object and the type.

        Args:
            obj_name (str): name of object
            obj_type (str): type of object - either "asset", "sensors"

        Returns:
            obj_pose (np.array): 4x4 object pose
        """
        assert obj_type in ["asset", "frame_transformer", "sensor"]

        if obj_type == "asset":
            # TODO: how do we handle multiple envs in a single scene
            obj_pos = np.array(self.env.unwrapped.scene[obj_name].data.root_pos_w.cpu().numpy()[0, :]) # Get the pose from the first environment
            obj_rot = np.array(self.env.unwrapped.scene[obj_name].data.root_quat_w.cpu().numpy()[0, :]) # Get the orienation quaternion from the first environment
        elif obj_type == "frame_transformer":
            obj_pos = self.env.unwrapped.scene[obj_name].data.target_pos_w[0, 0, :].cpu().numpy()
            obj_rot = self.env.unwrapped.scene[obj_name].data.target_quat_w[0, 0, :].cpu().numpy()
        elif obj_type == "sensor":
            raise NotImplementedError(f"Object type '{obj_type}' is not implemented.")
        else:
            raise NotImplementedError(f"Object type '{obj_type}' is not implemented.")
        
        # convert from w,x,y,z to x,y,z,w
        return PoseUtils.make_pose(obj_pos, PoseUtils.quat2mat(np.array([obj_rot[1], obj_rot[2], obj_rot[3], obj_rot[0]])))


class MG_PickPlace(IsaacLabInterface):
    """
    Corresponds to robosuite StackThree task and variants.
    """
    def get_object_poses(self):
        """
        Gets the pose of each object relevant to MimicGen data generation in the current scene.

        Returns:
            object_poses (dict): dictionary that maps object name (str) to object pose matrix (4x4 np.array)
        """

        # three relevant objects - three cubes
        return dict(
            object=self.get_object_pose(obj_name="object", obj_type="asset")
        )

    def get_subtask_term_signals(self):
        """
        Gets a dictionary of binary flags for each subtask in a task. The flag is 1
        when the subtask has been completed and 0 otherwise. MimicGen only uses this
        when parsing source demonstrations at the start of data generation, and it only
        uses the first 0 -> 1 transition in this signal to detect the end of a subtask.

        Returns:
            subtask_term_signals (dict): dictionary that maps subtask name to termination flag (0 or 1)
        """
        signals = dict()

        # first subtask is grasping cubeA (motion relative to cubeA)
        object_pose_mat = self.get_object_pose(obj_name="object", obj_type="asset")

        if object_pose_mat[2,3] > 0.05:
            signals["picked"] = 1
        else:
            signals["picked"] = 0

        # final subtask is placing cubeC on cubeA (motion relative to cubeA) - but final subtask signal is not needed
        return signals

class MG_Stack(IsaacLabInterface):
    """
    Corresponds to robosuite StackThree task and variants.
    """
    def get_object_poses(self):
        """
        Gets the pose of each object relevant to MimicGen data generation in the current scene.

        Returns:
            object_poses (dict): dictionary that maps object name (str) to object pose matrix (4x4 np.array)
        """
        # three relevant objects - three cubes
        return dict(
            cube_1=self.get_object_pose(obj_name="cube_1", obj_type="asset"),
            cube_2=self.get_object_pose(obj_name="cube_2", obj_type="asset"),
            cube_3=self.get_object_pose(obj_name="cube_3", obj_type="asset")
        )

    def grasp_1(self,
        diff_threshold: float = 0.04
    ) -> torch.Tensor:
        """Check if an object is grapsed."""

        focused_oject: RigidObject = self.env.unwrapped.scene["cube_2"]
        focused_oject_pose = focused_oject.data.root_pos_w

        end_effecotr: FrameTransformer = self.env.unwrapped.scene['ee_frame']
        end_effecotr_pose = end_effecotr.data.target_pos_w

        pose_diff = torch.norm(torch.norm(focused_oject_pose - end_effecotr_pose, dim=1), dim=1)

        gripper_closed = self.env.unwrapped.action_manager.action[:, -1] < 0 # -1: close, 1: open

        # print('object grasp_1: ', torch.logical_and(pose_diff < diff_threshold, gripper_closed))

        return torch.logical_and(pose_diff < diff_threshold, gripper_closed)

    
    def grasp_2(self,
        diff_threshold: float = 0.04
    ) -> torch.Tensor:
        """Check if an object is grapsed."""

        focused_oject: RigidObject = self.env.unwrapped.scene["cube_3"]
        focused_oject_pose = focused_oject.data.root_pos_w

        end_effecotr: FrameTransformer = self.env.unwrapped.scene['ee_frame']
        end_effecotr_pose = end_effecotr.data.target_pos_w

        pose_diff = torch.norm(torch.norm(focused_oject_pose - end_effecotr_pose, dim=1), dim=1)

        gripper_closed = self.env.unwrapped.action_manager.action[:, -1] < 0 # -1: close, 1: open

        # print('object grasp_2: ', torch.logical_and(pose_diff < diff_threshold, gripper_closed))

        return torch.logical_and(pose_diff < diff_threshold, gripper_closed)
    
    def stack_1(self,
        xy_threshold: float = 0.04,
        height_threshold: float = 0.005,
        height_diff: float = 0.0468,
    ) -> torch.Tensor:
        """Check if an object is grapsed."""

        focused_oject: RigidObject = self.env.unwrapped.scene["cube_1"]
        focused_oject_pose = focused_oject.data.root_pos_w

        target_oject: RigidObject = self.env.unwrapped.scene["cube_2"]
        target_oject_pose = target_oject.data.root_pos_w

        pose_diff = focused_oject_pose - target_oject_pose
        height_distance = torch.norm(pose_diff[:, 2:], dim=1)
        xy_distance = torch.norm(pose_diff[:, :2], dim=1)
        stacked = torch.logical_and(xy_distance < xy_threshold, torch.norm(height_distance - height_diff) < height_threshold)

        gripper_closed = self.env.unwrapped.action_manager.action[:, -1] < 0 # -1: close, 1: open
        return torch.logical_and(stacked, gripper_closed)

        print('object stacked: ', torch.logical_and(stacked, gripper_closed))

        return stacked

    def get_subtask_term_signals(self):
        """
        Gets a dictionary of binary flags for each subtask in a task. The flag is 1
        when the subtask has been completed and 0 otherwise. MimicGen only uses this
        when parsing source demonstrations at the start of data generation, and it only
        uses the first 0 -> 1 transition in this signal to detect the end of a subtask.
        Returns:
            subtask_term_signals (dict): dictionary that maps subtask name to termination flag (0 or 1)
        """
        signals = dict()
        
        if self.grasp_1():
            signals["grasp_1"] = 1
        else:
            signals["grasp_1"] = 0
        if self.grasp_2():
            signals["grasp_2"] = 1
        else:
            signals["grasp_2"] = 0
        if self.stack_1():
            signals["stack_1"] = 1
        else:
            signals["stack_1"] = 0
        # final subtask is placing cubeC on cubeA (motion relative to cubeA) - but final subtask signal is not needed
        return signals

