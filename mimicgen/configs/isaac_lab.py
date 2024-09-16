# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the NVIDIA Source Code License [see LICENSE for details].

"""
Task configs for robosuite.

See @PickPlace_Config below for an explanation of each parameter.
"""
import mimicgen
from mimicgen.configs.config import MG_Config


class PickPlace_Config(MG_Config):
    """
    Corresponds to robosuite Coffee task and variants.
    """
    NAME = "pick_place"
    TYPE = "isaac_lab"

    def task_config(self):
        """
        This function populates the `config.task` attribute of the config, 
        which has settings for each object-centric subtask in a task. Each 
        dictionary should have kwargs for the @add_subtask method in the 
        @MG_TaskSpec object.
        """
        self.task.task_spec.subtask_1 = dict(
            # Each subtask involves manipulation with respect to a single object frame. 
            # This string should specify the object for this subtask. The name should be 
            # consistent with the "datagen_info" from the environment interface and dataset.
            object_ref="object",
            # The "datagen_info" from the environment and dataset includes binary indicators 
            # for each subtask of the task at each timestep. This key should correspond
            # to the key in "datagen_info" that should be used to infer when this subtask 
            # is finished (e.g. on a 0 to 1 edge of the binary indicator). Should provide 
            # None for the final subtask.
            subtask_term_signal="picked",
            # if not None, specifies time offsets to be used during data generation when splitting 
            # a trajectory into subtask segments. On each data generation attempt, an offset is sampled
            # and added to the boundary defined by @subtask_term_signal.
            subtask_term_offset_range=None,
            # specifies how the source subtask segment should be selected during data generation 
            # from the set of source human demos
            selection_strategy="random",
            # optional keyword arguments for the selection strategy function used
            selection_strategy_kwargs=None,
            # amount of action noise to apply during this subtask
            action_noise=0.05,
            # number of interpolation steps to bridge previous subtask segment to this one
            num_interpolation_steps=5,
            # number of additional steps (with constant target pose of beginning of this subtask segment) to 
            # add to give the robot time to reach the pose needed to carry out this subtask segment
            num_fixed_steps=0,
            # if True, apply action noise during interpolation phase leading up to this subtask, as 
            # well as during the execution of this subtask
            apply_noise_during_interpolation=False,
        )
        self.task.task_spec.subtask_2 = dict(
            object_ref="object", 
            # end of final subtask does not need to be detected
            subtask_term_signal=None,
            subtask_term_offset_range=None,
            selection_strategy="random",
            selection_strategy_kwargs=None,
            action_noise=0.05,
            num_interpolation_steps=5,
            num_fixed_steps=0,
            apply_noise_during_interpolation=False,
        )
        # allow downstream code to completely replace the task spec from an external config
        self.task.task_spec.do_not_lock_keys()
