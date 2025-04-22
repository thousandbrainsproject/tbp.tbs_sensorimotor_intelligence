import logging


from tbp.monty.frameworks.environments import embodied_data as ED
from tbp.monty.frameworks.models.motor_policies import SurfacePolicy, SurfacePolicyCurvatureInformed
from tbp.monty.frameworks.actions.actions import (
    Action,
    MoveTangentially,
    SetAgentPose,
    SetSensorRotation,
)


class EnvironmentDataLoaderPerRotation(ED.EnvironmentDataLoader):
    """Dataloader for continual learning with one object across all rotations.
    
    This is very similar to EnvironmentDataLoaderPerObject, with key difference in when we cycle through objects.
    I've removed code related to num_distractors to make the code more readable for continual learning.
    """

    def __init__(self, object_names, object_init_sampler, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if isinstance(object_names, list):
            self.object_names = sorted(object_names) # Sort to match the order of ViT Continual Learning
            self.source_object_list = sorted(list(dict.fromkeys(object_names)))
        else:
            raise ValueError("Object names should be a list")
    
        self.create_semantic_mapping()
        self.object_init_sampler = object_init_sampler
        self.object_init_sampler.rng = self.rng
        self.object_params = self.object_init_sampler()
        self.current_object = 0
        self.n_objects = len(self.object_names)
        self.episodes = 0
        self.epochs = 0
        self.primary_target = None

    def pre_episode(self):
        super().pre_episode()
        self.reset_agent()

    def post_episode(self):
        super().post_episode()
        self.object_init_sampler.post_episode()
        self.object_params = self.object_init_sampler()
        self.episodes += 1
        self.set_primary_target(self.current_object, self.object_params)

    def pre_epoch(self):
        self.change_object_by_idx(self.current_object)
        self.set_primary_target(self.current_object, self.object_params)

    def post_epoch(self):
        self.epochs += 1
        self.object_init_sampler.post_epoch()
        self.object_params = self.object_init_sampler()
        self.cycle_object()
        self.set_primary_target(self.current_object, self.object_params)

    def create_semantic_mapping(self):
        """Create a unique semantic ID (positive integer) for each object.

        Used by Habitat for the semantic sensor.

        In addition, create a dictionary mapping back and forth between these IDs and
        the corresponding name of the object
        """
        assert set(self.object_names).issubset(
            set(self.source_object_list)
        ), "Semantic mapping requires primary targets sampled from source list"

        starting_integer = 1  # Start at 1 so that we can distinguish on-object semantic
        # IDs (>0) from being off object (semantic_id == 0 in Habitat by default)
        self.semantic_id_to_label = {
            i + starting_integer: label
            for i, label in enumerate(self.source_object_list)
        }
        self.semantic_label_to_id = {
            label: i + starting_integer
            for i, label in enumerate(self.source_object_list)
        }

    def cycle_object(self):
        """Remove the previous object(s) from the scene and add a new primary target.

        Also add any potential distractor objects.
        """
        next_object = (self.current_object + 1) % self.n_objects
        logging.info(
            f"\n\nGoing from {self.current_object} to {next_object} of {self.n_objects}"
        )
        self.change_object_by_idx(next_object)
    
    def change_object_by_idx(self, idx):
        """Update the primary target object in the scene based on the given index.

        The given `idx` is the index of the object in the `self.object_names` list,
        which should correspond to the index of the object in the `self.object_params`
        list.

        Also add any distractor objects if required.

        Args:
            idx: Index of the new object and ints parameters in object_params
        """
        assert idx <= self.n_objects, "idx must be <= self.n_objects"
        self.dataset.env.remove_all_objects()

        # Specify config for the primary target object and then add it
        init_params = self.object_params.copy()
        init_params.pop("euler_rotation")
        if "quat_rotation" in init_params.keys():
            init_params.pop("quat_rotation")
        init_params["semantic_id"] = self.semantic_label_to_id[self.object_names[idx]]

        # TODO clean this up with its own specific call i.e. Law of Demeter
        primary_target_obj = self.dataset.env.add_object(
            name=self.object_names[idx], **init_params
        )

        self.current_object = idx

    def set_primary_target(self, idx, object_params):
        self.current_object = idx
        self.primary_target = {
            "object": self.object_names[idx],
            "semantic_id": self.semantic_label_to_id[self.object_names[idx]],
            **object_params,
        }
        logging.info(f"New primary target: {self.primary_target}")

    def reset_agent(self):
        logging.debug("resetting agent------")
        self._observation, self.motor_system.state = self.dataset.reset()
        self._counter = 0

        # Make sure to also reset action variables when resetting agent during
        # pre-episode
        self._action = None
        self._amount = None
        self.motor_system.state[self.motor_system.agent_id]["motor_only_step"] = False

        return self._observation

# Below is mostly a copy-and-paste of InformedEnvironmentDataLoader from tbp.monty v0.1.0 
# It is there to inherit from EnvironmentDataLoaderPerRotation and not EnvironmentDataLoaderPerObject
class InformedEnvironmentDataLoader(EnvironmentDataLoaderPerRotation):
    """Dataloader that supports a policy which makes use of previous observation(s).

    Extension of the EnvironmentDataLoader where the actions can be informed by the
    observations. It passes the observation to the InformedPolicy class (which is an
    extension of the BasePolicy). This policy can then make use of the observation
    to decide on the next action.

    Also has the following, additional functionality; TODO refactor/separate these
    out as appropriate

    i) this dataloader allows for early stopping by adding the set_done
    method which can for example be called when the object is recognized.

    ii) the motor_only_step can be set such that the sensory module can
    later determine whether perceptual data should be sent to the learning module,
    or just fed back to the motor policy.

    iii) Handles different data-loader updates depending on whether the policy is
    based on the surface-agent or touch-agent

    iv) Supports hypothesis-testing "jump" policy
    """

    def __init__(self, *args, **kwargs):
        super(InformedEnvironmentDataLoader, self).__init__(*args, **kwargs)

    def __iter__(self):
        # Overwrite original because we don't want to reset agent at this stage
        # (already done in pre-episode)

        # TODO look into refactoring the parent __iter__ method so that we don't need
        # to use this fix

        return self

    def __next__(self):
        if self._counter == 0:
            return self.first_step()

        # Check if any LM's have output a goal-state (such as hypothesis-testing
        # goal-state)
        elif (
            self.motor_system.use_goal_state_driven_actions
            and self.motor_system.driving_goal_state is not None
        ):
            return self.execute_jump_attempt()

        # NOTE: terminal conditions are now handled in experiment.run_episode loop
        else:
            self._action = self.motor_system()

            # If entirely off object, use vision (i.e. view-finder)
            # TODO refactor so that this check is done in the motor-policy, and we
            # update the constraint separately/appropriately; i.e. the below
            # code should be as general as possible
            if isinstance(self.motor_system, SurfacePolicy) and self._action is None:
                self._action = self.motor_system.touch_object(
                    self._observation, view_sensor_id="view_finder"
                )

                self.motor_system.state[self.motor_system.agent_id][
                    "motor_only_step"
                ] = True

            self._observation, self.motor_system.state = self.dataset[self._action]

            # Check whether sensory information is just for feeding back to motor policy
            # TODO refactor so that the motor policy itself is making this update
            # when appropriate, not embodied_data
            if (
                (type(self.motor_system) == SurfacePolicy)
                or (type(self.motor_system) == SurfacePolicyCurvatureInformed)
            ) and self._action.name != "orient_vertical":
                self.motor_system.state[self.motor_system.agent_id][
                    "motor_only_step"
                ] = True
            else:
                self.motor_system.state[self.motor_system.agent_id][
                    "motor_only_step"
                ] = False

            self._counter += 1  # TODO clean up incrementing of counter

            return self._observation

    def pre_episode(self):
        super().pre_episode()
        if not self.dataset.env._agents[0].action_space_type == "surface_agent":
            on_target_object = self.get_good_view_with_patch_refinement()

    def first_step(self):
        """Carry out particular motor-system state updates required on the first step.

        TODO ?can get rid of this by appropriately initializing motor_only_step

        Returns:
            The observation from the first step.
        """
        # Return first observation after 'reset' before any action is applied
        self._counter += 1

        # Based on current code-base self._action will always be None when
        # the counter is 0
        assert self._action is None, "Setting of motor_only_step may need updating"

        # For first step of surface-agent policy, always bypass LM processing
        # For distant-agent policy, we still process the first sensation if it is
        # on the object
        self.motor_system.state[self.motor_system.agent_id]["motor_only_step"] = (
            isinstance(self.motor_system, SurfacePolicy)
        )

        return self._observation

    def get_good_view(
        self,
        sensor_id: str,
        allow_translation: bool = True,
        max_orientation_attempts: int = 1,
    ) -> bool:
        """Policy to get a good view of the object before an episode starts.

        Used by the distant agent to find the initial view of an object at the
        beginning of an episode with respect to a given sensor (the surface agent
        makes use of the `touch_object` method instead). Also currently used
        by the distant agent after a "jump" has been initialized by a model-based
        policy.

        First, the agent moves towards object until it fills a minimum of percentage
        (given by `motor_system.good_view_percentage`) of the sensor's field of view
        or the closest point of the object is less than a given distance
        (`motor_system.desired_object_distance`) from the sensor. This makes sure
        that big and small objects all fill similar amount of space in the sensor's
        field of view. Otherwise small objects may be too small to perform saccades or
        the sensor ends up inside of big objects. This step is performed by default
        but can be skipped by setting `allow_translation=False`.

        Second, the agent will then be oriented towards the object so that the
        sensor's central pixel is on-object. In the case of multi-object experiments,
        (i.e., when `num_distractors > 0`), there is an additional orientation step
        performed prior to the translational movement step.

        Args:
            sensor_id: The name of the sensor used to inform movements.
            allow_translation: Whether to allow movement toward the object via
                the motor systems's `move_close_enough` method. If `False`, only
                orientienting movements are performed. Default is `True`.
            max_orientation_attempts: The maximum number of orientation attempts
                allowed before giving up and returning `False` indicating that the
                sensor is not on the target object.

        Returns:
            Whether the sensor is on the target object.

        TODO M : move most of this to the motor systems, shouldn't be in embodied_data
            class
        """
        # TODO break up this method so that there is less code duplication
        # Start by ensuring the center of the patch is covering the primary target
        # object before we start moving forward; only done for multi-object experiments
        multiple_objects_present = False
        if multiple_objects_present:
            on_target_object = self.motor_system.is_on_target_object(
                self._observation,
                sensor_id,
                target_semantic_id=self.primary_target["semantic_id"],
                multiple_objects_present=multiple_objects_present,
            )
            if not on_target_object:
                actions = self.motor_system.orient_to_object(
                    self._observation,
                    sensor_id,
                    target_semantic_id=self.primary_target["semantic_id"],
                    multiple_objects_present=multiple_objects_present,
                )
                for action in actions:
                    self._observation, self.motor_system.state = self.dataset[action]

        if allow_translation:
            # Move closer to the object, if not already close enough
            action, close_enough = self.motor_system.move_close_enough(
                self._observation,
                sensor_id,
                target_semantic_id=self.primary_target["semantic_id"],
                multiple_objects_present=multiple_objects_present,
            )
            # Continue moving to a close distance to the object
            while not close_enough:
                logging.debug("moving closer!")
                self._observation, self.motor_system.state = self.dataset[action]
                action, close_enough = self.motor_system.move_close_enough(
                    self._observation,
                    sensor_id,
                    target_semantic_id=self.primary_target["semantic_id"],
                    multiple_objects_present=multiple_objects_present,
                )

        on_target_object = self.motor_system.is_on_target_object(
            self._observation,
            sensor_id,
            target_semantic_id=self.primary_target["semantic_id"],
            multiple_objects_present=multiple_objects_present,
        )
        num_attempts = 0
        while not on_target_object and num_attempts < max_orientation_attempts:
            actions = self.motor_system.orient_to_object(
                self._observation,
                sensor_id,
                target_semantic_id=self.primary_target["semantic_id"],
                multiple_objects_present=multiple_objects_present,
            )
            for action in actions:
                self._observation, self.motor_system.state = self.dataset[action]
            on_target_object = self.motor_system.is_on_target_object(
                self._observation,
                sensor_id,
                target_semantic_id=self.primary_target["semantic_id"],
                multiple_objects_present=multiple_objects_present,
            )
            num_attempts += 1

        return on_target_object

    def get_good_view_with_patch_refinement(self) -> bool:
        """Policy to get a good view of the object for the central patch.

        Used by the distant agent to move and orient toward an object such that the
        central patch is on-object. This is done by first moving and orienting the
        agent toward the object using the view finder. Then orienting movements are
        performed using the central patch (i.e., the sensor module with id
        "patch" or "patch_0") to ensure that the patch's central pixel is on-object.
        Up to 3 reorientation attempts are performed using the central patch.

        Also currently used by the distant agent after a "jump" has been initialized
        by a model-based policy.

        Returns:
            Whether the sensor is on the object.

        """
        self.get_good_view("view_finder")
        for patch_id in ("patch", "patch_0"):
            if patch_id in self._observation["agent_id_0"].keys():
                on_target_object = self.get_good_view(
                    patch_id,
                    allow_translation=False,  # only orientation movements
                    max_orientation_attempts=3,  # allow 3 reorientation attempts
                )
                break
        return on_target_object

    def execute_jump_attempt(self):
        """Attempt a hypothesis-testing "jump" onto a location of the object.

        Delegates to motor policy directly to determine specific jump actions.

        Returns:
            The observation from the jump attempt.
        """
        logging.debug(
            "Attempting a 'jump' like movement to evaluate an object hypothesis"
        )

        # Store the current location and orientation of the agent
        # If the hypothesis-guided jump is unsuccesful (e.g. to empty space,
        # or inside an object, we return here)
        pre_jump_state = self.motor_system.state[self.motor_system.agent_id]

        # Check that all sensors have identical rotations - this is because actions
        # currently update them all together; if this changes, the code needs
        # to be updated; TODO make this its own method
        for ii, current_sensor in enumerate(pre_jump_state["sensors"].keys()):
            if ii == 0:
                first_sensor = current_sensor
            assert np.all(
                pre_jump_state["sensors"][current_sensor]["rotation"]
                == pre_jump_state["sensors"][first_sensor]["rotation"]
            ), "Sensors are not identical in pose"

        # TODO In general what would be best/cleanest way of routing information,
        # e.g. perhaps the learning module should just pass a *displacement* (in
        # internal coordinates, and a target point-normal)
        # Could also consider making use of decide_location_for_movement (or
        # decide_location_for_movement_matching)

        (target_loc, target_np_quat) = self.motor_system.derive_habitat_goal_state()

        # Update observations and motor system-state based on new pose
        set_agent_pose = SetAgentPose(
            agent_id=self.motor_system.agent_id,
            location=target_loc,
            rotation_quat=target_np_quat,
        )
        self._observation, self.motor_system.state = self.dataset[set_agent_pose]

        # As above, but now also accounting for resetting the sensor pose; this
        # is necessary for the distant agent, which pivots the camera around
        # like a ball-and-socket joint; note the surface agent does not
        # modify this from the the unit quaternion and [0, 0, 0] position
        # anyways; further note this is globally applied to all sensors
        set_sensor_rotation = SetSensorRotation(
            agent_id=self.motor_system.agent_id,
            rotation_quat=quaternion.one,
        )
        self._observation, self.motor_system.state = self.dataset[set_sensor_rotation]

        # Check depth-at-center to see if the object is in front of us
        # As for methods such as touch_object, we use the view-finder
        depth_at_center = self.motor_system.get_depth_at_center(
            self._observation,
            view_sensor_id="view_finder",
            initial_pose=False,
        )

        # Save the potential post-jump state for later visualization (i.e. of
        # failed jumps)
        # TODO M when updating code to visualize graph-mismatch, can also ensure we
        # log this information as necessary
        # temp_motor_state_copy = self.motor_system.convert_motor_state()
        # self.motor_system.action_details["post_jump_pose"].append(
        #     temp_motor_state_copy
        # )

        # If depth_at_center < 1.0, there is a visible element within 1 meter of the
        # view-finder's central pixel)
        if depth_at_center < 1.0:
            self.handle_successful_jump()

        else:
            self.handle_failed_jump(pre_jump_state, first_sensor)

        # Regardless of whether movement was successful, counts as a step,
        # and we provide the observation to the next step of the motor policy
        self._counter += 1

        self.motor_system.state[self.motor_system.agent_id]["motor_only_step"] = True

        # Call post_action (normally taken care of __call__ within
        # self.motor_system()) - TODO refactor so that the whole of the hypothesis
        # driven jumps makes cleaner use of self.motor_system()
        self.motor_system.post_action(self.motor_system.action)

        return self._observation

    def handle_successful_jump(self):
        """Deal with the results of a successful hypothesis-testing jump.

        A successful jump is "on-object", i.e. the object is perceived by the sensor.
        """
        logging.debug(
            "Object visible, maintaining new pose for hypothesis-testing action"
        )

        if isinstance(self.motor_system, SurfacePolicy):
            # For the surface-agent policy, update last action as if we have
            # just moved tangentially
            # Results in us seemlessly transitioning into the typical
            # corrective movements (forward or orientation) of the surface-agent
            # policy
            self.motor_system.action = MoveTangentially(
                agent_id=self.motor_system.agent_id, distance=0.0, direction=[0, 0, 0]
            )

            # Store logging information about jump success; when doing refactor
            # of policy code, TODO cleanup where this is performed, and make
            # variable names more general; TODO also only log this when
            # we are doing detailed logging
            # TODO M clean up these action details loggings; this may need to remain
            # local to a "motor-system buffer" given that these are model-free
            # actions that have nothing to do with the LMs
            self.motor_system.action_details["pc_heading"].append("jump")
            self.motor_system.action_details["avoidance_heading"].append(False)
            self.motor_system.action_details["z_defined_pc"].append(None)

        else:
            self.get_good_view_with_patch_refinement()

    def handle_failed_jump(self, pre_jump_state, first_sensor):
        """Deal with the results of a failed hypothesis-testing jump.

        A failed jump is "off-object", i.e. the object is not perceived by the sensor.
        """
        logging.debug("No object visible from hypothesis jump, or inside object!")
        logging.debug("Returning to previous position")

        set_agent_pose = SetAgentPose(
            agent_id=self.motor_system.agent_id,
            location=pre_jump_state["position"],
            rotation_quat=pre_jump_state["rotation"],
        )
        self._observation, self.motor_system.state = self.dataset[set_agent_pose]

        # All sensors are updated globally by actions, and are therefore
        # identical
        set_sensor_rotation = SetSensorRotation(
            agent_id=self.motor_system.agent_id,
            rotation_quat=pre_jump_state["sensors"][first_sensor]["rotation"],
        )
        self._observation, self.motor_system.state = self.dataset[set_sensor_rotation]

        assert np.all(
            self.motor_system.state[self.motor_system.agent_id]["position"]
            == pre_jump_state["position"]
        ), "Failed to return agent to location"
        assert np.all(
            self.motor_system.state[self.motor_system.agent_id]["rotation"]
            == pre_jump_state["rotation"]
        ), "Failed to return agent to orientation"

        for current_sensor in self.motor_system.state[self.motor_system.agent_id][
            "sensors"
        ].keys():
            assert np.all(
                self.motor_system.state[self.motor_system.agent_id]["sensors"][
                    current_sensor
                ]["rotation"]
                == pre_jump_state["sensors"][current_sensor]["rotation"]
            ), "Failed to return sensor to orientation"

        # TODO explore reverting to an attempt with touch_object here,
        # only moving back to our starting location if this is unsuccessful
        # after e.g. 16 glances around where we arrived; NB however that
        # if we're inside the object, then we don't want to do this

