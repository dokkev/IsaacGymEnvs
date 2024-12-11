import numpy as np
import os
import torch

from isaacgym import gymtorch
from isaacgym import gymapi
from isaacgym import gymutil

from isaacgymenvs.utils.torch_jit_utils import quat_mul, quat_apply, to_torch, tensor_clamp, quat_conjugate, quat_to_angle_axis 
from isaacgymenvs.tasks.base.priv_info_task import PrivInfoVecTask



@torch.jit.script
def axisangle2quat(vec, eps=1e-6):
    """
    Converts scaled axis-angle to quat.
    Args:
        vec (tensor): (..., 3) tensor where final dim is (ax,ay,az) axis-angle exponential coordinates
        eps (float): Stability value below which small values will be mapped to 0

    Returns:
        tensor: (..., 4) tensor where final dim is (x,y,z,w) vec4 float quaternion
    """
    # type: (Tensor, float) -> Tensor
    # store input shape and reshape
    input_shape = vec.shape[:-1]
    vec = vec.reshape(-1, 3)

    # Grab angle
    angle = torch.norm(vec, dim=-1, keepdim=True)

    # Create return array
    quat = torch.zeros(torch.prod(torch.tensor(input_shape)), 4, device=vec.device)
    quat[:, 3] = 1.0

    # Grab indexes where angle is not zero an convert the input to its quaternion form
    idx = angle.reshape(-1) > eps
    quat[idx, :] = torch.cat([
        vec[idx, :] * torch.sin(angle[idx, :] / 2.0) / angle[idx, :],
        torch.cos(angle[idx, :] / 2.0)
    ], dim=-1)

    # Reshape and return output
    quat = quat.reshape(list(input_shape) + [4, ])
    return quat


class FrankaCubeThrow(PrivInfoVecTask):

    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
        self.cfg = cfg
        
        # Initialize gym
        self.gym = gymapi.acquire_gym()

        self.randomize = self.cfg["task"]["randomize"]
        self.max_episode_length = self.cfg["env"]["episodeLength"]

        self.action_scale = self.cfg["env"]["actionScale"]
        
        # Cube location Randomization Parameters
        self.init_cube_pos_noise = self.cfg["env"]["cubeInitPosNoise"]
        self.init_cube_ori_noise = self.cfg["env"]["cubeInitOriNoise"]
        self.goal_cube_pos_noise = self.cfg["env"]["cubeGoalPosNoise"]
        self.goal_cube_ori_noise = self.cfg["env"]["cubeGoalOriNoise"]


        self.franka_position_noise = self.cfg["env"]["frankaPositionNoise"]
        self.franka_rotation_noise = self.cfg["env"]["frankaRotationNoise"]
        self.franka_dof_noise = self.cfg["env"]["frankaDofNoise"]
        self.aggregate_mode = self.cfg["env"]["aggregateMode"]
        
        # Domain Randomization Parameters
        self.randomization_params = self.cfg["task"]["randomization_params"]

        # Create dicts to pass to reward function
        self.reward_settings = {
            "r_pos_scale": self.cfg["env"]["posRewardScale"],
            "r_ori_scale": self.cfg["env"]["oriRewardScale"],
            "r_contact_scale": self.cfg["env"]["contactRewardScale"],
            "r_vel_scale": self.cfg["env"]["velRewardScale"],
            "r_success_scale": self.cfg["env"]["successRewardScale"],
            "r_eef_approach_scale": self.cfg["env"]["eefApproachPenaltyScale"],
        }
        
        # print messages for priv info for each env
        self.enable_priv_info_print = self.cfg["env"]["enablePrivInfoPrint"]
     
        
        # include priviliged information in the observation space
        self.include_priv_info = self.cfg["env"]["includePrivInfo"]
        self.num_env_factors = self.cfg['env']['privInfoDim']

        # Controller type (OSC or joint torques)
        self.control_type = self.cfg["env"]["controlType"]
        assert self.control_type in {"osc", "jvel"},\
            "Invalid control type specified. Must be one of: {osc, jvel}"


        # dimensions
        # obs include: cube_pos(3) + cube_quat(4) + goal_cube_dist_pos(3)  + eef_pose (7) + [priv_info_dim]
        self.cfg["env"]["numObservations"] = 20

        if self.include_priv_info:
            self.cfg["env"]["numObservations"] += self.num_env_factors
    
        # self.cfg["env"]["numObservations"] = 17 if self.control_type == "osc" else 26
        # actions include: delta EEF if OSC (6)  + kp (6) (kd critically damped)
        self.cfg["env"]["numActions"] = 4 # 3 joint velocities + gripper pos
        
        
        # Values to be filled in at runtime
        self.states = {}                        # will be dict filled with relevant states to use for reward calculation
        self.handles = {}                       # will be dict mapping names to relevant sim handles
        self.num_dofs = None                    # Total number of DOFs per env
        self.actions = None                     # Current actions to be deployed
        self._init_cube_state = None           # Initial state of cube for the current env
        self._cube_state = None                # Current state of cube for the current env
        self._goal_cube_state = None           # Goal state of cube for the current env
        self._cube_id = None                   # Actor ID corresponding to cube for a given env
        
        self._eef_goal_state = None            # Goal state of end effector
        self.quat_desired = None                # Quaternion target for fixed ori control

        
        # Tensor placeholders
        self._root_state = None             # State of root body        (n_envs, 13)
        self._dof_state = None  # State of all joints       (n_envs, n_dof)
        self._q = None  # Joint positions           (n_envs, n_dof)
        self._qd = None                     # Joint velocities          (n_envs, n_dof)
        self._rigid_body_state = None  # State of all rigid bodies             (n_envs, n_bodies, 13)
        self._contact_forces = None     # Contact forces in sim
        self._eef_state = None  # end effector state (at grasping point)
        self._finger_state = None  # finger state
        self._j_eef = None  # Jacobian for end effector
        self._mm = None  # Mass matrix
        self._arm_control = None  # Tensor buffer for controlling arm
        self._gripper_control = None  # Tensor buffer for controlling gripper
        self._pos_control = None            # Position actions
        self._effort_control = None         # Torque actions
        self._franka_effort_limits = None        # Actuator effort limits for franka
        self._global_indices = None         # Unique indices corresponding to all envs in flattened array

        self.debug_viz = self.cfg["env"]["enableDebugVis"]

        self.up_axis = "z"
        self.up_axis_idx = 2
        self._steps_elapsed = 0 

        super().__init__(config=self.cfg, rl_device=rl_device, sim_device=sim_device, graphics_device_id=graphics_device_id, headless=headless, virtual_screen_capture=virtual_screen_capture, force_render=force_render)

        # Franka defaults
        self.franka_default_dof_pos = to_torch(
            [0, 1.0472, 0, -1.5708, 0, 2.61799, 0.7854, 0.020, 0.020 ], device=self.device
        )
        
        self.controlled_joints = torch.tensor([1, 3, 5], device=self.device)

        # Joint PI Velocity Gains
        self.kp_vel = to_torch([10.] * 7, device=self.device)
        self.ki_vel = to_torch([50] * 7, device=self.device)
        self.vel_i_error = torch.zeros(self.num_envs, 7, device=self.device)

        # Set control limits
        self.cmd_limit = to_torch([0.1, 0.1, 0.1, 0.5, 0.5, 0.5], device=self.device).unsqueeze(0) if \
        self.control_type == "osc" else self._franka_effort_limits[:7].unsqueeze(0)


        # Reset all environments
        self.reset_idx(torch.arange(self.num_envs, device=self.device))

        # Refresh tensors
        self._refresh()
        

    def create_sim(self):
        self.sim_params.up_axis = gymapi.UP_AXIS_Z
        self.sim_params.gravity.x = 0
        self.sim_params.gravity.y = 0
        self.sim_params.gravity.z = -9.81
        self.sim = super().create_sim(
            self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        self._create_ground_plane()
        self._create_envs(self.num_envs, self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))


        # apply domain randomization if true
        if self.randomize:
            self.apply_randomizations(self.randomization_params)

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)

    def _create_envs(self, num_envs, spacing, num_per_row):
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../assets")
        franka_asset_file = "urdf/franka_description/robots/franka_panda_gripper.urdf"
 
    
        if "asset" in self.cfg["env"]:
            asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), self.cfg["env"]["asset"].get("assetRoot", asset_root))
            franka_asset_file = self.cfg["env"]["asset"].get("assetFileNameFranka", franka_asset_file)

        # load franka asset
        asset_options = gymapi.AssetOptions()
        asset_options.flip_visual_attachments = True
        asset_options.fix_base_link = True
        asset_options.collapse_fixed_joints = False
        asset_options.disable_gravity = True
        asset_options.thickness = 0.001
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_EFFORT
        asset_options.use_mesh_materials = True
        franka_asset = self.gym.load_asset(self.sim, asset_root, franka_asset_file, asset_options)

        franka_dof_stiffness = to_torch([0, 0, 0, 0, 0, 0, 0, 5000., 5000.], dtype=torch.float, device=self.device)
        franka_dof_damping = to_torch([0, 0, 0, 0, 0, 0, 0, 1.0e2, 1.0e2], dtype=torch.float, device=self.device)

        # Create table asset
        table_x_offset = 1.0
        table_pos = [table_x_offset, 0.0, 1.0]
        table_thickness = 0.05
        table_opts = gymapi.AssetOptions()
        table_opts.fix_base_link = True
        table_asset = self.gym.create_box(self.sim, *[4.5, 1.2, table_thickness], table_opts)

        # Create table stand asset
        table_stand_height = 0.1
        table_stand_pos = [-0.5, 0.0, 1.0 + table_thickness / 2 + table_stand_height / 2]
        table_stand_opts = gymapi.AssetOptions()
        table_stand_opts.fix_base_link = True
        table_stand_asset = self.gym.create_box(self.sim, *[0.2, 0.2, table_stand_height], table_opts)
        rigid_shape_props_asset = self.gym.get_asset_rigid_shape_properties(table_asset)
        for element in rigid_shape_props_asset:
            element.friction = 0.01
        self.gym.set_asset_rigid_shape_properties(table_asset, rigid_shape_props_asset)


        cube_color = gymapi.Vec3(0.6, 0.1, 0.0)
        # load cube asset
        puck_asset_file = "urdf/cube.urdf"
        self.cube_size = 0.05
        cube_asset = self.gym.load_asset(self.sim,asset_root, puck_asset_file, gymapi.AssetOptions())
        
        
    
        self.num_franka_bodies = self.gym.get_asset_rigid_body_count(franka_asset)
        self.num_franka_dofs = self.gym.get_asset_dof_count(franka_asset)

        print("num franka bodies: ", self.num_franka_bodies)
        print("num franka dofs: ", self.num_franka_dofs)

        # set franka dof properties
        franka_dof_props = self.gym.get_asset_dof_properties(franka_asset)
        self.franka_dof_lower_limits = []
        self.franka_dof_upper_limits = []
        self._franka_effort_limits = []
        for i in range(self.num_franka_dofs):
            franka_dof_props['driveMode'][i] = gymapi.DOF_MODE_POS if i > 6 else gymapi.DOF_MODE_EFFORT
            if self.physics_engine == gymapi.SIM_PHYSX:
                franka_dof_props['stiffness'][i] = franka_dof_stiffness[i]
                franka_dof_props['damping'][i] = franka_dof_damping[i]
            else:
                franka_dof_props['stiffness'][i] = 7000.0
                franka_dof_props['damping'][i] = 50.0

            self.franka_dof_lower_limits.append(franka_dof_props['lower'][i])
            self.franka_dof_upper_limits.append(franka_dof_props['upper'][i])
            self._franka_effort_limits.append(franka_dof_props['effort'][i])

        self.franka_dof_lower_limits = to_torch(self.franka_dof_lower_limits, device=self.device)
        self.franka_dof_upper_limits = to_torch(self.franka_dof_upper_limits, device=self.device)
        self._franka_effort_limits = to_torch(self._franka_effort_limits, device=self.device)
        self.franka_dof_speed_scales = torch.ones_like(self.franka_dof_lower_limits)
        self.franka_dof_speed_scales[[7, 8]] = 0.1
        franka_dof_props['effort'][7] = 200
        franka_dof_props['effort'][8] = 200

        # Define start pose for franka
        franka_start_pose = gymapi.Transform()
        franka_start_pose.p = gymapi.Vec3(-0.45, 0.0, 1.0 + table_thickness / 2 + table_stand_height)
        franka_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        # Define start pose for table
        table_start_pose = gymapi.Transform()
        table_start_pose.p = gymapi.Vec3(*table_pos)
        table_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
        self._table_surface_pos = np.array(table_pos) + np.array([0, 0, table_thickness / 2])
        self.reward_settings["table_height"] = self._table_surface_pos[2]

        # Define start pose for table stand
        table_stand_start_pose = gymapi.Transform()
        table_stand_start_pose.p = gymapi.Vec3(*table_stand_pos)
        table_stand_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        # Define start pose for cubes (doesn't really matter since they're get overridden during reset() anyways)
        init_cube_pose = gymapi.Transform()
        init_cube_pose.p = gymapi.Vec3(0.0, 0.0, 0.0)
        init_cube_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
        
        # compute aggregate size
        num_franka_bodies = self.gym.get_asset_rigid_body_count(franka_asset)
        num_franka_shapes = self.gym.get_asset_rigid_shape_count(franka_asset)
        max_agg_bodies = num_franka_bodies + 4     # for table, table stand, cube, goal cube
        max_agg_shapes = num_franka_shapes + 4     # 1 for table, table stand, cube, goal cube
        self.frankas = []
        self.envs = []
        

        # Create environments
        for i in range(self.num_envs):
            
            # create env instance
            env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)

            # Create actors and define aggregate group appropriately depending on setting
            # NOTE: franka should ALWAYS be loaded first in sim!
            if self.aggregate_mode >= 3:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            # Create franka
            # Potentially randomize start pose
            if self.franka_position_noise > 0:
                rand_xy = self.franka_position_noise * (-1. + np.random.rand(2) * 2.0)
                franka_start_pose.p = gymapi.Vec3(-0.45 + rand_xy[0], 0.0 + rand_xy[1],
                                                 1.0 + table_thickness / 2 + table_stand_height)
            if self.franka_rotation_noise > 0:
                rand_rot = torch.zeros(1, 3)
                rand_rot[:, -1] = self.franka_rotation_noise * (-1. + np.random.rand() * 2.0)
                new_quat = axisangle2quat(rand_rot).squeeze().numpy().tolist()
                franka_start_pose.r = gymapi.Quat(*new_quat)
            franka_actor = self.gym.create_actor(env_ptr, franka_asset, franka_start_pose, "franka", i, 0, 0)
            self.gym.set_actor_dof_properties(env_ptr, franka_actor, franka_dof_props)

            if self.aggregate_mode == 2:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            # Create table
            table_actor = self.gym.create_actor(env_ptr, table_asset, table_start_pose, "table", i, 0, 0)
            table_stand_actor = self.gym.create_actor(env_ptr, table_stand_asset, table_stand_start_pose, "table_stand",
                                                      i, 0, 0)

            if self.aggregate_mode == 1:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            # Create cubes
            self._cube_id = self.gym.create_actor(env_ptr, cube_asset, init_cube_pose, "cube", i, 0, 0)

            # Set colors
            self.gym.set_rigid_body_color(env_ptr, self._cube_id, 0, gymapi.MESH_VISUAL, cube_color)

            

            if self.aggregate_mode > 0:
                self.gym.end_aggregate(env_ptr)

            # Store the created env pointers
            self.envs.append(env_ptr)
            self.frankas.append(franka_actor)

        # Setup init state buffer
        self._init_cube_state = torch.zeros(self.num_envs, 13, device=self.device)
        self._goal_cube_state = torch.zeros(self.num_envs, 13, device=self.device)
        self._eef_goal_state = torch.zeros(self.num_envs, 13, device=self.device)

        # Setup data
        self.init_data()

    def init_data(self):
        # Setup sim handles
        env_ptr = self.envs[0]
        franka_handle = 0
        self.handles = {
            # Franka
            "hand": self.gym.find_actor_rigid_body_handle(env_ptr, franka_handle, "panda_hand"),
            
            # Franka Gripper
            "finger": self.gym.find_actor_rigid_body_handle(env_ptr, franka_handle, "panda_rightfinger_tip"),

            # Cube
            "cube_body_handle": self.gym.find_actor_rigid_body_handle(self.envs[0], self._cube_id, "box"),
        }

        # Get total DOFs
        self.num_dofs = self.gym.get_sim_dof_count(self.sim) // self.num_envs

        # Setup tensor buffers
        _actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        _dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        _rigid_body_state_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self._root_state = gymtorch.wrap_tensor(_actor_root_state_tensor).view(self.num_envs, -1, 13)
        self._dof_state = gymtorch.wrap_tensor(_dof_state_tensor).view(self.num_envs, -1, 2)
        self._rigid_body_state = gymtorch.wrap_tensor(_rigid_body_state_tensor).view(self.num_envs, -1, 13)
        self._q = self._dof_state[..., 0]
        self._qd = self._dof_state[..., 1]
        self._eef_state = self._rigid_body_state[:, self.handles["hand"], :]
        self._finger_state = self._rigid_body_state[:, self.handles["finger"], :]
        _jacobian = self.gym.acquire_jacobian_tensor(self.sim, "franka")
        jacobian = gymtorch.wrap_tensor(_jacobian)
        hand_joint_index = self.gym.get_actor_joint_dict(env_ptr, franka_handle)['panda_hand_joint']
        self._j_eef = jacobian[:, hand_joint_index, :, :7]
        _massmatrix = self.gym.acquire_mass_matrix_tensor(self.sim, "franka")
        mm = gymtorch.wrap_tensor(_massmatrix)
        self._mm = mm[:, :7, :7]
        self._cube_state = self._root_state[:, self._cube_id, :]
        
        # contact time
        self.contact_time = torch.zeros(self.num_envs, device=self.device)
   
        # Initialize states
        self.states.update({
            "cube_size": torch.ones_like(self._eef_state[:, 0]) * self.cube_size,
        })
        


        # Initialize actions
        self._pos_control = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)
        self._effort_control = torch.zeros_like(self._pos_control)

        # Initialize control
        self._arm_control = self._effort_control[:, :7]
        self._gripper_control = self._pos_control[:, 7:9]


        # Initialize indices
        self._global_indices = torch.arange(self.num_envs * 4, dtype=torch.int32,
                                           device=self.device).view(self.num_envs, -1)

    def _update_states(self):
        self.states.update({
            # Franka 
            "q": self._q[:, :],
            "q_gripper": self._q[:, -2:],
            "eef_pos": self._eef_state[:, :3],
            "eef_quat": self._eef_state[:, 3:7],
            "eef_vel": self._eef_state[:, 7:],
            "finger_pos" : self._finger_state[:, :3],
            "finger_quat" : self._finger_state[:, 3:7],

            # Object Observable Information
            "cube_pos": self._cube_state[:, :3],
            "cube_quat": self._cube_state[:, 3:7],
            "cube_vel": self._cube_state[:, 7:10],
            
            "cube_contact": self._cube_state[:, :3] - self._finger_state[:, :3], # cube to eef pos diff
            
            "contact_time": self.contact_time,
            
            "goal_cube_pos": self._goal_cube_state[:, :3],
            "goal_cube_quat": self._goal_cube_state[:, 3:7],
            "cube_to_goal_cube_pos": self._goal_cube_state[:, :3] - self._cube_state[:, :3],
            
        })

    def _refresh(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_jacobian_tensors(self.sim)
        self.gym.refresh_mass_matrix_tensors(self.sim)

        # Refresh states
        self._update_states()

    def compute_reward(self, actions):
        self.rew_buf[:], self.reset_buf[:], self.extras['success'] = compute_franka_reward(
            self.reset_buf, self.progress_buf, self.actions, self.states, self.reward_settings, self.max_episode_length
        )

    def compute_observations(self):
        self._refresh()

        cube_pos=self.states["cube_pos"]
        cube_quat=self.states["cube_quat"]
        eef_pos=self.states["eef_pos"]
        eef_quat=self.states["eef_quat"]

        cube_vel=self.states["cube_vel"]
        
        # compute current cube to goal cube position
        cube_pos_diff = self._goal_cube_state[:, :3] - cube_pos
        
        # TODO: compute current cube to goal cube quaternion
        
        # Observable Information
        obs = [cube_pos, cube_quat, eef_pos, eef_quat, cube_vel, cube_pos_diff]
        
        # Include priv info in the observation space
        if self.include_priv_info:
            obs.append(self.priv_info_buf)
            
          

        # Concatenate all observations
        self.obs_buf = torch.cat(obs, dim=-1)

        # maxs = {ob: torch.max(self.states[ob]).item() for ob in obs}
        return self.obs_buf
    
    def store_proprio_hist(self):
        """
        Store the proprioceptive history of the cube (cube states) in the proprioception buffer. 
        """
        
        # get cube pos and quat
        cube_states = torch.cat([self.states["cube_pos"], self.states["cube_quat"]], dim=1)  # [num_envs, 7]

        # proprio_hist_buf = [num_envs] x [prop_hist_len] x [prop_dim]
        cube_states_dim = cube_states.shape[1] # 7
        prop_his_buf_dim = self.proprio_hist_buf.shape[2] #[prop_dim] 32 (hardcoded val from `_allocate_task_buffer`)
                
        # check dimensions of the cube_states and self.proprio_hist_buf
        if cube_states_dim > prop_his_buf_dim:
            raise ValueError(f"Proprioception buffer dimension mismatch! Cube state dim: {cube_states_dim} > Proprioception buffer dim: {prop_his_buf_dim}")
        
        # if prop hist buffer's prop dim is greater than cube state dim, pad the cube_states with zeros to match the prop hist buffer dim
        elif cube_states_dim < prop_his_buf_dim:
            padding = torch.zeros((self.num_envs, (prop_his_buf_dim - cube_states_dim)), device=self.device, dtype=torch.float)
            cube_states = torch.cat([cube_states, padding], dim=1) # shape: [num_envs, prop_his_buf_dim]
            
        # update the proprio_hist_buf
        # Shift the buffer to the left by one to discard the oldest data
        self.proprio_hist_buf = torch.roll(self.proprio_hist_buf, shifts=-1, dims=1)
        
        # append the new cube state to the buffer
        self.proprio_hist_buf[:, -1, :] = cube_states
        
        
        # Print the shape and example data of the proprio_hist_buf for debugging
        # print(f"proprio_hist_buf shape: {self.proprio_hist_buf.shape}")
        # print(f"proprio_hist_buf example data (first env): {self.proprio_hist_buf[0]}")
        

    def reset_idx(self, env_ids):
        
        if self.randomize:
            self.apply_randomizations(self.randomization_params)

            #  store prvi info in priv_info_buf [[env_id], [mass, friction, com_x, com_y, com_z]]
            self._store_priv_info(env_ids)
            

        
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        
        # Reset cube state
        self.contact_time[env_ids] = 0.0

        # if not self._i:
        self._reset_init_cube_state(env_ids=env_ids, check_valid=False)
        # self._i = True

        # Write these new init states to the sim states
        self._cube_state[env_ids] = self._init_cube_state[env_ids]
        self._goal_cube_state[env_ids] = self._goal_cube_state[env_ids]
        
        # Reset agent
        reset_noise = torch.rand((len(env_ids), 9), device=self.device)
        pos = tensor_clamp(
            self.franka_default_dof_pos.unsqueeze(0) +
            self.franka_dof_noise * 2.0 * (reset_noise - 0.5),
            self.franka_dof_lower_limits.unsqueeze(0), self.franka_dof_upper_limits)

        # Overwrite gripper init pos (no noise since these are always position controlled)
        pos[:, -2:] = self.franka_default_dof_pos[-2:]

        # Reset the internal obs accordingly
        self._q[env_ids, :] = pos
        self._qd[env_ids, :] = torch.zeros_like(self._qd[env_ids])

        # Set any position control to the current position, and any vel / effort control to be 0
        # NOTE: Task takes care of actually propagating these controls in sim using the SimActions API
        self._pos_control[env_ids, :] = pos
        self._effort_control[env_ids, :] = torch.zeros_like(pos)

        # Deploy updates
        multi_env_ids_int32 = self._global_indices[env_ids, 0].flatten()
        self.gym.set_dof_position_target_tensor_indexed(self.sim,
                                                        gymtorch.unwrap_tensor(self._pos_control),
                                                        gymtorch.unwrap_tensor(multi_env_ids_int32),
                                                        len(multi_env_ids_int32))
        self.gym.set_dof_actuation_force_tensor_indexed(self.sim,
                                                        gymtorch.unwrap_tensor(self._effort_control),
                                                        gymtorch.unwrap_tensor(multi_env_ids_int32),
                                                        len(multi_env_ids_int32))
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self._dof_state),
                                              gymtorch.unwrap_tensor(multi_env_ids_int32),
                                              len(multi_env_ids_int32))

        # Update cube states
        multi_env_ids_cubes_int32 = self._global_indices[env_ids, -1:].flatten()
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim, gymtorch.unwrap_tensor(self._root_state),
            gymtorch.unwrap_tensor(multi_env_ids_cubes_int32), len(multi_env_ids_cubes_int32))
        
        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0
        
        # visualize goal cube state
        axes_geom = gymutil.AxesGeometry(0.1)
        # Create a wireframe sphere
        sphere_rot = gymapi.Quat.from_euler_zyx(0.5 * 3.14, 0, 0)
        sphere_pose = gymapi.Transform(r=sphere_rot)
        sphere_geom = gymutil.WireframeSphereGeometry(0.02, 12, 12, sphere_pose, color=(1, 1, 0))

    def _store_priv_info(self, env_ids):
 
        for env_id in env_ids:
            env_ptr = self.envs[env_id]
            cube_handle = self.gym.find_actor_handle(env_ptr, "cube")
            cube_rb_props = self.gym.get_actor_rigid_body_properties(env_ptr, cube_handle)
            cube_shape_props = self.gym.get_actor_rigid_shape_properties(env_ptr, cube_handle)
            
            table_shape_props = self.gym.get_actor_rigid_shape_properties(
                env_ptr,
                self.gym.find_actor_handle(env_ptr, "table")
            )
            
            #isaacgym.gymapi.RigidBodyProperties
            for i, rb_prop in enumerate(cube_rb_props):
                cube_mass = rb_prop.mass # float (kg)
                cube_com = rb_prop.com # Vec3
                # cube_inertia = rb_prop.inertia # Mat33 == [Vec3, Vec3, Vec3]
                
            #isaacgym.gymapi.RigidShapeProperties
            for i, shape_prop in enumerate(cube_shape_props):
                cube_friction = shape_prop.friction
                # cube_rolling_friction = shape_prop.rolling_friction
                # cube_torsion_friction = shape_prop.torsion_friction
                # cube_compliance = shape_prop.compliance
                # cube_restitution = shape_prop.restitution # [0,1]
            
            for i, shape_prop in enumerate(table_shape_props):
                table_friction = shape_prop.friction
    
                
            # store in priv_info_buf
            # self.priv_info_buf[env_id, 0] = cube_mass
            # self.priv_info_buf[env_id, 1] = cube_friction
            # self.priv_info_buf[env_id, 2] = cube_com.x
            # self.priv_info_buf[env_id, 3] = cube_com.y
            # self.priv_info_buf[env_id, 4] = cube_com.z
            self.priv_info_buf[env_id, 0] = cube_friction
            
            
            if self.enable_priv_info_print:
                print(f"Env {env_id}, Cube Privileged Info:")
                print(f"  Mass = {cube_mass}")
                print(f"  CoM = {cube_com.x}, {cube_com.y}, {cube_com.z}")
                print(f"  Friction = {cube_friction}")
                print(f"  Table Friction = {table_friction}")
                # print(f" Inertia = {cube_inertia.x}, {cube_inertia.y}, {cube_inertia.z}")
             
        
    def _reset_init_cube_state(self, env_ids, check_valid=True):
        """
        Reset the cube's position based on self.startPositionNoise and self.startRotationNoise.
        Populates the appropriate self._init_cube_state.
        """

        # If env_ids is None, reset all environments
        if env_ids is None:
            env_ids = torch.arange(start=0, end=self.num_envs, device=self.device, dtype=torch.long)

        # Initialize buffer to hold sampled values
        num_resets = len(env_ids)
        sampled_init_cube_state = torch.zeros(num_resets, 13, device=self.device)
        sampled_goal_cube_state = torch.zeros(num_resets, 13, device=self.device)
    

        # Sample position and orientation for the cube
        centered_cube_xy_state = torch.tensor(self._table_surface_pos[:2], device=self.device, dtype=torch.float32)
        cube_height = self.states["cube_size"]
        
        # add offet to the centered_cube_xy_state
        init_x_offset = -0.1 - 0.8
        init_y_offset = 0.0
        init_cube_xy_state = centered_cube_xy_state + torch.tensor([init_x_offset, init_y_offset], device=self.device, dtype=torch.float32)
        
        # add offset to the centered_cube_xy_state
        goal_x_offset = 2.0
        goal_y_offset = 0.0
        # goal_cube_xy_state = centered_cube_xy_state + torch.tensor([goal_x_offset, goal_y_offset], device=self.device, dtype=torch.float32)
        goal_cube_xy_state = torch.tensor([goal_x_offset, goal_y_offset], device=self.device, dtype=torch.float32)
        
        # Gripper position
        eef_pos = self._eef_state[env_ids, :3]

        # set the cube's initial position to match the gripper's position
        sampled_init_cube_state[:, 0] = 0.216
        sampled_init_cube_state[:, 1] = 0
        # sampled_init_cube_state[:, 2] = eef_pos[:, 2]
        
        sampled_init_cube_state[:, 2] = self._table_surface_pos[2] + cube_height[env_ids] / 2
        

        #sample orientation
        sampled_init_cube_state[:, 6] = 1.0
        sampled_goal_cube_state[:, 6] = 1.0

        # Sample x, y values with noise for the goal cube
        sampled_goal_cube_state[:, :2] = goal_cube_xy_state.unsqueeze(0) + \
                                    2.0 * self.goal_cube_pos_noise * (torch.rand(num_resets, 
                                                                                 2, device=self.device) - 0.5)
        sampled_goal_cube_state[:, 2] = self._table_surface_pos[2] + cube_height[env_ids] / 2

        # Set the new sampled values as the initial state for the cube
        self._init_cube_state[env_ids, :] = sampled_init_cube_state
        self._goal_cube_state[env_ids, :] = sampled_goal_cube_state

    def _compute_osc_torques(self, dpose):
  
        # Solve for Operational Space Control # Paper: khatib.stanford.edu/publications/pdfs/Khatib_1987_RA.pdf
        # Helpful resource: studywolf.wordpress.com/2013/09/17/robot-control-4-operation-space-control/
        q, qd = self._q[:, :7], self._qd[:, :7]
        mm_inv = torch.inverse(self._mm)
        m_eef_inv = self._j_eef @ mm_inv @ torch.transpose(self._j_eef, 1, 2)
        m_eef = torch.inverse(m_eef_inv)

        # Transform our cartesian action `dpose` into joint torques `u`
        u = torch.transpose(self._j_eef, 1, 2) @ m_eef @ (
                self.kp * dpose - self.kd * self.states["eef_vel"]).unsqueeze(-1)

        # Nullspace control torques `u_null` prevents large changes in joint configuration
        # They are added into the nullspace of OSC so that the end effector orientation remains constant
        # roboticsproceedings.org/rss07/p31.pdf
        j_eef_inv = m_eef @ self._j_eef @ mm_inv
        u_null = self.kd_null * -qd + self.kp_null * (
                (self.franka_default_dof_pos[:7] - q + np.pi) % (2 * np.pi) - np.pi)
        u_null[:, 7:] *= 0
        u_null = self._mm @ u_null.unsqueeze(-1)
        u += (torch.eye(7, device=self.device).unsqueeze(0) - torch.transpose(self._j_eef, 1, 2) @ j_eef_inv) @ u_null

        # Clip the values to be within valid effort range
        u = tensor_clamp(u.squeeze(-1),
                         -self._franka_effort_limits[:7].unsqueeze(0), self._franka_effort_limits[:7].unsqueeze(0))

        return u
    
    def _compute_jvel_torques(self, jvel):
        qd = self._qd[:, :7]

        # Get uncontrolled joint indices

        uncontrolled_joints = torch.tensor([i for i in range(7) if i not in self.controlled_joints], device=self.device)
        
        # Initialize velocity errors
        vel_error = torch.zeros_like(qd)
        
        # For controlled joints, error is commanded velocity minus current velocity
        vel_error[:, self.controlled_joints] = jvel[:, self.controlled_joints] - qd[:, self.controlled_joints]
        
        # vel_error[:, self.controlled_joints] = 0 - qd[:, self.controlled_joints]
        
        # For uncontrolled joints, error is 0 minus current velocity (to maintain zero velocity)
        vel_error[:, uncontrolled_joints] = 0 - qd[:, uncontrolled_joints]
        
        # Update integral error for all joints
        self.vel_i_error += vel_error
        
        # Apply PID control to all joints
        u = self.kp_vel * vel_error + self.ki_vel * self.vel_i_error

        # Clip the values to be within valid effort range
        u = tensor_clamp(u, -self._franka_effort_limits[:7].unsqueeze(0), self._franka_effort_limits[:7].unsqueeze(0))
        
        return u

    def pre_physics_step(self, actions):
        self.actions = actions.clone().to(self.device)
        
        # split actions into arm and gripper actions
        u_arm, u_gripper = self.actions[:, :-1], self.actions[:, -1]
        # self.actions = torch.zeros_like(actions, device=self.device)
        # self.actions[:, 0] = 1.0
        
        # Create a zero tensor for all joints
        joint_cmd = torch.zeros((self.num_envs, 7), device=self.device)
        
        # Map the 3 action values to their corresponding joints (2,4,6 → indices 1,3,5)
        joint_cmd[:, self.controlled_joints] = u_arm
        
        # Scale using the full command limit
        joint_cmd = joint_cmd * self.cmd_limit / self.action_scale

        
        if self.control_type == "jvel":
            u_arm = self._compute_jvel_torques(jvel=joint_cmd)
        elif self.control_type == "osc":
            u_arm = self._compute_osc_torques(dpose=joint_cmd)
           
        # Control arm
        self._arm_control[:, :] = u_arm  # Apply control with variable kp and kd
        
        # Control gripper
        u_fingers = torch.zeros_like(self._gripper_control)
        u_fingers[:, 0] = torch.where(u_gripper >= 0.0, self.franka_dof_upper_limits[-2].item(),
                                      self.franka_dof_lower_limits[-2].item())
        u_fingers[:, 1] = torch.where(u_gripper >= 0.0, self.franka_dof_upper_limits[-1].item(),
                                      self.franka_dof_lower_limits[-1].item())
        # Write gripper command to appropriate tensor buffer
        self._gripper_control[:, :] = u_fingers

        # Deploy actions
        self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self._pos_control))
        self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self._effort_control))
        
        # print postion of the cube
        # print(self.states["cube_pos"])



    def step(self, actions: torch.Tensor):
        """Step the physics of the environment.

        Copied here to allow for primitive actions.

        Args:
            actions: actions to apply
        Returns:
            Observations, rewards, resets, info
            Observations are dict of observations (currently only one member called 'obs')
        """


        # randomize actions
        if self.dr_randomizations.get('actions', None):
            actions = self.dr_randomizations['actions']['noise_lambda'](actions)

        action_tensor = torch.clamp(actions, -self.clip_actions, self.clip_actions)

        # apply actions
        self.pre_physics_step(action_tensor)

        # step physics and render each frame
        for i in range(self.control_freq_inv):
            if self.force_render:
                self.render()
            self.gym.simulate(self.sim)

        # to fix!
        if self.device == 'cpu':
            self.gym.fetch_results(self.sim, True)

        # compute observations, rewards, resets, ...
        self.post_physics_step()

        self.control_steps += 1

        # fill time out buffer: set to 1 if we reached the max episode length AND the reset buffer is 1. Timeout == 1 makes sense only if the reset buffer is 1.
        self.timeout_buf = (self.progress_buf >= self.max_episode_length - 1) & (self.reset_buf != 0)

        # randomize observations
        if self.dr_randomizations.get('observations', None):
            self.obs_buf = self.dr_randomizations['observations']['noise_lambda'](self.obs_buf)

        self.extras["time_outs"] = self.timeout_buf.to(self.rl_device)

        self.obs_dict["obs"] = torch.clamp(self.obs_buf, -self.clip_obs, self.clip_obs).to(self.rl_device)

        # asymmetric actor-critic
        if self.num_states > 0:
            self.obs_dict["states"] = self.get_state()

        return self.obs_dict, self.rew_buf.to(self.rl_device), self.reset_buf.to(self.rl_device), self.extras
    


    def post_physics_step(self):
        self.progress_buf += 1
        self.randomize_buf += 1
        
        # contact time
        in_contact = (torch.norm(self.states["cube_contact"], dim=-1) < 0.2).float()  # Adjust threshold as needed
        self.contact_time = (self.contact_time + in_contact) * (1 - self.reset_buf.float())

        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            self.reset_idx(env_ids)

        self.compute_observations()
        self.compute_reward(self.actions)
        self.store_proprio_hist()

        # debug viz
        if self.viewer and self.debug_viz:
            self.gym.clear_lines(self.viewer)
            self.gym.refresh_rigid_body_state_tensor(self.sim)

            # Grab relevant states to visualize
            eef_pos = self.states["finger_pos"]
            eef_rot = self.states["finger_quat"]
            cube_pos = self.states["cube_pos"]
            cube_rot = self.states["cube_quat"]
            goal_cube_pos = self.states["goal_cube_pos"]
            goal_cube_rot = self.states["goal_cube_quat"]
            
      

            # Plot visualizations
            for i in range(self.num_envs):
                for pos, rot in zip((eef_pos, cube_pos, goal_cube_pos), (eef_rot, cube_rot, goal_cube_rot)):
                    px = (pos[i] + quat_apply(rot[i], to_torch([1, 0, 0], device=self.device) * 0.2)).cpu().numpy()
                    py = (pos[i] + quat_apply(rot[i], to_torch([0, 1, 0], device=self.device) * 0.2)).cpu().numpy()
                    pz = (pos[i] + quat_apply(rot[i], to_torch([0, 0, 1], device=self.device) * 0.2)).cpu().numpy()

                    p0 = pos[i].cpu().numpy()
                    self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], px[0], px[1], px[2]], [0.85, 0.1, 0.1])
                    self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], py[0], py[1], py[2]], [0.1, 0.85, 0.1])
                    self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], pz[0], pz[1], pz[2]], [0.1, 0.1, 0.85])
  
    def batch_reward_fn(self, obs):

        # Observable Information
        # obs = [cube_pos, cube_quat, eef_pos, eef_quat, cube_vel, cube_pos_diff]
        
        cube_pos = obs[:, :, :3]
        cube_quat = obs[:, :, 3:7]
        eef_pos = obs[:, :, 7:10]
        eef_quat = obs[:, :, 10:14]
        cube_vel = obs[:, :, 14:17]
    
        cube_pos_diff = obs[:, :, 17:20]
        delta_pos = torch.norm(cube_pos_diff, dim=-1)
        pos_reward = 1.0 - torch.tanh(10.0 * delta_pos)  # Scale based on distance

        cube_contact = cube_pos - eef_pos
        contact_reward = 1.0 - torch.tanh(10.0 * torch.norm(cube_contact, dim=-1))

        success_threshold = 0.05  # Success threshold for distance to goal
        success_condition = delta_pos < success_threshold
        success_reward = success_condition * self.max_episode_length

        rewards = (self.reward_settings["r_pos_scale"] * pos_reward +
                self.reward_settings["r_contact_scale"] * contact_reward + 
                self.reward_settings["r_success_scale"] * success_reward)     
        
        return rewards.detach()                    

#####################################################################
###=========================jit functions=========================###
#####################################################################



@torch.jit.script
@torch.jit.script
def compute_franka_reward(
    reset_buf, progress_buf, actions, states, reward_settings, max_episode_length
):
    # type: (Tensor, Tensor, Tensor, Dict[str, Tensor], Dict[str, float], float) -> Tuple[Tensor, Tensor, Tensor]

    # Get relevant states
    cube_pos = states["cube_pos"]
    cube_quat = states["cube_quat"]
    goal_pos = states["goal_cube_pos"]
    goal_quat = states["goal_cube_quat"]
    cube_vel = states["cube_vel"]
    eef_pos = states["eef_pos"]

    # Distance to goal
    delta_pos = torch.norm(cube_pos - goal_pos, dim=-1)

    # 1. Distance Reward (increases as cube gets closer to goal)
    distance_reward = reward_settings["r_pos_scale"] * (1.0 - torch.tanh(10.0 * delta_pos))

    # 2. Throwing Direction Reward (alignment between velocity and goal direction)
    goal_dir = goal_pos - cube_pos
    goal_dir = goal_dir / (torch.norm(goal_dir, dim=-1, keepdim=True) + 1e-7)  # Normalize
    vel_dir = cube_vel / (torch.norm(cube_vel, dim=-1, keepdim=True) + 1e-7)  # Normalize
    dir_alignment = torch.sum(goal_dir * vel_dir, dim=-1)  # Dot product for alignment
    direction_reward = reward_settings["r_vel_scale"] * torch.clamp(dir_alignment, min=0.0)

    # 3. Velocity Magnitude Reward
    vel_magnitude = torch.norm(cube_vel, dim=-1)
    velocity_reward = reward_settings["r_vel_scale"] * torch.tanh(0.5 * vel_magnitude)

    # 4. Success Reward
    success_threshold = 0.05
    terminal_velocity_threshold = 0.001
    
    # Success conditions
    success_condition1 = torch.norm(cube_vel, p=2, dim=-1) < terminal_velocity_threshold
    success_condition2 = delta_pos < success_threshold 
    success_condition3 = progress_buf > 0
    success_condition = success_condition1 & success_condition2 & success_condition3
    
    success_reward = torch.where(success_condition, 
                                reward_settings["r_success_scale"], 
                                torch.zeros_like(distance_reward))

    # Optional: End Effector Penalty (to prevent gripper from blocking throw)
    ee_goal_dist = torch.norm(eef_pos - goal_pos, dim=-1)


    # Combine rewards
    rewards = (distance_reward + 
              direction_reward + 
              velocity_reward + 
              success_reward)
            #   ee_penalty)

    # Update reset buffer
    reset_buf = torch.where((progress_buf >= max_episode_length - 1) | success_condition, 
                           torch.ones_like(reset_buf), 
                           reset_buf)

    return rewards.detach(), reset_buf, success_condition



