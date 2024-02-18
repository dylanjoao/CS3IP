import os
import math
from isaacgym import gymapi
from isaacgym import gymutil
from isaacgym import gymtorch
import torch
import numpy as np

# Initialize gym
gym = gymapi.acquire_gym()

# Parse arguments
args = gymutil.parse_arguments(description="Visualize Transforms")

# configure sim
sim_params = gymapi.SimParams()
sim_params.dt = 1 / 60
sim_params.substeps = 2
sim_params.up_axis = gymapi.UP_AXIS_Z
sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.8)

if args.physics_engine == gymapi.SIM_PHYSX:
    sim_params.physx.solver_type = 1
    sim_params.physx.num_position_iterations = 4
    sim_params.physx.num_velocity_iterations = 1
    sim_params.physx.num_threads = args.num_threads
    sim_params.physx.use_gpu = args.use_gpu

sim_params.use_gpu_pipeline = False
if args.use_gpu_pipeline:
    print("WARNING: Forcing CPU pipeline.")

sim = gym.create_sim(args.compute_device_id, args.graphics_device_id, args.physics_engine, sim_params)

if sim is None:
    print("*** Failed to create sim")
    quit()

# Add ground plane
plane_params = gymapi.PlaneParams()
# configure the ground plane
plane_params = gymapi.PlaneParams()
plane_params.normal = gymapi.Vec3(0, 0, 1) # z-up!
plane_params.distance = 0
plane_params.static_friction = 1
plane_params.dynamic_friction = 1
plane_params.restitution = 0
gym.add_ground(sim, plane_params)

# Create viewer
viewer = gym.create_viewer(sim, gymapi.CameraProperties())
if viewer is None:
    print("*** Failed to create viewer")
    quit()

torso_pose = gymapi.Transform()
torso_pose.p = gymapi.Vec3(0.3, 0.0, 0.5)
torso_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

wall_pose = gymapi.Transform()
wall_pose.p = gymapi.Vec3(1.0, 0.0, 1.5)
wall_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

### Asset handling
asset_root = "./assets"
asset_options = gymapi.AssetOptions()
asset_options.fix_base_link = True

torso_asset_file = "mjcf/nv_torso.xml"
torso_asset = gym.load_mjcf(sim, asset_root, torso_asset_file, asset_options)

wall_dims = gymapi.Vec3(0.0, 2.0, 3.0)
wall_asset = gym.create_box(sim, wall_dims.x, wall_dims.y, wall_dims.z, asset_options)

hold_asset_file = "mjcf/hold.xml"
hold_asset = gym.load_mjcf(sim, asset_root, hold_asset_file, asset_options)

# Set up the environment grid
num_envs = 1
spacing = 1.5
env_lower = gymapi.Vec3(-spacing, -spacing, 0.0)
env_upper = gymapi.Vec3(spacing, spacing, 3.0)

gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_R, "reset")

# cache some common handles for later use
envs = []




print("Creating %d environments" % num_envs)
num_per_row = int(math.sqrt(num_envs))

torso_handle = None
for i in range(num_envs):
    # Create env
    env = gym.create_env(sim, env_lower, env_upper, num_per_row)
    envs.append(env)

    torso_handle = gym.create_actor(env, torso_asset, torso_pose, "torso", i, 0)
    wall_handle = gym.create_actor(env, wall_asset, wall_pose, "wall", i, 0)


    hold_pose = gymapi.Transform()
    hold_pose.p = gymapi.Vec3(.975, 0.0, 0.0)
    hold_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
    
    num_holds = 5
    for j in range(num_holds):
        hold_pose.p.z = j * 0.5 + 0.5
        hold_handle = gym.create_actor(env, hold_asset, hold_pose, "hold", i, 0)
        gym.set_rigid_body_color(env, hold_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, gymapi.Vec3(1, 0, 0))

gym.viewer_camera_look_at(viewer, envs[0], gymapi.Vec3(-3, -3, 3), gymapi.Vec3(1.5, 1, 0))

initial_state = np.copy(gym.get_sim_rigid_body_states(sim, gymapi.STATE_ALL))

print(gym.get_sim_actor_count(sim))



while not gym.query_viewer_has_closed(viewer):

    for evt in gym.query_viewer_action_events(viewer):
        if evt.action == "reset" and evt.value > 0:
            gym.set_sim_rigid_body_states(sim, initial_state, gymapi.STATE_ALL)

    # Step the physics
    gym.simulate(sim)
    gym.fetch_results(sim, True)

    gym.clear_lines(viewer)

    # Update the viewer
    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, True)

    # Wait for dt to elapse in real time.
    # This synchronizes the physics simulation with the rendering rate.
    gym.sync_frame_time(sim)

print("Exiting")
gym.destroy_viewer(viewer)
gym.destroy_sim(sim)
