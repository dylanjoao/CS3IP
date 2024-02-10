from gymnasium.envs.registration import register
register(
    id='TorsoClimb-v0',
    entry_point='torso_climb.env:TorsoClimbEnv',
    # max_episode_steps=1000
)