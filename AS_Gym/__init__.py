from gym.envs.registration import register

register(
    id='AS_Gym-v0',
    entry_point='AS_Gym.envs:AS_GymEnv',
)
