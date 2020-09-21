from ray import tune

args = {
    'mode': 'cui',
    'step': 1
}

tune.run(
    "PPO",
    config={
        "env": "gym_sumo.envs:SumoEnv",
        # "env_config": {**args}
    }
)
