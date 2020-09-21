import gym

# default
# args = {
#     'step_length': 0.01,
#     'isgraph': True,
#     'area': 'nishiwaseda',
#     'carnum': 100,
#     'mode': 'gui' (or 'cui'),
#     'simlation_step': 100
# }
args = {
    'carnum': 10,
}


class SumoAgent:
    def __init__(self):
        self.env = gym.make('gym_sumo:sumo-v0', **args)

    def policy(self, observation):
        """detemine action of agent

        Args:
            observation (gym.Space.Box(4, 2)): condition of sumo environment

        Returns:
            list: agent action
                  action[0] = direction of agent
                  action[1] = accel or decel of agent
        """
        action = [1, -0.1]
        return action
