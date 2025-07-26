class AgentTrainer(object):
    def __init__(self, name, model, obs_shape, act_space, args):
        raise NotImplementedError()

    def action(self, obs):
        raise NotImplementedError()

    def process_experience(self, obs, act, rew, new_obs, done, terminal):
        raise NotImplementedError()

    def preupdate(self):
        raise NotImplementedError()

    def update(self, agents):
        raise NotImplementedError()