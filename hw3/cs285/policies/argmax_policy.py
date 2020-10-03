import numpy as np


class ArgMaxPolicy(object):

    def __init__(self, critic):
        self.critic = critic

    def get_action(self, obs):
        if len(obs.shape) > 3: # since obs is length*width*height, so shape >3 means there are batch axis
            observation = obs
        else:
            observation = obs[None]
        
        ## TODO return the action that maxinmizes the Q-value ---------------------
        # at the current observation as the output
        # actions = TODO
        q_value = self.critic.qa_values(observation)
        action = q_value.argmax(axis=1)

        return action.squeeze()