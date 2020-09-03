import os

from torchvision.utils import save_image

from gym import utils
from gym.envs.robotics import fetch_env
import numpy as np
from vae.import_vae import goal_set_fetch_pick

# Ensure we get the path separator correct on windows
MODEL_XML_PATH = os.path.join('fetch', 'pick_and_place.xml')


class FetchPickAndPlaceEnv(fetch_env.FetchEnv, utils.EzPickle):
    def __init__(self, reward_type='sparse'):
        initial_qpos = {
            'robot0:slide0': 0.405,
            'robot0:slide1': 0.48,
            'robot0:slide2': 0.0,
            'object0:joint': [1.25, 0.53, 0.4, 1., 0., 0., 0.],
        }
        fetch_env.FetchEnv.__init__(
            self, MODEL_XML_PATH, has_object=True, block_gripper=False, n_substeps=20,
            gripper_extra_height=0.2, target_in_the_air=True, target_offset=0.0,
            obj_range=0.15, target_range=0.15, distance_threshold=0.05,
            initial_qpos=initial_qpos, reward_type=reward_type)
        utils.EzPickle.__init__(self)

    def _sample_goal(self):
        goal = goal_set_fetch_pick[np.random.randint(5)]
        goal = self.fetch_pick_vae.format(goal)
        save_image(goal.cpu().view(-1, 3, self.img_size, self.img_size), 'videos/goal/goal.png')
        x, y = self.fetch_pick_vae.encode(goal)
        goal = self.fetch_pick_vae.reparameterize(x, y)
        goal = goal.detach().cpu().numpy()
        goal = np.squeeze(goal)
        return goal.copy()

    def _get_image(self, img_name='default'):
        local_vae = self.fetch_pick_vae
        np.array(self.render(mode='rgb_array',
                             width=84, height=84))
        rgb_array = np.array(self.render(mode='rgb_array',
                                         width=84, height=84))
        tensor = local_vae.format(rgb_array)
        x, y = local_vae.encode(tensor)
        obs = local_vae.reparameterize(x, y)
        obs = obs.detach().cpu().numpy()
        obs = np.squeeze(obs)
        save_image(tensor.cpu().view(-1, 3, 84, 84), img_name)
        return obs
