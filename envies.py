from abc import ABC
import gymnasium
import numpy as np
import random
from r4c.envy import RLEnvy, FiniteActionsRLEnvy, CASRLEnvy
from typing import List, Optional



class SimpleBoardGame(FiniteActionsRLEnvy):
    """ SimpleBoardGame has N fields on the board,
    task is to get into each field once, after that game is won """

    def __init__(self, board_size=4, render=False, **kwargs):

        self.board_size = board_size
        super().__init__(**kwargs)

        self.kwargs = kwargs
        self.kwargs['board_size'] = board_size
        self.render = render

    def build_renderable(self) -> "RLEnvy":
        return SimpleBoardGame(**self.kwargs, render=True)

    @property
    def observation(self) -> List[int]:
        return [] + self.state

    def sample_action(self) -> int:
        return random.sample(self.get_valid_actions(), k=1)[0]

    def _lost_episode(self) -> bool:
        return max(self.state) > 1

    def has_won(self) -> bool:
        return set(self.state) == {1}

    def is_terminal(self) -> bool:
        return self._lost_episode() or self.has_won()

    def run(self, action:int) -> float:
        self.state[action] += 1
        reward = 1 if not self._lost_episode() else -1
        if self.render:
            print(self.state)
        return reward

    def reset_with_seed(self, seed:int):
        """ seed is not used since SimpleBoardGame is deterministic """
        self.state = [0] * self.board_size
        return self.state

    @property
    def max_steps(self) -> Optional[int]:
        return self.board_size

    def observation_vector(self, observation:List[int]) -> np.ndarray:
        return np.asarray(observation, dtype=int)

    def get_valid_actions(self) -> List[int]:
        return list(range(self.board_size))


class GymBasedEnvy(RLEnvy, ABC):
    """ GymBasedEnvy is an abstract to easily build RLEnvy based on Gymnasium Envy """

    GYM_KWARGS = {'id': '__GYM_ENVY_NAME__'}

    def __init__(
            self,
            max_steps=  500,  # you can override default of Gym Envy
            render=     False,
            **kwargs):

        render_mode = "human" if render else None
        self.gym_envy = gymnasium.make(render_mode=render_mode, **self.GYM_KWARGS)

        super().__init__(**kwargs)

        self.gym_envy._max_episode_steps = max_steps
        self._max_steps = max_steps
        self.is_over = False
        self.step = 0

        self.kwargs = kwargs
        self.kwargs['max_steps'] = max_steps

    def build_renderable(self) -> "RLEnvy":
        return type(self)(**self.kwargs, render=True)

    def sample_action(self) -> object:
        return self.gym_envy.action_space.sample()

    def _lost_episode(self) -> bool:
        return self.is_over and self.step < self.max_steps

    def has_won(self) -> bool:
        return self.is_over and self.step >= self.max_steps

    def is_terminal(self) -> bool:
        return self._lost_episode() or self.has_won()

    def _override_step_reward(self, reward:float) -> float:
        """ allows to override default step reward """
        return reward

    def run(self, action:int) -> float:
        next_state, reward, terminated, truncated, info = self.gym_envy.step(action)
        self.step += 1
        self.state = next_state
        game_over = terminated or truncated
        self.is_over = game_over
        return self._override_step_reward(float(reward))

    def reset_with_seed(self, seed:int):
        self.state, _ = self.gym_envy.reset(seed=seed)
        self.is_over = False
        self.step = 0
        return self.state

    @property
    def max_steps(self) -> int:
        return self._max_steps

    def observation_vector(self, observation:np.ndarray) -> np.ndarray:
        return observation.astype(float)


class CartPoleEnvy(GymBasedEnvy, FiniteActionsRLEnvy):

    GYM_KWARGS = {'id': 'CartPole-v1'}

    def __init__(
            self,
            step_reward=    1.0,    # reward returned every step
            won_reward=     100,    # reward given for won episode (last action)
            lost_reward=   -100.0,  # reward given for lost episode (last action)
            max_steps=      500,    # you can override default 500 of CartPole-v1
            **kwargs):

        super().__init__(max_steps=max_steps, **kwargs)

        self.step_reward = float(step_reward)
        self.won_reward = float(won_reward)
        self.lost_reward = float(lost_reward)

        self.kwargs.update({
            'step_reward':  step_reward,
            'won_reward':   won_reward,
            'lost_reward':  lost_reward})

    def _override_step_reward(self, reward:float) -> float:
        reward = self.step_reward
        if self._lost_episode(): reward = self.lost_reward
        if self.has_won(): reward = self.won_reward
        return reward

    def get_valid_actions(self) -> List[int]:
        return list(range(self.gym_envy.action_space.n))


class AcrobotEnvy(GymBasedEnvy, FiniteActionsRLEnvy):

    GYM_KWARGS = {'id': 'Acrobot-v1'}

    def __init__(
            self,
            end_game_reward=    100,
            max_steps=          500,
            **kwargs):
        super().__init__(max_steps=max_steps, **kwargs)
        self.end_game_reward = float(end_game_reward)
        self.kwargs['end_game_reward'] = self.end_game_reward

    def _override_step_reward(self, reward:float) -> float:
        return self.end_game_reward if self.has_won() else reward

    def get_valid_actions(self) -> List[int]:
        return list(range(self.gym_envy.action_space.n))


class LunarLanderEnvy(GymBasedEnvy, CASRLEnvy):
    """ Continuous action space envy: action np.array([main, lateral]).
    The main engine will be turned off completely if main < 0 and the throttle scales affinely
    from 50% to 100% for 0 <= main <= 1 (in particular, the main engine doesn’t work with less than 50% power).
    Similarly, if -0.5 < lateral < 0.5, the lateral boosters will not fire at all.
    If lateral < -0.5, the left booster will fire, and if lateral > 0.5, the right booster will fire.
    Again, the throttle scales affinely from 50% to 100% between -1 and -0.5 (and 0.5 and 1, respectively). """

    GYM_KWARGS = {'id':'LunarLander-v2', 'continuous':True}

    def __init__(self, max_steps=500, **kwargs):
        super().__init__(max_steps=max_steps, **kwargs)

    @property
    def action_width(self) -> int:
        return 2