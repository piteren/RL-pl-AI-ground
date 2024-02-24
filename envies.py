import gym
import numpy as np
from r4c.envy import FiniteActionsRLEnvy
from typing import List, Optional



class SimpleBoardGame(FiniteActionsRLEnvy):
    """ SimpleBoardGame has N fields on the board,
    task is to get into each field once, after that game is won """

    def __init__(self, board_size= 4, **kwargs):
        super().__init__(**kwargs)
        self.__board_size = board_size
        self.__state: Optional[List[int]] = None
        self.reset()


    def _lost_episode(self) -> bool:
        return max(self.__state) > 1


    def has_won(self) -> bool:
        return set(self.__state) == {1}


    def is_terminal(self) -> bool:
        return self._lost_episode() or self.has_won()


    def run(self, action: int) -> float:
        self.__state[action] += 1
        reward = 1 if not self._lost_episode() else -1
        return reward


    def get_observation(self) -> List[int]:
        return [] + self.__state

    # INFO: seed is not used since SimpleBoardGame is deterministic
    def reset_with_seed(self, seed:int):
        self.__state = [0] * self.__board_size

    @property
    def max_steps(self) -> Optional[int]:
        return self.__board_size

    def render(self):
        print(self.__state)

    def observation_vector(self, observation:List[int]) -> np.ndarray:
        return np.asarray(observation, dtype=int)

    def get_valid_actions(self) -> List[int]:
        return list(range(self.__board_size))


class CartPoleEnvy(FiniteActionsRLEnvy):

    GYM_ENVY_NAME = 'CartPole-v1'
    # https://www.gymlibrary.dev/environments/classic_control/cart_pole/

    def __init__(
            self,
            step_reward=    1.0,    # reward returned every step
            won_reward=     100,    # reward given for won episode (last action)
            lost_reward=   -100.0,  # reward given for lost episode (last action)
            max_steps=      500,    # you can override default 500 of CartPole-v1
            **kwargs):

        self.__gym_envy = gym.make(CartPoleEnvy.GYM_ENVY_NAME)
        self.__gym_envy._max_episode_steps = max_steps

        super().__init__(**kwargs)

        self.__step_reward = float(step_reward)
        self.__won_reward = float(won_reward)
        self.__lost_reward = float(lost_reward)
        self.__is_over = False
        self.reset()

    def _lost_episode(self) -> bool:
        return self.__is_over and self.__gym_envy._elapsed_steps < self.max_steps

    def has_won(self) -> bool:
        return self.__is_over and self.__gym_envy._elapsed_steps >= self.max_steps

    def is_terminal(self) -> bool:
        return self._lost_episode() or self.has_won()

    def run(self, action:int) -> float:

        next_state, reward, game_over, info = self.__gym_envy.step(action)
        self.__is_over = game_over

        reward = self.__step_reward
        if self._lost_episode(): reward = self.__lost_reward
        if self.has_won(): reward = self.__won_reward

        return reward

    def get_observation(self) -> np.ndarray:
        return self.__gym_envy.state

    def reset_with_seed(self, seed:int):
        self.__gym_envy.reset(seed=seed)
        self.__is_over = False

    @property
    def max_steps(self) -> Optional[int]:
        return self.__gym_envy._max_episode_steps

    def render(self):
        self.__gym_envy.render()

    def observation_vector(self, observation:np.ndarray) -> np.ndarray:
        return np.asarray(observation, dtype=np.float32)

    def get_valid_actions(self) -> List[int]:
        return list(range(self.__gym_envy.action_space.n))


class AcrobotEnvy(FiniteActionsRLEnvy):

    GYM_ENVY_NAME = 'Acrobot-v1'

    def __init__(self, end_game_reward=100, **kwargs):

        self.__gym_envy = gym.make(AcrobotEnvy.GYM_ENVY_NAME)

        super().__init__(**kwargs)

        self.__end_game_reward = float(end_game_reward)
        self.__is_over = False
        self.reset()

    def _lost_episode(self) -> bool:
        return self.__is_over and self.__gym_envy._elapsed_steps < self.max_steps

    def has_won(self) -> bool:
        return self.__is_over and self.__gym_envy._elapsed_steps >= self.max_steps

    def is_terminal(self) -> bool:
        return self._lost_episode() or self.has_won()

    def run(self, action:int) -> float:
        next_state, r, game_over, info = self.__gym_envy.step(action)
        self.__is_over = game_over
        reward = self.__end_game_reward if self.has_won() else r
        return reward

    def get_observation(self) -> np.ndarray:
        return self.__gym_envy.state

    def reset_with_seed(self, seed: int):
        self.__gym_envy.reset(seed=seed)
        self.__is_over = False

    @property
    def max_steps(self) -> Optional[int]:
        return self.__gym_envy._max_episode_steps

    def render(self):
        self.__gym_envy.render()

    def observation_vector(self, observation:np.ndarray) -> np.ndarray:
        return np.asarray(observation, dtype=np.float32)

    def get_valid_actions(self) -> List[int]:
        return list(range(self.__gym_envy.action_space.n))