import gym
import numpy as np
from r4c.envy import FiniteActionsRLEnvy
from typing import List, Optional, Tuple



class SimpleBoardGame(FiniteActionsRLEnvy):
    """
        gots N fields on the board,
        task is to get into each field once, after that game is won
    """

    def __init__(self, board_size= 4, **kwargs):

        self.__board_size = board_size

        FiniteActionsRLEnvy.__init__(self, **kwargs)

        self.__state: Optional[List[int]] = None
        self.reset()

    def _lost_episode(self) -> bool:
        return max(self.__state) > 1

    def _won_episode(self) -> bool:
        return set(self.__state) == {1}

    def _is_terminal(self) -> bool:
        return self._lost_episode() or self._won_episode()

    def run(self, action: int) -> Tuple[float,bool,bool]:

        if self._is_terminal(): self.reset()

        self.__state[action] += 1

        reward = 1 if not self._lost_episode() else -1
        return reward, self._is_terminal(), self._won_episode()

    def get_observation(self) -> List[int]:
        return [] + self.__state

    def _reset_with_seed(self, seed:int):
        # seed is not used since SimpleBoardGame is deterministic
        self.__state = [0] * self.__board_size

    def get_max_steps(self) -> Optional[int]:
        return self.__board_size

    def render(self):
        print(self.__state)

    def prep_observation_vec(self, observation:List[int]) -> np.ndarray:
        return np.asarray(observation, dtype=float)

    def get_valid_actions(self) -> List[int]:
        return list(range(self.__board_size))


class CartPoleEnvy(FiniteActionsRLEnvy):

    GYM_ENVY_NAME = 'CartPole-v1'
    # https://www.gymlibrary.dev/environments/classic_control/cart_pole/

    def __init__(
            self,
            reward_scale=   1.0,    # CartPole-v1 gives 1 for every step, we set reward == 1*reward_scale
            won_reward=     100,    # reward given for won episode (last action)
            lost_penalty=   -100.0, # reward given for lost episode (last action)
            max_steps=      500,    # you can override default 500 of CartPole-v1
            **kwargs):

        self.__gym_envy = gym.make(CartPoleEnvy.GYM_ENVY_NAME)
        self.__gym_envy._max_episode_steps = max_steps

        FiniteActionsRLEnvy.__init__(self, **kwargs)

        self.__reward_scale = float(reward_scale)
        self.__won_reward = float(won_reward)
        self.__lost_penalty = float(lost_penalty)
        self.__is_over = False
        self.reset()

    def _lost_episode(self) -> bool:
        return self.__is_over and self.__gym_envy._elapsed_steps < self.get_max_steps()

    def _won_episode(self) -> bool:
        return self.__is_over and self.__gym_envy._elapsed_steps >= self.get_max_steps()

    def _is_terminal(self) -> bool:
        return self._lost_episode() or self._won_episode()

    def run(self, action:int) -> Tuple[float,bool,bool]:

        if self._is_terminal(): self.reset()

        next_state, r, game_over, info = self.__gym_envy.step(action)
        self.__is_over = game_over

        reward = self.__reward_scale * r
        if self._lost_episode(): reward = self.__lost_penalty
        if self._won_episode(): reward = self.__won_reward
        return reward, self._is_terminal(), self._won_episode()

    def get_observation(self) -> np.ndarray:
        return self.__gym_envy.state

    def _reset_with_seed(self, seed:int):
        self.__gym_envy.reset(seed=seed)
        self.__is_over = False

    def get_max_steps(self) -> Optional[int]:
        return self.__gym_envy._max_episode_steps

    def render(self):
        self.__gym_envy.render()

    def prep_observation_vec(self, observation:np.ndarray) -> np.ndarray:
        return np.asarray(observation, dtype=np.float32)

    def get_valid_actions(self) -> List[int]:
        return list(range(self.__gym_envy.action_space.n))


class AcrobotEnvy(FiniteActionsRLEnvy):

    GYM_ENVY_NAME = 'Acrobot-v1'

    def __init__(self, end_game_reward=100, **kwargs):

        self.__gym_envy = gym.make(AcrobotEnvy.GYM_ENVY_NAME)

        FiniteActionsRLEnvy.__init__(self, **kwargs)

        self.__end_game_reward = float(end_game_reward)
        self.__is_over = False
        self.reset()

    def _lost_episode(self) -> bool:
        return self.__is_over and self.__gym_envy._elapsed_steps < self.get_max_steps()

    def _won_episode(self) -> bool:
        return self.__is_over and self.__gym_envy._elapsed_steps >= self.get_max_steps()

    def _is_terminal(self) -> bool:
        return self._lost_episode() or self._won_episode()

    def run(self, action:int) -> Tuple[float,bool,bool]:

        if self._is_terminal(): self.reset()

        next_state, r, game_over, info = self.__gym_envy.step(action)
        self.__is_over = game_over

        won = self._won_episode()
        reward = self.__end_game_reward if won else r
        return reward, self._is_terminal(), won

    def get_observation(self) -> np.ndarray:
        return self.__gym_envy.state

    def _reset_with_seed(self, seed: int):
        self.__gym_envy.reset(seed=seed)
        self.__is_over = False

    def get_max_steps(self) -> Optional[int]:
        return self.__gym_envy._max_episode_steps

    def render(self):
        self.__gym_envy.render()

    def prep_observation_vec(self, observation:np.ndarray) -> np.ndarray:
        return np.asarray(observation, dtype=np.float32)

    def get_valid_actions(self) -> List[int]:
        return list(range(self.__gym_envy.action_space.n))


if __name__ == "__main__":

    for et in [
        SimpleBoardGame,
        CartPoleEnvy,
        AcrobotEnvy
    ]:
        print(et)
        envy = et(seed=123)
        obs = envy.get_observation()
        print(type(obs), obs)
        obs_vec = envy.prep_observation_vec(obs)
        print(type(obs_vec), obs_vec)
