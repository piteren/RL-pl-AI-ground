import unittest

from envies import SimpleBoardGame, CartPoleEnvy, AcrobotEnvy, LunarLanderEnvy
ALL_ENVIES = (
    SimpleBoardGame,
    CartPoleEnvy,
    AcrobotEnvy,
    LunarLanderEnvy,
)

class TestEnvies(unittest.TestCase):

    def test_base(self):

        for et in ALL_ENVIES:
            print(f'Envy: {et}')
            envy = et()
            obs = envy.observation
            print(f'> observation:{obs} type:{type(obs)}')
            obs_vec = envy.observation_vector(obs)
            print(f'> observation_vector:{obs_vec} type:{type(obs_vec)}')
            print()

    def test_more(self):

        for et in ALL_ENVIES:

            envy = et()
            print(f'Envy: {et}')
            print(f'> observation: {envy.observation}')
            print(f'> has_won: {envy.has_won()}')

            action = envy.sample_action()
            reward = envy.run(action)
            print(f'action:{action} reward:{reward}, observation:{envy.observation}')
            action = envy.sample_action()
            reward = envy.run(action)
            print(f'action:{action} reward:{reward}, observation:{envy.observation}')
            print()

    def test_run(self):

        max_steps = 100
        for et in ALL_ENVIES:

            envy = et(render=True)

            step = 0
            while not envy.is_terminal() and step < max_steps:
                action = envy.sample_action()
                reward = envy.run(action)
                print(step, action, reward)
                step += 1