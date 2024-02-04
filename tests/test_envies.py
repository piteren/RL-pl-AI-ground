import unittest

from envies import SimpleBoardGame, CartPoleEnvy, AcrobotEnvy


class TestEnvies(unittest.TestCase):

    def test_base(self):

        for et in [
            SimpleBoardGame,
            CartPoleEnvy,
            AcrobotEnvy
        ]:
            print(et)
            envy = et()
            obs = envy.get_observation()
            print(type(obs), obs)
            obs_vec = envy.observation_vector(obs)
            print(type(obs_vec), obs_vec)

    def test_CartPoleEnvy(self):

        for et in [
            SimpleBoardGame,
            CartPoleEnvy,
            AcrobotEnvy
        ]:

            envy = et()
            print(envy.num_actions(), envy.get_valid_actions())
            print(envy.get_observation())
            print(envy.has_won())

            envy.run(0)
            print(envy.get_observation())
            envy.run(1)
            print(envy.get_observation())
