import unittest

from envies import CartPoleEnvy


class TestCartPoleEnvy(unittest.TestCase):

    def test_CartPoleEnvy(self):

        envy = CartPoleEnvy()
        print(envy.num_actions(), envy.get_valid_actions())
        print(envy.get_observation())
        print(envy.get_last_action_reward())
        print(envy.lost_episode())
        print(envy.won_episode())

        envy.run(0)
        print(envy.get_observation())
        envy.run(1)
        print(envy.get_observation())
