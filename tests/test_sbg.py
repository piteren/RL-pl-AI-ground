import unittest

from envies import SimpleBoardGame


class TestSBG(unittest.TestCase):

    def test_SimpleBoardGame(self):

        game = SimpleBoardGame(board_size=5)
        print(game.num_actions(), game.get_valid_actions())
        self.assertTrue(game.num_actions() == 5)
        self.assertFalse(game.lost_episode())

        game.run(0)
        obs = game.get_observation()
        print(obs)
        self.assertTrue(obs[0] == 1)
        self.assertFalse(game.lost_episode())

        game.run(0)
        self.assertTrue(game.lost_episode())
        game.reset()
        obs = game.get_observation()
        print(obs)
        self.assertFalse(game.lost_episode())

        game.run(0)
        game.run(1)
        game.run(2)
        game.run(3)
        print(game.get_observation())
        self.assertFalse(game.won_episode())
        self.assertFalse(game.lost_episode())
        game.run(4)
        print(game.get_observation())
        self.assertTrue(game.won_episode())
        self.assertFalse(game.lost_episode())
