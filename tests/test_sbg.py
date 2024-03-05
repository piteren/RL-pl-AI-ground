import unittest

from envies import SimpleBoardGame


class TestSimpleBoardGame(unittest.TestCase):

    def test_base(self):
        game = SimpleBoardGame(board_size=5)
        print(game.num_actions, game.get_valid_actions(), game.observation)
        self.assertTrue(game.num_actions == 5)
        self.assertFalse(game.is_terminal())

    def test_run(self):
        game = SimpleBoardGame(board_size=5)
        game.run(0)
        obs = game.observation
        print(obs)
        self.assertTrue(obs == [1, 0, 0, 0, 0])
        self.assertFalse(game.is_terminal())

    def test_terminal_reset(self):
        game = SimpleBoardGame(board_size=5)
        game.run(0)
        game.run(0)
        obs = game.observation
        print(obs)
        self.assertTrue(game.is_terminal())

        game.reset()
        obs = game.observation
        print(obs)
        self.assertFalse(game.is_terminal())

    def test_won(self):
        game = SimpleBoardGame(board_size=5)
        game.run(0)
        game.run(1)
        game.run(2)
        game.run(3)
        print(game.observation)
        self.assertFalse(game.has_won())
        self.assertFalse(game.is_terminal())
        game.run(4)
        print(game.observation)
        self.assertTrue(game.has_won())
        self.assertTrue(game.is_terminal())
