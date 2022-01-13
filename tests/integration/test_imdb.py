import unittest

from composing_datasets import ImdbDataset


class TestImdbDataset(unittest.TestCase):
    def test_dataset(self):
        with self.subTest("train"):
            dataset = ImdbDataset("train")
            self.assertEqual(25000, len(dataset))
        with self.subTest("test"):
            dataset = ImdbDataset("test")
            self.assertEqual(25000, len(dataset))
