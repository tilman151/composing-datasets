import unittest

from composing_datasets import ImdbDataset, RevtokImdbDataset


class TestImdbDataset(unittest.TestCase):
    def test_dataset(self):
        with self.subTest("train"):
            dataset = ImdbDataset("train")
            self.assertEqual(25000, len(dataset))
        with self.subTest("test"):
            dataset = ImdbDataset("test")
            self.assertEqual(25000, len(dataset))

    def test_revtok_dataset(self):
        with self.subTest("train"):
            dataset = RevtokImdbDataset("train")
            self.assertEqual(25000, len(dataset))
        with self.subTest("test"):
            dataset = RevtokImdbDataset("test")
            self.assertEqual(25000, len(dataset))
