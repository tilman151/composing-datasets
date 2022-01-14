import unittest

from composing_datasets import TextClassificationDataset, ImdbDataset


class TestImdbDataset(unittest.TestCase):
    def test_dataset(self):
        with self.subTest("train"):
            dataset = TextClassificationDataset(ImdbDataset("train"))
            self.assertEqual(25000, len(dataset))
        with self.subTest("test"):
            dataset = TextClassificationDataset(ImdbDataset("test"))
            self.assertEqual(25000, len(dataset))
