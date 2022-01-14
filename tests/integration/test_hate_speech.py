import unittest

from composing_datasets import TextClassificationDataset, HateSpeechDataset


class TestHateSpeechDataset(unittest.TestCase):
    def test_dataset(self):
        dataset = TextClassificationDataset(HateSpeechDataset())
        self.assertEqual(24783, len(dataset))
