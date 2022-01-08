import unittest

from composing_datasets import HateSpeechDataset


class TestHateSpeechDataset(unittest.TestCase):
    def test_dataset(self):
        dataset = HateSpeechDataset()
        self.assertEqual(24783, len(dataset))
