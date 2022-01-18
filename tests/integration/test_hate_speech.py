import unittest

from composing_datasets import HateSpeechDataset, RevtokHateSpeechDataset


class TestHateSpeechDataset(unittest.TestCase):
    def test_dataset(self):
        dataset = HateSpeechDataset()
        self.assertEqual(24783, len(dataset))

    def test_revtok_dataset(self):
        dataset = RevtokHateSpeechDataset()
        self.assertEqual(24783, len(dataset))
