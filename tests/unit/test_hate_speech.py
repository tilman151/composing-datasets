import unittest
from unittest import mock

from composing_datasets import HateSpeechDataset


class TestHateSpeechDataset(unittest.TestCase):
    MOCK_CSV = """,count,hate_speech,offensive_language,neither,class,tweet
        0,3,0,0,3,2,Tweet 1
        1,3,0,3,0,1,Tweet 2
        2,3,0,3,0,0,Tweet 3"""

    def test_loading_data(self):
        mock_open = mock.mock_open(read_data=self.MOCK_CSV)
        with mock.patch("composing_datasets.dataset.open", new=mock_open):
            dataset = HateSpeechDataset()
            text, labels = zip(*dataset)
            self.assertListEqual([2, 1, 0], list(labels))
            self.assertListEqual(["Tweet 1", "Tweet 2", "Tweet 3"], list(text))
