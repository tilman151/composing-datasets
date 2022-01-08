import unittest
from unittest import mock

import responses
import torch
from responses import matchers

from composing_datasets.dataset import HateSpeechDataset, _download_data


@mock.patch("os.makedirs")
@mock.patch("os.path.exists", return_value=True)
class TestHateSpeechDataset(unittest.TestCase):
    MOCK_CSV = """,count,hate_speech,offensive_language,neither,class,tweet
        0,3,0,0,3,2,Tweet 1
        1,3,0,3,0,1,Tweet 2
        2,3,0,3,0,0,Tweet 3"""

    @mock.patch("composing_datasets.dataset._download_data")
    def test_download(self, mock_download, mock_exists, mock_makedirs):
        with self.subTest("not downloaded"):
            mock_exists.return_value = False
            HateSpeechDataset()

            mock_makedirs.assert_called_with(HateSpeechDataset.DATA_ROOT, exist_ok=True)
            mock_exists.assert_called_with(HateSpeechDataset.DATA_FILE)
            mock_download.assert_called_with(
                HateSpeechDataset.DOWNLOAD_URL, HateSpeechDataset.DATA_FILE
            )

        mock_exists.reset_mock()
        mock_makedirs.reset_mock()
        mock_download.reset_mock()

        with self.subTest("already downloaded"):
            mock_exists.return_value = True
            HateSpeechDataset()

            mock_makedirs.assert_called_with(HateSpeechDataset.DATA_ROOT, exist_ok=True)
            mock_exists.assert_called_with(HateSpeechDataset.DATA_FILE)
            mock_download.assert_not_called()

    def test_loading_data(self, mock_exists, mock_makedirs):
        mock_open = mock.mock_open(read_data=self.MOCK_CSV)
        with mock.patch("composing_datasets.dataset.open", new=mock_open):
            dataset = HateSpeechDataset()
            self.assertListEqual([0, 1, 1], dataset.labels)
            self.assertListEqual(
                [["tweet", "1"], ["tweet", "2"], ["tweet", "3"]], dataset.text
            )

    def test_get_item(self, mock_exists, mock_makedirs):
        mock_open = mock.mock_open(read_data=self.MOCK_CSV)
        with mock.patch("composing_datasets.dataset.open", new=mock_open):
            dataset = HateSpeechDataset()
            expected_samples = [
                (torch.tensor([0, 1]), torch.tensor(0)),
                (torch.tensor([0, 2]), torch.tensor(1)),
                (torch.tensor([0, 3]), torch.tensor(1)),
            ]
            for (exp_ids, exp_label), (ids, label) in zip(expected_samples, dataset):
                self.assertTrue(torch.all(exp_ids == ids))
                self.assertEqual(exp_label, label)
                self.assertEqual(torch.long, ids.dtype)
                self.assertEqual(torch.long, label.dtype)

    def test_len(self, mock_exists, mock_makedirs):
        mock_open = mock.mock_open(read_data=self.MOCK_CSV)
        with mock.patch("composing_datasets.dataset.open", new=mock_open):
            dataset = HateSpeechDataset()
            self.assertEqual(3, len(dataset))


class TestDownloadData(unittest.TestCase):
    def setUp(self):
        responses.add(
            responses.GET,
            "https://test.com",
            headers={"content-length": "2048"},
            body="0" * 2048,
            match=[matchers.request_kwargs_matcher({"stream": True})],
        )

    @responses.activate
    def test_download_data(self):
        mock_open = mock.mock_open()
        with mock.patch("composing_datasets.dataset.open", new=mock_open):
            _download_data("https://test.com", "mock/file")
            mock_open.assert_called_with("mock/file", mode="wb")
            mock_open().write.assert_has_calls([mock.call(b"0" * 1024)] * 2)
