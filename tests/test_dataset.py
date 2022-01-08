import unittest
from unittest import mock

import responses
from responses import matchers

from composing_datasets.dataset import HateSpeechDataset, _download_data


class TestHateSpeechDataset(unittest.TestCase):
    MOCK_CSV = """,count,hate_speech,offensive_language,neither,class,tweet
        0,3,0,0,3,2,Tweet 1
        1,3,0,3,0,1,Tweet 2
        2,3,0,3,0,0,Tweet 3"""

    @mock.patch("composing_datasets.dataset._download_data")
    @mock.patch("os.makedirs")
    @mock.patch("os.path.exists")
    def test_download(self, mock_exists, mock_makedirs, mock_download):
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

    @mock.patch("composing_datasets.dataset._download_data")
    @mock.patch("os.makedirs")
    @mock.patch("os.path.exists", return_value=True)
    def test_loading_data(self, mock_exists, mock_makedirs, mock_download):
        mock_open = mock.mock_open(read_data=self.MOCK_CSV)
        with mock.patch("composing_datasets.dataset.open", new=mock_open):
            dataset = HateSpeechDataset()
            self.assertListEqual([0, 1, 1], dataset.labels)
            self.assertListEqual(["Tweet 1", "Tweet 2", "Tweet 3"], dataset.text)


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
