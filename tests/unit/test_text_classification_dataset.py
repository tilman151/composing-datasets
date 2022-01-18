import unittest
from unittest import mock

import responses
import revtok
import torch
import torchtext
from responses import matchers

from composing_datasets.dataset import (
    TextClassificationDataset,
    _RevtokTokenizer,
    _download_data,
    _fetch_data,
    _extract_data_if_archive,
)


class DummyDataset(TextClassificationDataset):
    DOWNLOAD_URL = "http://example.com"
    DATA_ROOT = "foo/bar"
    DATA_FILE = "foo/bar/baz.csv"

    def _load_data(self):
        return ["Tweet 1", "Tweet 2", "Tweet 3"], [2, 1, 0]


@mock.patch("os.makedirs")
@mock.patch("os.path.exists", return_value=True)
class TestTextClassificationDataset(unittest.TestCase):
    @mock.patch("composing_datasets.dataset._download_data")
    def test_download(self, mock_download, mock_exists, mock_makedirs):
        with self.subTest("not downloaded"):
            mock_exists.return_value = False
            DummyDataset()

            mock_makedirs.assert_called_with(DummyDataset.DATA_ROOT, exist_ok=True)
            mock_exists.assert_called_with(DummyDataset.DATA_FILE)
            mock_download.assert_called_with(
                DummyDataset.DOWNLOAD_URL, DummyDataset.DATA_ROOT
            )

        mock_exists.reset_mock()
        mock_makedirs.reset_mock()
        mock_download.reset_mock()

        with self.subTest("already downloaded"):
            mock_exists.return_value = True
            DummyDataset()

            mock_makedirs.assert_called_with(DummyDataset.DATA_ROOT, exist_ok=True)
            mock_exists.assert_called_with(DummyDataset.DATA_FILE)
            mock_download.assert_not_called()

    def test_get_item(self, mock_exists, mock_makedirs):
        dataset = DummyDataset()
        expected_samples = [
            (torch.tensor([0, 1]), torch.tensor(2)),
            (torch.tensor([0, 2]), torch.tensor(1)),
            (torch.tensor([0, 3]), torch.tensor(0)),
        ]
        for (exp_ids, exp_label), (ids, label) in zip(expected_samples, dataset):
            self.assertTrue(torch.all(exp_ids == ids))
            self.assertEqual(exp_label, label)
            self.assertEqual(torch.long, ids.dtype)
            self.assertEqual(torch.long, label.dtype)

    def test_len(self, mock_exists, mock_makedirs):
        dataset = DummyDataset()
        self.assertEqual(3, len(dataset))

    def test_oov(self, mock_exists, mock_makedirs):
        dataset = DummyDataset()
        exp_default_index = len(dataset.vocab)
        self.assertListEqual([exp_default_index], dataset.vocab(["foo"]))

    def test_setting_tokenizer(self, mock_exists, mock_makedirs):
        with self.subTest("default tokenizer"):
            dataset = DummyDataset()
            self.assertIs(
                torchtext.data.utils._basic_english_normalize, dataset.tokenizer
            )
        with self.subTest("revtok tokenizer"):
            dataset = DummyDataset("revtok")
            self.assertIs(revtok.tokenize, dataset.tokenizer)
        with self.subTest("unknown tokenizer"):
            self.assertRaises(ValueError, DummyDataset, "foobar")
        with self.subTest("revtok with kwargs"):
            dataset = DummyDataset("revtok", decap=True, split_punctuation=False)
            self.assertIsInstance(dataset.tokenizer, _RevtokTokenizer)
            self.assertDictEqual(
                {"decap": True, "split_punctuation": False}, dataset.tokenizer.kwargs
            )


class TestDownloadData(unittest.TestCase):
    def setUp(self):
        responses.add(
            responses.GET,
            "https://test.com/example.csv",
            headers={"content-length": "2048"},
            body="0" * 2048,
            match=[matchers.request_kwargs_matcher({"stream": True})],
        )

    @mock.patch("composing_datasets.dataset._extract_data_if_archive")
    @mock.patch("composing_datasets.dataset._fetch_data")
    def test_download_data(self, mock_fetch, mock_extract):
        _download_data("https://test.com/example.csv", "mock/file")
        mock_fetch.assert_called_with(
            "https://test.com/example.csv", "mock/file/example.csv"
        )
        mock_extract.assert_called_with("mock/file/example.csv", "mock/file")

    @responses.activate
    def test_fetch_data(self):
        mock_open = mock.mock_open()
        with mock.patch("composing_datasets.dataset.open", new=mock_open):
            _fetch_data("https://test.com/example.csv", "mock/file/example.csv")
            mock_open.assert_called_with("mock/file/example.csv", mode="wb")
            mock_open().write.assert_has_calls([mock.call(b"0" * 1024)] * 2)

    def test_extract_data_if_archive(self):
        with self.subTest("zip"):
            with mock.patch("zipfile.ZipFile") as mock_zip_file:
                _extract_data_if_archive("foo/bar/example.zip", "foo/bar")
                mock_zip_file.assert_called_with("foo/bar/example.zip", mode="r")
                mock_zip_file().__enter__().extractall.assert_called_with("foo/bar")
        with self.subTest("targ.gz"):
            with mock.patch("tarfile.open") as mock_tar_file:
                _extract_data_if_archive("foo/bar/example.tar.gz", "foo/bar")
                mock_tar_file.assert_called_with("foo/bar/example.tar.gz", mode="r:gz")
                mock_tar_file().__enter__().extractall.assert_called_with("foo/bar")
