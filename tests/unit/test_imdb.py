import os
import shutil
import tempfile
import unittest
from unittest import mock

from composing_datasets import ImdbDataset


class TestImdbDataset(unittest.TestCase):
    _tmp_folder = tempfile.mkdtemp()
    _data_file_patcher = mock.patch(
        "composing_datasets.dataset.ImdbDataset.DATA_FILE",
        new_callable=mock.PropertyMock,
    )
    _data_root_patcher = mock.patch(
        "composing_datasets.dataset.ImdbDataset.DATA_ROOT",
        new_callable=mock.PropertyMock,
    )

    @classmethod
    def setUpClass(cls):
        cls._create_split_dir(cls._tmp_folder, "train")
        cls._create_split_dir(cls._tmp_folder, "test")

        mock_data_root = cls._data_root_patcher.start()
        mock_data_file = cls._data_file_patcher.start()

        mock_data_root.return_value = cls._tmp_folder
        mock_data_file.return_value = os.path.join(cls._tmp_folder, "aclImdb")

    @classmethod
    def _create_split_dir(cls, tmp_folder, split):
        pos_dir = os.path.join(tmp_folder, "aclImdb", split, "pos")
        cls._create_dir_with_file(pos_dir, file_content=split, rating=10)

        neg_dir = os.path.join(tmp_folder, "aclImdb", split, "neg")
        cls._create_dir_with_file(neg_dir, file_content=split, rating=1)

    @classmethod
    def _create_dir_with_file(cls, dir_path, file_content, rating):
        os.makedirs(dir_path)
        with open(os.path.join(dir_path, f"1_{rating}.txt"), mode="wt") as f:
            f.write(file_content)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls._tmp_folder)
        cls._data_root_patcher.stop()
        cls._data_file_patcher.stop()

    def test_selecting_split(self):
        with self.subTest("train"):
            dataset = ImdbDataset("train")
            self.assertEqual("train", dataset.split)
        with self.subTest("test"):
            dataset = ImdbDataset("test")
            self.assertEqual("test", dataset.split)

    def test_loading_data(self):
        with self.subTest("train"):
            dataset = ImdbDataset("train")
            self.assertListEqual([10, 1], dataset.labels)
            self.assertListEqual(["train", "train"], dataset.text)
        with self.subTest("test"):
            dataset = ImdbDataset("test")
            self.assertListEqual([10, 1], dataset.labels)
            self.assertListEqual(["test", "test"], dataset.text)
