import csv
import os
import re
import tarfile
import zipfile
from abc import ABCMeta, abstractmethod
from typing import Tuple, List, Dict, Optional, Callable, Any
from urllib.parse import urlparse

import requests
import revtok
import torch
import torchtext
from torch.utils.data import Dataset
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(__file__)


class TextClassificationDataset(Dataset, metaclass=ABCMeta):
    """Text classification dataset base class."""

    DOWNLOAD_URL: str
    DATA_ROOT: str
    DATA_FILE: str

    def __init__(self, tokenizer: Optional[str] = None, **kwargs) -> None:
        """
        This is a base class for text classification datasets.

        The dataset provides IDs and labels for blobs of text as long tensors. The
        dataset files are downloaded to the project's data folder if not already
        present. Each text is split with a tokenizer specified through a string
        argument. The vocabulary of all tokens with a default index is built
        afterwards.

        :param tokenizer: One of 'spacy', 'moses', 'toktok', 'revtok', 'subword'
                          or 'basic_english'. Defaults to the last one.
        """
        os.makedirs(self.DATA_ROOT, exist_ok=True)
        if not os.path.exists(self.DATA_FILE):
            _download_data(self.DOWNLOAD_URL, self.DATA_ROOT)
        self.text, self.labels = self._load_data()

        self.tokenizer = self._get_tokenizer(tokenizer, kwargs)
        self.tokens = [self.tokenizer(text) for text in self.text]

        self.vocab = torchtext.vocab.build_vocab_from_iterator(self.tokens)
        self.vocab.set_default_index(len(self.vocab))
        self.token_ids = [self._tokens_to_tensor(tokens) for tokens in self.tokens]
        self.labels = [torch.tensor(label, dtype=torch.long) for label in self.labels]

    @abstractmethod
    def _load_data(self) -> Tuple[List[str], List[int]]:
        """Load all data and return a list of strings and a list of int labels."""
        pass

    def _get_tokenizer(
        self, tokenizer: Optional[str], kwargs: Dict[Any, Any]
    ) -> Callable:
        if tokenizer is None:
            tokenizer = torchtext.data.get_tokenizer("basic_english")
        elif not kwargs:
            tokenizer = torchtext.data.get_tokenizer(tokenizer)
        elif tokenizer == "revtok":
            tokenizer = _RevtokTokenizer(kwargs)
        else:
            raise ValueError(f"Unknown tokenizer '{tokenizer}'.")

        return tokenizer

    def _tokens_to_tensor(self, tokens: List[str]) -> torch.Tensor:
        return torch.tensor([self.vocab[token] for token in tokens], dtype=torch.long)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return token ids and label of the requested sample as long tensors."""
        return self.token_ids[index], self.labels[index]

    def __len__(self) -> int:
        """Number of tweets in the dataset."""
        return len(self.text)


class _RevtokTokenizer:
    """Class wrapper for revtok tokenizer."""

    def __init__(self, kwargs: Dict[Any, Any]) -> None:
        self.kwargs = kwargs

    def __call__(self, x: str) -> List[str]:
        return revtok.tokenize(x, **self.kwargs)


class HateSpeechDataset(TextClassificationDataset):
    """
    This class provides token IDs and labels for the hate speech dataset sourced
    from twitter.

    The dataset files are downloaded to the projects data folder if not already
    present. Each tweet is split with the torchtext basic english tokenizer. The
    vocabulary of all tokens with a default index is built afterwards. The class
    labels correspond to hate speech (0), offensive (1) and neither (2).
    """

    DOWNLOAD_URL: str = "https://raw.githubusercontent.com/t-davidson/hate-speech-and-offensive-language/master/data/labeled_data.csv"
    DATA_ROOT: str = os.path.normpath(
        os.path.join(SCRIPT_DIR, "..", "data", "hate_speech")
    )
    DATA_FILE: str = os.path.join(DATA_ROOT, "labeled_data.csv")

    def _load_data(self) -> Tuple[List[str], List[int]]:
        data = self._read_data()
        text, labels = self._process_data(data)

        return text, labels

    def _read_data(self) -> Dict[str, List[str]]:
        data = {"class": [], "text": []}
        with open(self.DATA_FILE, mode="rt") as f:
            reader = csv.DictReader(f)
            for row in reader:
                data["class"].append(row["class"])
                data["text"].append(row["tweet"])

        return data

    def _process_data(self, data: Dict[str, List[str]]) -> Tuple[List[str], List[int]]:
        clean_text = data["text"]
        clean_labels = [int(label) for label in data["class"]]

        return clean_text, clean_labels


class ImdbDataset(TextClassificationDataset):
    """Class for the Large Movie Review Dataset (imdb) dataset."""

    DOWNLOAD_URL: str = "http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
    DATA_ROOT: str = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "data", "imdb"))
    DATA_FILE: str = os.path.join(DATA_ROOT, "aclImdb")

    CLASS_PATTERN = re.compile(r"\d+_(?P<rating>\d+).txt")

    def __init__(self, split: str, tokenizer: Optional[str] = None) -> None:
        """
        This class loads the Large Movie Review Dataset (imdb) dataset.

        The dataset provides IDs and labels for movie reviews as long tensors. The
        dataset files are downloaded to the project's data folder if not already
        present. Each review is split with a tokenizer specified through a string
        argument. The vocabulary of all tokens with a default index is built
        afterwards.

        :param split: either 'train' or 'test' to select the split
        :param tokenizer: One of 'spacy', 'moses', 'toktok', 'revtok', 'subword'
                          or 'basic_english'. Defaults to the last one.
        """
        if split not in ["train", "test"]:
            raise ValueError("Unknown split supplied. Use either 'train' or 'test'.")
        self.split = split
        super(ImdbDataset, self).__init__(tokenizer)

    def _load_data(self) -> Tuple[List[str], List[int]]:
        pos_data = self._read_data(os.path.join(self.DATA_FILE, self.split, "pos"))
        neg_data = self._read_data(os.path.join(self.DATA_FILE, self.split, "neg"))
        pos_data["text"].extend(neg_data["text"])
        pos_data["class"].extend(neg_data["class"])
        text, labels = self._process_data(pos_data)

        return text, labels

    def _read_data(self, input_folder) -> Dict[str, List[str]]:
        data = {"class": [], "text": []}
        for file_name in sorted(os.listdir(input_folder)):
            data["class"].append(self._get_rating_from_file_name(file_name))
            with open(os.path.join(input_folder, file_name), mode="rt") as f:
                data["text"].append(f.read())

        return data

    def _get_rating_from_file_name(self, file_name):
        if (match := self.CLASS_PATTERN.match(file_name)) is not None:
            rating = match.group("rating")
        else:
            raise RuntimeError(
                f"Filename '{file_name}' does not match the schema '<id>_<rating>.txt'."
            )

        return rating

    def _process_data(self, data: Dict[str, List[str]]) -> Tuple[List[str], List[int]]:
        clean_text = data["text"]
        clean_labels = [int(label) for label in data["class"]]

        return clean_text, clean_labels


def _download_data(url: str, output_folder: str) -> None:
    """Download the content of a URL to a folder and extract if it is archive."""
    file_name = os.path.basename(urlparse(url).path)
    output_file_path = os.path.join(output_folder, file_name)
    _fetch_data(url, output_file_path)
    _extract_data_if_archive(output_file_path, output_folder)


def _fetch_data(url, output_path):
    response = requests.get(url, stream=True)
    with open(output_path, mode="wb") as f:
        download_size = int(response.headers["content-length"]) // 1024
        for data in tqdm(response.iter_content(chunk_size=1024), total=download_size):
            f.write(data)


def _extract_data_if_archive(archive_path, output_folder):
    if archive_path.endswith(".zip"):
        with zipfile.ZipFile(archive_path, mode="r") as archive:
            archive.extractall(output_folder)
    elif archive_path.endswith("tar.gz"):
        with tarfile.open(archive_path, mode="r:gz") as archive:
            archive.extractall(output_folder)
