import csv
import os
from typing import Tuple, List, Dict, Optional, Callable

import requests
import torch
import torchtext.data
from torch.utils.data import Dataset
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(__file__)


class HateSpeechDataset(Dataset):
    """Automated Hate Speech Detection and the Problem of Offensive Language dataset."""

    DOWNLOAD_URL: str = "https://raw.githubusercontent.com/t-davidson/hate-speech-and-offensive-language/master/data/labeled_data.csv"
    DATA_ROOT: str = os.path.normpath(
        os.path.join(SCRIPT_DIR, "..", "data", "hate_speech")
    )
    DATA_FILE: str = os.path.join(DATA_ROOT, "labeled_data.csv")

    def __init__(self, tokenizer: Optional[str] = None) -> None:
        """
        This class provides token IDs and labels for the hate speech dataset sourced
        from twitter.

        The dataset files are downloaded to the projects data folder if not already
        present. Each tweet is split with the torchtext basic english tokenizer. The
        vocabulary of all tokens with a default index is built afterwards. The class
        labels correspond to hate speech (0), offensive (1) and neither (2).
        """
        os.makedirs(self.DATA_ROOT, exist_ok=True)
        if not os.path.exists(self.DATA_FILE):
            _download_data(self.DOWNLOAD_URL, self.DATA_FILE)
        self.text, self.labels = self._load_data()

        self.tokenizer = self._get_tokenizer(tokenizer)
        self.tokens = [self.tokenizer(text) for text in self.text]

        self.vocab = torchtext.vocab.build_vocab_from_iterator(self.tokens)
        self.vocab.set_default_index(len(self.vocab))
        self.token_ids = [self._tokens_to_tensor(tokens) for tokens in self.tokens]
        self.labels = [torch.tensor(label, dtype=torch.long) for label in self.labels]

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

    def _get_tokenizer(self, tokenizer: Optional[str]) -> Callable:
        if tokenizer is None:
            tokenizer = torchtext.data.get_tokenizer("basic_english")
        else:
            tokenizer = torchtext.data.get_tokenizer(tokenizer)

        return tokenizer

    def _tokens_to_tensor(self, tokens: List[str]) -> torch.Tensor:
        return torch.tensor([self.vocab[token] for token in tokens], dtype=torch.long)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return token ids and label of the requested sample as long tensors."""
        return self.token_ids[index], self.labels[index]

    def __len__(self) -> int:
        """Number of tweets in the dataset."""
        return len(self.text)


def _download_data(url: str, output_path: str) -> None:
    """Download the content of a URL to a file."""
    response = requests.get(url, stream=True)
    with open(output_path, mode="wb") as f:
        download_size = int(response.headers["content-length"]) // 1024
        for data in tqdm(response.iter_content(chunk_size=1024), total=download_size):
            f.write(data)
