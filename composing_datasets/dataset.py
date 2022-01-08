import csv
import os
from typing import Tuple, List, Dict

import requests
import torch
import torchtext.data
from torch.utils.data import Dataset
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(__file__)


class HateSpeechDataset(Dataset):
    DOWNLOAD_URL: str = "https://raw.githubusercontent.com/t-davidson/hate-speech-and-offensive-language/master/data/labeled_data.csv"
    DATA_ROOT: str = os.path.normpath(
        os.path.join(SCRIPT_DIR, "..", "data", "hate_speech")
    )
    DATA_FILE: str = os.path.join(DATA_ROOT, "labeled_data.csv")

    def __init__(self) -> None:
        os.makedirs(self.DATA_ROOT, exist_ok=True)
        if not os.path.exists(self.DATA_FILE):
            _download_data(self.DOWNLOAD_URL, self.DATA_FILE)
        self.text, self.labels = self._load_data()

        self.tokenizer = torchtext.data.get_tokenizer("basic_english")
        self.tokens = [self.tokenizer(text) for text in self.text]

        self.vocab = torchtext.vocab.build_vocab_from_iterator(self.tokens)
        self.token_ids = [self._tokens_to_tensor(tokens) for tokens in self.tokens]
        self.labels = [torch.tensor(label, dtype=torch.long) for label in self.labels]

    def _load_data(self):
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

    def _tokens_to_tensor(self, tokens: List[str]) -> torch.Tensor:
        return torch.tensor([self.vocab[token] for token in tokens], dtype=torch.long)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.token_ids[index], self.labels[index]

    def __len__(self) -> int:
        return len(self.text)


def _download_data(url: str, output_path: str) -> None:
    response = requests.get(url, stream=True)
    with open(output_path, mode="wb") as f:
        download_size = int(response.headers["content-length"]) // 1024
        for data in tqdm(response.iter_content(chunk_size=1024), total=download_size):
            f.write(data)
