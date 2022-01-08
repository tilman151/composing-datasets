import csv
import os
from typing import Tuple

import requests
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(__file__)


class HateSpeechDataset(Dataset):
    DOWNLOAD_URL: str = "https://raw.githubusercontent.com/t-davidson/hate-speech-and-offensive-language/master/data/labeled_data.csv"
    DATA_ROOT: str = os.path.normpath(
        os.path.join(SCRIPT_DIR, "..", "data", "hate_speech")
    )
    DATA_FILE: str = os.path.join(DATA_ROOT, "labeled_data.csv")
    CLASS_NON_OFFENSIVE: str = "2"

    def __init__(self) -> None:
        os.makedirs(self.DATA_ROOT, exist_ok=True)
        if not os.path.exists(self.DATA_FILE):
            _download_data(self.DOWNLOAD_URL, self.DATA_FILE)
        self.text, self.labels = self._load_data()

    def _load_data(self):
        data = self._read_data()
        text, labels = self._process_data(data)

        return text, labels

    def _read_data(self):
        data = {"class": [], "text": []}
        with open(self.DATA_FILE, mode="rt") as f:
            reader = csv.DictReader(f)
            for row in reader:
                data["class"].append(row["class"])
                data["text"].append(row["tweet"])

        return data

    def _process_data(self, data):
        clean_text = data["text"]
        clean_labels = []
        for text, label in zip(data["text"], data["class"]):
            if label == self.CLASS_NON_OFFENSIVE:
                clean_labels.append(0)
            else:
                clean_labels.append(1)

        return clean_text, clean_labels

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        pass

    def __len__(self) -> int:
        pass


def _download_data(url: str, output_path: str) -> None:
    response = requests.get(url, stream=True)
    with open(output_path, mode="wb") as f:
        download_size = int(response.headers["content-length"]) // 1024
        for data in tqdm(response.iter_content(chunk_size=1024), total=download_size):
            f.write(data)
