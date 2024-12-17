import re
from typing import List, Tuple, Set, Iterable

from torch import set_num_threads
import torch

from models import BiLSTM_CRF

MODEL_PATH = 'bilstm_crf_model.pth'
EMBEDDING_DIM = 128
HIDDEN_DIM = 256


def tokenize(text: str) -> list[tuple[str, int, int]]:
    """
    Tokenize text
    :param text: Raw text
    :return: List of tokens with positions
    """
    tokens = [(m.group(), m.start(), m.end()) for m in re.finditer(r'\w+|[^\w\s]', text)]
    return tokens


def load_model_and_vocab(model_path, embedding_dim, hidden_dim, pad_idx, device):
    """
    Load model and vocab for predictions
    """
    checkpoint = torch.load(model_path, map_location=device)

    token2idx = checkpoint["token2idx"]
    label2idx = checkpoint["label2idx"]

    model = BiLSTM_CRF(vocab_size=len(token2idx), tagset_size=len(label2idx),
                       embedding_dim=embedding_dim, hidden_dim=hidden_dim, pad_idx=pad_idx)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)

    print(f"Модель и словари успешно загружены из {model_path}")
    return model, token2idx, label2idx


class Solution:
    def __init__(self, model=None, token2idx=None, label2idx=None, device=None):
        if model is None:
            set_num_threads(8)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model_path = "bilstm_crf_model.pth"
            checkpoint = torch.load(model_path, map_location=device)
            self.token2idx = checkpoint["token2idx"]
            self.idx2label = {v: k for k, v in checkpoint["label2idx"].items()}

            self.model = BiLSTM_CRF(vocab_size=len(self.token2idx),
                                    tagset_size=len(self.idx2label),
                                    embedding_dim=EMBEDDING_DIM,
                                    hidden_dim=HIDDEN_DIM,
                                    pad_idx=self.token2idx["<PAD>"])
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.model.to(device)
        else:
            self.device = device
            self.token2idx = token2idx
            self.idx2label = {v: k for k, v in label2idx.items()}
            self.model = model
        self.model.eval()

    @staticmethod
    def bio_to_entities(tokens: List[Tuple[str, int, int]], bio_labels: List[str]) -> Set[Tuple[int, int, str]]:
        """
        Convert BIO tags into entities
        """
        entities = set()
        current_entity = None
        current_start = None
        current_end = None

        for (token, start, end), label in zip(tokens, bio_labels):
            if label.startswith("B-"):
                if current_entity:
                    entities.add((current_start, current_end, current_entity))
                current_entity = label[2:]
                current_start = start
                current_end = end
            elif label.startswith("I-") and current_entity == label[2:]:
                current_end = end
                continue
            else:
                if current_entity:
                    entities.add((current_start, current_end, current_entity))
                    current_entity = None

        if current_entity:
            entities.add((current_start, current_end, current_entity))

        return entities

    def predict(self, texts: List[str]) -> Iterable[Set[Tuple[int, int, str]]]:
        predictions = []

        for text in texts:
            tokens = tokenize(text)
            token_indices = [self.token2idx.get(token, self.token2idx["<PAD>"]) for token, _, _ in tokens]

            tokens_tensor = torch.tensor(token_indices).unsqueeze(0).to(self.device)
            mask = tokens_tensor != self.token2idx["<PAD>"]

            with torch.no_grad():
                predictions_idx = self.model(tokens_tensor, mask=mask)[0]

            bio_labels = [self.idx2label[idx] for idx in predictions_idx]

            entities = self.bio_to_entities(tokens, bio_labels)
            predictions.append(entities)

        return predictions
