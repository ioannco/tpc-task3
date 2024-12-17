import os
from collections import Counter
from typing import Iterable

import torch
from sklearn.metrics import precision_score, recall_score, f1_score
from torch import optim
from torch.nn.utils.rnn import pad_sequence
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from models import NERDataset, BiLSTM_CRF
from solution import MODEL_PATH, EMBEDDING_DIM, HIDDEN_DIM, tokenize, Solution

TRAIN_PATH = 'dataset/train/'
EPOCHS = 20
BATCH_SIZE = 32
LEARNING_RATE = 0.01


def extract_annotations(ann_file: Iterable[str]) -> list[tuple[str, int, int]]:
    """
    Extract annotations from file with entities
    :param ann_file: File with entities
    :return: List of entities with positions
    """
    entities = list()
    for line in ann_file:
        if not line.startswith("T"):
            continue
        index, label_info, entity = line.split("\t")
        try:
            label, start_pos, end_pos = label_info.split(" ")
            start_pos = int(start_pos)
            end_pos = int(end_pos)
            entities.append((label, start_pos, end_pos))
        except ValueError:
            continue
    return entities


def load_dataset(path: str) -> list[tuple[str, list[tuple[str, int, int]]]]:
    """
    Load dataset from path.
    Dataset must include pairs of .txt and .ann files with text and entities respectively.
    :param path: Dataset folder path
    :return: List of tuples of texts and entities with positions
    """
    dataset = list()
    for file in os.listdir(path):
        if file.endswith(".txt"):
            with open(path + file, "r", encoding="utf-8") as text_file:
                with open(path + file.replace(".txt", ".ann"), "r", encoding="utf-8") as ann_file:
                    entities = extract_annotations(ann_file)
                dataset.append((text_file.read(), entities))
    return dataset


def token_label_2idx(data: list[tuple[list[str], list[str]]]) -> tuple[dict[str, int], dict[str, int]]:
    """
    Construct vocabularies for tokens and labels
    :param data: List of tuples of tokens and labels
    :return: token2idx and label2idx
    """
    token2idx = {'<PAD>': 0}
    idx = 1
    for text in data:
        for token in text[0]:
            if token not in token2idx:
                token2idx[token] = idx
                idx += 1
    label2idx = {'O': 0}
    idx = 1
    for text in data:
        for label in text[1]:
            if label not in label2idx:
                label2idx[label] = idx
                idx += 1
    return token2idx, label2idx


def collate_fn(batch):
    """
    Collate function for a batch
    :param batch: Batch
    :return: Padded tokens, padded labels and a mask
    """
    tokens, labels = zip(*batch)

    filtered_batch = [(t, l) for t, l in zip(tokens, labels) if t.numel() > 0]
    if len(filtered_batch) == 0:
        return None, None, None

    tokens_filtered, labels_filtered = zip(*filtered_batch)
    tokens_padded = pad_sequence(tokens_filtered, batch_first=True, padding_value=0)
    labels_padded = pad_sequence(labels_filtered, batch_first=True, padding_value=0)
    mask = tokens_padded != 0

    if mask.size(0) > 0 and mask.size(1) > 0:
        mask[:, 0] = True

    return tokens_padded, labels_padded, mask


def weighted_loss(loss, tags, weights, mask):
    """
    Взвешенный loss для CRF с использованием тензорных операций на GPU.
    :param loss: Текущий loss из CRF.
    :param tags: Метки тегов в батче.
    :param weights: Веса для классов (тензор на GPU).
    :param mask: Маска для непаддинговых токенов.
    :return: Loss с учетом весов классов.
    """
    # Извлекаем веса для тегов с учетом GPU
    weights_tensor = weights[tags]  # Используем индексы tags для выбора весов классов

    # Применяем маску
    mask = mask.float()  # Преобразуем маску в float
    weighted_loss = weights_tensor * mask

    # Усредняем loss с учетом маски
    return (loss * weighted_loss).sum() / mask.sum()



def train_model(model, train_loader, optimizer, weights, scheduler, token2idx, label2idx, epochs, device) -> None:
    """
    Train NER model with weighted loss on GPU.
    """
    print(device)

    # Переносим веса классов на GPU
    weights_tensor = torch.tensor([weights[i] for i in sorted(weights.keys())]).to(device)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        print(f"Epoch {epoch + 1}/{epochs}")

        for batch_idx, (tokens, labels, mask) in enumerate(train_loader):
            if tokens is None:
                print(f"Skipping empty batch {batch_idx + 1}")
                continue

            tokens = tokens.to(device)
            labels = labels.to(device)
            mask = mask.to(device)

            optimizer.zero_grad()

            # Исходный loss из CRF
            loss = model(tokens, labels, mask)

            # Применяем взвешивание классов на GPU
            loss = weighted_loss(loss, labels, weights_tensor, mask)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            print(f"  Batch {batch_idx + 1} Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(train_loader)
        scheduler.step(avg_loss)
        print(f"Epoch {epoch + 1} completed. Avg Loss: {avg_loss:.4f}")

        validate_model(model, token2idx, label2idx, device)


def validate_model(model, token2idx, label2idx, device, dataset_path='dataset/test/'):
    dev_dataset = load_dataset(dataset_path)
    texts = [data[0] for data in dev_dataset]

    solution = Solution(model, token2idx, label2idx, device)

    predictions = solution.predict(texts)
    answers = [set((start, end, lbl) for lbl, start, end in data[1]) for data in dev_dataset]

    all_labels = set()
    for labels in answers:
        all_labels |= set(label for start, end, label in labels)

    calculate_metrics(predictions, answers, all_labels)

def save_model_and_vocab(model, token2idx, label2idx, save_path=MODEL_PATH):
    """
    Save model and vocab for predictions
    """
    save_dict = {
        "model_state_dict": model.state_dict(),
        "token2idx": token2idx,
        "label2idx": label2idx
    }
    torch.save(save_dict, save_path)
    print(f"Модель и словари успешно сохранены в {save_path}!")


def calculate_metrics(test_ans: list[set], user_ans: list[set], all_labels: set):
    """
    Рассчитывает micro-averaged precision, recall и F1 с использованием sklearn.
    :param test_ans: Список наборов истинных меток
    :param user_ans: Список наборов предсказанных меток
    :param all_labels: Множество всех меток (для фильтрации)
    """
    y_true = []
    y_pred = []

    # Преобразуем данные в BIO-метки
    for true_set, pred_set in zip(test_ans, user_ans):
        for label in all_labels:
            # Создаём бинарные метки для всех классов
            true_binary = {(start, end) for start, end, lbl in true_set if lbl == label}
            pred_binary = {(start, end) for start, end, lbl in pred_set if lbl == label}

            for span in true_binary | pred_binary:
                y_true.append(1 if span in true_binary else 0)
                y_pred.append(1 if span in pred_binary else 0)

    # Вычисляем метрики
    precision = precision_score(y_true, y_pred, average='micro')
    recall = recall_score(y_true, y_pred, average='micro')
    f1 = f1_score(y_true, y_pred, average='micro')

    print(f"Precision (micro): {precision:.4f}")
    print(f"Recall (micro): {recall:.4f}")
    print(f"F1 Score (micro): {f1:.4f}")

    return f1, precision, recall


def load_data_and_train_model():
    """
    Main training function
    """
    dataset = load_dataset(TRAIN_PATH)
    bio_data = [convert_to_bio(data) for data in dataset]

    label_counts = Counter()
    for _, labels in bio_data:
        label_counts.update(labels)
    total_count = sum(label_counts.values())
    weights = {label: total_count / count for label, count in label_counts.items()}
    total_weight = sum(weights.values())

    token2idx, label2idx = token_label_2idx(bio_data)
    weights = {label2idx[label]: weight / total_weight for label, weight in weights.items()}

    train_dataset = NERDataset(bio_data, token2idx, label2idx)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                              collate_fn=collate_fn, shuffle=True, drop_last=True)

    model = BiLSTM_CRF(vocab_size=len(token2idx), tagset_size=len(label2idx),
                       embedding_dim=EMBEDDING_DIM, hidden_dim=HIDDEN_DIM, pad_idx=token2idx["<PAD>"])
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.5, verbose=True)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = model.to(device)
    train_model(model, train_loader, optimizer, weights, scheduler, token2idx, label2idx, EPOCHS, device)

    save_model_and_vocab(model, token2idx, label2idx)


def convert_to_bio(dataset_tuple: tuple[str, list[tuple[str, int, int]]]) -> tuple[list[str], list[str]]:
    """
    Convert dataset to BIO format B-[label] means start of the named entity, I-[label] means continuation of the entity,
    O means that the token is not a named entity
    :param dataset_tuple: Dataset
    :return:
    """
    text, entities = dataset_tuple
    tokens = tokenize(text)
    bio_labels = []
    j = 0

    for token, start_pos, end_pos in tokens:
        if j >= len(entities) or end_pos <= entities[j][1]:
            bio_labels.append("O")
            continue
        if start_pos >= entities[j][1] and end_pos <= entities[j][2]:
            tag = f'B-{entities[j][0]}' if start_pos == entities[j][1] else f'I-{entities[j][0]}'
            bio_labels.append(tag)
        else:
            j += 1
            if j < len(entities) and start_pos >= entities[j][1] and end_pos <= entities[j][2]:
                tag = f'B-{entities[j][0]}' if start_pos == entities[j][1] else f'I-{entities[j][0]}'
                bio_labels.append(tag)
            else:
                bio_labels.append("O")

    return [token[0] for token in tokens], bio_labels

if __name__ == "__main__":
    load_data_and_train_model()
