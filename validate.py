from typing import Tuple

from solution import Solution
from train import load_dataset, calculate_metrics

VAL_PATH = 'dataset/dev/'

def debug_print(ans_set, pred_set):
    ans_list = list(ans_set)
    ans_list.sort()

    pred_list = list(pred_set)
    pred_list.sort()

    print(ans_list)
    print(pred_list)


def validate(dataset_path: str = "dataset/train/") -> Tuple[float, float, float]:
    dev_dataset = load_dataset(dataset_path)
    texts = [data[0] for data in dev_dataset]

    solution = Solution()

    predictions = solution.predict(texts)
    answers = [set((start, end, lbl) for lbl, start, end in data[1]) for data in dev_dataset]


    for pred, ans, text in zip(predictions, answers, texts):
        pred_with_text = [set((start, end, label, text[start:end]) for start, end, label in pred)]
        ans_with_text = [set((start, end, label, text[start:end]) for start, end, label in ans)]
        #debug_print(ans_with_text, pred_with_text)

    all_labels = set()
    for labels in answers:
        all_labels |= set(label for start, end, label in labels)

    return calculate_metrics(predictions, answers, all_labels)

if __name__ == '__main__':
    validate("dataset/train/")