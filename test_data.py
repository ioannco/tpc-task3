from solution import tokenize
from validate import validate
from train import load_dataset, extract_annotations, convert_to_bio


def test_extract_annotations():
    annotations = ["T10\tPERSON 93 104\tБарак Обама"]
    annotations = extract_annotations(annotations)
    assert len(annotations) == 1
    assert annotations[0][0] == "PERSON"
    assert annotations[0][1] == 93
    assert annotations[0][2] == 104


def test_load_dataset_except_nonnull():
    dataset = load_dataset("dataset/train/")
    assert dataset is not None


def test_tokenizer():
    text = "ФБР арестовало подозреваемого."
    tokens = tokenize(text)
    print(tokens)
    assert tokens is not None
    assert len(tokens) == 4
    assert text[tokens[1][1]:tokens[1][2]] == "арестовало"


def test_tokenizer_integrity():
    dataset = load_dataset("dataset/train/")
    for dataset_file in dataset:
        tokens = tokenize(dataset_file[0])
        for token in tokens:
            assert len(token) == 3

def test_bio():
    text = "ФБР арестовало подозреваемого Барака Обаму."
    annotations = ["T1\tORG 0 3\tФБР", "T2\tPERSON 30 42\tБарака Обаму"]
    annotations = extract_annotations(annotations)
    bio = convert_to_bio((text, annotations))
    assert bio[1] == ['B-ORG', 'O', 'O', 'B-PERSON', 'I-PERSON', 'O']
    print(bio)

def test_convert_to_bio():
    dataset = load_dataset("dataset/train/")
    bio = [convert_to_bio(dataset_file) for dataset_file in dataset]
    print([(token, tag) for token, tag in zip(bio[0][0], bio[0][1])])
    assert bio is not None

def test_validate_solution():
    f1, precision, recall = validate()
    assert f1 > 0.37