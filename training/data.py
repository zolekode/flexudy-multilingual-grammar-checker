from typing import List
from sentence_transformers import SentenceTransformer
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import pandas as pd
import json
import torch


class DataImporter:

    def __init__(self, parameters: dict):
        self.__parameters = parameters

        self.__train_sequences = None

        self.__test_sequences = None

        self.__train_labels = None

        self.__test_labels = None

        self.__load_data()

    @staticmethod
    def load_parameters(path_to_parameters: str) -> dict:
        with open(path_to_parameters, "r") as fp:
            parameters = json.load(fp)

        return parameters

    def __load_data(self):
        train_data = self.__load_multiple_datasets(phase_key="train")

        train_texts, test_texts, train_labels, test_labels = self.__get_train_test_sets(train_data)

        train_texts = train_texts.tolist()

        test_texts = test_texts.tolist()

        train_labels = train_labels.tolist()

        test_labels = test_labels.tolist()

        test_labels = [int(label) for label in test_labels]

        train_labels = [int(label) for label in train_labels]

        self.__train_sequences = train_texts

        self.__test_sequences = test_texts

        self.__train_labels = train_labels

        self.__test_labels = test_labels

    def __get_train_test_sets(self, train_data):
        train_texts = train_data[self.__parameters["text_column"]]

        train_labels = train_data[self.__parameters["label_column"]]

        if len(self.__parameters["data_paths"]["dev"]) == 0:
            return train_test_split(train_texts, train_labels, test_size=(1 - self.__parameters["train_split"]),
                                    stratify=train_labels)

        dev_set = self.__load_multiple_datasets("dev")

        test_texts = dev_set[self.__parameters["text_column"]]

        test_labels = dev_set[self.__parameters["label_column"]]

        return train_texts, test_texts, train_labels, test_labels

    def __load_multiple_datasets(self, phase_key: str = "train"):
        phases = self.__parameters["data_paths"][phase_key]

        dataframes = []

        for phase in phases:
            dataframes.append(pd.read_csv(phase["path"], encoding="utf-8", sep=phase["separator"]))

        data = pd.concat(dataframes)

        data = data.drop_duplicates()

        data[self.__parameters["label_column"]] = pd.to_numeric(data[self.__parameters["label_column"]],
                                                                errors="coerce",
                                                                downcast="integer")

        data = data.sample(frac=1.0).reset_index(drop=True)

        print("PHASE: " + phase_key.upper() + "\n")

        print(data.describe())

        return data

    def get_training_data(self) -> tuple:
        return self.__train_sequences, self.__train_labels

    def get_test_data(self) -> tuple:
        return self.__test_sequences, self.__test_labels


class SequenceClassificationDataset(Dataset):

    def __init__(self, sequences, labels, sequence_encoder):
        self.__sequences = sequences

        self.__labels = labels

        self.__sequence_encoder = sequence_encoder

        self.__data_size = len(self.__sequences)

    def __len__(self):
        return self.__data_size

    def __getitem__(self, index):
        sequence = self.__sequences[index]

        sequence_embedding = self.__sequence_encoder.encode([sequence])

        sequence_embedding = torch.tensor(sequence_embedding, dtype=torch.float32)

        label = self.__labels[index]

        label = torch.tensor([label], dtype=torch.float32)

        return sequence_embedding, label


class SequenceClassificationDataModule:

    def __init__(self, data_importer: DataImporter, sequence_encoder: SentenceTransformer, shuffle_data: bool = True,
                 batch_size: int = 32, num_workers: int = 0):
        self.__sequence_encoder = sequence_encoder

        self.__data_importer = data_importer

        self.__num_workers = num_workers

        self.__batch_size = batch_size

        self.__shuffle_data = shuffle_data

    def get_train_data_loader(self):
        sequences, labels = self.__data_importer.get_training_data()

        return self.__get_data_loader(sequences, labels)

    def get_test_data_loader(self):
        sequences, labels = self.__data_importer.get_test_data()

        return self.__get_data_loader(sequences, labels)

    def __get_data_loader(self, sequences: List[str], labels: list) -> DataLoader:
        dataset = SequenceClassificationDataset(sequences, labels, self.__sequence_encoder)

        data_loader = DataLoader(dataset, batch_size=self.__batch_size, shuffle=self.__shuffle_data,
                                 num_workers=self.__num_workers)

        return data_loader
