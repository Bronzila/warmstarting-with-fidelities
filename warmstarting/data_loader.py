from tokenize import String
from typing import Tuple
import os
import openml
import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler, TensorDataset, Dataset



class TrainingSet(Dataset):
    def __init__(self,X,Y):
        self.X = X
        self.Y = Y                  

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return [self.X[idx], self.Y[idx]]

class DataHandler:
    """
    This class is to get the dataset, save it, seperate it into train and test sets.
    """

    def __init__(self,):
        self.X = None
        self.y = None

    def set_dataset(self, openml_id: int = 61) -> None:
        """ Saves the dataset with the given path

        Parameters
        ----------
        openml_id (int): openml task id to download

        Returns
        -------
        X (numpy.array): X values
        y (numpy.array): y values

        """
        PATH = "datasets/openml_id_" + str(openml_id)
        if os.path.isdir(PATH):
            with open(PATH + "/X.npy", 'rb') as f:
                X = np.load(f)
            with open(PATH + "/y.npy", 'rb') as f:
                y = np.load(f)
        else:
            dataset = openml.datasets.get_dataset(openml_id)
            X, y, categorical_indicator, attribute_names = dataset.get_data(
                dataset_format="array", target=dataset.default_target_attribute
            )
            self._save_dataset(PATH, X, y)

        self.X = X
        self.y = y
        return X, y

    def _save_dataset(self, path: String, X: np.array, y: np.array) -> None:
        """ Saves the dataset with the given path

        Parameters
        ----------
        path (str): path to save the data
        X (numpy.array): X values
        y (numpy.array): y values

        Returns
        -------
        None

        """
        if not os.path.isdir("datasets"):
            os.mkdir("datasets")
        os.mkdir(path)
        os.chdir(path)
        with open("X.npy", 'wb') as f:
            np.save(f, X)
        with open("y.npy", 'wb') as f:
            np.save(f, y)

    def get_train_and_val_set(
        self,
        validation_split: float = 0.2,
        shuffle_dataset: bool = True,
        batch_size: int = 64,
        seed: int = 42,
        device:torch.device = torch.device('cuda')) -> Tuple[DataLoader, DataLoader]:

        """ Splits the dataset into X_train, y_train, X_val, y_val
        and converts them to torch.Tensor

        Parameters
        ----------
        validation_split (float): validation data size
        shuffle_dataset (bool): if data will be shuffled
        batch_size (int): batch size
        seed: (int): numpy seed
        device: (torch.device): device to run (cuda/cpu)

        Returns
        -------
        Tuple
            (X_train: Tensor, y_train: Tensor, X_val: Tensor, y_val: Tensor)

        Raises
        ------
        ValueError
            validation_split must be between 0 and 1
        """

        if validation_split <= 0 or validation_split >= 1:
            raise ValueError("validation_split must be between 0 and 1")

        indices = list(range(len(self.X)))
        split = int(np.floor(validation_split * len(self.X)))
        if shuffle_dataset:
            np.random.seed(seed)
            np.random.shuffle(indices)
        train_indices, val_indices = indices[split:], indices[:split]

        self.X_train = torch.from_numpy(self.X[train_indices]).to(device)
        self.y_train = torch.from_numpy(self.y[train_indices]).to(device)
        self.X_val = torch.from_numpy(self.X[val_indices]).to(device)
        self.y_val = torch.from_numpy(self.y[val_indices]).to(device)

        training_dataset = TrainingSet(self.X_train, self.y_train)
        train_dataloader = torch.utils.data.DataLoader(training_dataset, batch_size=batch_size)

        val_data = TrainingSet(self.X_val, self.y_val)
        valid_dataloader = torch.utils.data.DataLoader(val_data, batch_size=batch_size)

        return train_dataloader, valid_dataloader
