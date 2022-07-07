from tokenize import String
from typing import Tuple
from pathlib import Path
import os
import openml
import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler, TensorDataset, Dataset
import torchvision
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
from torchvision.transforms import RandomCrop
import torchvision.transforms as transforms
import random




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
        subset_ratio: float = 1.0,
        old_percentage: float = 0.2,
        new_percentage: float = 0.4,
        shuffle_subset: bool = True,
        only_new_data: bool = True,
        device: torch.device = torch.device('cuda')) -> Tuple[DataLoader, DataLoader]:

        """ Splits the dataset into X_train, y_train, X_val, y_val
        and converts them to torch.Tensor

        Parameters
        ----------
        validation_split (float): validation data size
        shuffle_dataset (bool): if data will be shuffled
        batch_size (int): batch size
        seed: (int): numpy seed
        subset_ratio: (float): subset ratio
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

        if subset_ratio < 0.2 or subset_ratio > 1:
            raise ValueError("subset_size must be between 0.2 and 1")

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
        
        random.seed(seed)

        if only_new_data == True and shuffle_subset == True:
            old_subset_size = int(old_percentage * len(self.X_train))
            old_indices = random.sample(range(0, len(self.X_train)), old_subset_size)
            total_indices = list(range(0, len(self.X_train)))
            diff_indices = list(set(total_indices)^set(old_indices))
            new_subset_size = int(new_percentage * len(diff_indices))
            final_indices = random.sample(diff_indices, new_subset_size)

        elif only_new_data == False and shuffle_subset == False:
            subset_size = int(new_percentage * len(self.X_train))
            indices = list(range(0, subset_size))
       
        elif only_new_data == True and shuffle_subset == False:
            start = int(old_percentage * len(self.X_train))
            end = int(new_percentage * len(self.X_train))
            final_indices = list(range(start, end))
        else:
            subset_size = int(new_percentage * len(self.X_train))
            final_indices = random.sample(range(0, len(self.X_train)), subset_size)
     
        training_dataset = TrainingSet(self.X_train, self.y_train)
        sub_training_dataset = torch.utils.data.Subset(training_dataset, final_indices)
        train_dataloader = torch.utils.data.DataLoader(sub_training_dataset, batch_size=batch_size)

        val_dataset = TrainingSet(self.X_val, self.y_val)
        sub_val_dataset = torch.utils.data.Subset(val_dataset, final_indices)
        valid_dataloader = torch.utils.data.DataLoader(sub_val_dataset, batch_size=batch_size)

        return train_dataloader, valid_dataloader

class PyTorchDatasetManager():
    """ Base Class for PyTorch datasets
    """

    def __init__(self):
        self._save_to = 'datasets/torch'

    def load(self):
        """ Loads data from data directory as defined in
        config_file.data_directory
        """
        raise NotImplementedError()

    def get_train_and_val_set(self,
     validation_split: float = 0.2,
     batch_size: int = 60,
     seed: int = 42,
     subset_ratio: float = 1.0,
     shuffle_dataset: bool = True,
     device: torch.device = torch.device('cuda')):
        """ Splits the data into train and validation sets and created the dataloaders.

        Parameters
        ----------
            validation_split (float, optional): Ratio of the validation set
            batch_size (int, optional): Batch size
        
        Returns
        -------
            train_dataloader (DataLoader): Train data loader
            val_dataloader (DataLoader): Validation data loader
        """
        print("Loading the data...")
        self._load()
        torch.manual_seed(seed=seed)
        size = len(self.dataset)
        train_set, val_set = torch.utils.data.random_split(self.dataset, [int(size * (1 - validation_split)), int(size * validation_split)])

        subset_size = int(subset_ratio * len(train_set))
        indices = random.sample(range(0, len(train_set)), subset_size)
        sub_training_dataset = torch.utils.data.Subset(train_set, indices)
        train_dataloader = torch.utils.data.DataLoader(sub_training_dataset, batch_size=batch_size)
        
        subset_size = int(subset_ratio * len(val_set))
        indices = random.sample(range(0, len(val_set)), subset_size)
        sub_val_dataset = torch.utils.data.Subset(val_set, indices)
        val_dataloader = torch.utils.data.DataLoader(sub_val_dataset, batch_size=batch_size)

        return train_dataloader, val_dataloader

        
class CIFAR10Data(PyTorchDatasetManager):
    """ Class loading the Cifar10 data set. """

    def __init__(self):
        super(CIFAR10Data, self).__init__()

    def _load(self, download: bool = True):
        self.dataset = datasets.CIFAR10(
            root=self._save_to,
            train=True,
            download=download,
            transform=ToTensor()
        )

class CIFAR100Data(PyTorchDatasetManager):
    """ Class loading the Cifar10 data set. """

    def __init__(self):
        super(CIFAR100Data, self).__init__()

    def _load(self, download: bool = True):
        self.dataset = datasets.CIFAR100(
            root=self._save_to,
            train=True,
            download=download,
            transform=ToTensor()
        )

class Country211Data(PyTorchDatasetManager):
    """ Class loading the Country211 data set. """

    def __init__(self):
        super(Country211Data, self).__init__()

    def _load(self, download: bool = True):
        self.dataset = datasets.Country211(
            root=self._save_to,
            download=download,
            # transform=
            # target_transform=
        )


class EMNISTData(PyTorchDatasetManager):
    """ Class loading the EMNIST data set. """

    def __init__(self):
        super(EMNISTData, self).__init__()

    def _load(self, download: bool = True):
        self.dataset = datasets.EMNIST(
            root=self._save_to,
            train=True,
            download=download,
            split="mnist",
            transform=transforms.CenterCrop(10)
            # target_transform=
        )
