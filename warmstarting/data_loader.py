from tokenize import String
from typing import Tuple
from pathlib import Path
import os
import openml
import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler, TensorDataset, Dataset
import torch
from torch.utils.data import Dataset
from torchvision import datasets
import torchvision.transforms as transforms
import random




class TrainingSet(Dataset):
    def __init__(self,X,Y):
        super().__init__()
        self.X = X
        self.Y = Y                  

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return [self.X[idx], self.Y[idx]]

class DataHandler:
    """
    Abstract class for data loader
    """
    def __init__(self):
        pass

class OpenMLDATA(DataHandler):
    """
    This class is to get the openml datasets, save it, seperate it into train and test sets.
    """

    def __init__(self,):
        super().__init__()
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

    def get_number_of_classes(self):
        return len(np.unique(self.y))

    def get_train_and_val_set(
        self,
        validation_split: float = 0.2,
        shuffle_dataset: bool = True,
        batch_size: int = 64,
        seed: int = 42,
        old_ratio: float = 0,
        new_ratio: float = 1,
        shuffle_subset: bool = False,
        only_new_data: bool = False,
        device: torch.device = torch.device('cpu')) -> Tuple[DataLoader, DataLoader]:

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
        
        random.seed(seed)

        if only_new_data and shuffle_subset:
            old_subset_size = int(old_ratio * len(self.X_train))
            old_indices = random.sample(range(len(self.X_train)), old_subset_size)
            total_indices = list(range(len(self.X_train)))
            diff_indices = list(set(total_indices) ^ set(old_indices))
            new_subset_size = int(new_ratio * len(diff_indices))
            final_indices = random.sample(diff_indices, new_subset_size)

        elif not only_new_data and not shuffle_subset:
            subset_size = int(new_ratio * len(self.X_train))
            final_indices = list(range(subset_size))
       
        elif only_new_data and not shuffle_subset:
            start = int(old_ratio * len(self.X_train))
            end = int(new_ratio * len(self.X_train))
            final_indices = list(range(start, end))
        else:
            subset_size = int(new_ratio * len(self.X_train))
            final_indices = random.sample(range(len(self.X_train)), subset_size)
     
        training_dataset = TrainingSet(self.X_train, self.y_train)
        sub_training_dataset = torch.utils.data.Subset(training_dataset, final_indices)
        train_dataloader = torch.utils.data.DataLoader(sub_training_dataset, batch_size=batch_size)

        val_dataset = TrainingSet(self.X_val, self.y_val)
        valid_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size)

        return train_dataloader, valid_dataloader

class PyTorchDatasetManager(DataHandler):
    """ Base Class for PyTorch datasets
    """

    def __init__(self):
        super().__init__()
        self._save_to = 'datasets/torch'
        self.dataset = None
        print("Loading the data...")

    def load(self):
        """ Loads data from data directory as defined in
        config_file.data_directory
        """
        raise NotImplementedError()

    def get_number_of_classes(self):
        if self.dataset is not None:
            return len(self.dataset.classes)
        else:
            return -1

    def get_train_and_val_set(self,
     validation_split: float = 0.2,
     batch_size: int = 60,
     seed: int = 42,
     shuffle_dataset: bool = True,
     device: torch.device = torch.device('cpu'),
     old_ratio: float = 0,
     new_ratio: float = 1,
     shuffle_subset: bool = False,
     only_new_data: bool = False):
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

        torch.manual_seed(seed=seed)
        size = len(self.dataset)
        train_set, val_set = torch.utils.data.random_split(self.dataset, [int(size * (1 - validation_split)), int(size * validation_split)])

        val_dataloader = torch.utils.data.DataLoader(val_set, batch_size=batch_size)
 
        if only_new_data and shuffle_subset:
            old_subset_size = int(old_ratio * len(train_set))
            old_indices = random.sample(range(len(train_set)), old_subset_size)
            total_indices = list(range(len(train_set)))
            diff_indices = list(set(total_indices) ^ set(old_indices))
            new_subset_size = int(new_ratio * len(diff_indices))
            final_indices = random.sample(diff_indices, new_subset_size)

        elif not only_new_data and not shuffle_subset:
            subset_size = int(new_ratio * len(train_set))
            final_indices = list(range(subset_size))
       
        elif only_new_data and not shuffle_subset:
            start = int(old_ratio * len(train_set))
            end = int(new_ratio * len(train_set))
            final_indices = list(range(start, end))
        else:
            subset_size = int(new_ratio * len(train_set))
            final_indices = random.sample(range(len(train_set)), subset_size)
     
        sub_training_dataset = torch.utils.data.Subset(train_set, final_indices)
        train_dataloader = torch.utils.data.DataLoader(sub_training_dataset, batch_size=batch_size)

        return train_dataloader, val_dataloader

        
class CIFAR10Data(PyTorchDatasetManager):
    """ Class loading the Cifar10 data set. """

    def __init__(self, transform: transforms= None, target_transform: transforms=None):
        super(CIFAR10Data, self).__init__()
        self.transform = transform
        self.target_transform = target_transform
        self._load()

    def _load(self, download: bool = True):
        self.dataset = datasets.CIFAR10(
            root=self._save_to,
            train=True,
            download=download,
            transform=self.transform,
            target_transform=self.target_transform
        )

class CIFAR100Data(PyTorchDatasetManager):
    """ Class loading the Cifar10 data set. """

    def __init__(self, transform: transforms= None, target_transform: transforms=None):
        super(CIFAR100Data, self).__init__()
        self.transform = transform
        self.target_transform = target_transform
        self._load()

    def _load(self, download: bool = True):
        self.dataset = datasets.CIFAR100(
            root=self._save_to,
            train=True,
            download=download,
            transform=self.transform,
            target_transform=self.target_transform
        )

class Country211Data(PyTorchDatasetManager):
    """ Class loading the Country211 data set. """

    def __init__(self, transform: transforms= None, target_transform: transforms=None):
        super(Country211Data, self).__init__()
        self.transform = transform
        self.target_transform = target_transform
        self._load()

    def _load(self, download: bool = True):
        self.dataset = datasets.Country211(
            root=self._save_to,
            download=download,
            transform=self.transform,
            target_transform=self.target_transform
        )


class EMNISTData(PyTorchDatasetManager):
    """ Class loading the EMNIST data set. """

    def __init__(self, split: String="mnist", transform: transforms= None, target_transform: transforms=None):
        super(EMNISTData, self).__init__()
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        self._load()

    def _load(self, download: bool = True):
        self.dataset = datasets.EMNIST(
            root=self._save_to,
            train=True,
            download=download,
            split=self.split,
            transform=self.transform,
            target_transform=self.target_transform
        )

class MNISTData(PyTorchDatasetManager):
    """ Class loading the EMNIST data set. """

    def __init__(self, transform: transforms= None, target_transform: transforms=None):
        super(MNISTData, self).__init__()
        self.transform = transform
        self.target_transform = target_transform
        self._load()

    def _load(self, download: bool = True):
        self.dataset = datasets.MNIST(
            root=self._save_to,
            train=True,
            download=download,
            transform=self.transform,
            target_transform=self.target_transform
        )
