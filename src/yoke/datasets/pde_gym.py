"""Datasets for data from the PDEgym https://huggingface.co/camlab-ethz."""

from typing import Callable, Union
import sys

import lightning.pytorch as L
import netCDF4
import numpy as np
import torch
from torch.utils.data import DataLoader


class NavierStokes(torch.utils.data.Dataset):
    """Wrapper for Navier-Stokes datasets https://huggingface.co/datasets/camlab-ethz.

    This dataset returns trajectories from pre-assembled Navier-Stokes (NS) dataset files.

    Args:
        data_file (str): Location of the assembled data file from the NS dataset.
        seq_len (int): Number of consecutive frames to return. This includes the
            starting frame.
        timeIDX_offset (int): Time index offset between items in the returned trajectory.
        transform (Callable): Transform applied to loaded data sequence before returning.
    """

    def __init__(
        self,
        data_file: str,
        seq_len: int = 2,
        timeIDX_offset: Union[int, list[int], tuple[int]] = None,
        transform: Callable = None,
        *args,
        **kwargs,
    ) -> None:
        """Initialization for NS sequential dataset."""
        self.seq_len = seq_len
        self.transform = transform
        self.rng = np.random.default_rng()

        # Load the data.
        self.data = np.array(
            netCDF4.Dataset(filename=data_file, mode="r").variables["velocity"]
        )

        # Define default time offsets.
        traj_len = self.data.shape[1]
        if timeIDX_offset is None:
            timeIDX_offset = list(range(-traj_len, traj_len + 1))

        # Create a list of valid sequence indices.
        timeIDX_offset = (
            [timeIDX_offset] if isinstance(timeIDX_offset, int) else timeIDX_offset
        )
        self.timeIDX_offset = timeIDX_offset
        valid_seq = []
        for dt in timeIDX_offset:
            for t0 in range(traj_len):
                if dt == 0:
                    valid_seq.append([0 for _ in range(seq_len)])
                else:
                    tf = t0 + seq_len * dt
                    if (tf < traj_len) and (tf > 0):
                        valid_seq.append(slice(t0, tf, dt))

        self.valid_seq = valid_seq
        self.n_traj = self.data.shape[0]
        self.n_samples = len(valid_seq) * self.n_traj

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return self.n_samples

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return a sequence of consecutive frames of a trajectory."""
        ind_slice = self.valid_seq[index % len(self.valid_seq)]
        img_seq = torch.from_numpy(self.data[index % self.n_traj, ind_slice].copy())

        # Apply transforms if requested.
        if self.transform is not None:
            img_seq = self.transform(img_seq)

        # Determine step size.
        if isinstance(ind_slice, slice):
            dt = ind_slice.step
        else:
            dt = 0

        return img_seq, torch.tensor(dt, dtype=torch.float32)


class PDEDataModule(L.LightningDataModule):
    """Lightning data module for generic PDEgym datasets.

    Args:
        ds_name (str): Name of desired dataset in src.yoke.datasets.pde_gym
        ds_params_train (dict): Keyword arguments passed to dataset initializer
            to generate the training dataset.
        dl_params_train (dict): Keyword arguments passed to training dataloader.
        ds_params_val (dict): Keyword arguments passed to dataset initializer
            to generate the validation dataset.
        dl_params_val (dict): Keyword arguments passed to validation dataloader.
        ds_params_test (dict): Keyword arguments passed to dataset initializer
            to generate the testing dataset.
        dl_params_test (dict): Keyword arguments passed to testing dataloader.
    """

    def __init__(
        self,
        ds_name: str,
        ds_params_train: dict,
        dl_params_train: dict,
        ds_params_val: dict,
        dl_params_val: dict,
        ds_params_test: dict = None,
        dl_params_test: dict = None,
    ) -> None:
        """PDEDataModule initialization method."""
        super().__init__()
        self.ds_name = ds_name

        self.ds_params_train = ds_params_train
        self.dl_params_train = dl_params_train

        self.ds_params_val = ds_params_val
        self.dl_params_val = dl_params_val

        self.ds_params_test = ds_params_test
        self.dl_params_test = dl_params_test

    def setup(self, stage: str = None) -> None:
        """Data module setup called on all devices."""
        # Currently, PDE datasets are fast to instantiate.  As such, to
        # facilitate "dynamic" datasets that may change throughout
        # training, dataset instantiation is done on-the-fly when
        # preparing dataloaders.
        pass

    def train_dataloader(self) -> DataLoader:
        """Prepare the training dataset and dataloader."""
        ds_ref = getattr(sys.modules[__name__], self.ds_name)
        ds_train = ds_ref(**self.ds_params_train)
        self.ds_train = ds_train
        return DataLoader(dataset=ds_train, **self.dl_params_train)

    def val_dataloader(self) -> DataLoader:
        """Prepare the validation dataset and dataloader."""
        ds_ref = getattr(sys.modules[__name__], self.ds_name)
        ds_val = ds_ref(**self.ds_params_val)
        self.ds_val = ds_val
        return DataLoader(dataset=ds_val, **self.dl_params_val)

    def test_dataloader(self) -> DataLoader:
        """Prepare the testing dataset and dataloader."""
        ds_ref = getattr(sys.modules[__name__], self.ds_name)
        ds_test = ds_ref(**self.ds_params_test)
        self.ds_test = ds_test
        return DataLoader(dataset=ds_test, **self.dl_params_test)
