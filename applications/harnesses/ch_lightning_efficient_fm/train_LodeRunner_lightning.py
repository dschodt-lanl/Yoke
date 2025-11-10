"""Training for Lightning-wrapped LodeRunner on LSC material densities.

This version of training uses only the lsc240420 data with only per-material
density along with the velocity field. A single timestep is input, a single
timestep is predicted. The number of input variables is fixed throughout
training.

`lightning` is used to train a LightningModule wrapper for LodeRunner to allow
multi-node, multi-GPU, distributed data-parallel training.

"""

#############################################
# Packages
#############################################
import argparse
import os
import re

import lightning.pytorch as L
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
import torch
import torch.nn as nn
import torchvision.transforms.v2 as tforms
import numpy as np
import pandas as pd

from yoke.models.vit.swin.bomberman import LodeRunner, Lightning_LodeRunner
from yoke.datasets.lsc_dataset import LSCDataModule
from yoke.datasets.transforms import ResizePadCrop
import yoke.torch_training_utils as tr
from yoke.lr_schedulers import CosineWithWarmupScheduler
from yoke.helpers import cli
import yoke.scheduled_sampling
from yoke.losses.masked_loss import CroppedLoss2D

from yoke.models.cnn.cnn import LodeRunnerCNN

#############################################
# Inputs
#############################################
descr_str = (
    "Trains lightning-wrapped LodeRunner on multi-timestep input and output of the "
    "lsc240420 per-material density fields."
)
parser = argparse.ArgumentParser(
    prog="Initial LodeRunner Training", description=descr_str, fromfile_prefix_chars="@"
)
parser = cli.add_default_args(parser=parser)
parser = cli.add_filepath_args(parser=parser)
parser = cli.add_computing_args(parser=parser)
parser = cli.add_model_args(parser=parser)
parser = cli.add_training_args(parser=parser)
parser = cli.add_cosine_lr_scheduler_args(parser=parser)
parser = cli.add_scheduled_sampling_args(parser=parser)
parser.add_argument(
    "--cache_dir",
    action="store",
    type=str,
    default=None,
    help="Path to .h5 caches for lsc sequential dataset.",
)
parser.add_argument(
    "--cache_file_train",
    action="store",
    type=str,
    default=None,
    help="h5 cache file for lsc sequential training dataset.",
)
parser.add_argument(
    "--cache_file_val",
    action="store",
    type=str,
    default=None,
    help="h5 cache file for lsc sequential validation dataset.",
)
parser.add_argument(
    "--channel_data_dir",
    action="store",
    type=str,
    default=None,
    help="Path to files containing channel names and normalizations.",
)
parser.add_argument(
    "--channel_file",
    action="store",
    type=str,
    default=None,
    help="File defining channels and normalizations.",
)
parser.add_argument(
    "--precision",
    action="store",
    type=str,
    default="32-true",
    help="Training precision passed to Lightning Trainer.",
)
parser.add_argument(
    "--dtype",
    action="store",
    type=str,
    default="float32",
    help="Torch datatype for loaded data.",
)
parser.add_argument(
    "--loss",
    action="store",
    type=str,
    default="MSELoss",
    help="Name of loss in torch.nn to use for training.",
)
parser.add_argument(
    "--n_channels_train",
    action="store",
    type=int,
    default=8,
    help="Number of channels for model to be trained on.",
)
parser.add_argument(
    "--n_channels_val",
    action="store",
    type=int,
    default=8,
    help="Number of channels for model to be validated on.",
)
parser.add_argument(
    "--eval_on_same_channels",
    action="store",
    type=int,
    default=1,
    help="Evaluate on same channels as used to train. Using 0==False, 1==True due to argparse quirks.",
)
parser.add_argument(
    "--model",
    action="store",
    type=str,
    default="cnn",
    help="Model type: (cnn, vit)",
)
parser.add_argument(
    "--n_downsample",
    action="store",
    type=int,
    default=0,
    help="Number of downsampling stages when args.model=='cnn'",
)
parser.add_argument(
    "--gradient_clip_val",
    action="store",
    type=float,
    default=1.0,
    help="Gradient clipping value to clip absolute value of gradients.",
)

# Change some default filepaths.
parser.set_defaults(
    train_filelist="lsc240420_prefixes_train_80pct.txt",
    validation_filelist="lsc240420_prefixes_validation_10pct.txt",
    test_filelist="lsc240420_prefixes_test_10pct.txt",
)


#############################################
#############################################
if __name__ == "__main__":
    # Set precision for tensor core speedup potential.
    torch.set_float32_matmul_precision("medium")

    #############################################
    # Process Inputs
    #############################################
    args = parser.parse_args()

    # Data Paths
    train_filelist = os.path.join(args.FILELIST_DIR, args.train_filelist)
    validation_filelist = os.path.join(args.FILELIST_DIR, args.validation_filelist)
    test_filelist = os.path.join(args.FILELIST_DIR, args.test_filelist)

    #############################################
    # Check Devices
    #############################################
    print("\n")
    print("Slurm & Device Information")
    print("=========================================")
    print("Slurm Job ID:", os.environ["SLURM_JOB_ID"])
    print("Pytorch Cuda Available:", torch.cuda.is_available())
    print("GPU ID:", os.environ["SLURM_JOB_GPUS"])
    print("Number of System CPUs:", os.cpu_count())
    print("Number of CPUs per GPU:", os.environ["SLURM_JOB_CPUS_PER_NODE"])
    print("\n")
    print("Model Training Information")
    print("=========================================")

    #############################################
    # Initialize Model
    #############################################
    image_size = (
        args.image_size if args.scaled_image_size is None else args.scaled_image_size
    )
    ch_file = pd.read_csv(os.path.join(args.channel_data_dir, args.channel_file))
    hydro_fields = list(ch_file["channel"])
    if args.model.lower() == "cnn":
        model = LodeRunnerCNN(
            default_vars=hydro_fields,
            embed_dim=args.embed_dim,
            n_downsample=args.n_downsample,  # int(np.log2(args.scale_factor / 0.25)),
        )
    elif args.model.lower() == "vit":
        model = LodeRunner(
            default_vars=hydro_fields,
            image_size=image_size,
            patch_size=(5, 5),
            embed_dim=args.embed_dim,
            emb_factor=2,
            num_heads=8,
            block_structure=tuple(args.block_structure),
            window_sizes=[(2, 2) for _ in range(4)],
            patch_merge_scales=[(2, 2) for _ in range(3)],
        )
    else:
        raise ValueError(f"Model type {args.model} not available!")

    #############################################
    # Initialize Data
    #############################################
    # Define some arg-dependent flags and parameters.
    persistent_workers_train = args.n_channels_train == len(hydro_fields)
    persistent_workers_val = args.n_channels_val == len(hydro_fields)
    if args.n_channels_train < len(hydro_fields):
        reload_dataloaders_every_n_epochs = 1
    elif args.n_channels_val < len(hydro_fields):
        reload_dataloaders_every_n_epochs = args.TRAIN_PER_VAL
    else:
        reload_dataloaders_every_n_epochs = 0

    # Prepare datasets and datamodule.
    means = list(ch_file["mean"])
    scales = list(ch_file["scale"])
    transform = torch.nn.Sequential(
        ResizePadCrop(
            interp_kwargs={"scale_factor": args.scale_factor},
            scaled_image_size=args.scaled_image_size,
            pad_position=("bottom", "right"),
        ),
        tforms.Normalize(
            mean=means,
            std=scales,
        ),
    )
    dtype = getattr(torch, args.dtype)
    ds_params = {
        "LSC_NPZ_DIR": args.LSC_NPZ_DIR,
        "seq_len": args.seq_len,
        "timeIDX_offset": [1],
        "half_image": True,
        "transform": transform,
        "hydro_fields": hydro_fields,
        "dtype": dtype,
    }
    dl_params = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "prefetch_factor": args.prefetch_factor,
    }
    lsc_datamodule = LSCDataModule(
        ds_name="LSC_rho2rho_sequential_DataSet",
        ds_params_train=ds_params
        | {
            "file_prefix_list": train_filelist,
            "path_to_cache": os.path.join(args.cache_dir, args.cache_file_train),
        },
        ds_params_val=ds_params
        | {
            "file_prefix_list": validation_filelist,
            "path_to_cache": os.path.join(args.cache_dir, args.cache_file_val),
        },
        dl_params_train=dl_params
        | {
            "shuffle": True,
            "persistent_workers": persistent_workers_train,
            "pin_memory": persistent_workers_train,  # intentionally same as persisent workers
        },
        dl_params_val=dl_params
        | {
            "shuffle": False,
            "persistent_workers": persistent_workers_val,
            "pin_memory": persistent_workers_val,
        },
    )

    # Define a datamodule callback for scheduled sampling.
    class Scheduler(L.Callback):
        def __init__(
            self,
            hydro_fields: str,
            means: list[float],
            scales: list[float],
            n_channels_train: int = 8,
            n_channels_val: int = 8,
            eval_on_same_channels: bool = True,
        ) -> None:
            self.hydro_fields = hydro_fields
            self.n_channels_train = n_channels_train
            self.n_channels_val = n_channels_val
            self.means = means
            self.scales = scales
            self.eval_on_same_channels = eval_on_same_channels

        def on_train_epoch_end(self, trainer, model):
            if self.n_channels_train < len(self.hydro_fields):
                h_fields_inds = torch.tensor(
                    np.random.choice(
                        np.arange(len(self.hydro_fields)),
                        size=self.n_channels_train,
                        replace=False,
                    )
                ).sort()[0]
            else:
                h_fields_inds = torch.arange(len(self.hydro_fields))
            model.in_vars_train.copy_(h_fields_inds)
            model.out_vars_train.copy_(h_fields_inds)
            h_fields_sample = [self.hydro_fields[n] for n in h_fields_inds]
            transform = torch.nn.Sequential(
                ResizePadCrop(
                    interp_kwargs={"scale_factor": args.scale_factor},
                    scaled_image_size=args.scaled_image_size,
                    pad_position=("bottom", "right"),
                ),
                tforms.Normalize(
                    mean=[self.means[n] for n in h_fields_inds],
                    std=[self.scales[n] for n in h_fields_inds],
                ),
            )
            if trainer.datamodule.ds_params_train is not None:
                trainer.datamodule.ds_params_train["hydro_fields"] = h_fields_sample
                trainer.datamodule.ds_params_train["transform"] = transform

        def on_validation_epoch_end(self, trainer, model):
            if self.eval_on_same_channels:
                # Validation should be done on same channel subset as training.
                h_fields_inds = model.in_vars_train.clone()
            elif self.n_channels_val < len(self.hydro_fields):
                h_fields_inds = torch.tensor(
                    np.random.choice(
                        np.arange(len(self.hydro_fields)),
                        size=self.n_channels_val,
                        replace=False,
                    )
                ).sort()[0]
            else:
                h_fields_inds = torch.arange(len(self.hydro_fields))
            model.in_vars_val.copy_(h_fields_inds)
            model.out_vars_val.copy_(h_fields_inds)
            h_fields_sample = [self.hydro_fields[n] for n in h_fields_inds]
            transform = torch.nn.Sequential(
                ResizePadCrop(
                    interp_kwargs={"scale_factor": args.scale_factor},
                    scaled_image_size=args.scaled_image_size,
                    pad_position=("bottom", "right"),
                ),
                tforms.Normalize(
                    mean=[self.means[n] for n in h_fields_inds],
                    std=[self.scales[n] for n in h_fields_inds],
                ),
            )
            if trainer.datamodule.ds_params_val is not None:
                trainer.datamodule.ds_params_val["hydro_fields"] = h_fields_sample
                trainer.datamodule.ds_params_val["transform"] = transform

        def on_fit_start(self, trainer, model):
            self.on_train_epoch_end(trainer=trainer, model=model)
            self.on_validation_epoch_end(trainer=trainer, model=model)

    ch_subsampling_schedule = Scheduler(
        hydro_fields=hydro_fields,
        means=means,
        scales=scales,
        n_channels_train=args.n_channels_train,
        n_channels_val=args.n_channels_val,
        eval_on_same_channels=bool(args.eval_on_same_channels),
    )

    #############################################
    # Lightning wrap
    #############################################
    # Define a cropped loss function (used to ignore padding on rescaled images).
    loss_mask = torch.zeros(args.scaled_image_size, dtype=dtype)
    scaled_image_size = np.array(args.scaled_image_size)
    valid_im_size = np.floor(args.scale_factor * np.array(args.image_size)).astype(int)
    loss_fxn = getattr(nn, args.loss)(reduction="none")
    loss = CroppedLoss2D(
        loss_fxn=loss_fxn,
        crop=(
            0,
            0,
            min(valid_im_size[0], args.scaled_image_size[0]),
            min(valid_im_size[1], args.scaled_image_size[1]),
        ),  # corresponds to pad_position=("bottom", "right") in ResizePadCrop
    )

    # Prepare the Lightning module.
    lm_kwargs = {
        "model": model,
        "in_vars_train": torch.arange(args.n_channels_train),
        "out_vars_train": torch.arange(args.n_channels_train),
        "in_vars_val": torch.arange(args.n_channels_val),
        "out_vars_val": torch.arange(args.n_channels_val),
        "loss_train": loss,
        "lr_scheduler": CosineWithWarmupScheduler,
        "scheduler_params": {
            "warmup_steps": args.warmup_steps,
            "anchor_lr": args.anchor_lr,
            "terminal_steps": args.terminal_steps,
            "num_cycles": args.num_cycles,
            "min_fraction": args.min_fraction,
        },
        "scheduled_sampling_scheduler": getattr(yoke.scheduled_sampling, args.schedule)(
            initial_schedule_prob=args.initial_schedule_prob,
            decay_param=args.decay_param,
            minimum_schedule_prob=args.minimum_schedule_prob,
        ),
        "use_pushforward": bool(args.use_pushforward),
        "gradient_clip_val": args.gradient_clip_val,
    }
    valid_checkpoint = (args.checkpoint is not None) and (
        args.checkpoint not in ("", "nan")
    )
    if args.continuation or (not valid_checkpoint) or args.only_load_backbone:
        L_loderunner = Lightning_LodeRunner(**lm_kwargs)
    else:
        # This condition is used to load pretrained weights without continuing training.
        L_loderunner = Lightning_LodeRunner.load_from_checkpoint(
            checkpoint_path=args.checkpoint,
            strict=False,
            **lm_kwargs,
        )

    # Load U-Net backbone if needed.
    if args.only_load_backbone and valid_checkpoint:
        ckpt = torch.load(args.checkpoint, map_location=torch.device("cpu"))
        unet_weights = {
            k.replace("model.unet.", ""): v
            for k, v in ckpt["state_dict"].items()
            if k.startswith("model.unet.")
        }
        L_loderunner.model.unet.load_state_dict(unet_weights)

    # Freeze the U-Net backbone.
    if args.freeze_backbone:
        tr.freeze_torch_params(L_loderunner.model.unet)

    # Prepare Lightning logger.
    logger = TensorBoardLogger(save_dir="./")

    # Determine starting and final epochs for this round of training.
    # Format: study{args.studyIDX:03d}_epoch={epoch:04d}_val_loss={val_loss:.4f}.ckpt
    if args.continuation:
        starting_epoch = (
            int(re.search(r"epoch=(?P<epoch>\d+)_", args.checkpoint)["epoch"]) + 1
        )
    else:
        starting_epoch = 0

    # Prepare trainer.
    checkpoint_callback = ModelCheckpoint(
        save_top_k=10,
        every_n_epochs=args.TRAIN_PER_VAL,
        monitor="val_loss",
        mode="min",
        dirpath="./checkpoints",
        filename=f"study{args.studyIDX:03d}" + "_{epoch:04d}_{val_loss:.4f}",
        save_last=True,
    )
    checkpoint_callback.CHECKPOINT_NAME_LAST = (
        f"study{args.studyIDX:03d}" + "_{epoch:04d}_{val_loss:.4f}-last"
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")
    final_epoch = min(starting_epoch + args.cycle_epochs, args.total_epochs) - 1
    trainer = L.Trainer(
        max_epochs=final_epoch + 1,
        limit_train_batches=args.train_batches,
        check_val_every_n_epoch=args.TRAIN_PER_VAL,
        limit_val_batches=args.val_batches,
        reload_dataloaders_every_n_epochs=reload_dataloaders_every_n_epochs,
        gradient_clip_val=None if args.use_pushforward else args.gradient_clip_val,
        accelerator="gpu",
        devices=args.Ngpus,  # Number of GPUs per node
        num_nodes=args.Knodes,
        strategy="ddp",
        precision=args.precision,
        enable_progress_bar=True,
        logger=logger,
        log_every_n_steps=min(
            1024,
            args.train_batches,
        ),  # arbitrary choice of 1024 minimum
        callbacks=[ch_subsampling_schedule, checkpoint_callback, lr_monitor],
    )

    # Run training using Lightning.
    if args.continuation:
        ckpt_path = args.checkpoint
    else:
        ckpt_path = None
    trainer.fit(
        L_loderunner,
        datamodule=lsc_datamodule,
        ckpt_path=ckpt_path,
    )

    #############################################
    # Continue if Necessary
    #############################################
    # Run only in main process, otherwise we'll get NGPUs copies of the chain due
    # to the way Lightning tries to parallelize the script.
    if trainer.is_global_zero:
        FINISHED_TRAINING = (final_epoch + 1) >= args.total_epochs
        if not FINISHED_TRAINING:
            new_slurm_file = tr.continuation_setup(
                checkpoint_callback.last_model_path,
                args.studyIDX,
                last_epoch=final_epoch + 1,
            )
            os.system(f"sbatch {new_slurm_file}")
