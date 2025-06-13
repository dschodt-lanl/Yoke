"""Training for Lightning-wrapped LodeRunner on LSC material densities.

This version of training uses only the lsc240420 data with only per-material
density along with the velocity field. A single timestep is input, a single
timestep is predicted. The number of input variables is fixed throughout
training.

`lightning` is used to train a LightningModule wrapper for LodeRunner to allow
multi-node, multi-GPU, distributed data-parallel training.  Furthermore, this
version of the harness uses a YAML configuration file to allow highly-flexible
parameter settings.

"""

import argparse
import os

import hydra
import torch
from omegaconf import OmegaConf

import yoke.torch_training_utils as tr


# Prepare command line argument parser.
descr_str = (
    "Trains lightning-wrapped LodeRunner on multi-timestep input and output of the "
    "lsc240420 per-material density fields."
)
parser = argparse.ArgumentParser(
    prog="Initial LodeRunner Training", description=descr_str, fromfile_prefix_chars="@"
)
parser.add_argument(
    "--config",
    action="store",
    type=str,
    default="./default_parameters.yaml",
    help="YAML file containing study hyperparameters.",
)


if __name__ == "__main__":
    # Set precision for tensor core speedup potential.
    torch.set_float32_matmul_precision("medium")

    # Process CLI inputs.
    args = parser.parse_args()

    # Load parameter config.
    cfg = OmegaConf.load(args.config)

    # Instantiate datamodule.
    lsc_datamodule = hydra.utils.instantiate(cfg.datamodule)

    # Prepare the Lightning module.
    if cfg.continuation or (cfg.checkpoint is None) or cfg.only_load_backbone:
        L_loderunner = hydra.utils.instantiate(cfg.lightning_module)
    else:
        # This condition is used to load pretrained weights without continuing training.
        module_ref = hydra.utils.get_class(cfg.lightning_module._target_)
        module_kwargs = cfg.lightning_module
        module_kwargs.pop("_target_", None)  # remove _target_ from kwargs
        L_loderunner = module_ref.load_from_checkpoint(
            checkpoint_path=cfg.checkpoint, strict=False, **module_kwargs
        )

    # Load backbone weights.
    if cfg.only_load_backbone and os.path.exists(cfg.checkpoint):
        ckpt = torch.load(cfg.checkpoint, map_location=torch.device("cpu"))
        unet_weights = {
            k.replace("model.unet.", ""): v
            for k, v in ckpt["state_dict"].items()
            if k.startswith("model.unet.")
        }
        L_loderunner.model.unet.load_state_dict(unet_weights)

    # Freeze the backbone.
    if cfg.freeze_backbone:
        tr.freeze_torch_params(L_loderunner.model.unet)

    # Prepare trainer.
    trainer = hydra.utils.instantiate(cfg.trainer)

    # Run training using Lightning.
    if cfg.continuation:
        ckpt_path = cfg.checkpoint
    else:
        ckpt_path = None
    trainer.fit(
        L_loderunner,
        datamodule=lsc_datamodule,
        ckpt_path=ckpt_path,
    )
