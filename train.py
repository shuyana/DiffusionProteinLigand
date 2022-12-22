import os
import warnings
from argparse import ArgumentParser
from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from dpl.data import PDBbindDataModule
from dpl.model import DiffusionModel


def main(args):
    pl.seed_everything(args.seed, workers=True)
    args.save_dir.mkdir(parents=True)

    datamodule = PDBbindDataModule.from_argparse_args(args)
    model = DiffusionModel(args)
    trainer = pl.Trainer.from_argparse_args(
        args,
        accelerator="auto",
        precision=16,
        strategy="ddp_find_unused_parameters_false",
        logger=WandbLogger(save_dir=args.save_dir, project="DiffusionProteinLigand"),
        callbacks=[
            ModelCheckpoint(
                filename="{epoch:03d}-{val_loss:.2f}",
                monitor="val_loss",
                save_top_k=3,
                save_last=True,
            )
        ],
        default_root_dir=args.save_dir,
        max_epochs=-1,
    )
    trainer.fit(model, datamodule=datamodule)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser = PDBbindDataModule.add_argparse_args(parser)
    parser = DiffusionModel.add_argparse_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--save_dir", type=Path, required=True)
    args = parser.parse_args()

    # https://github.com/Lightning-AI/lightning/issues/5558#issuecomment-1199306489
    warnings.filterwarnings("ignore", "Detected call of", UserWarning)

    main(args)
