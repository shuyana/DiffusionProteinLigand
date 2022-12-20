from argparse import ArgumentParser, Namespace
from typing import Mapping, Union

import einops
import pytorch_lightning as pl
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from torch_ema import ExponentialMovingAverage

from .modules import (
    AtomEmbedding,
    BondEmbedding,
    FoldingBlock,
    Linear,
    RadialBasisProjection,
    SinusoidalProjection,
)
from .protein import RESIDUE_TYPES
from .utils import (
    angstrom_to_nanometre,
    nanometre_to_angstrom,
    nearest_bin,
    pseudo_beta,
    remove_mean,
)


class DiffusionModel(pl.LightningModule):
    def __init__(self, args: Union[Namespace, Mapping]):
        super().__init__()
        if isinstance(args, Mapping):
            args = Namespace(**args)
        self.no_cb_distogram = args.no_cb_distogram
        self.esm_dim = args.esm_dim
        self.time_dim = args.time_dim
        self.dist_dim = args.dist_dim
        self.single_dim = args.single_dim
        self.pair_dim = args.pair_dim
        self.head_dim = args.head_dim
        self.num_heads = args.num_heads
        self.transition_factor = args.transition_factor
        self.num_blocks = args.num_blocks
        self.max_bond_distance = args.max_bond_distance
        self.max_relpos = args.max_relpos
        self.gamma_0 = args.gamma_0
        self.gamma_1 = args.gamma_1
        self.num_steps = args.num_steps
        self.learning_rate = args.learning_rate
        self.warmup_steps = args.warmup_steps
        self.ema_decay = args.ema_decay
        self.embed_atom_feats = AtomEmbedding(self.single_dim)
        self.embed_bond_feats = BondEmbedding(self.pair_dim)
        self.embed_bond_distance = nn.Embedding(
            self.max_bond_distance + 1, self.pair_dim
        )
        self.embed_residue_type = nn.Embedding(len(RESIDUE_TYPES), self.single_dim)
        self.embed_residue_esm = nn.Sequential(
            nn.LayerNorm(self.esm_dim, elementwise_affine=False),
            Linear(self.esm_dim, self.single_dim, bias=False, init="normal"),
        )
        self.embed_relpos = nn.Embedding(self.max_relpos * 2 + 1, self.pair_dim)
        self.embed_cb_distogram = nn.Embedding(39, self.pair_dim)
        self.embed_dist = nn.Sequential(
            RadialBasisProjection(self.dist_dim),
            Linear(self.dist_dim, self.pair_dim, bias=False, init="normal"),
        )
        self.embed_gamma = nn.Sequential(
            SinusoidalProjection(self.time_dim),
            Linear(self.time_dim, self.pair_dim, bias=False, init="normal"),
        )
        self.folding_blocks = nn.ModuleList(
            [
                FoldingBlock(
                    self.single_dim,
                    self.pair_dim,
                    self.head_dim,
                    self.num_heads,
                    self.transition_factor,
                )
                for _ in range(self.num_blocks)
            ]
        )
        self.weight_radial = nn.Sequential(
            nn.LayerNorm(self.pair_dim, elementwise_affine=False),
            Linear(self.pair_dim, self.pair_dim, init="relu"),
            nn.ReLU(),
            Linear(self.pair_dim, 1, bias=False, init="final"),
        )

        self.ema = ExponentialMovingAverage(self.parameters(), decay=self.ema_decay)

        self.save_hyperparameters(args)

    @staticmethod
    def add_argparse_args(parent_parser: ArgumentParser) -> ArgumentParser:
        parser = parent_parser.add_argument_group("DiffusionModel")
        parser.add_argument("--no_cb_distogram", action="store_true")
        parser.add_argument("--esm_dim", type=int, default=1280)
        parser.add_argument("--time_dim", type=int, default=256)
        parser.add_argument("--dist_dim", type=int, default=256)
        parser.add_argument("--single_dim", type=int, default=512)
        parser.add_argument("--pair_dim", type=int, default=64)
        parser.add_argument("--head_dim", type=int, default=16)
        parser.add_argument("--num_heads", type=int, default=4)
        parser.add_argument("--transition_factor", type=int, default=4)
        parser.add_argument("--num_blocks", type=int, default=12)
        parser.add_argument("--max_bond_distance", type=int, default=7)
        parser.add_argument("--max_relpos", type=int, default=32)
        parser.add_argument("--gamma_0", type=float, default=-10.0)
        parser.add_argument("--gamma_1", type=float, default=10.0)
        parser.add_argument("--num_steps", type=int, default=64)
        parser.add_argument("--learning_rate", type=float, default=4e-4)
        parser.add_argument("--warmup_steps", type=int, default=1000)
        parser.add_argument("--ema_decay", type=float, default=0.999)
        return parent_parser

    def to(self, *args, **kwargs):
        out = torch._C._nn._parse_to(*args, **kwargs)
        self.ema.to(device=out[0], dtype=out[1])
        return super().to(*args, **kwargs)

    def on_save_checkpoint(self, checkpoint):
        checkpoint["ema_state_dict"] = self.ema.state_dict()

    def on_load_checkpoint(self, checkpoint):
        self.ema.load_state_dict(checkpoint["ema_state_dict"])

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        lr_scheduler_config = {
            "scheduler": torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=1.0 / self.warmup_steps,
                total_iters=self.warmup_steps - 1,
            ),
            "interval": "step",
        }
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler_config}

    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)
        self.ema.update(self.parameters())

    def training_step(self, batch, batch_idx):
        batch, x, mask = self.prepare_batch(batch)
        batch_size = x.size(0)
        num_nodes = einops.reduce(mask > 0.5, "b i -> b", "sum")
        t = torch.rand(batch_size, device=self.device)
        diffusion_loss = self.diffusion_loss(batch, x, mask, t)
        loss = torch.mean(diffusion_loss / num_nodes)
        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
            batch_size=batch_size,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        batch, x, mask = self.prepare_batch(batch)
        batch_size = x.size(0)
        num_nodes = einops.reduce(mask > 0.5, "b i -> b", "sum")
        t = torch.rand(batch_size, device=self.device)
        with self.ema.average_parameters():
            diffusion_loss = self.diffusion_loss(batch, x, mask, t)
        loss = torch.mean(diffusion_loss / num_nodes)
        self.log(
            "val_loss",
            loss,
            on_epoch=True,
            sync_dist=True,
            batch_size=batch_size,
        )

    def predict_step(self, batch, batch_idx):
        with self.ema.average_parameters():
            x = self.sample(batch)
        return x

    def forward(self, batch, z, mask, gamma):
        atom_feats = batch["atom_feats"]
        atom_mask = batch["atom_mask"]
        bond_feats = batch["bond_feats"]
        bond_mask = batch["bond_mask"]
        bond_distance = batch["bond_distance"]
        residue_type = batch["residue_type"]
        residue_mask = batch["residue_mask"]
        residue_esm = batch["residue_esm"]
        residue_chain_index = batch["residue_chain_index"]
        residue_index = batch["residue_index"]
        residue_atom_pos = batch["residue_atom_pos"]
        residue_atom_mask = batch["residue_atom_mask"]
        atom_mask_2d = atom_mask.unsqueeze(-1) * atom_mask.unsqueeze(-2)
        residue_mask_2d = residue_mask.unsqueeze(-1) * residue_mask.unsqueeze(-2)
        relpos = residue_index.unsqueeze(-1) - residue_index.unsqueeze(-2)
        chain_mask = (
            residue_chain_index.unsqueeze(-1) == residue_chain_index.unsqueeze(-2)
        ).float()
        cb_pos, cb_mask = pseudo_beta(residue_atom_pos, residue_atom_mask)
        cb_distogram = nearest_bin(
            torch.linalg.norm(
                cb_pos.unsqueeze(-2) - cb_pos.unsqueeze(-3),
                dim=-1,
            ),
            39,
            3.25,
            52.0,
        )
        cb_mask_2d = cb_mask.unsqueeze(-1) * cb_mask.unsqueeze(-2)
        if self.no_cb_distogram:
            cb_mask_2d.zero_()
        mask_2d = mask.unsqueeze(-1) * mask.unsqueeze(-2)
        radial = z.unsqueeze(-2) - z.unsqueeze(-3)
        dist = torch.linalg.norm(radial, dim=-1)
        scaled_gamma = (gamma + 10.0) / 20.0

        single = atom_mask.unsqueeze(-1) * self.embed_atom_feats(atom_feats)
        single += residue_mask.unsqueeze(-1) * (
            self.embed_residue_type(residue_type) + self.embed_residue_esm(residue_esm)
        )
        pair = atom_mask_2d.unsqueeze(-1) * (
            bond_mask.unsqueeze(-1) * self.embed_bond_feats(bond_feats)
            + self.embed_bond_distance(bond_distance.clamp(max=self.max_bond_distance))
        )
        pair += residue_mask_2d.unsqueeze(-1) * (
            chain_mask.unsqueeze(-1)
            * self.embed_relpos(
                self.max_relpos
                + relpos.clamp(min=-self.max_relpos, max=self.max_relpos)
            )
            + cb_mask_2d.unsqueeze(-1) * self.embed_cb_distogram(cb_distogram)
        )
        pair += mask_2d.unsqueeze(-1) * (
            self.embed_dist(dist) + self.embed_gamma(scaled_gamma[:, None, None])
        )

        checkpoint_fn = checkpoint if pair.requires_grad else lambda f, *args: f(*args)
        for block in self.folding_blocks:
            single, pair = checkpoint_fn(block, single, pair, mask)
        pair = 0.5 * (pair + einops.rearrange(pair, "b i j h -> b j i h"))

        w = self.weight_radial(pair)
        r = radial * torch.rsqrt(
            torch.sum(torch.square(radial), -1, keepdim=True) + 1e-4
        )
        noise_pred = einops.reduce(
            mask_2d.unsqueeze(-1) * w * r,
            "b i j xyz -> b i xyz",
            "sum",
        )
        noise_pred = remove_mean(noise_pred, mask)

        return noise_pred

    @torch.inference_mode()
    def sample(self, batch):
        batch, x, mask = self.prepare_batch(batch)
        batch_size = x.size(0)
        time_steps = torch.linspace(
            1.0, 0.0, steps=self.num_steps + 1, device=self.device
        )

        z_t = remove_mean(torch.randn_like(x), mask)

        for i in range(self.num_steps):
            t = torch.broadcast_to(time_steps[i], (batch_size,))
            s = torch.broadcast_to(time_steps[i + 1], (batch_size,))
            gamma_t = self.gamma(t)
            gamma_s = self.gamma(s)
            alpha_sq_t = torch.sigmoid(-gamma_t)
            alpha_sq_s = torch.sigmoid(-gamma_s)
            sigma_t = torch.sqrt(torch.sigmoid(gamma_t))
            c = -torch.expm1(gamma_s - gamma_t)
            noise_pred = self(batch, z_t, mask, gamma_t)
            mean = torch.sqrt(alpha_sq_s / alpha_sq_t)[:, None, None] * (
                z_t - (sigma_t * c)[:, None, None] * noise_pred
            )
            std = torch.sqrt((1.0 - alpha_sq_s) * c)[:, None, None]
            noise = remove_mean(torch.randn_like(z_t), mask)
            z_t = mean + std * noise

        gamma_0 = self.gamma(z_t.new_zeros((batch_size)))
        alpha_0 = torch.sqrt(torch.sigmoid(-gamma_0))
        sigma_0 = torch.sqrt(torch.sigmoid(gamma_0))
        noise = remove_mean(torch.randn_like(z_t), mask)
        x = (z_t + sigma_0 * noise) / alpha_0

        pos = nanometre_to_angstrom(x)
        return pos

    def prepare_batch(self, batch):
        atom_pos = batch["atom_pos"]
        atom_mask = batch["atom_mask"]
        residue_ca_pos = batch["residue_atom_pos"][:, :, 1]
        residue_mask = batch["residue_mask"]
        pos = (
            atom_mask.unsqueeze(-1) * atom_pos
            + residue_mask.unsqueeze(-1) * residue_ca_pos
        )
        x = angstrom_to_nanometre(pos)
        mask = atom_mask + residue_mask
        return batch, x, mask

    def gamma(self, t):
        return self.gamma_0 + (self.gamma_1 - self.gamma_0) * t

    def diffusion_loss(self, batch, x, mask, t):
        with torch.enable_grad():
            t = t.clone().detach().requires_grad_(True)
            gamma_t = self.gamma(t)
            grad_gamma_t = torch.autograd.grad(gamma_t.sum(), t, create_graph=True)[0]
        gamma_t = gamma_t.detach()
        alpha_t = torch.sqrt(torch.sigmoid(-gamma_t))
        sigma_t = torch.sqrt(torch.sigmoid(gamma_t))
        noise = remove_mean(torch.randn_like(x), mask)
        z_t = alpha_t[:, None, None] * x + sigma_t[:, None, None] * noise
        noise_pred = self(batch, z_t, mask, gamma_t)
        loss = (
            0.5
            * grad_gamma_t
            * einops.reduce(
                mask.unsqueeze(-1) * torch.square(noise_pred - noise),
                "b i xyz -> b",
                "sum",
            )
        )
        return loss
