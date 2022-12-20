import dataclasses
import itertools
import warnings
from argparse import ArgumentParser
from operator import itemgetter
from pathlib import Path
from typing import Tuple

import numpy as np
import pytorch_lightning as pl
import torch
from rdkit import Chem
from torch.utils.data import DataLoader

from dpl.data import RepeatDataset, collate_fn, ligand_to_data, protein_to_data
from dpl.model import DiffusionModel
from dpl.mol import get_mol_positions, mol_from_file, update_mol_positions
from dpl.protein import (
    RESIDUE_TYPES,
    Protein,
    protein_from_pdb_file,
    protein_from_sequence,
    proteins_to_pdb_file,
)
from dpl.tmalign import run_tmalign


def compute_residue_esm(protein: Protein) -> torch.Tensor:
    esm_model, esm_alphabet = torch.hub.load(
        "facebookresearch/esm:main", "esm2_t33_650M_UR50D"
    )
    esm_model.cuda().eval()
    esm_batch_converter = esm_alphabet.get_batch_converter()

    data = []
    for chain, _ in itertools.groupby(protein.chain_index):
        sequence = "".join(
            [RESIDUE_TYPES[aa] for aa in protein.aatype[protein.chain_index == chain]]
        )
        data.append(("", sequence))
    batch_tokens = esm_batch_converter(data)[2].cuda()
    with torch.inference_mode():
        results = esm_model(batch_tokens, repr_layers=[esm_model.num_layers])
    token_representations = results["representations"][esm_model.num_layers].cpu()
    residue_representations = []
    for i, (_, sequence) in enumerate(data):
        residue_representations.append(token_representations[i, 1 : len(sequence) + 1])
    residue_esm = torch.cat(residue_representations, dim=0)
    assert residue_esm.size(0) == len(protein.aatype)
    return residue_esm


def update_pos(
    protein: Protein, ligand: Chem.Mol, pos: np.ndarray
) -> Tuple[Protein, Chem.Mol]:
    atom_pos = np.zeros_like(protein.atom_pos)
    atom_pos[:, 1] = pos[ligand.GetNumAtoms() :]
    atom_mask = np.zeros_like(protein.atom_mask)
    atom_mask[:, 1] = 1.0
    protein = dataclasses.replace(protein, atom_pos=atom_pos, atom_mask=atom_mask)
    ligand = update_mol_positions(ligand, pos[: ligand.GetNumAtoms()])
    return protein, ligand


def main(args):
    pl.seed_everything(args.seed, workers=True)
    args.output_dir.mkdir(parents=True)

    # Model
    model = DiffusionModel.load_from_checkpoint(
        args.ckpt_path, num_steps=args.num_steps
    )

    # Inputs
    if args.protein.endswith(".pdb"):
        protein = protein_from_pdb_file(args.protein)
    else:
        if not model.no_cb_distogram:
            raise ValueError(
                "When using the protein structure-dependent model, the protein structure must be provided with a PDB file."
            )
        protein = protein_from_sequence(args.protein)

    if args.ligand.endswith(".sdf") or args.ligand.endswith(".mol2"):
        ligand = mol_from_file(args.ligand)
    else:
        ligand = Chem.MolFromSmiles(args.ligand)
    ligand = update_mol_positions(ligand, np.zeros((ligand.GetNumAtoms(), 3)))

    total_num_atoms = len(protein.aatype) + ligand.GetNumAtoms()
    print(f"Total number of atoms: {total_num_atoms}")
    if total_num_atoms > 384:
        warnings.warn("Too many atoms. May take a long time for sample generation.")

    data = {
        **ligand_to_data(ligand),
        **protein_to_data(protein, residue_esm=compute_residue_esm(protein)),
    }

    ref_protein = protein_from_pdb_file(args.ref_path) if args.ref_path else None

    # Generate samples
    trainer = pl.Trainer.from_argparse_args(
        args,
        accelerator="auto",
        default_root_dir=args.output_dir,
        max_epochs=-1,
    )
    positions = trainer.predict(
        model,
        dataloaders=DataLoader(
            RepeatDataset(data, args.num_samples),
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            collate_fn=collate_fn,
        ),
    )
    positions = torch.cat(positions, dim=0).detach().cpu().numpy()

    # Save samples
    sample_proteins, sample_ligands = [], []
    tmscores = []
    for pos in positions:
        sample_protein, sample_ligand = update_pos(protein, ligand, pos)
        if ref_protein is None:
            warnings.warn("Using the first sample as a reference. The resulting structures may be mirror images.")
            ref_protein = sample_protein
        tmscore, t, R = max(
            run_tmalign(sample_protein, ref_protein),
            run_tmalign(sample_protein, ref_protein, mirror=True),
            key=itemgetter(0),
        )
        sample_proteins.append(
            dataclasses.replace(
                sample_protein, atom_pos=t + sample_protein.atom_pos @ R
            )
        )
        sample_ligands.append(
            update_mol_positions(
                sample_ligand, t + get_mol_positions(sample_ligand) @ R
            )
        )
        tmscores.append(tmscore)
    proteins_to_pdb_file(sample_proteins, args.output_dir / "sample_protein.pdb")
    with Chem.SDWriter(str(args.output_dir / "sample_ligand.sdf")) as w:
        for sample_ligand in sample_ligands:
            w.write(sample_ligand)
    with open(args.output_dir / "sample_tmscores.txt", "w") as f:
        for tmscore in tmscores:
            f.write(str(tmscore) + "\n")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--num_steps", type=int, default=64)
    parser.add_argument("-c", "--ckpt_path", type=Path, required=True)
    parser.add_argument("-o", "--output_dir", type=Path, required=True)
    parser.add_argument("-p", "--protein", type=str, required=True)
    parser.add_argument("-l", "--ligand", type=str, required=True)
    parser.add_argument("-n", "--num_samples", type=int, required=True)
    parser.add_argument("-r", "--ref_path", type=Path)
    args = parser.parse_args()

    main(args)
