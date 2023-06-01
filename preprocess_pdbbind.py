import itertools
from argparse import ArgumentParser
from pathlib import Path
from typing import List

import torch
from rdkit import rdBase
from tqdm import tqdm

from dpl.data import ligand_to_data, protein_to_data
from dpl.mol import mol_from_file
from dpl.protein import RESIDUE_TYPES, protein_from_pdb_file


def main(args):
    rdBase.DisableLog("rdApp.*")
    input_dir = args.data_dir / "PDBBind_processed"
    if not input_dir.is_dir():
        raise ValueError(f"The PDBbind dataset not found: {input_dir}.")
    output_dir = args.data_dir / "PDBBind_processed_cache"
    output_dir.mkdir(parents=True)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model, alphabet = torch.hub.load("facebookresearch/esm:main", "esm2_t33_650M_UR50D")
    model.to(device).eval()
    batch_converter = alphabet.get_batch_converter()

    pdb_ids: List[str] = []
    with open(args.data_dir / "train_pdb_ids", "r") as f:
        pdb_ids.extend(line.strip() for line in f.readlines())
    with open(args.data_dir / "val_pdb_ids", "r") as f:
        pdb_ids.extend(line.strip() for line in f.readlines())
    with open(args.data_dir / "test_pdb_ids", "r") as f:
        pdb_ids.extend(line.strip() for line in f.readlines())

    for pdb_id in tqdm(pdb_ids):
        ligand_path = input_dir / pdb_id / f"{pdb_id}_ligand.sdf"
        try:
            ligand = mol_from_file(ligand_path)
        except ValueError:
            ligand = mol_from_file(ligand_path.with_suffix(".mol2"))

        protein_path = input_dir / pdb_id / f"{pdb_id}_protein_processed.pdb"
        protein = protein_from_pdb_file(protein_path)

        data = []
        for chain, _ in itertools.groupby(protein.chain_index):
            sequence = "".join(
                [RESIDUE_TYPES[aa] for aa in protein.aatype[protein.chain_index == chain]]
            )
            data.append(("", sequence))
        batch_tokens = batch_converter(data)[2].to(device)
        with torch.inference_mode():
            results = model(batch_tokens, repr_layers=[model.num_layers])
        token_representations = results["representations"][model.num_layers].cpu()
        residue_representations = []
        for i, (_, sequence) in enumerate(data):
            residue_representations.append(
                token_representations[i, 1 : len(sequence) + 1]
            )
        residue_esm = torch.cat(residue_representations, dim=0)
        assert residue_esm.size(0) == len(protein.aatype)

        output_path = output_dir / pdb_id
        output_path.mkdir()
        torch.save(ligand_to_data(ligand), output_path / "ligand_data.pt")
        torch.save(
            protein_to_data(protein, residue_esm=residue_esm),
            output_path / "protein_data.pt",
        )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--data_dir", type=Path, default="data")
    args = parser.parse_args()

    main(args)
