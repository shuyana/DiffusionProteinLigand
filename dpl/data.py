from pathlib import Path
from typing import Any, List, Mapping, Optional, Sequence, Union

import pytorch_lightning as pl
import torch
from rdkit import Chem
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, default_collate

from .features import ALLOWABLE_BOND_FEATURES, featurize_atom, featurize_bond
from .mol import get_mol_positions
from .protein import Protein, protein_to_ca_mol


def ligand_to_data(ligand: Chem.Mol, **kwargs: Any) -> Mapping[str, Any]:
    num_atoms = ligand.GetNumAtoms()
    atom_feats = torch.stack(
        [featurize_atom(atom) for atom in ligand.GetAtoms()], dim=0
    )
    atom_mask = torch.ones(num_atoms)
    atom_pos = torch.from_numpy(get_mol_positions(ligand))
    bond_feats = torch.zeros(
        num_atoms, num_atoms, len(ALLOWABLE_BOND_FEATURES), dtype=torch.long
    )
    bond_mask = torch.zeros(num_atoms, num_atoms)
    for i in range(num_atoms):
        for j in range(num_atoms):
            bond = ligand.GetBondBetweenAtoms(i, j)
            if bond is not None:
                bond_feats[i, j] = featurize_bond(bond)
                bond_mask[i, j] = 1.0
    bond_distance = torch.tensor(Chem.GetDistanceMatrix(ligand), dtype=torch.long)
    return {
        "ligand_mol": ligand,
        "num_atoms": num_atoms,
        "atom_feats": atom_feats,
        "atom_mask": atom_mask,
        "atom_pos": atom_pos,
        "bond_feats": bond_feats,
        "bond_mask": bond_mask,
        "bond_distance": bond_distance,
        **kwargs,
    }


def protein_to_data(prot: Protein, **kwargs: Any) -> Mapping[str, Any]:
    num_residues = len(prot.aatype)
    residue_type = torch.from_numpy(prot.aatype)
    residue_mask = torch.ones(num_residues)
    residue_chain_index = torch.from_numpy(prot.chain_index)
    residue_index = torch.from_numpy(prot.residue_index)
    residue_atom_pos = torch.from_numpy(prot.atom_pos)
    residue_atom_mask = torch.from_numpy(prot.atom_mask)
    return {
        "protein_mol": protein_to_ca_mol(prot),
        "num_residues": num_residues,
        "residue_type": residue_type,
        "residue_mask": residue_mask,
        "residue_chain_index": residue_chain_index,
        "residue_index": residue_index,
        "residue_atom_pos": residue_atom_pos,
        "residue_atom_mask": residue_atom_mask,
        **kwargs,
    }


def collate_fn(data_list: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    N = max(map(lambda d: d["num_atoms"] + d["num_residues"], data_list))
    batch = {}
    for k, v in data_list[0].items():
        if k.startswith("atom_"):
            feat_pad = (0, 0) * (v.dim() - 1)
            batch[k] = default_collate(
                [F.pad(d[k], feat_pad + (0, N - d["num_atoms"])) for d in data_list]
            )
        elif k.startswith("bond_"):
            feat_pad = (0, 0) * (v.dim() - 2)
            batch[k] = default_collate(
                [F.pad(d[k], feat_pad + (0, N - d["num_atoms"]) * 2) for d in data_list]
            )
        elif k.startswith("residue_"):
            feat_pad = (0, 0) * (v.dim() - 1)
            batch[k] = default_collate(
                [
                    F.pad(
                        d[k],
                        feat_pad
                        + (d["num_atoms"], N - d["num_atoms"] - d["num_residues"]),
                    )
                    for d in data_list
                ]
            )
        elif k.endswith("_mol"):
            batch[k] = [data[k] for data in data_list]
        else:
            batch[k] = default_collate([data[k] for data in data_list])
    return batch


class RepeatDataset(Dataset):
    def __init__(self, data: Mapping[str, Any], repeat: int):
        super().__init__()
        self.data = data
        self.repeat = repeat

    def __len__(self):
        return self.repeat

    def __getitem__(self, index: int) -> Mapping[str, Any]:
        return self.data


class PDBbindDataset(Dataset):
    def __init__(self, root_dir: Union[str, Path], pdb_ids: Sequence[str]):
        super().__init__()
        if isinstance(root_dir, str):
            root_dir = Path(root_dir)
        self.root_dir = root_dir
        self.pdb_ids = pdb_ids

    def __len__(self):
        return len(self.pdb_ids)

    def __getitem__(self, index: int) -> Mapping[str, Any]:
        pdb_id = self.pdb_ids[index]
        ligand_data = torch.load(self.root_dir / pdb_id / "ligand_data.pt")
        protein_data = torch.load(self.root_dir / pdb_id / "protein_data.pt")
        return {"pdb_id": pdb_id, **ligand_data, **protein_data}


class PDBbindDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: Union[str, Path] = "data",
        batch_size: int = 1,
        num_workers: int = 1,
    ):
        super().__init__()
        if isinstance(data_dir, str):
            data_dir = Path(data_dir)
        self.data_dir = data_dir
        self.cache_dir = data_dir / "PDBBind_processed_cache"
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage: Optional[str] = None) -> None:
        self.train_pdb_ids: List[str] = []
        with open(self.data_dir / "train_pdb_ids", "r") as f:
            self.train_pdb_ids.extend(line.strip() for line in f.readlines())
        self.val_pdb_ids: List[str] = []
        with open(self.data_dir / "val_pdb_ids", "r") as f:
            self.val_pdb_ids.extend(line.strip() for line in f.readlines())
        self.test_pdb_ids: List[str] = []
        with open(self.data_dir / "test_pdb_ids", "r") as f:
            self.test_pdb_ids.extend(line.strip() for line in f.readlines())

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            PDBbindDataset(self.cache_dir, self.train_pdb_ids),
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            PDBbindDataset(self.cache_dir, self.val_pdb_ids),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            PDBbindDataset(self.cache_dir, self.test_pdb_ids),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
        )
