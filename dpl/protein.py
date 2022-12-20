# This file is adopted from DeepMind Technologies Limited.
#
# Copyright 2021 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import dataclasses
import io
import re
from pathlib import Path
from typing import Iterable, List, Union

import numpy as np
from Bio.PDB.PDBParser import PDBParser
from rdkit import Chem

# fmt: off
RESIDUE_TYPES = [
    "A", "R", "N", "D", "C", "Q", "E", "G", "H", "I",
    "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V"
]
RESIDUE_TYPE_INDEX = {name: index for index, name in enumerate(RESIDUE_TYPES)}

RESIDUE_NAMES = [
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL"
]
RESIDUE_NAME_INDEX = {name: index for index, name in enumerate(RESIDUE_NAMES)}

RESIDUE_ATOMS = [
    "N", "CA", "C", "CB", "O", "CG", "CG1", "CG2", "OG", "OG1", "SG", "CD",
    "CD1", "CD2", "ND1", "ND2", "OD1", "OD2", "SD", "CE", "CE1", "CE2", "CE3",
    "NE", "NE1", "NE2", "OE1", "OE2", "CH2", "NH1", "NH2", "OH", "CZ", "CZ2",
    "CZ3", "NZ", "OXT"
]
RESIDUE_ATOM_INDEX = {name: index for index, name in enumerate(RESIDUE_ATOMS)}

PDB_CHAIN_IDS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
# fmt: on


@dataclasses.dataclass(frozen=True)
class Protein:
    chain_index: np.ndarray
    residue_index: np.ndarray
    aatype: np.ndarray
    atom_pos: np.ndarray
    atom_mask: np.ndarray


def protein_from_pdb_string(pdb_str: str) -> Protein:
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("", io.StringIO(pdb_str))
    first_model = next(structure.get_models())

    chain_ids = []
    residue_index = []
    aatype = []
    atom_pos = []
    atom_mask = []
    for chain in first_model:
        for residue in chain.get_residues():
            if residue.id[0] != " ":  # hetfield
                continue
            if residue.id[2] != " ":  # insertion code
                raise ValueError("Insertion codes are not supperted.")
            pos = np.zeros((len(RESIDUE_ATOMS), 3), dtype=np.float32)
            mask = np.zeros((len(RESIDUE_ATOMS),), dtype=np.float32)
            for atom in residue:
                if atom.name in RESIDUE_ATOM_INDEX:
                    atom_index = RESIDUE_ATOM_INDEX[atom.name]
                else:
                    continue
                pos[atom_index] = atom.get_coord()
                mask[atom_index] = 1.0
            chain_ids.append(chain.id)
            residue_index.append(residue.id[1])
            aatype.append(RESIDUE_NAME_INDEX[residue.get_resname()])
            atom_pos.append(pos)
            atom_mask.append(mask)
    unique_chain_ids = list(np.unique(chain_ids))
    chain_index = [unique_chain_ids.index(chain_id) for chain_id in chain_ids]

    return Protein(
        chain_index=np.array(chain_index, dtype=np.int64),
        residue_index=np.array(residue_index, dtype=np.int64),
        aatype=np.array(aatype, dtype=np.int64),
        atom_pos=np.array(atom_pos, dtype=np.float32),
        atom_mask=np.array(atom_mask, dtype=np.float32),
    )


def protein_from_pdb_file(pdb_path: Union[str, Path]) -> Protein:
    with open(pdb_path, "r") as f:
        pdb_str = f.read()
    return protein_from_pdb_string(pdb_str)


def proteins_from_pdb_file(pdb_path: Union[str, Path]) -> List[Protein]:
    with open(pdb_path, "r") as f:
        pdb_str = f.read()
    proteins = []
    for s in re.split(r"ENDMDL.+?\n", pdb_str):
        if s == "":
            continue
        m = re.match(r"MODEL.+?\n", s)
        if m is not None:
            s = s[m.end() :]
        proteins.append(protein_from_pdb_string(s))
    return proteins


def protein_to_pdb_string(prot: Protein) -> str:
    pdb_lines = []
    atom_index = 1
    for i in range(prot.chain_index.shape[0]):
        chain_id = PDB_CHAIN_IDS[prot.chain_index[i]]
        residue_index = prot.residue_index[i]
        residue_name = RESIDUE_NAMES[prot.aatype[i]]
        for pos, mask, atom_name in zip(
            prot.atom_pos[i], prot.atom_mask[i], RESIDUE_ATOMS
        ):
            if mask < 0.5:
                continue
            record_type = "ATOM"
            alt_loc = ""
            insertion_code = ""
            occupancy = 1.0
            bfactor = 0.0
            element = atom_name[0]
            if len(atom_name) < 4:
                atom_name = " " + atom_name.ljust(3)
            charge = ""
            pdb_lines.append(
                f"{record_type:<6}{atom_index:>5} {atom_name}{alt_loc:>1}"
                f"{residue_name:>3} {chain_id:>1}"
                f"{residue_index:>4}{insertion_code:>1}   "
                f"{pos[0]:>8.3f}{pos[1]:>8.3f}{pos[2]:>8.3f}"
                f"{occupancy:>6.2f}{bfactor:>6.2f}          "
                f"{element:>2}{charge:>2}"
            )

            atom_index += 1
    pdb_lines = [line.ljust(80) for line in pdb_lines]
    return "\n".join(pdb_lines) + "\n"


def protein_to_pdb_file(prot: Protein, pdb_path: Union[str, Path]) -> None:
    pdb_str = protein_to_pdb_string(prot)
    with open(pdb_path, "w") as f:
        f.write(pdb_str)


def proteins_to_pdb_file(
    proteins: Iterable[Protein], pdb_path: Union[str, Path]
) -> None:
    pdb_str = ""
    for model_id, prot in enumerate(proteins, 1):
        pdb_str = pdb_str + f"MODEL      {model_id:>3}".ljust(80) + "\n"
        pdb_str = pdb_str + protein_to_pdb_string(prot)
        pdb_str = pdb_str + "ENDMDL".ljust(80) + "\n"
    with open(pdb_path, "w") as f:
        f.write(pdb_str)


def protein_from_sequence(sequence: str) -> Protein:
    aatype = np.array([RESIDUE_TYPE_INDEX[s] for s in sequence], dtype=np.int64)
    N = len(aatype)
    chain_index = np.zeros((N,), dtype=np.int64)
    residue_index = np.arange(N, dtype=np.int64)
    atom_pos = np.zeros((N, len(RESIDUE_ATOMS), 3), dtype=np.float32)
    atom_mask = np.zeros((N, len(RESIDUE_ATOMS)), dtype=np.float32)
    atom_mask[:, 1] = 1.0
    return Protein(
        chain_index=chain_index,
        residue_index=residue_index,
        aatype=aatype,
        atom_pos=atom_pos,
        atom_mask=atom_mask,
    )


def protein_to_sequence(prot: Protein) -> str:
    return "".join([RESIDUE_TYPES[aa] for aa in prot.aatype])


def protein_to_ca_mol(prot: Protein) -> Chem.Mol:
    ca_atom_mask = np.zeros_like(prot.atom_mask)
    ca_atom_mask[:, 1] = 1.0
    ca_prot = dataclasses.replace(prot, atom_mask=ca_atom_mask * prot.atom_mask)
    return Chem.MolFromPDBBlock(protein_to_pdb_string(ca_prot))
