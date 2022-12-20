# This file is adopted from Hannes Stärk.
#
# MIT License
#
# Copyright (c) 2022 Hannes Stärk
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from typing import Any, List, Mapping

import torch
from rdkit import Chem

# fmt: off
ALLOWABLE_ATOM_FEATURES: Mapping[str, List[Any]] = {
    "atomic_num": list(range(1, 119)) + ["misc"],  # type: ignore[list-item]
    "chirality": [
        "CHI_UNSPECIFIED",
        "CHI_TETRAHEDRAL_CW",
        "CHI_TETRAHEDRAL_CCW",
        "CHI_OTHER",
    ],
    "degree": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, "misc"],
    "formal_charge": [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, "misc"],
    "num_hs": [0, 1, 2, 3, 4, 5, 6, 7, 8, "misc"],
    "num_radical_e": [0, 1, 2, 3, 4, "misc"],
    "hybridization": ["SP", "SP2", "SP3", "SP3D", "SP3D2", "misc"],
    "is_aromatic": [False, True],
    "is_in_ring": [False, True],
}


ALLOWABLE_BOND_FEATURES: Mapping[str, List[Any]] = {
    "bond_type": ["SINGLE", "DOUBLE", "TRIPLE", "AROMATIC", "misc"],
    "stereo": [
        "STEREONONE",
        "STEREOZ",
        "STEREOE",
        "STEREOCIS",
        "STEREOTRANS",
        "STEREOANY",
    ],
    "is_conjugated": [False, True],
}
# fmt: on


def safe_index(allowable_list: List[Any], value: Any) -> int:
    try:
        return allowable_list.index(value)
    except ValueError:
        assert allowable_list[-1] == "misc"
        return len(allowable_list) - 1


def featurize_atom(atom: Chem.Atom) -> torch.Tensor:
    return torch.tensor(
        [
            safe_index(ALLOWABLE_ATOM_FEATURES["atomic_num"], atom.GetAtomicNum()),
            ALLOWABLE_ATOM_FEATURES["chirality"].index(str(atom.GetChiralTag())),
            safe_index(ALLOWABLE_ATOM_FEATURES["degree"], atom.GetTotalDegree()),
            safe_index(
                ALLOWABLE_ATOM_FEATURES["formal_charge"], atom.GetFormalCharge()
            ),
            safe_index(ALLOWABLE_ATOM_FEATURES["num_hs"], atom.GetTotalNumHs()),
            safe_index(
                ALLOWABLE_ATOM_FEATURES["num_radical_e"], atom.GetNumRadicalElectrons()
            ),
            safe_index(
                ALLOWABLE_ATOM_FEATURES["hybridization"], str(atom.GetHybridization())
            ),
            ALLOWABLE_ATOM_FEATURES["is_aromatic"].index(atom.GetIsAromatic()),
            ALLOWABLE_ATOM_FEATURES["is_in_ring"].index(atom.IsInRing()),
        ],
        dtype=torch.long,
    )


def featurize_bond(bond: Chem.Bond) -> torch.Tensor:
    return torch.tensor(
        [
            safe_index(ALLOWABLE_BOND_FEATURES["bond_type"], str(bond.GetBondType())),
            ALLOWABLE_BOND_FEATURES["stereo"].index(str(bond.GetStereo())),
            ALLOWABLE_BOND_FEATURES["is_conjugated"].index(bond.GetIsConjugated()),
        ],
        dtype=torch.long,
    )
