import itertools
import os
import subprocess
import tempfile
from typing import Tuple

import numpy as np

from .protein import Protein, protein_to_pdb_file


def run_tmalign(
    prb: Protein, ref: Protein, mirror: bool = False
) -> Tuple[float, np.ndarray, np.ndarray]:
    with tempfile.TemporaryDirectory() as tmp_dir:
        prb_path = os.path.join(tmp_dir, "prb.pdb")
        ref_path = os.path.join(tmp_dir, "ref.pdb")
        protein_to_pdb_file(prb, prb_path)
        protein_to_pdb_file(ref, ref_path)
        cmd = ["TMalign", prb_path, ref_path]
        cmd += ["-outfmt", "2"]
        if mirror:
            cmd += ["-mirror", "1"]
        matrix_path = os.path.join(tmp_dir, "matrix.txt")
        cmd += ["-m", matrix_path]
        try:
            output = subprocess.check_output(cmd).decode()
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"TMalign failed: {e}")
        line = output.splitlines()[1]
        tmscore = float(line.split()[3])  # TM2
        t, R = np.empty((3,)), np.empty((3, 3))
        with open(matrix_path, "r") as f:
            for i, line in enumerate(itertools.islice(f, 2, 5)):
                t[i], R[0, i], R[1, i], R[2, i] = map(float, line.split()[1:])
        if mirror:
            R = np.array([[1, 0, 0], [0, 1, 0], [0, 0, -1]]) @ R
        return tmscore, t, R
