# End-to-end protein-ligand complex structure generation with diffusion-based generative models

## Setup Environment
Clone this repository and install dependencies:
```bash
git clone https://github.com/shuyana/DiffusionProteinLigand.git
cd DiffusionProteinLigand
conda env create -f environment.yml
conda activate dpl
```

Download model parameters:
```bash
gdown --fuzzy --folder https://drive.google.com/drive/u/1/folders/1AAJ4P5EmQtwle9_eSeNMcF-KMWObksxZ
```

Additionally, TMalign is required to align generated structures.
You can install it as follows:
```bash
wget https://zhanggroup.org/TM-align/TMalign.cpp
g++ -static -O3 -ffast-math -lm -o TMalign TMalign.cpp
chmod +x TMalign
export PATH="/path/to/TMalign:$PATH"
```

## Sample generation
Generate complex structures with the protein structure-free model:
```bash
python generate.py \
    --ckpt_path "checkpoints/DPL_v1.ckpt" \
    --output_dir "workdir/generate/example_DPL" \
    --protein "LSEQLKHCNGILKELLSKKHAAYAWPFYKPVDASALGLHDYHDIIKHPMDLSTVKRKMENRDYRDAQEFAADVRLMFSNCYKYNPPDHDVVAMARKLQDVFEFRYAKMPD" \
    --ligand "Cc1ccc2c(c1c3cc(cc4c3nc([nH]4)C5CC5)c6c(noc6C)C)cccn2" \
    --num_samples 8
```

Alternatively, the protein structure-dependent model can be used:
```bash
wget https://files.rcsb.org/download/6MOA.pdb
python generate.py \
    --ckpt_path "checkpoints/DPLS_v1.ckpt" \
    --output_dir "workdir/generate/example_DPLS" \
    --protein "6MOA.pdb" \
    --ligand "Cc1ccc2c(c1c3cc(cc4c3nc([nH]4)C5CC5)c6c(noc6C)C)cccn2" \
    --num_samples 8
```
Note that an input protein structure must be given as a PDB file in this case.

Besides, you can specify a reference protein structure to be used for the alignment of results:
```bash
python generate.py \
    --ckpt_path "checkpoints/DPL_v1.ckpt" \
    --output_dir "workdir/generate/example_DPL_ref" \
    --protein "LSEQLKHCNGILKELLSKKHAAYAWPFYKPVDASALGLHDYHDIIKHPMDLSTVKRKMENRDYRDAQEFAADVRLMFSNCYKYNPPDHDVVAMARKLQDVFEFRYAKMPD" \
    --ligand "Cc1ccc2c(c1c3cc(cc4c3nc([nH]4)C5CC5)c6c(noc6C)C)cccn2" \
    --num_samples 8 \
    --ref_path "6MOA.pdb"
```
This is used only for alignment and does not affect the generation process itself.

Run time can be reduced by changing num_steps from the default value of 64:
```bash
python generate.py \
    --ckpt_path "checkpoints/DPL_v1.ckpt" \
    --output_dir "workdir/generate/example_DPL_fast" \
    --protein "LSEQLKHCNGILKELLSKKHAAYAWPFYKPVDASALGLHDYHDIIKHPMDLSTVKRKMENRDYRDAQEFAADVRLMFSNCYKYNPPDHDVVAMARKLQDVFEFRYAKMPD" \
    --ligand "Cc1ccc2c(c1c3cc(cc4c3nc([nH]4)C5CC5)c6c(noc6C)C)cccn2" \
    --num_samples 8 \
    --ref_path "6MOA.pdb" \
    --num_steps 24
```

## Training
Code for training will soon be available.

## Citation
    @article{Nakata2022,
        author = {Nakata, Shuya and Mori, Yoshiharu and Tanaka, Shigenori},
        doi = {10.1101/2022.12.20.521309},
        url = {https://doi.org/10.1101/2022.12.20.521309},
        title = {End-to-end protein-ligand complex structure generation with diffusion-based generative models},
        journal = {bioRxiv},
        year = {2022},
        publisher = {Cold Spring Harbor Laboratory},
    }

## Acknowledgements
Our work is based on the following repositories:
- https://github.com/deepmind/alphafold
- https://github.com/facebookresearch/esm
- https://github.com/HannesStark/EquiBind
- https://github.com/RosettaCommons/trRosetta2
