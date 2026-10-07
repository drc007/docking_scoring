# Docking_scoring

A Jupyter notebook for docking ligands into a protein with **smina** and rescoring the poses with the **AEV-PLIG** binding-affinity model. It covers conformer generation, binding-site preparation, rigid or flexible docking, pose viewing, affinity prediction and export.

The example is the SARS-CoV-2 main protease (Mpro) fragment structure **5RHD**. A fully run copy of the notebook is in [`DockingScoring.html`](DockingScoring.html) (download it and open it in a browser to see the outputs and 3D views).

## Workflow

1. **Ligands.** Load an SDF of 3D structures, check that the names are unique, and browse the molecules with a slider.
2. **Conformers.** Generate conformers with RDKit srETKDGv3 (or ETKDGv3, ETKDG, ETDG), pruned by RMSD, MMFF94-minimised and filtered by an energy window. Undefined stereocentres can optionally be enumerated.
3. **Binding site.** Either:
   - load a **protein–ligand complex PDB** and pick the ligand that defines the site, or
   - supply **prepared files**: a protein without the ligand plus the extracted ligand.
4. **Images.** Ray-traced PyMOL images of the whole protein with the ligand bound, and of the ligand in the binding site.
5. **Docking.** An smina input panel for rigid docking, flexible side chains, or `--minimize`.
6. **Results.** Step through the poses in 3D with a slider.
7. **AEV-PLIG.** Predict the pK of every pose, and score the ligand from the PDB in its crystal pose as a reference.
8. **Export.** Write the best pose of each ligand as PDB files and as a PyMOL session.

## Setup

Run these commands from the repository folder.

### 1. Conda environment and Jupyter kernel

```bash
conda env create -f environment.yml
```

```bash
conda run -n docking_scoring python -m ipykernel install --user --name docking_scoring --display-name "Python (docking_scoring)"
```

`environment.yml` is portable (macOS and Linux) and installs current versions. `docking_scoring.yml` pins the exact versions this was developed and tested with on an Apple-silicon Mac:

| Package | Tested version |
|---|---|
| Python | 3.12 |
| RDKit | 2026.09 |
| PyTorch | 2.10 (conda-forge) |
| torch-geometric | 2.8 |
| torchani | 2.9 |
| PyMOL (open source) | 3.1 |
| Open Babel | 3.2 |
| py3Dmol | 2.5 |
| ipywidgets | 8.1 |
| JupyterLab | 4.6 |

> **Install PyTorch from conda-forge, not pip.** The pip wheel bundles its own OpenMP library, which clashes with conda's and crashes on import with `OMP: Error #15`.

### 2. AEV-PLIG trained models

The trained models (about 300 MB) are not stored in this repository. Download them from the official [AEV-PLIG repository](https://github.com/oxpig/AEV-PLIG):

```bash
conda run -n docking_scoring python aev_plig/download_models.py
```

The script fetches the 10-model `model_GATv2Net_ligsim90_fep_benchmark` ensemble and its scaler from a fixed upstream commit, checks each file's checksum, and saves them to `aev_plig/output/trained_models/`.

### 3. smina

smina is not included in `environment.yml`: the conda-forge smina package (2020 build) needs an old version of Boost that forces an old RDKit. Install it in one of two ways:

- **A separate conda environment**, which the notebook finds automatically:

  ```bash
  conda create -n smina -c conda-forge smina
  ```

- **The prebuilt binary** from https://sourceforge.net/projects/smina/. Make it executable (`chmod +x`), then put it on your `PATH` or set the `SMINA` environment variable to its full path.

The notebook looks for smina in this order:

1. the `SMINA` environment variable;
2. `smina`, `smina.static` or `smina.osx.12` in the notebook's environment or on the `PATH`;
3. a conda environment called `smina`;
4. `/usr/local/smina.osx.12` and other common locations.

You can also type the path into the smina panel.

### 4. Run

Open `DockingScoring.ipynb` in JupyterLab with the **Python (docking_scoring)** kernel.

- Each step has a panel with a **Run** or **Apply** button. Running all cells sets the panels up but does not start the long jobs.
- To run every step with the default settings, set `AUTO_RUN = True` in the first code cell.

## Project layout

```
DockingScoring.ipynb         the notebook
DockingScoring.html          a fully run copy of the notebook (static)
docking_tools.py             all helper code (panels, viewers, docking, scoring, export)
environment.yml              conda environment (portable)
docking_scoring.yml          exact tested environment (macOS arm64)
aev_plig/                    AEV-PLIG prediction code (models: aev_plig/download_models.py)
example/                     5RHD example inputs; results are written here when you run the notebook
```

Generated results are not tracked in git (see `.gitignore`). Run the notebook to recreate them, or see `DockingScoring.html`.

## Inputs

### Ligand SDF

- 3D structures, ideally with hydrogens.
- Each record needs a name on its first line.
- smina keeps only that name, and the results are grouped by it, so different structures must not share a name.

The notebook checks for duplicate names. If it finds any, it offers two fixes:

- use an SD field with a unique value for every molecule (e.g. `ID`, `IDNUMBER`, `Record_ID`) as the ligand ID, or
- assign unique names automatically (`name`, `name_2`, …).

The original name is kept in the `original_name` property, and the input file is not modified.

### Binding site

**Complex PDB mode**

- Lists every HET group (waters excluded) and pre-selects the largest; choose another if the file contains several ligands.
- Bond orders for the extracted ligand come from:
  - its hydrogens and formal charges (RDKit), when hydrogens are present;
  - a SMILES template, if you paste one;
  - Open Babel, when there are no hydrogens.
- The protein file keeps the HELIX and SHEET records, but other HET groups and waters are removed. Metal ions are kept unless you untick that option.

**Prepared-files mode** uses your protein and ligand PDB files as they are.

> **Note:** The two modes can give different smina scores for the same ligand. For example, `proteinNoligand.pdb` keeps the waters and DMS molecules, which add contacts. Compare runs that used the same mode.

## Outputs

Docked poses are stored as SDF files. The best pose of each ligand is also exported as PDB files (ligand only, and protein–ligand complex) and as a PyMOL session file.

Every output is written next to the input it was derived from.

| File | Location |
|---|---|
| `<input>_confs.sdf` | next to the ligand SDF |
| `<stem>_protein.pdb`, `<stem>_ligand_<RES><chain><num>.pdb/.sdf` | next to the complex PDB |
| `<stem>_complex.png`, `<stem>_site.png` | next to the binding-site input |
| `<input>_docked.sdf` / `_flexdocked.sdf` / `_minimized.sdf`, plus `.log` | next to the ligand SDF |
| `<output>_flexres.pdb` (flexible docking: one model per pose) | next to the docked SDF |

All prediction results go to a `predictions/` folder next to the docked SDF:

| File | Contents |
|---|---|
| `<docked>_summary.csv` | every pose: smina score, AEV-PLIG pK (10-model mean ± std), best-pose flags |
| `<docked>_best_poses_table.csv` | best pose per ligand, plus the PDB ligand row (AEV-PLIG pK and smina scores) |
| `<docked>_scored.sdf`, `<docked>_best_poses.sdf` | poses annotated with pK |
| `best_poses_pdb/<docked>/` | best pose per ligand as `_ligand.pdb` and `_complex.pdb` (ligand = chain L, resname LIG) |
| `<docked>_best_poses.pse` | PyMOL session of the best poses |
| `<docked>_smina_vs_aevplig.png` | smina score vs pK; the dashed line is the PDB ligand |
| `poses/<docked>/` | per-pose AEV-PLIG inputs (and per-pose receptors after flexible docking) |
| `pdb_ligand/` | the PDB ligand as scored by smina, its minimised pose and the log |
| `<docked>_aevplig_input.csv`, `<docked>_predictions.csv`, logs | raw AEV-PLIG input and output |

Copies of the docked SDF, receptor, reference ligand, conformers and images are also placed in `predictions/`, so the folder is a self-contained record.

## Flexible docking

- Choose **Flexible side chains** in the smina panel.
- **Automatic** selection takes every residue within a set distance of the ligand (default 5 Å).
- **Manual** selection lets you pick residues from a list, or type them (`A:41,A:145`).
- Gly, Ala and Pro are excluded, because their side chains have no rotatable bonds.

**It is much slower than rigid docking.** For the example's 33 conformers:

- rigid docking took about 40 s;
- flexible docking with 4 residues at exhaustiveness 8 took about 8½ min.

The automatic 5 Å selection picks 13 residues for 5RHD, which would be far slower again.

smina writes the flexed side chains for every pose. They come out in the same order as the poses, but without PDB atom names. The notebook maps them back onto the receptor and rebuilds a complete receptor for each pose, which is used:

- in the 3D viewer, where the flexed side chains are magenta and the original ones grey;
- by AEV-PLIG, which scores each pose against its own receptor;
- in the PDB and PyMOL exports.

> **Note:** The flexible-docking smina score appears to include the flexed side chains, so its values (around −8 to −11 kcal/mol for the example) are not comparable with rigid scores or with the PDB-ligand row. AEV-PLIG pK values are comparable between modes.

## AEV-PLIG

`aev_plig/` contains the AEV-PLIG prediction code (`process_and_predict.py` and its modules) from https://github.com/oxpig/AEV-PLIG, licensed under the BSD 3-Clause licence in `aev_plig/LICENSE.txt`. The trained `model_GATv2Net_ligsim90_fep_benchmark` ensemble (10 models) is downloaded separately (see Setup).

Changes for the current library versions:

- torchani 2.9 changed its `SpeciesConverter`, so the species index is now computed directly as atomic number − 1 (equivalent).
- `pkg_resources` is replaced by `importlib.metadata`.
- Model weights are loaded with `torch.load(..., weights_only=False)`, which torch ≥ 2.6 requires.
- A new `--output_dir` option controls where the predictions are written.
- Dataset paths are handled with `os.path.splitext`.
- RDKit's deprecated `GetExplicitValence()` is replaced.

**Regression check:** on AEV-PLIG's example dataset, all 241 predictions from the patched code in the new environment match the original code in a Python 3.8 / torch 2.4 environment to within 2×10⁻⁶.

### Hydrogens

**Before scoring, all hydrogens are added to each ligand pose**, because AEV-PLIG was trained on fully protonated ligands and one of its atom features counts attached hydrogens. smina writes polar hydrogens only. Scoring the same poses with every hydrogen removed changes the predicted pK by up to about 1 unit.

### Citation

Valsson *et al.*, "Narrowing the gap between machine learning scoring functions and free energy perturbation using augmented data", *Communications Chemistry* (2025), https://doi.org/10.1038/s42004-025-01428-y. The licence is in `aev_plig/LICENSE.txt`.

## Display options

- **Protein colouring** uses the RasMol secondary-structure scheme by default: helix magenta-red, sheet yellow, coil and turn white. Coil is drawn light grey so it shows on a white background.
  - Set `dt.PROTEIN_COLOURING` to `"rainbow"` or `"grey"` for other schemes.
  - Set `dt.RASMOL_COIL = "#FFFFFF"` for strict RasMol white coil.
- **Pose viewers** have toggles for the binding-site residues, the ligand from the PDB, and (after flexible docking) the original side chains.
- **Binding-site image** has a toggle to hide the side chains.

## Known limitations

- Running the notebook from the command line with `jupyter nbconvert --execute` stalls on the slider cells; this is an nbclient issue. All the steps still complete and their files are written. Use JupyterLab interactively.
- Turns are coloured as coil unless the PDB file has TURN records, because neither 3Dmol.js nor PyMOL assigns turns.
- AEV-PLIG accepts only the elements C, N, O, S, P, F, Cl, Br, I and B. Ligands with any other element are skipped, and the notebook reports how many were rejected.

## Licence

The notebook, `docking_tools.py` and documentation are released under the MIT licence (see `LICENSE`).

The bundled AEV-PLIG code in `aev_plig/` keeps its own BSD 3-Clause licence (see `aev_plig/LICENSE.txt`). The 5RHD structure comes from the Protein Data Bank (https://www.rcsb.org/structure/5RHD).
