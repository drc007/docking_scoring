"""Helper functions for DockingScoring.ipynb.

Conformer generation (RDKit), ligand extraction from a protein-ligand complex,
PyMOL images, an ipywidgets input panel for smina, pose browsers and
AEV-PLIG binding-affinity prediction.

Rule used throughout: every output file is written to the folder of the
input file it was derived from.
"""

import base64
import os
import re
import platform
import shlex
import shutil
import subprocess
import sys
from collections import OrderedDict
from pathlib import Path

import ipywidgets as widgets
import numpy as np
import pandas as pd
import py3Dmol
from IPython.display import HTML, SVG, display
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Draw, rdDetermineBonds, rdDistGeom
from rdkit.Chem.EnumerateStereoisomers import (EnumerateStereoisomers,
                                               StereoEnumerationOptions)

PROJECT_DIR = Path(__file__).resolve().parent
AEV_PLIG_DIR = PROJECT_DIR / "aev_plig"
AEV_PLIG_MODEL = "model_GATv2Net_ligsim90_fep_benchmark"
def find_smina():
    """Locate the smina executable: $SMINA, then smina / smina.static /
    smina.osx.12 in this conda environment or on the PATH, then a sibling
    conda environment called "smina", then common install locations."""
    names = ("smina", "smina.static", "smina.osx.12")
    candidates = [os.environ.get("SMINA")]
    candidates += [os.path.join(sys.prefix, "bin", n) for n in names]  # conda env (kernel PATH may lack it)
    # a separate conda env named "smina" (conda create -n smina -c conda-forge smina)
    candidates += [os.path.join(os.path.dirname(sys.prefix), "smina", "bin", "smina")]
    candidates += [shutil.which(n) for n in names]
    candidates += ["/usr/local/smina.osx.12", "/usr/local/bin/smina.osx.12", "/usr/local/bin/smina",
                   "/opt/homebrew/bin/smina"]
    for c in candidates:
        if c and os.path.isfile(c) and os.access(c, os.X_OK):
            return c
    return "smina"  # not found: the panel shows this and reports it when run


DEFAULT_SMINA = find_smina()

WATER_NAMES = {"HOH", "WAT", "H2O", "DOD", "TIP", "TIP3", "SOL"}
METAL_NAMES = {"ZN", "MG", "CA", "MN", "FE", "FE2", "CU", "CU1", "CO", "NI",
               "NA", "K", "CD", "HG"}


# --------------------------------------------------------------------------
# Environment information
# --------------------------------------------------------------------------
OLD_ENV_VERSIONS = {  # conda env "docking" used by ConformationGenerationDocking.ipynb
    "python": "3.8.19", "rdkit": "2024.3.5", "torch": "2.4.1",
    "torch_geometric": "2.6.1", "torchani": "2.2.4", "py3Dmol": "2.5.4",
    "pandas": "2.0.3", "numpy": "1.24.4", "scipy": "1.10.1",
    "sklearn": "1.3.2", "biopandas": "0.5.1", "qcelemental": "0.30.1",
    "ipywidgets": "8.1.5", "matplotlib": "3.7.3", "pymol": "-", "openbabel": "-",
}


def library_versions():
    """Table of library versions: old docking env vs this kernel."""
    import importlib
    new = {"python": platform.python_version()}
    for name in OLD_ENV_VERSIONS:
        if name == "python":
            continue
        try:
            if name == "pymol":
                from pymol import cmd
                new[name] = cmd.get_version()[0]
            elif name == "openbabel":
                from openbabel import openbabel as ob
                new[name] = ob.OBReleaseVersion()
            else:
                new[name] = importlib.import_module(name).__version__
        except Exception:
            new[name] = "not installed"
    return pd.DataFrame({"old env (docking)": OLD_ENV_VERSIONS,
                         "this env": new})


def cpu_info():
    """Return (logical cores, performance cores, efficiency cores)."""
    total = os.cpu_count() or 1
    perf = eff = None
    if sys.platform == "darwin":
        def _sysctl(key):
            try:
                return int(subprocess.check_output(["sysctl", "-n", key],
                                                   text=True).strip())
            except Exception:
                return None
        perf = _sysctl("hw.perflevel0.physicalcpu")
        eff = _sysctl("hw.perflevel1.physicalcpu")
    return total, perf, eff


def cpu_note():
    total, perf, eff = cpu_info()
    note = f"{total} cores available"
    if perf:
        note += f" ({perf} performance + {eff or 0} efficiency)"
    return note


# --------------------------------------------------------------------------
# Protein colouring
# --------------------------------------------------------------------------
# RasMol "colour structure": alpha helix magenta-red, beta sheet yellow,
# turn pale blue, everything else (coil) white. Coil is drawn light grey by
# default because white is invisible on a white page; set RASMOL_COIL to
# "#FFFFFF" for strict RasMol colours. Neither 3Dmol.js nor PyMOL assigns
# turns, so turns are shown as coil unless the PDB file has TURN records.
RASMOL_SS = {"helix": "#FF0080", "sheet": "#FFC800", "turn": "#6080FF", "coil": "#FFFFFF"}
RASMOL_COIL = "#C8C8C8"
PROTEIN_COLOURING = "rasmol"  # "rasmol" (secondary structure), "rainbow" (N->C) or "grey"


def _hex_rgb(h):
    h = h.lstrip("#")
    return [int(h[i:i + 2], 16) for i in (0, 2, 4)]


def cartoon_style(colouring=None, opacity=0.85):
    """py3Dmol cartoon style for the chosen protein colouring."""
    colouring = colouring or PROTEIN_COLOURING
    if colouring == "rasmol":
        colour = {"colorscheme": {"prop": "ss", "map": {"h": RASMOL_SS["helix"], "s": RASMOL_SS["sheet"],
                                                        "c": RASMOL_COIL, "": RASMOL_COIL}}}
    elif colouring == "rainbow":
        colour = {"color": "spectrum"}
    else:
        colour = {"color": "lightgrey"}
    return {"cartoon": {**colour, "opacity": opacity}}


def colour_protein_pymol(cmd, sel, colouring=None):
    """Colour the cartoon of a PyMOL selection by RasMol secondary structure,
    rainbow or grey. Uses the cartoon_color setting, so atom colours (sticks)
    are unaffected."""
    colouring = colouring or PROTEIN_COLOURING
    if colouring == "rasmol":
        cmd.set_color("ras_helix", _hex_rgb(RASMOL_SS["helix"]))
        cmd.set_color("ras_sheet", _hex_rgb(RASMOL_SS["sheet"]))
        cmd.set_color("ras_coil", _hex_rgb(RASMOL_COIL))
        cmd.set("cartoon_color", "ras_coil", sel)
        cmd.set("cartoon_color", "ras_helix", f"({sel}) and ss H")
        cmd.set("cartoon_color", "ras_sheet", f"({sel}) and ss S")
    elif colouring == "rainbow":
        cmd.spectrum("count", "rainbow", f"({sel}) and name CA")
    else:
        cmd.set("cartoon_color", "grey80", sel)


def colour_legend_html(colouring=None):
    colouring = colouring or PROTEIN_COLOURING
    if colouring != "rasmol":
        return ""
    sw = lambda c, t: (f"<span style='display:inline-block;width:12px;height:12px;background:{c};"
                       f"border:1px solid #888;margin:0 3px 0 10px;vertical-align:middle'></span>{t}")
    return ("<span style='font-size:90%'>Protein (RasMol secondary structure):"
            + sw(RASMOL_SS["helix"], "helix") + sw(RASMOL_SS["sheet"], "sheet") + sw(RASMOL_COIL, "coil/turn")
            + "</span>")


# --------------------------------------------------------------------------
# Input ligands
# --------------------------------------------------------------------------
def load_sdf(path, warn=True):
    """Read an SDF keeping hydrogens; report unreadable or unnamed records.
    Duplicate names are checked separately (duplicate_names / LigandNamePanel)."""
    mols = []
    for i, m in enumerate(Chem.SDMolSupplier(str(path), removeHs=False)):
        if m is None:
            if warn:
                print(f"Warning: failed to read molecule {i} in {path}")
            continue
        if not m.GetProp("_Name").strip():
            if warn:
                print(f"Warning: no name for molecule {i} in {path}; using mol_{i}")
            m.SetProp("_Name", f"mol_{i}")
        mols.append(m)
    return mols


# --------------------------------------------------------------------------
# Ligand names / IDs
# --------------------------------------------------------------------------
# smina keeps only the molecule name, and results are grouped per ligand by
# name, so different structures must not share a name.
ID_FIELD_HINTS = ("id", "idnumber", "id_number", "record_id", "recordid", "compound_id", "cmpd_id",
                  "compoundid", "mol_id", "molid", "molecule_id", "ligand_id", "reg_no", "regno",
                  "registry", "registry_number", "corp_id", "chembl_id", "pubchem_cid", "cid",
                  "zinc_id", "catalog_id", "catalogue_id", "supplier_id", "code", "name", "title")


def _canonical(m):
    return Chem.MolToSmiles(Chem.RemoveHs(m, sanitize=False))


def duplicate_names(mols):
    """{name: [indices]} for names shared by *different* structures
    (records with the same name and structure, e.g. conformers, are fine)."""
    by_name = OrderedDict()
    for i, m in enumerate(mols):
        by_name.setdefault(m.GetProp("_Name"), []).append(i)
    return OrderedDict((n, idx) for n, idx in by_name.items()
                       if len(idx) > 1 and len({_canonical(mols[i]) for i in idx}) > 1)


def candidate_id_fields(mols):
    """SD fields present (non-empty) on every molecule with all values unique,
    identifier-like names (ID, IDNUMBER, Record_ID, ...) first."""
    if not mols:
        return []
    common = set(mols[0].GetPropNames())
    for m in mols[1:]:
        common &= set(m.GetPropNames())
    fields = []
    for f in common:
        vals = [m.GetProp(f).strip() for m in mols]
        if all(vals) and len(set(vals)) == len(vals):
            fields.append(f)
    norm = lambda f: re.sub(r"[\s\-]+", "_", f.strip().lower())
    def rank(f):
        n = norm(f)
        if n in ID_FIELD_HINTS:
            return (0, ID_FIELD_HINTS.index(n))
        if n.endswith(("id", "_id", "number", "no")) or "id" in n.split("_"):
            return (1, 0)
        return (2, 0)
    return sorted(fields, key=lambda f: (rank(f), f.lower()))


def apply_names_from_field(mols, field):
    """Use an SD field as the ligand name; the original name is kept in
    the 'original_name' property."""
    for m in mols:
        m.SetProp("original_name", m.GetProp("_Name"))
        m.SetProp("_Name", m.GetProp(field).strip())
    return mols


def assign_unique_names(mols):
    """Different structures sharing a name become <name>_2, <name>_3, ...
    (the first keeps its name); the original name is kept in 'original_name'."""
    seen, taken = {}, {m.GetProp("_Name") for m in mols}
    for m in mols:
        name, smi = m.GetProp("_Name"), _canonical(m)
        variants = seen.setdefault(name, {})
        if smi not in variants:
            if not variants:
                variants[smi] = name
            else:
                k = len(variants) + 1
                while f"{name}_{k}" in taken:
                    k += 1
                variants[smi] = f"{name}_{k}"
                taken.add(variants[smi])
        m.SetProp("original_name", name)
        m.SetProp("_Name", variants[smi])
    return mols


class LigandNamePanel:
    """Check the input ligands for duplicate names; if any, offer a unique
    identifier field from the SDF as the ligand ID, or automatic unique names."""

    AUTO = "Assign unique names automatically (name, name_2, name_3, ...)"

    def __init__(self, mols):
        self.mols = mols
        self.out = widgets.Output()
        self.dups = duplicate_names(mols)
        self.fields = candidate_id_fields(mols) if self.dups else []
        options = [(f"Use SD field '{f}' as the ligand ID  (e.g. {', '.join(m.GetProp(f).strip() for m in mols[:3])})", f)
                   for f in self.fields] + [(self.AUTO, self.AUTO)]
        self.choice = widgets.RadioButtons(options=options, value=options[0][1], layout=widgets.Layout(width="95%"))
        self.button = widgets.Button(description="Apply", button_style="success", icon="check")
        self.button.on_click(lambda _: self.apply())
        self.resolved = not self.dups

    def report_html(self):
        if not self.dups:
            return (f"<span style='color:#1a7f37'>&#10004; All {len(self.mols)} ligand names are unique.</span>")
        rows = "".join(f"<li><b>{n}</b>: {len(idx)} different structures (records {', '.join(str(i + 1) for i in idx)})</li>"
                       for n, idx in self.dups.items())
        fields = (f"Unique identifier field(s) found in the SDF: <b>{', '.join(self.fields)}</b>."
                  if self.fields else "No SD field with a unique value for every molecule was found.")
        return ("<div style='background:#fff3cd;color:#664d03;border-left:5px solid #e0a800;padding:6px 10px'>"
                "<b>&#9888; Duplicate ligand names.</b> smina keeps only the name, and results are grouped by "
                f"name, so these would be merged:<ul style='margin:4px 0'>{rows}</ul>{fields} "
                "Choose how to name the ligands and press <b>Apply</b>.</div>")

    def widget(self):
        items = [widgets.HTML(self.report_html())]
        if self.dups:
            items += [self.choice, self.button]
        return widgets.VBox(items + [self.out])

    def apply(self, choice=None):
        choice = choice or self.choice.value
        self.out.clear_output()
        with self.out:
            if choice == self.AUTO:
                assign_unique_names(self.mols)
                how = "automatic unique names"
            else:
                apply_names_from_field(self.mols, choice)
                how = f"SD field '{choice}'"
            self.dups = duplicate_names(self.mols)
            self.resolved = not self.dups
            names = [m.GetProp("_Name") for m in self.mols]
            print(f"Ligand names now from {how} (original names kept in 'original_name'):")
            print("  " + ", ".join(names))
        return self


def to_2d(mol):
    """Hydrogen-free copy with fresh 2D coordinates, for depiction."""
    m = Chem.Mol(Chem.RemoveHs(mol, sanitize=False))
    try:
        m.UpdatePropertyCache(strict=False)
    except Exception:
        pass
    AllChem.Compute2DCoords(m)
    return m


def mol_svg(mol, size=(350, 260), legend=""):
    m = Chem.RemoveHs(mol, sanitize=False)
    try:
        m.UpdatePropertyCache(strict=False)
    except Exception:
        pass
    m = Chem.Mol(m)
    AllChem.Compute2DCoords(m)
    d = Draw.rdMolDraw2D.MolDraw2DSVG(*size)
    d.drawOptions().addStereoAnnotation = True
    d.DrawMolecule(m, legend=legend)
    d.FinishDrawing()
    return d.GetDrawingText()


def _nav_slider(n, description):
    slider = widgets.IntSlider(value=0, min=0, max=max(n - 1, 0), step=1,
                               description=description, continuous_update=False,
                               layout=widgets.Layout(width="420px"))
    prev_b = widgets.Button(description="◀ Prev", layout=widgets.Layout(width="80px"))
    next_b = widgets.Button(description="Next ▶", layout=widgets.Layout(width="80px"))
    prev_b.on_click(lambda _: setattr(slider, "value", max(slider.min, slider.value - 1)))
    next_b.on_click(lambda _: setattr(slider, "value", min(slider.max, slider.value + 1)))
    return slider, widgets.HBox([prev_b, slider, next_b])


def molecule_browser(mols):
    """Slider to step through input molecules: 2D depiction + 3D view."""
    if not mols:
        print("No molecules to show")
        return
    slider, nav = _nav_slider(len(mols), "Molecule")
    out = widgets.Output()

    def show(i):
        out.clear_output()
        m = mols[i]
        with out:
            print(f"{i + 1}/{len(mols)}  {m.GetProp('_Name')}   "
                  f"heavy atoms: {m.GetNumHeavyAtoms()}   "
                  f"SMILES: {Chem.MolToSmiles(Chem.RemoveHs(m, sanitize=False))}")
            display(SVG(mol_svg(m)))
            if m.GetNumConformers() and m.GetConformer().Is3D():
                v = py3Dmol.view(width=450, height=300)
                v.addModel(Chem.MolToMolBlock(m), "sdf")
                v.setStyle({"stick": {}})
                v.zoomTo()
                v.show()

    slider.observe(lambda c: show(c["new"]), names="value")
    display(widgets.VBox([nav, out]))
    show(0)


# --------------------------------------------------------------------------
# Conformer generation
# --------------------------------------------------------------------------
EMBED_METHODS = OrderedDict([
    ("srETKDGv3 (recommended)", rdDistGeom.srETKDGv3),
    ("ETKDGv3", rdDistGeom.ETKDGv3),
    ("ETKDG (old default)", rdDistGeom.ETKDG),
    ("ETDG (no torsion knowledge)", rdDistGeom.ETDG),
])


def _heavy_rms_matrix(mol, conf_ids):
    """Symmetry-aware heavy-atom RMSD between conformers (after alignment)."""
    heavy = Chem.RemoveHs(mol)
    n = len(conf_ids)
    rms = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            r = AllChem.GetBestRMS(heavy, heavy, conf_ids[i], conf_ids[j],
                                   maxMatches=1000)
            rms[i, j] = rms[j, i] = r
    return rms


def generate_conformers(mols, out_path, method="srETKDGv3 (recommended)",
                        num_confs=10, prune_rms=0.5, optimize=True,
                        energy_window=10.0, enumerate_stereo=False,
                        seed=42, max_out=None):
    """Generate a diverse, low-energy conformer ensemble for each molecule.

    1. optional enumeration of unassigned stereocentres
    2. embedding with the chosen ETKDG variant (RMS pruning during embedding)
    3. MMFF94 optimisation (UFF fallback)
    4. discard conformers more than `energy_window` kcal/mol above the minimum
    5. remove post-optimisation duplicates (heavy-atom RMSD < prune_rms)
    6. keep at most `max_out` conformers per molecule (lowest energy first)
    """
    rows = []
    with Chem.SDWriter(str(out_path)) as w:
        for mol in mols:
            name = mol.GetProp("_Name")
            base = Chem.Mol(mol)
            if base.GetNumConformers() and base.GetConformer().Is3D():
                Chem.AssignStereochemistryFrom3D(base)
            variants = [base]
            if enumerate_stereo:
                opts = StereoEnumerationOptions(onlyUnassigned=True, unique=True,
                                                maxIsomers=16)
                variants = list(EnumerateStereoisomers(Chem.RemoveHs(base), options=opts)) or [base]
            for vi, var in enumerate(variants):
                vname = name if len(variants) == 1 else f"{name}_iso{vi + 1}"
                m = Chem.AddHs(Chem.RemoveHs(var), addCoords=False)
                params = EMBED_METHODS[method]()
                params.randomSeed = seed
                params.pruneRmsThresh = prune_rms
                params.numThreads = 0
                cids = list(rdDistGeom.EmbedMultipleConfs(m, num_confs, params))
                if not cids:
                    params.useRandomCoords = True
                    cids = list(rdDistGeom.EmbedMultipleConfs(m, num_confs, params))
                if not cids:
                    print(f"Warning: embedding failed for {vname}")
                    rows.append({"name": vname, "embedded": 0, "kept": 0})
                    continue
                n_embedded = len(cids)
                energies = {c: np.nan for c in cids}
                ff = ""
                if optimize:
                    if AllChem.MMFFHasAllMoleculeParams(m):
                        res = AllChem.MMFFOptimizeMoleculeConfs(m, numThreads=0, maxIters=2000)
                        ff = "MMFF94"
                    else:
                        res = AllChem.UFFOptimizeMoleculeConfs(m, numThreads=0, maxIters=2000)
                        ff = "UFF"
                    energies = {c.GetId(): e for c, (_, e) in zip(m.GetConformers(), res)}
                    emin = min(energies.values())
                    cids = [c for c in cids if energies[c] - emin <= energy_window]
                cids.sort(key=lambda c: energies[c] if optimize else 0)
                # post-optimisation duplicate removal
                if optimize and prune_rms > 0 and len(cids) > 1:
                    rms = _heavy_rms_matrix(m, cids)
                    keep = []
                    for i in range(len(cids)):
                        if all(rms[i, j] >= prune_rms for j in keep):
                            keep.append(i)
                    cids = [cids[i] for i in keep]
                if max_out:
                    cids = cids[:max_out]
                emin = min(energies[c] for c in cids) if optimize else 0
                for k, c in enumerate(cids):
                    m.SetProp("_Name", vname)
                    m.SetProp("conformer", str(k + 1))
                    if optimize:
                        m.SetProp(f"{ff}_energy", f"{energies[c]:.3f}")
                        m.SetProp("rel_energy", f"{energies[c] - emin:.3f}")
                    w.write(m, confId=c)
                rows.append({"name": vname, "embedded": n_embedded, "kept": len(cids),
                             "force field": ff or "none"})
    df = pd.DataFrame(rows)
    print(f"Wrote {int(df['kept'].sum())} conformers to {out_path}")
    return df


class ConformerPanel:
    """Widgets for generate_conformers(); output is <input stem>_confs.sdf
    in the folder of the input SDF."""

    def __init__(self, mols, input_sdf):
        self.mols = mols
        self.input_sdf = Path(input_sdf)
        self.out_path = self.input_sdf.with_name(self.input_sdf.stem + "_confs.sdf")
        style = {"description_width": "190px"}
        self.method = widgets.Dropdown(options=list(EMBED_METHODS), description="Embedding method", style=style,
                                    layout=widgets.Layout(width="480px"))
        self.num_confs = widgets.BoundedIntText(value=20, min=1, max=1000, description="Conformers to embed", style=style)
        self.prune = widgets.BoundedFloatText(value=0.5, min=0, max=5, step=0.1, description="RMS prune threshold (Å)", style=style)
        self.optimize = widgets.Checkbox(value=True, description="MMFF94 optimise (UFF fallback)",
                                         layout=widgets.Layout(width="95%"))
        self.window = widgets.BoundedFloatText(value=10.0, min=0, max=100, description="Energy window (kcal/mol)", style=style)
        self.max_out = widgets.BoundedIntText(value=0, min=0, max=1000, description="Max kept per molecule (0 = all)", style=style)
        self.stereo = widgets.Checkbox(value=False, description="Enumerate unassigned stereocentres",
                                       layout=widgets.Layout(width="95%"))
        self.seed = widgets.IntText(value=42, description="Random seed", style=style)
        self.button = widgets.Button(description="Generate conformers", button_style="success", icon="play")
        self.out = widgets.Output()
        self.summary = None
        self.button.on_click(lambda _: self.run())

    def widget(self):
        return widgets.VBox([self.method, self.num_confs, self.prune, self.optimize, self.window,
                             self.max_out, self.stereo, self.seed,
                             widgets.HTML(f"Output: <code>{self.out_path}</code>"), self.button, self.out])

    def run(self):
        self.out.clear_output()
        with self.out:
            dups = duplicate_names(self.mols)
            if dups:
                print("Duplicate ligand names for different structures: " + ", ".join(dups) +
                      "\nResolve them in the ligand-name check above (Apply), then generate conformers.")
                return None
            self.summary = generate_conformers(
                self.mols, self.out_path, method=self.method.value, num_confs=self.num_confs.value,
                prune_rms=self.prune.value, optimize=self.optimize.value, energy_window=self.window.value,
                enumerate_stereo=self.stereo.value, seed=self.seed.value, max_out=self.max_out.value or None)
            display(self.summary)
        return self.summary


# --------------------------------------------------------------------------
# Ionisation states with QupKake (micro-pKa prediction, GFN2-xTB + GNN)
# --------------------------------------------------------------------------
# QupKake (Abarbanel & Hutchison, JCTC 2024) runs in its own conda env
# ("qupkake", Python 3.9, xtb 6.4.1 built from source). It predicts micro-pKa
# values for the sites of the NEUTRAL molecule: "basic" = pKa of the
# conjugate acid of a site, "acidic" = deprotonation of a site.
QUPKAKE_ENV = "qupkake"

# Basic sites QupKake sometimes reports that are not basic in reality
# (seen for this ligand series): sulphonamide N, amide/imide N, and amine N
# bonded to an aromatic ring (aryl amines, pKa ~3-5).
IMPLAUSIBLE_BASIC_SMARTS = {
    "sulphonamide N": "[NX3][SX4](=O)=O",
    "amide/imide N": "[NX3][CX3]=[OX1,SX1]",
    "aryl-amine N": "[NX3;!a;!$(N=*)][a]",
}


def _conda_base():
    exe = os.environ.get("CONDA_EXE")
    if exe:
        return Path(exe).resolve().parent.parent
    for c in (Path.home() / "miniconda3", Path.home() / "anaconda3", Path.home() / "miniforge3",
              Path("/opt/miniconda3"), Path("/opt/homebrew/Caskroom/miniforge/base")):
        if (c / "etc" / "profile.d" / "conda.sh").exists():
            return c
    return None


def qupkake_available(env=QUPKAKE_ENV):
    base = _conda_base()
    return bool(base and ((base / "envs" / env / "bin" / "qupkake").exists()))


def _neutral_heavy(mol):
    """Heavy-atom copy (same atom order and coordinates) with charges
    neutralised where possible (RDKit Uncharger)."""
    from rdkit.Chem.MolStandardize import rdMolStandardize
    heavy = Chem.RemoveHs(mol)
    return rdMolStandardize.Uncharger().uncharge(heavy)


def run_qupkake(mols, work_dir, n_proc=None, env=QUPKAKE_ENV, timeout=3600):
    """Predict micro-pKa sites for molecules (neutralised first) with QupKake.

    Returns a DataFrame: name, smiles (neutral, as submitted), idx (atom index
    in that SMILES), site_atom, pka_type ('acidic'/'basic'), pka.
    Molecules with no ionisable site have no rows; failures are listed in
    the 'failed' attribute of the returned frame."""
    work_dir = Path(work_dir).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    base = _conda_base()
    if not qupkake_available(env):
        raise RuntimeError(f"QupKake not found (conda env '{env}' with the qupkake command)")
    rows = [{"name": m.GetProp("_Name"), "smiles": Chem.MolToSmiles(_neutral_heavy(m))} for m in mols]
    pd.DataFrame(rows).to_csv(work_dir / "qupkake_input.csv", index=False)
    n_proc = n_proc or max(1, min(len(rows), (os.cpu_count() or 2) - 2))
    root = work_dir / "qupkake_root"
    shutil.rmtree(root, ignore_errors=True)
    script = f"""set -e
source "{base}/etc/profile.d/conda.sh"
conda activate {env}
cd "{work_dir}"
REAL_XTB="${{XTBPATH:-$(command -v xtb)}}"
# QupKake passes -P <n> to every xtb call; force one thread per xtb process
cat > xtb1 <<EOF
#!/bin/bash
args=(); skip=0
for a in "\\$@"; do
  if [ \\$skip = 1 ]; then args+=("1"); skip=0; continue; fi
  [ "\\$a" = "-P" ] && skip=1
  args+=("\\$a")
done
exec "$REAL_XTB" "\\${{args[@]}}"
EOF
chmod +x xtb1
export XTBPATH="$PWD/xtb1"
qupkake file qupkake_input.csv -s smiles -n name -mp {n_proc} -r qupkake_root -o qupkake_output.sdf
"""
    (work_dir / "run_qupkake.sh").write_text(script)
    with open(work_dir / "qupkake.log", "w") as log:
        rc = subprocess.run(["bash", str(work_dir / "run_qupkake.sh")], stdout=log,
                            stderr=subprocess.STDOUT, timeout=timeout).returncode
    out_sdf = root / "output" / "qupkake_output.sdf"
    if rc != 0 or not out_sdf.exists():
        raise RuntimeError(f"QupKake failed (exit {rc}); see {work_dir / 'qupkake.log'}")
    smiles = {r["name"]: r["smiles"] for r in rows}
    sites = []
    for m in Chem.SDMolSupplier(str(out_sdf), removeHs=False):
        if m is None:
            continue
        name = m.GetProp("_Name")
        idx = int(m.GetProp("idx"))
        pka = float(str(m.GetProp("pka")).replace("tensor(", "").rstrip(")"))
        sites.append({"name": name, "smiles": smiles.get(name, ""), "idx": idx,
                      "site_atom": m.GetAtomWithIdx(idx).GetSymbol(),
                      "pka_type": m.GetProp("pka_type"), "pka": pka})
    df = pd.DataFrame(sites, columns=["name", "smiles", "idx", "site_atom", "pka_type", "pka"])
    errors = (root / "logs" / "error_log.txt")
    failed = [r["name"] for r in rows if errors.exists() and r["name"] in errors.read_text()]
    df.attrs["failed"] = failed
    df.to_csv(work_dir / "qupkake_sites.csv", index=False)
    return df


def _implausible_basic(neutral, idx):
    for label, smarts in IMPLAUSIBLE_BASIC_SMARTS.items():
        patt = Chem.MolFromSmarts(smarts)
        if any(match[0] == idx for match in neutral.GetSubstructMatches(patt)):
            return label
    return None


def ionise_from_sites(mol, sites, ph=7.4, filter_implausible=True):
    """Apply QupKake predictions to a molecule, keeping atom order and 3D
    coordinates (hydrogens rebuilt). Returns (ionised_mol, record dict).

    Rule: on the neutral molecule, protonate the most basic plausible site if
    its pKa > pH and deprotonate the most acidic site if its pKa < pH (at most
    one of each); other sites that also cross the pH are reported."""
    name = mol.GetProp("_Name")
    neutral = _neutral_heavy(mol)
    rec = {"name": name, "input_charge": Chem.GetFormalCharge(mol), "sites": "", "ignored": "",
           "protonated": "", "deprotonated": "", "also_crossing_pH": "", "predicted_charge": None}
    s = sites[sites.name == name]
    if len(s):
        qk = Chem.MolFromSmiles(s.smiles.iloc[0])
        match = neutral.GetSubstructMatch(qk)   # match[k] = our atom index of QupKake atom k
        if not match:
            raise ValueError(f"{name}: QupKake structure does not match the input")
        s = s.assign(atom=[match[i] for i in s.idx])
    else:
        s = s.assign(atom=[])
    lab = lambda r: f"{'b' if r.pka_type == 'basic' else 'a'}:{neutral.GetAtomWithIdx(int(r.atom)).GetSymbol()}{int(r.atom) + 1}={r.pka:.2f}"
    rec["sites"] = "; ".join(lab(r) for r in s.sort_values("pka").itertuples())
    basic = s[s.pka_type == "basic"].copy()
    if filter_implausible and len(basic):
        why = [_implausible_basic(neutral, int(a)) for a in basic.atom]
        rec["ignored"] = "; ".join(f"{lab(r)} ({w})" for r, w in zip(basic.itertuples(), why) if w)
        basic = basic[[w is None for w in why]]
    acidic = s[s.pka_type == "acidic"]
    rw = Chem.RWMol(neutral)
    crossing = []
    if len(basic) and basic.pka.max() > ph:
        b = basic.loc[basic.pka.idxmax()]
        a = rw.GetAtomWithIdx(int(b.atom))
        a.SetNumExplicitHs(a.GetTotalNumHs() + 1); a.SetNoImplicit(True)
        a.SetFormalCharge(a.GetFormalCharge() + 1)
        rec["protonated"] = lab(b)
        crossing += [lab(r) for r in basic.itertuples() if r.pka > ph and r.Index != b.name]
    if len(acidic) and acidic.pka.min() < ph:
        c = acidic.loc[acidic.pka.idxmin()]
        a = rw.GetAtomWithIdx(int(c.atom))
        if a.GetTotalNumHs() > 0:
            a.SetNumExplicitHs(a.GetTotalNumHs() - 1); a.SetNoImplicit(True)
            a.SetFormalCharge(a.GetFormalCharge() - 1)
            rec["deprotonated"] = lab(c)
        crossing += [lab(r) for r in acidic.itertuples() if r.pka < ph and r.Index != c.name]
    rec["also_crossing_pH"] = "; ".join(crossing)
    new = rw.GetMol()
    Chem.SanitizeMol(new)
    new = Chem.AddHs(new, addCoords=new.GetNumConformers() > 0)
    for p in mol.GetPropNames():
        new.SetProp(p, mol.GetProp(p))
    new.SetProp("_Name", name)
    rec["predicted_charge"] = Chem.GetFormalCharge(new)
    new.SetProp("qupkake_pH", f"{ph:g}")
    new.SetProp("qupkake_sites", rec["sites"])
    new.SetProp("qupkake_charge", str(rec["predicted_charge"]))
    rec["smiles"] = Chem.MolToSmiles(Chem.RemoveHs(new))
    return new, rec


def predict_ionisation(mols, work_dir, ph=7.4, filter_implausible=True, n_proc=None):
    """Run QupKake on molecules and return (ionised mols, table)."""
    sites = run_qupkake(mols, work_dir, n_proc=n_proc)
    out, recs = [], []
    for m in mols:
        if m.GetProp("_Name") in sites.attrs.get("failed", []):
            rec = {"name": m.GetProp("_Name"), "input_charge": Chem.GetFormalCharge(m),
                   "predicted_charge": Chem.GetFormalCharge(m), "note": "QupKake failed - input state kept",
                   "smiles": Chem.MolToSmiles(Chem.RemoveHs(m))}
            out.append(Chem.Mol(m)); recs.append(rec)
            continue
        new, rec = ionise_from_sites(m, sites, ph, filter_implausible)
        out.append(new); recs.append(rec)
    table = pd.DataFrame(recs)
    table["input_smiles"] = [Chem.MolToSmiles(Chem.RemoveHs(m)) for m in mols]
    table.insert(1, "changed", table.input_smiles != table.smiles)  # same charge can still mean a different site
    return out, table


class ProtonationPanel:
    """Predict ionisation states with QupKake and write them as a new SDF
    (<input>_pH<pH>.sdf) that the rest of the notebook docks."""

    def __init__(self, mols, input_sdf):
        self.mols, self.input_sdf = mols, Path(input_sdf)
        style = {"description_width": "160px"}
        self.ph = widgets.BoundedFloatText(value=7.4, min=0, max=14, step=0.1, description="pH", style=style)
        self.filter = widgets.Checkbox(value=True, layout=widgets.Layout(width="95%"),
                                       description="Ignore implausible basic sites (sulphonamide, amide and aryl-amine N)")
        self.use = widgets.Checkbox(value=True, layout=widgets.Layout(width="95%"),
                                    description="Dock the ligands in their predicted ionisation states")
        self.button = widgets.Button(description="Run QupKake", button_style="success", icon="play")
        self.out = widgets.Output()
        self.table = None
        self.ionised = None
        self.button.on_click(lambda _: self.run())
        ok = qupkake_available()
        self.status = widgets.HTML("" if ok else
                                   "<span style='color:#b00'>QupKake not found (conda env 'qupkake'); "
                                   "the input ionisation states will be used.</span>")
        self.button.disabled = not ok

    @property
    def output_sdf(self):
        return self.input_sdf.with_name(f"{self.input_sdf.stem}_pH{self.ph.value:g}.sdf")

    def widget(self):
        return widgets.VBox([self.status, self.ph, self.filter, self.use, self.button, self.out])

    def run(self):
        self.out.clear_output()
        with self.out:
            print("Running QupKake (GFN2-xTB + GNN); about 30 s for 10 molecules ...")
            work = self.input_sdf.parent / "qupkake" / self.input_sdf.stem
            self.ionised, self.table = predict_ionisation(self.mols, work, self.ph.value, self.filter.value)
            with Chem.SDWriter(str(self.output_sdf)) as w:
                for m in self.ionised:
                    w.write(m)
            self.table.to_csv(self.output_sdf.with_suffix(".csv"), index=False)
            print(f"Ionisation states written to {self.output_sdf} (table: {self.output_sdf.with_suffix('.csv').name}; "
                  f"QupKake files in {work})")
            display(self.table[["name", "changed", "input_charge", "predicted_charge", "protonated", "deprotonated",
                                "ignored", "also_crossing_pH", "input_smiles", "smiles"]])
        return self.table

    def docking_input(self):
        """(molecules, SDF path) to dock: predicted states if run and selected."""
        if self.use.value and self.ionised is not None:
            return self.ionised, self.output_sdf
        return self.mols, self.input_sdf


def ionise_reference_ligand(mol, work_dir, ph=7.4, filter_implausible=True):
    """Predicted ionisation state of the PDB ligand, keeping its crystal
    coordinates. Returns (mol, table)."""
    mols, table = predict_ionisation([mol], work_dir, ph, filter_implausible)
    return mols[0], table


# --------------------------------------------------------------------------
# Protein-ligand complex handling
# --------------------------------------------------------------------------
def _res_key(line):
    return (line[17:20].strip(), line[21], line[22:26].strip(), line[26].strip())


# --------------------------------------------------------------------------
# Structure files: PDB or mmCIF (gemmi), HET-group classification
# --------------------------------------------------------------------------
SOLVENT_ADDITIVES = {
    "DMS", "DMF", "DMX", "EDO", "GOL", "PEG", "PGE", "PG4", "1PE", "P6G", "PE4", "PE8", "2PE", "MPD",
    "MRD", "IPA", "EOH", "MOH", "BU3", "ACT", "ACY", "ACE", "FMT", "SO4", "PO4", "NO3", "SCN", "AZI",
    "CIT", "FLC", "TRS", "MES", "EPE", "BME", "DTT", "IMD", "TAR", "MLI", "SIN", "BTB", "B3P", "NHE",
    "CAC", "BCT", "CO3", "PG0", "OXY", "PEO", "HEZ", "DOD", "UNX"}
ION_NAMES = METAL_NAMES | {"CL", "BR", "IOD", "F", "LI", "RB", "CS", "SR", "BA", "AL", "CU1", "NI2",
                           "CO3", "HO", "YB", "GD", "SM", "PT", "AU", "AG"}
CAP_NAMES = {"NME", "NH2", "ACE", "FOR"}  # terminal caps on the polymer, not ligands
GENERIC_LIGAND_CODES = {"LIG", "UNL", "UNK", "UNX", "DRG", "INH", "MOL", "XXX", "LG1", "LG2", "INI"}
_STRUCTURE_CACHE = {}


def _structure_kind(path):
    """'pdb' or 'cif' (also for .gz); anything else is treated as PDB."""
    name = Path(path).name.lower()
    name = name[:-3] if name.endswith(".gz") else name
    return "cif" if name.endswith((".cif", ".mmcif")) else "pdb"


def structure_stem(path):
    """Stem used for files derived from a structure; mmCIF input gets a
    '_cif' suffix so its outputs never overwrite those from a PDB file of the
    same entry (the two may use different coordinate frames)."""
    name = Path(path).name
    for ext in (".gz", ".cif", ".mmcif", ".pdb", ".ent"):
        if name.lower().endswith(ext):
            name = name[: -len(ext)]
    return name + ("_cif" if _structure_kind(path) == "cif" else "")


def read_structure_text(path):
    """(PDB-format text, {residue name: chemical name}) for a PDB or mmCIF
    file. mmCIF (optionally gzipped) is converted with gemmi, keeping
    HELIX/SHEET records, formal charges and alternate locations."""
    path = Path(path)
    key = (str(path.resolve()), path.stat().st_mtime)
    if key in _STRUCTURE_CACHE:
        return _STRUCTURE_CACHE[key]
    names = {}
    if _structure_kind(path) == "cif" or path.suffix.lower() == ".gz":
        import gemmi
        st = gemmi.read_structure(str(path))
        st.setup_entities()
        if len(st) > 1:
            del st[1:]  # first model only
        text = st.make_pdb_string()
        if _structure_kind(path) == "cif":
            block = gemmi.cif.read(str(path)).sole_block()
            for row in block.find("_chem_comp.", ["id", "name"]):
                names[row[0]] = gemmi.cif.as_string(row[1]).strip()
    else:
        text = path.read_text()
    hetnam = {}
    for line in text.splitlines():  # HETNAM records of PDB files (with continuation lines)
        if line.startswith("HETNAM"):
            code, part = line[11:14].strip(), line[15:70].strip()
            prev = hetnam.get(code, "")
            hetnam[code] = (prev + ("" if not prev or prev.endswith("-") else " ") + part) if line[8:10].strip() else part
    for code, name in hetnam.items():
        names.setdefault(code, name)
    _STRUCTURE_CACHE[key] = (text, names)
    return text, names


def het_category(resname, n_atoms):
    if resname in WATER_NAMES:
        return "water"
    if resname in CAP_NAMES:
        return "terminal cap"
    if resname in ION_NAMES or n_atoms == 1:
        return "ion"
    if resname in SOLVENT_ADDITIVES:
        return "solvent/additive"
    return "ligand"


def list_het_groups(path, include_all=False):
    """HET groups in a PDB or mmCIF file, ligands first (largest first).

    Waters are never listed. Solvents, crystallisation additives, ions and
    terminal caps are listed only with include_all=True. Alternate locations
    count once. Columns: resname, chain, resnum, icode, atoms, name, category."""
    text, names = read_structure_text(path)
    groups = OrderedDict()
    for line in text.splitlines():
        if line.startswith("HETATM") and line[16] in (" ", "A"):
            key = _res_key(line)
            if key[0] in WATER_NAMES:
                continue
            groups[key] = groups.get(key, 0) + 1
    rows = []
    for k, n in groups.items():
        cat = het_category(k[0], n)
        rows.append({"resname": k[0], "chain": k[1], "resnum": k[2], "icode": k[3], "atoms": n,
                     "name": names.get(k[0], ""), "category": cat})
    df = pd.DataFrame(rows, columns=["resname", "chain", "resnum", "icode", "atoms", "name", "category"])
    if not include_all:
        df = df[df.category == "ligand"]
    if not df.empty:
        df = df.assign(_lig=(df.category != "ligand")).sort_values(
            ["_lig", "atoms"], ascending=[True, False], kind="stable").drop(columns="_lig").reset_index(drop=True)
    return df


def het_label(row):
    name = str(row.get("name", "") or "")
    name = (name[:45] + "…") if len(name) > 46 else name
    label = f"{row['resname']} {row['chain']} {row['resnum']}{row['icode']}"
    if name:
        label += f" – {name}"
    label += f" ({row['atoms']} atoms)"
    if row.get("category", "ligand") != "ligand":
        label += f" [{row['category']}]"
    return label


_CCD_CACHE = {}


def ccd_smiles(code, timeout=10):
    """SMILES for a PDB Chemical Component Dictionary code from RCSB
    (None if offline, unknown, or a generic code like LIG/UNL)."""
    code = code.upper()
    if code in GENERIC_LIGAND_CODES:
        return None
    if code not in _CCD_CACHE:
        import json
        import urllib.request
        try:
            with urllib.request.urlopen(f"https://data.rcsb.org/rest/v1/core/chemcomp/{code}",
                                        timeout=timeout) as r:
                d = json.load(r).get("rcsb_chem_comp_descriptor", {})
            _CCD_CACHE[code] = d.get("SMILES_stereo") or d.get("SMILES")
        except Exception:
            _CCD_CACHE[code] = None
    return _CCD_CACHE[code]


def _bond_orders_from_template(mol, smiles):
    template = Chem.MolFromSmiles(smiles)
    if template is None:
        raise ValueError(f"Could not parse SMILES template: {smiles}")
    heavy = Chem.RemoveHs(mol, sanitize=False)
    for b in heavy.GetBonds():
        b.SetBondType(Chem.BondType.SINGLE)
        b.SetIsAromatic(False)
    for a in heavy.GetAtoms():
        a.SetFormalCharge(0)
        a.SetIsAromatic(False)
        a.SetNoImplicit(False)
    if template.GetNumHeavyAtoms() != heavy.GetNumHeavyAtoms():
        raise ValueError(f"SMILES template has {template.GetNumHeavyAtoms()} heavy atoms, "
                         f"the ligand has {heavy.GetNumHeavyAtoms()}")
    fixed = AllChem.AssignBondOrdersFromTemplate(template, heavy)
    return Chem.AddHs(fixed, addCoords=True)


def _bond_orders_openbabel(pdb_block):
    from openbabel import pybel
    # blank the formal-charge columns so Open Babel perceives bonds from geometry alone
    pdb_block = "".join(l[:78].ljust(78) + "\n" if l.startswith(("ATOM", "HETATM")) else l
                        for l in pdb_block.splitlines(True))
    ob_mol = pybel.readstring("pdb", pdb_block)
    ob_mol.addh()
    mol = Chem.MolFromMolBlock(ob_mol.write("mol"), removeHs=False)
    if mol is None:
        raise ValueError("Open Babel bond-order perception failed")
    return mol


def extract_ligand(pdb_path, het, smiles=None, use_ccd=True):
    """Extract one HET group (a row of list_het_groups) from a PDB or mmCIF
    file as an RDKit molecule (first alternate location only).

    Bond orders, in order of preference:
      1. the SMILES template, if given (its charges are kept, so it can set
         the protonation state);
      2. RDKit DetermineBondOrders when hydrogens are present (net charge from
         the file's formal charges);
      3. the ligand's PDB Chemical Component Dictionary SMILES from RCSB, if
         the code is a registered one and its heavy atoms match (no hydrogens,
         e.g. RCSB X-ray files; neutral form);
      4. Open Babel perception from the heavy-atom geometry.
    Returns (mol, ligand_pdb_block, method).
    """
    key = (het["resname"], het["chain"], str(het["resnum"]), het["icode"])
    lines, serials = [], set()
    text, _ = read_structure_text(pdb_path)
    all_lines = text.splitlines(True)
    for line in all_lines:
        if line.startswith("HETATM") and _res_key(line) == key:
            alt = line[16]
            if alt not in (" ", "A"):
                continue
            lines.append(line[:16] + " " + line[17:])
            serials.add(int(line[6:11]))
    conect = []
    for line in all_lines:
        if line.startswith("CONECT"):
            nums = [int(line[i:i + 5]) for i in range(6, len(line.rstrip()), 5)
                    if line[i:i + 5].strip()]
            if nums and nums[0] in serials:
                partners = [n for n in nums[1:] if n in serials]
                if partners:
                    conect.append("CONECT" + "".join(f"{n:5d}" for n in [nums[0]] + partners) + "\n")
    block = "".join(lines) + "".join(conect) + "END\n"

    raw = Chem.MolFromPDBBlock(block, removeHs=False, sanitize=False,
                               proximityBonding=not conect)
    if raw is None:
        raise ValueError("RDKit could not read the ligand atoms")
    has_h = any(a.GetAtomicNum() == 1 for a in raw.GetAtoms())

    if smiles:
        mol, method = _bond_orders_from_template(raw, smiles), "SMILES template"
    elif has_h:
        mol = Chem.RWMol(raw)
        charge = sum(a.GetFormalCharge() for a in mol.GetAtoms())
        for b in mol.GetBonds():
            b.SetBondType(Chem.BondType.SINGLE)
        for a in mol.GetAtoms():
            a.SetFormalCharge(0)
        try:
            rdDetermineBonds.DetermineBondOrders(mol, charge=charge)
            mol = mol.GetMol()
            Chem.SanitizeMol(mol)
            method = f"RDKit DetermineBondOrders (net charge {charge:+d})"
        except Exception:
            mol, method = _bond_orders_openbabel(block), "Open Babel perception"
    else:
        mol = method = None
        ccd = ccd_smiles(key[0]) if use_ccd else None
        if ccd:
            try:
                mol = _bond_orders_from_template(raw, ccd)
                method = f"PDB Chemical Component Dictionary template for {key[0]} (RCSB; neutral form, H added)"
            except Exception:
                mol = None
        if mol is None:
            mol, method = _bond_orders_openbabel(block), "Open Babel perception (H added)"
    Chem.AssignStereochemistryFrom3D(mol)
    name = f"{key[0]}_{key[1]}{key[2]}{key[3]}"
    mol.SetProp("_Name", name)
    return mol, block, method


def write_complex_split(pdb_path, het, mol, ligand_block,
                        keep_waters=False, keep_metals=True):
    """Write <stem>_protein.pdb and <stem>_ligand_<RES><chain><num>.pdb/.sdf
    next to the complex PDB. Other HET groups are removed (metals optional)."""
    pdb_path = Path(pdb_path)
    folder, stem = pdb_path.parent, structure_stem(pdb_path)
    tag = f"{het['resname']}{het['chain']}{het['resnum']}{het['icode']}"
    prot_path = folder / f"{stem}_protein.pdb"
    lig_pdb = folder / f"{stem}_ligand_{tag}.pdb"
    lig_sdf = folder / f"{stem}_ligand_{tag}.sdf"

    kept = []
    text, _ = read_structure_text(pdb_path)
    for line in text.splitlines(True):
        rec = line[:6]
        if rec.startswith(("ATOM", "HETATM")):
            if line[16] not in (" ", "A"):
                continue  # keep the first alternate location only
            line = line[:16] + " " + line[17:]
        if rec.startswith(("ATOM", "HELIX", "SHEET", "TURN")):
            kept.append(line)
        elif rec == "HETATM":
            resn = line[17:20].strip()
            if (keep_waters and resn in WATER_NAMES) or \
               (keep_metals and resn in METAL_NAMES):
                kept.append(line)
        elif rec.startswith("TER"):
            kept.append(line)
    with open(prot_path, "w") as f:
        f.writelines(kept)
        f.write("END\n")
    with open(lig_pdb, "w") as f:
        f.write(ligand_block)
    with Chem.SDWriter(str(lig_sdf)) as w:
        w.write(mol)
    print(f"Protein for docking : {prot_path}  ({sum(l.startswith('ATOM') for l in kept)} ATOM records)")
    print(f"Ligand (PDB)        : {lig_pdb}")
    print(f"Ligand (SDF)        : {lig_sdf}")
    return prot_path, lig_pdb, lig_sdf


class BindingSiteSetup:
    """Choose the receptor and the ligand that defines the binding site.

    Mode "Complex PDB": load a protein-ligand complex (PDB or mmCIF), pick
    the ligand that defines the site (all ligands are listed, largest
    pre-selected; solvents/ions/additives on request), and write
    <stem>_protein.pdb + <stem>_ligand_*.pdb/.sdf next to the input
    (<stem> gets '_cif' for mmCIF input).
    Mode "Prepared files": use an existing protein (no ligand) PDB and an
    extracted ligand PDB, as in the original notebook.
    After apply(): .protein, .ligand_pdb (autobox), .ligand_ref (display),
    .image_prefix are set.
    """

    def __init__(self, complex_pdb, protein_pdb, ligand_pdb, mode="Complex PDB"):
        style = {"description_width": "160px"}
        wide = widgets.Layout(width="95%")
        self.mode = widgets.ToggleButtons(options=[("Complex structure (PDB/mmCIF)", "Complex PDB"),
                                                   ("Prepared files", "Prepared files")], value=mode,
                                          description="Input type", style={**style, "button_width": "230px"})
        self.complex_pdb = widgets.Text(value=str(complex_pdb), description="Complex (PDB/mmCIF)", style=style,
                                        layout=wide, continuous_update=False)
        self.load_button = widgets.Button(description="Reload ligands", icon="refresh")
        self.show_all = widgets.Checkbox(value=False, description="Also list solvents, ions, additives and caps",
                                         layout=widgets.Layout(width="95%"))
        self.het = widgets.Dropdown(description="Site-defining ligand", style=style, layout=widgets.Layout(width="95%"))
        self.het_info = widgets.HTML()
        self.smiles = widgets.Text(value="", placeholder="optional: SMILES to assign bond orders",
                                   description="SMILES template", style=style, layout=wide)
        self.keep_waters = widgets.Checkbox(value=False, description="Keep waters in protein",
                                            layout=widgets.Layout(width="260px"))
        self.keep_metals = widgets.Checkbox(value=True, description="Keep metal ions in protein",
                                            layout=widgets.Layout(width="260px"))
        self.protein_pdb = widgets.Text(value=str(protein_pdb), description="Protein (no ligand)", style=style, layout=wide)
        self.ligand_pdb_w = widgets.Text(value=str(ligand_pdb), description="Ligand from protein", style=style, layout=wide)
        self.apply_button = widgets.Button(description="Use this binding site", button_style="success", icon="check")
        self.out = widgets.Output()
        self.complex_box = widgets.VBox([self.complex_pdb, widgets.HBox([self.load_button, self.show_all]),
                                         self.het, self.het_info, self.smiles,
                                         widgets.HBox([self.keep_waters, self.keep_metals])])
        self.prepared_box = widgets.VBox([self.protein_pdb, self.ligand_pdb_w])
        self.protein = self.ligand_pdb = self.ligand_ref = self.image_prefix = None
        self.ligand_mol = None
        self.mode.observe(self._toggle, names="value")
        self.load_button.on_click(lambda _: self.load_hets())
        self.complex_pdb.observe(lambda _: self.load_hets(), names="value")
        self.show_all.observe(lambda _: self.load_hets(), names="value")
        self.het.observe(lambda _: self._update_het_info(), names="value")
        self.apply_button.on_click(lambda _: self.apply())
        self._toggle()
        self.load_hets()

    def _toggle(self, *_):
        cplx = self.mode.value == "Complex PDB"
        self.complex_box.layout.display = "" if cplx else "none"
        self.prepared_box.layout.display = "none" if cplx else ""

    def load_hets(self):
        """List the ligands in the complex; pre-select the largest."""
        self._auto = None
        try:
            self._hets = list_het_groups(self.complex_pdb.value, include_all=self.show_all.value)
            if self._hets.empty and not self.show_all.value:   # no ligand: show everything instead
                self._hets = list_het_groups(self.complex_pdb.value, include_all=True)
            self._error = None
        except Exception as e:
            self._hets, self._error = pd.DataFrame(), str(e)
        self._auto = 0 if len(self._hets) else None  # ligands first, largest first
        self.het.options = [(het_label(r), i) for i, r in self._hets.iterrows()]
        if len(self._hets):
            self.het.value = 0
        self._update_het_info()

    def _update_het_info(self):
        if self._error:
            self.het_info.value = f"<span style='color:#b00'>Could not read {self.complex_pdb.value}: {self._error}</span>"
            return
        if not len(self._hets):
            self.het_info.value = "<span style='color:#b00'>No HET groups found.</span>"
            return
        n_lig = int((self._hets.category == "ligand").sum())
        auto = het_label(self._hets.iloc[self._auto])
        if self.het.value == self._auto:
            msg = (f"<b>Automatically selected:</b> {auto} &ndash; the largest ligand. "
                   f"{'Choose another from the list if needed. ' if len(self._hets) > 1 else ''}")
        else:
            msg = f"<b>Your choice</b> (automatic selection was {auto}). "
        msg += f"<i>{n_lig} ligand(s) listed"
        msg += "; solvents, ions, additives and caps included.</i>" if self.show_all.value else \
               "; waters, solvents, ions, additives and terminal caps hidden &ndash; tick the box to list them.</i>"
        self.het_info.value = msg

    def widget(self):
        return widgets.VBox([self.mode, self.complex_box, self.prepared_box, self.apply_button, self.out])

    def apply(self):
        self.out.clear_output()
        with self.out:
            if self.mode.value == "Complex PDB":
                pdb = Path(self.complex_pdb.value)
                het = self._hets.iloc[self.het.value]
                mol, block, method = extract_ligand(pdb, het, self.smiles.value.strip() or None)
                prot, lig_pdb, lig_sdf = write_complex_split(pdb, het, mol, block,
                                                             self.keep_waters.value, self.keep_metals.value)
                self.protein, self.ligand_pdb, self.ligand_ref = prot, lig_pdb, lig_sdf
                self.image_prefix = pdb.parent / structure_stem(pdb)
            else:
                self.protein = Path(self.protein_pdb.value)
                self.ligand_pdb = self.ligand_ref = Path(self.ligand_pdb_w.value)
                self.image_prefix = self.protein.parent / self.protein.stem
                for f in (self.protein, self.ligand_pdb):
                    if not f.exists():
                        raise FileNotFoundError(f)
                mol = method = None
                hets = list_het_groups(self.ligand_pdb, include_all=True)
                if len(hets):
                    mol, _, method = extract_ligand(self.ligand_pdb, hets.iloc[0])
                print(f"Protein for docking : {self.protein}")
                print(f"Ligand from protein : {self.ligand_pdb}")
            self.ligand_mol = mol
            if mol is not None:
                smi = Chem.MolToSmiles(Chem.RemoveHs(mol))
                print(f"Bond orders: {method}\nSMILES: {smi}")
                display(SVG(mol_svg(mol, size=(400, 300), legend=mol.GetProp("_Name"))))
                v = py3Dmol.view(width=450, height=300)
                v.addModel(Chem.MolToMolBlock(mol), "sdf")
                v.setStyle({"stick": {}})
                v.zoomTo()
                v.show()
        return self


# --------------------------------------------------------------------------
# Images (headless PyMOL)
# --------------------------------------------------------------------------
def make_images(protein_pdb, ligand_file, out_prefix, width=1200, height=900,
                site_cutoff=5.0, colouring=None, site_side_chains=True):
    """Ray-traced PNGs: whole protein with ligand, and ligand in binding site.
    Saved as <out_prefix>_complex.png and <out_prefix>_site.png.
    site_side_chains=False hides the binding-site side chains (and the polar
    contacts drawn to them) in the site image; residue labels are kept."""
    import pymol2
    from pymol import util
    complex_png = f"{out_prefix}_complex.png"
    site_png = f"{out_prefix}_site.png"
    with pymol2.PyMOL() as p:
        cmd = p.cmd
        cmd.load(str(protein_pdb), "prot")
        cmd.load(str(ligand_file), "lig")
        cmd.remove("hydro and prot")
        cmd.bg_color("white")
        cmd.set("ray_opaque_background", 1)
        cmd.set("antialias", 2)
        cmd.set("ray_shadows", 0)
        cmd.set("cartoon_transparency", 0)
        # --- whole protein + ligand
        cmd.hide("everything")
        cmd.show("cartoon", "prot")
        colour_protein_pymol(cmd, "prot", colouring)
        cmd.show("spheres", "lig and not hydro")
        cmd.color("grey30", "lig and elem C")
        util.cnc("lig", _self=cmd)
        cmd.orient("prot")
        cmd.zoom("prot", -4)
        cmd.png(complex_png, width=width, height=height, dpi=150, ray=1)
        # --- binding site close-up
        cmd.hide("everything")
        cmd.select("site", f"byres (prot within {site_cutoff} of lig) and not solvent")
        cmd.show("cartoon", "prot")
        cmd.set("cartoon_transparency", 0.6)
        colour_protein_pymol(cmd, "prot", colouring)
        if site_side_chains:
            cmd.show("sticks", "site and not name N+C+O and not hydro")
        cmd.color("lightblue", "site and elem C")
        cmd.show("sticks", "lig and not (hydro and (elem H and neighbor elem C))")
        cmd.set("stick_radius", 0.14, "site")
        cmd.set("stick_radius", 0.28, "lig")
        cmd.color("green", "lig and elem C")
        util.cnc("site or lig", _self=cmd)
        if site_side_chains:
            cmd.distance("hbonds", "lig", "site", mode=2, cutoff=3.5)
            cmd.hide("labels", "hbonds")
            cmd.color("orange", "hbonds")
        cmd.set("label_size", 14)
        cmd.set("label_position", (0, 0, 4))
        cmd.set("label_color", "black")
        cmd.set("float_labels", 1)
        cmd.label("site and name CA", '"%s%s" % (resn, resi)')
        cmd.orient("lig")
        cmd.zoom("lig", 5)
        cmd.png(site_png, width=width, height=height, dpi=150, ray=1)
    return complex_png, site_png


def show_images_side_by_side(paths, titles=None, width=480):
    """Display PNGs side by side (embedded, so they persist in the notebook)."""
    titles = titles or [Path(p).name for p in paths]
    cells = []
    for p, t in zip(paths, titles):
        b64 = base64.b64encode(Path(p).read_bytes()).decode()
        cells.append(f'<figure style="margin:4px;text-align:center">'
                     f'<img src="data:image/png;base64,{b64}" width="{width}"/>'
                     f'<figcaption>{t}</figcaption></figure>')
    display(HTML('<div style="display:flex;flex-wrap:wrap">' + "".join(cells) + "</div>"))


# --------------------------------------------------------------------------
# Flexible side chains
# --------------------------------------------------------------------------
FLEX_EXCLUDED = {"GLY", "ALA", "PRO"}  # no rotatable side-chain torsions


def _pdb_element(line):
    el = re.sub(r"[^A-Za-z]", "", line[76:80])
    if not el:
        el = re.sub(r"[^A-Za-z]", "", line[12:16])[:1]
    return el.capitalize()


def _is_h(el):
    return el.upper() in ("H", "D")


def _ligand_heavy_coords(path):
    path = Path(path)
    if path.suffix.lower() in (".sdf", ".mol"):
        m = next(iter(Chem.SDMolSupplier(str(path), removeHs=False)))
        conf = m.GetConformer()
        return np.array([list(conf.GetAtomPosition(a.GetIdx())) for a in m.GetAtoms()
                         if a.GetAtomicNum() > 1])
    xyz = [[float(l[30:38]), float(l[38:46]), float(l[46:54])]
           for l in open(path) if l.startswith(("ATOM", "HETATM")) and not _is_h(_pdb_element(l))]
    return np.array(xyz)


def residues_near_ligand(protein_pdb, ligand_file, cutoff=10.0):
    """Protein residues with any heavy atom within `cutoff` Å of the ligand's
    heavy atoms, nearest first. Column `flexres` is smina's chain:resid format."""
    lig = _ligand_heavy_coords(ligand_file)
    res = OrderedDict()
    for line in open(protein_pdb):
        if line.startswith("ATOM") and not _is_h(_pdb_element(line)):
            key = (line[21], line[22:26].strip(), line[26].strip())
            res.setdefault(key, {"resname": line[17:20].strip(), "xyz": []})
            res[key]["xyz"].append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
    rows = []
    for (chain, resnum, icode), r in res.items():
        d = np.linalg.norm(np.array(r["xyz"])[:, None, :] - lig[None, :, :], axis=-1).min()
        if d <= cutoff:
            rows.append({"resname": r["resname"], "chain": chain, "resnum": resnum, "icode": icode,
                         "min_dist": round(float(d), 2), "can_flex": r["resname"] not in FLEX_EXCLUDED,
                         "flexres": f"{chain}:{resnum}"})
    return pd.DataFrame(rows).sort_values("min_dist").reset_index(drop=True) if rows else pd.DataFrame(
        columns=["resname", "chain", "resnum", "icode", "min_dist", "can_flex", "flexres"])


def _protein_heavy_atoms(protein_text):
    """(residue keys, coordinates) for heavy ATOM records."""
    keys, xyz = [], []
    for line in protein_text.splitlines():
        if line.startswith("ATOM") and not _is_h(_pdb_element(line)):
            keys.append((line[21], line[22:26].strip()))
            xyz.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
    return keys, np.array(xyz)


def _residues_within(prot_atoms, coords, cutoff=5.0):
    keys, xyz = prot_atoms
    d = np.linalg.norm(xyz[:, None, :] - coords[None, :, :], axis=-1).min(axis=1)
    return sorted({keys[i] for i in np.where(d <= cutoff)[0]}, key=lambda k: (k[0], int(re.sub(r"\D", "", k[1]) or 0)))


def flex_file_for(docked_sdf):
    """Flexible-residue PDB that smina writes alongside a docked SDF (--out_flex)."""
    docked_sdf = Path(docked_sdf)
    return docked_sdf.with_name(docked_sdf.stem + "_flexres.pdb")


def load_flex_models(flex_pdb):
    """Split smina's --out_flex PDB into one model per pose (same order as the
    poses in the docked SDF). Returns [{'name', 'block'}]."""
    models, cur, name = [], [], ""
    for line in open(flex_pdb):
        if line.startswith("MODEL"):
            cur, name = [], ""
        elif line.startswith("COMPND"):
            name = line[6:].strip()
        elif line.startswith(("ATOM", "HETATM", "CONECT")):
            cur.append(line)
        elif line.startswith("ENDMDL"):
            models.append({"name": name, "block": "".join(cur)})
            cur = []
    if cur:
        models.append({"name": name, "block": "".join(cur)})
    return models


def _atoms_from_lines(lines):
    atoms = []
    for i, l in enumerate(lines):
        if l.startswith(("ATOM", "HETATM")):
            atoms.append({"i": i, "name": l[12:16], "el": _pdb_element(l),
                          "key": (l[21], l[22:26].strip(), l[26].strip()),
                          "xyz": np.array([float(l[30:38]), float(l[38:46]), float(l[46:54])])})
    return atoms


def _heavy_graph(atoms):
    import networkx as nx
    g = nx.Graph()
    for k, a in enumerate(atoms):
        g.add_node(k, el=a["el"])
    xyz = np.array([a["xyz"] for a in atoms])
    d = np.linalg.norm(xyz[:, None, :] - xyz[None, :, :], axis=-1)
    for i in range(len(atoms)):
        for j in range(i + 1, len(atoms)):
            if d[i, j] < 1.95:
                g.add_edge(i, j)
    return g


def build_flexed_receptor(protein_text, flex_block):
    """Receptor PDB text with the flexed side chains of one pose.

    smina's --out_flex loses the PDB atom names, so each flexible residue is
    matched onto the receptor residue by bonded-graph (sub)isomorphism of the
    heavy atoms, anchored on the atoms smina keeps fixed (CA, C, CB); the
    receptor keeps its original atom names (needed by AEV-PLIG). Hydrogens on
    moved atoms are replaced by smina's polar hydrogens.
    Returns (pdb_text, [(resname, chain, resnum) flexed], [unmatched chain:resnum]).
    """
    from networkx.algorithms import isomorphism
    lines = protein_text.splitlines(True)
    rigid = _atoms_from_lines(lines)
    flex = _atoms_from_lines(flex_block.splitlines(True))
    by_res = OrderedDict()
    for a in rigid:
        by_res.setdefault(a["key"], []).append(a)
    flex_by_res = OrderedDict()
    for a in flex:
        flex_by_res.setdefault((a["key"][0], a["key"][1], ""), []).append(a)

    new_xyz, drop, insert_after, flexed, unmatched = {}, set(), {}, [], []
    for key, f_atoms in flex_by_res.items():
        r_atoms = by_res.get(key) or next((v for k, v in by_res.items() if k[:2] == key[:2]), None)
        if not r_atoms:
            unmatched.append(f"{key[0]}:{key[1]}")
            continue
        r_heavy = [a for a in r_atoms if not _is_h(a["el"])]
        f_heavy = [a for a in f_atoms if not _is_h(a["el"])]
        gm = isomorphism.GraphMatcher(_heavy_graph(r_heavy), _heavy_graph(f_heavy),
                                      node_match=lambda x, y: x["el"] == y["el"])
        best, best_score = None, -1
        for n, m in enumerate(gm.subgraph_monomorphisms_iter()):
            score = sum(np.linalg.norm(r_heavy[r]["xyz"] - f_heavy[f]["xyz"]) < 0.1 for r, f in m.items())
            if score > best_score:
                best, best_score = m, score
            if n > 500:
                break
        if best is None:
            unmatched.append(f"{key[0]}:{key[1]}")
            continue
        moved = set()
        for r, f in best.items():
            ra = r_heavy[r]
            new_xyz[ra["i"]] = f_heavy[f]["xyz"]
            if np.linalg.norm(ra["xyz"] - f_heavy[f]["xyz"]) > 0.1:
                moved.add(ra["i"])
        # drop receptor hydrogens attached to moved atoms; add smina's polar H
        for h in (a for a in r_atoms if _is_h(a["el"])):
            nearest = min(r_heavy, key=lambda a: np.linalg.norm(a["xyz"] - h["xyz"]))
            if nearest["i"] in moved:
                drop.add(h["i"])
        template = lines[r_atoms[-1]["i"]]
        h_lines = []
        for k, h in enumerate((a for a in f_atoms if _is_h(a["el"])), start=1):
            x, y, z = h["xyz"]
            h_lines.append(f"{template[:12]}{(' HX' + str(k))[:4]:<4}{template[16:30]}"
                           f"{x:8.3f}{y:8.3f}{z:8.3f}{template[54:76]} H  \n")
        insert_after[r_atoms[-1]["i"]] = h_lines
        flexed.append((lines[r_atoms[0]["i"]][17:20].strip(), key[0], key[1]))

    out = []
    for i, l in enumerate(lines):
        if i in drop:
            continue
        if i in new_xyz:
            x, y, z = new_xyz[i]
            l = f"{l[:30]}{x:8.3f}{y:8.3f}{z:8.3f}{l[54:]}"
        out.append(l)
        out.extend(insert_after.get(i, []))
    return "".join(out), flexed, unmatched


# --------------------------------------------------------------------------
# Run provenance
# --------------------------------------------------------------------------
def file_sha256(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def smina_version(smina):
    try:
        r = subprocess.run([str(smina), "--version"], capture_output=True, text=True, timeout=30)
        return (r.stdout + r.stderr).strip().splitlines()[0]
    except Exception as e:
        return f"unknown ({e})"


def software_versions():
    import importlib
    out = {"python": platform.python_version()}
    for name in ("rdkit", "numpy", "pandas", "torch", "torch_geometric", "torchani"):
        try:
            out[name] = importlib.import_module(name).__version__
        except Exception:
            out[name] = "not installed"
    return out


def _now():
    import datetime
    return datetime.datetime.now().astimezone().isoformat(timespec="seconds")


def smina_run_info(cmd, panel):
    """Everything needed to repeat a smina run: version, command, settings,
    platform and SHA-256 of the input files."""
    inputs = {"receptor": panel.receptor.value, "ligands": panel.ligands.value}
    if not panel.minimize.value:
        inputs["autobox_ligand"] = panel.autobox.value
    return {
        "created": _now(),
        "smina": {"path": panel.smina.value, "version": smina_version(panel.smina.value)},
        "command": shlex.join(cmd),
        "settings": {
            "mode": "minimize" if panel.minimize.value else ("flexible" if panel.flexible else "rigid"),
            "seed": panel.seed.value, "cpu": panel.cpu.value,
            "exhaustiveness": panel.exhaustiveness.value, "num_modes": panel.num_modes.value,
            "energy_range": panel.energy_range.value, "autobox_add": panel.autobox_add.value,
            "scoring": panel.scoring_arg or "vina",
            "flexres": panel.flexres() if panel.flexible else [],
            "extra_arguments": panel.extra.value.strip(),
        },
        "inputs": {k: {"path": v, "sha256": file_sha256(v) if Path(v).exists() else None}
                   for k, v in inputs.items()},
        "platform": {"system": platform.system(), "release": platform.release(),
                     "machine": platform.machine(), "logical_cpus": os.cpu_count()},
        "box": (None if panel.minimize.value or any(a in panel.extra.value for a in ("--center_", "--size_"))
                else box_check(panel.autobox.value, panel.autobox_add.value, panel.ligands.value)[0]),
        "note": ("With this smina build (Vina 1.1.2 based) the same --seed gave identical poses "
                 "for --cpu 1-24; record kept so the run can be repeated exactly."),
    }


# --------------------------------------------------------------------------
# smina panel
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# Search-box checks
# --------------------------------------------------------------------------
BOX_EDGE_MARGIN = 1.0   # Å: poses with a heavy atom closer than this to a box face are flagged
BOX_TIGHT = 2.0         # Å: a ligand longer than (smallest box side - this) is reported as tight
VINA_MAX_VOLUME = 27000  # Å^3: Vina warns above this search volume


def _all_atom_coords(path):
    path = Path(path)
    if path.suffix.lower() in (".sdf", ".mol"):
        m = next(iter(Chem.SDMolSupplier(str(path), removeHs=False, sanitize=False)))
        return m.GetConformer().GetPositions()
    return np.array([[float(l[30:38]), float(l[38:46]), float(l[46:54])]
                     for l in open(path) if l.startswith(("ATOM", "HETATM"))])


def autobox_bounds(autobox_ligand, autobox_add):
    """(min corner, max corner) of smina's --autobox_ligand box: the bounding
    box of ALL atoms in the file (hydrogens included - verified with smina
    dc3dfab) expanded by autobox_add on every side."""
    xyz = _all_atom_coords(autobox_ligand)
    return xyz.min(0) - autobox_add, xyz.max(0) + autobox_add


_SPAN_CACHE = {}


def ligand_spans(sdf):
    """{name: longest heavy-atom distance (Å) over all its conformers in sdf}."""
    sdf = Path(sdf)
    key = (str(sdf.resolve()), sdf.stat().st_mtime)
    if key not in _SPAN_CACHE:
        spans = {}
        RDLogger.DisableLog("rdApp.warning")
        try:
            for m in Chem.SDMolSupplier(str(sdf), removeHs=False, sanitize=False):
                if m is None or not m.GetNumConformers():
                    continue
                P = m.GetConformer().GetPositions()[[a.GetAtomicNum() > 1 for a in m.GetAtoms()]]
                d = float(np.linalg.norm(P[:, None] - P[None], axis=-1).max()) if len(P) > 1 else 0.0
                n = m.GetProp("_Name")
                spans[n] = max(spans.get(n, 0.0), d)
        finally:
            RDLogger.EnableLog("rdApp.warning")
        _SPAN_CACHE[key] = spans
    return _SPAN_CACHE[key]


def box_check(autobox_ligand, autobox_add, ligands_sdf):
    """Compare each ligand's longest span with the search box.
    Returns (box dict, DataFrame name/span/status, suggested autobox_add or None)."""
    lo, hi = autobox_bounds(autobox_ligand, autobox_add)
    size = hi - lo
    smallest = float(size.min())
    rows = []
    for name, span in ligand_spans(ligands_sdf).items():
        status = ("longer than smallest box side" if span > smallest else
                  "tight" if span > smallest - BOX_TIGHT else "ok")
        rows.append({"name": name, "span": round(span, 1), "status": status})
    df = pd.DataFrame(rows, columns=["name", "span", "status"])
    suggest = None
    if len(df) and (df.status != "ok").any():
        ref_extent = float((size - 2 * autobox_add).min())       # reference ligand along its narrowest axis
        need = (df.span.max() + BOX_TIGHT - ref_extent) / 2
        suggest = float(np.ceil(need * 2) / 2)                    # round up to 0.5 Å
    box = {"min": [round(float(v), 3) for v in lo], "max": [round(float(v), 3) for v in hi],
           "size": [round(float(v), 2) for v in size], "volume": round(float(np.prod(size)))}
    return box, df, suggest


def box_from_run_record(record):
    """Search box of a smina run: stored in its run record, or rebuilt from
    the recorded autobox ligand and autobox_add (older records)."""
    if not record:
        return None
    if record.get("box"):
        return record["box"]
    st, inp = record.get("settings", {}), record.get("inputs", {})
    path = inp.get("autobox_ligand", {}).get("path")
    if st.get("mode") in ("rigid", "flexible") and path and Path(path).exists():
        lo, hi = autobox_bounds(path, float(st.get("autobox_add", 4.0)))
        size = hi - lo
        return {"min": lo.tolist(), "max": hi.tolist(), "size": size.tolist(), "volume": float(np.prod(size))}
    return None


def box_margins(mols, box):
    """Smallest distance (Å) from any heavy atom of each pose to a box face;
    negative = outside the box."""
    lo, hi = np.array(box["min"]), np.array(box["max"])
    out = []
    for m in mols:
        P = m.GetConformer().GetPositions()[[a.GetAtomicNum() > 1 for a in m.GetAtoms()]]
        out.append(float(np.minimum(P - lo, hi - P).min()))
    return out


FLEX_WARNING = (
    "<div style='background:#fff3cd;color:#664d03;border-left:5px solid #e0a800;padding:6px 10px;margin:4px 0'>"
    "<b>&#9888; Flexible docking is much slower than rigid docking</b> &ndash; typically 10&ndash;100&times; "
    "longer, and the cost rises steeply with the number of flexible residues. Keep the selection small "
    "(&le; 6 residues if possible) and consider a lower exhaustiveness or fewer conformers for a first pass. "
    "Gly, Ala and Pro are excluded (no side-chain torsions).</div>")


class SminaPanel:
    """ipywidgets input panel for smina, pre-filled with the old notebook's settings.
    Supports rigid docking, flexible side-chain docking and --minimize."""

    def __init__(self, receptor, autobox_ligand, ligands, smina=DEFAULT_SMINA):
        total, perf, eff = cpu_info()
        style = {"description_width": "150px"}
        wide = widgets.Layout(width="95%")
        self.smina = widgets.Text(value=str(smina), description="smina executable", style=style, layout=wide)
        self.receptor = widgets.Text(value=str(receptor), description="Receptor (-r)", style=style, layout=wide)
        self.autobox = widgets.Text(value=str(autobox_ligand), description="Autobox ligand", style=style, layout=wide)
        self.ligands = widgets.Text(value=str(ligands), description="Ligands (-l)", style=style, layout=wide)
        self.output = widgets.Text(description="Output (-o)", style=style, layout=wide)
        self.exhaustiveness = widgets.BoundedIntText(value=20, min=1, max=256, description="exhaustiveness", style=style)
        self.seed = widgets.IntText(value=0, description="seed", style=style)
        self.cpu = widgets.BoundedIntText(value=total, min=1, max=total, description="cpu", style=style)
        self.cpu_label = widgets.HTML(f"<i>{cpu_note()}. smina uses up to this many threads per ligand; with the same seed the poses do not depend on it (tested 1-24).</i>")
        self.autobox_add = widgets.BoundedFloatText(value=4.0, min=0, max=20, step=0.5, description="autobox_add (Å)", style=style)
        self.num_modes = widgets.BoundedIntText(value=9, min=1, max=100, description="num_modes", style=style)
        self.energy_range = widgets.BoundedFloatText(value=3.0, min=0, max=20, description="energy_range", style=style)
        self.scoring = widgets.Dropdown(options=["vina (default)", "vinardo", "dkoes_scoring", "ad4_scoring"],
                                        value="vina (default)", description="scoring", style=style)
        self.minimize = widgets.Checkbox(value=False, description="--minimize (local energy minimisation only, no search)",
                                         layout=widgets.Layout(width="95%"))
        self.minimize_note = widgets.HTML(
            "<i>--minimize optimises the supplied coordinates in place (rigid receptor); use it on poses already "
            "in the site (e.g. the crystal ligand SDF or a docked SDF), not on freshly generated conformers.</i>")
        # --- flexible docking
        self.mode = widgets.ToggleButtons(options=["Rigid receptor", "Flexible side chains (slower)"],
                                          value="Rigid receptor", description="Docking mode",
                                          style={**style, "button_width": "230px"})
        self.flex_select = widgets.RadioButtons(
            options=["Automatic: all residues within a distance of the ligand", "Manual: choose residues"],
            description="Flexible residues", style=style, layout=widgets.Layout(width="95%"))
        self.flex_dist = widgets.BoundedFloatText(value=5.0, min=1.0, max=15.0, step=0.5,
                                                  description="Distance to ligand (Å)", style=style)
        self.flex_find = widgets.Button(description="Find residues", icon="refresh")
        self.flex_list = widgets.SelectMultiple(options=[], rows=12, description="Residues", style=style,
                                                layout=widgets.Layout(width="480px"))
        self.flex_help = widgets.HTML("<i>Manual mode: Ctrl/Cmd-click to select residues (listed nearest first, "
                                      "up to 10 Å from the ligand); add others below.</i>")
        self.flex_extra = widgets.Text(value="", placeholder="e.g. A:41,A:145", description="Add residues",
                                       style=style, layout=wide)
        self.flex_summary = widgets.HTML()
        self.flex_box = widgets.VBox([widgets.HTML(FLEX_WARNING), self.flex_select,
                                      widgets.HBox([self.flex_dist, self.flex_find]),
                                      self.flex_list, self.flex_help, self.flex_extra, self.flex_summary])
        self.extra = widgets.Text(value="", description="extra arguments", style=style, layout=wide)
        self.box_info = widgets.HTML()
        self.preview = widgets.Textarea(layout=widgets.Layout(width="95%", height="90px"), disabled=True)
        self.run_button = widgets.Button(description="Run smina", button_style="success", icon="play")
        self.log = widgets.Output(layout=widgets.Layout(max_height="300px", overflow="auto", border="1px solid #ccc"))
        self.last_output = None
        self._near = pd.DataFrame()

        for w in [self.smina, self.receptor, self.autobox, self.ligands, self.output, self.exhaustiveness,
                  self.seed, self.cpu, self.autobox_add, self.num_modes, self.energy_range, self.scoring,
                  self.minimize, self.mode, self.flex_list, self.flex_extra, self.extra]:
            w.observe(self._update, names="value")
        for w in (self.ligands, self.minimize, self.mode):
            w.observe(self._default_output, names="value")
        self.flex_dist.observe(lambda _: self.find_flex_residues(), names="value")
        self.flex_select.observe(lambda _: self.find_flex_residues(), names="value")
        self.flex_find.on_click(lambda _: self.find_flex_residues())
        self.run_button.on_click(lambda _: self.run())
        self.find_flex_residues()
        self._default_output()

    # ---- flexible residues
    @property
    def flexible(self):
        return self.mode.value.startswith("Flexible") and not self.minimize.value

    def find_flex_residues(self):
        """(Re)list residues near the autobox ligand; in automatic mode select
        all flexible-capable residues within the distance cutoff."""
        auto = self.flex_select.value.startswith("Automatic")
        try:
            near = residues_near_ligand(self.receptor.value, self.autobox.value,
                                        cutoff=max(10.0, self.flex_dist.value))
        except Exception as e:
            self.flex_summary.value = f"<span style='color:#b00'>Could not list residues: {e}</span>"
            return
        self._near = near[near.can_flex]
        previous = set(self.flex_list.value)
        self.flex_list.options = [(f"{r.resname} {r.chain}{r.resnum}{r.icode}  ({r.min_dist:.1f} Å)", r.flexres)
                                  for r in self._near.itertuples()]
        if auto:
            chosen = [r.flexres for r in self._near.itertuples() if r.min_dist <= self.flex_dist.value]
        else:
            chosen = [o for _, o in self.flex_list.options if o in previous] or \
                     [r.flexres for r in self._near.itertuples() if r.min_dist <= self.flex_dist.value]
        self.flex_list.value = tuple(chosen)
        self.flex_list.layout.display = "none" if auto else ""
        self.flex_dist.disabled = not auto
        self.flex_help.layout.display = "none" if auto else ""
        self._update()

    def flexres(self):
        res = list(self.flex_list.value)
        for r in re.split(r"[,\s]+", self.flex_extra.value.strip()):
            if r and r not in res:
                res.append(r)
        return res

    @property
    def scoring_arg(self):
        """--scoring value, or None for smina's default (vina)."""
        return None if self.scoring.value.startswith("vina") else self.scoring.value

    @property
    def flex_output(self):
        return flex_file_for(self.output.value)

    # ---- command
    def _default_output(self, *_):
        lig = Path(self.ligands.value)
        stem = lig.stem.replace("_confs", "")
        suffix = "_minimized" if self.minimize.value else ("_flexdocked" if self.flexible else "_docked")
        self.output.value = str(lig.parent / f"{stem}{suffix}.sdf")
        self._update()

    def _update(self, *_):
        self.flex_box.layout.display = "" if self.mode.value.startswith("Flexible") else "none"
        if self.mode.value.startswith("Flexible"):
            res = self.flexres()
            names = {r.flexres: f"{r.resname} {r.chain}{r.resnum} ({r.min_dist:.1f} Å)" for r in self._near.itertuples()}
            colour = "#b00" if len(res) > 6 else "inherit"
            note = " &ndash; <b>many residues: expect a very long run</b>" if len(res) > 6 else ""
            if self.minimize.value:
                note += " (ignored with --minimize)"
            self.flex_summary.value = (f"<span style='color:{colour}'><b>{len(res)} flexible residues</b>{note}: "
                                       f"{', '.join(names.get(r, r) for r in res) or 'none'}</span>")
        self.preview.value = shlex.join(self.command())
        self._update_box_info()

    def _update_box_info(self):
        """Pre-docking check: does every ligand fit the search box?"""
        if self.minimize.value:
            self.box_info.value = "<i>--minimize: no search box.</i>"
            return
        if any(a in self.extra.value for a in ("--center_", "--size_")):
            self.box_info.value = "<i>Box set explicitly in extra arguments; box check skipped.</i>"
            return
        if not Path(self.ligands.value).exists():
            self.box_info.value = "<i>Box check: generate the conformers first.</i>"
            return
        try:
            box, df, suggest = box_check(self.autobox.value, self.autobox_add.value, self.ligands.value)
        except Exception as e:
            self.box_info.value = f"<i>Box check not available: {e}</i>"
            return
        x, y, z = box["size"]
        msg = (f"<b>Search box</b> {x:.1f} × {y:.1f} × {z:.1f} Å ({box['volume']:,} Å³; autobox ligand + "
               f"{self.autobox_add.value:g} Å). ")
        if box["volume"] > VINA_MAX_VOLUME:
            msg += f"<span style='color:#b00'>Larger than {VINA_MAX_VOLUME:,} Å³ – raise exhaustiveness.</span> "
        if len(df):
            bad = df[df.status != "ok"]
            msg += f"Longest ligand: {df.loc[df.span.idxmax(), 'name']} ({df.span.max():.1f} Å; smallest box side {min(box['size']):.1f} Å). "
            if len(bad):
                items = ", ".join(f"{r.name} {r.span:.1f} Å ({r.status})" for r in bad.itertuples())
                msg = ("<div style='background:#fff3cd;color:#664d03;border-left:5px solid #e0a800;padding:6px 10px'>"
                       + msg + f"<br><b>&#9888; May be limited by the box:</b> {items}. "
                       f"Consider <b>autobox_add ≥ {suggest:g} Å</b> (and a higher exhaustiveness).</div>")
            else:
                msg += "<span style='color:#1a7f37'>&#10004; All ligands fit with ≥ "\
                       f"{BOX_TIGHT:g} Å to spare.</span>"
        self.box_info.value = msg

    def command(self):
        c = [self.smina.value, "-r", self.receptor.value, "-l", self.ligands.value,
             "-o", self.output.value, "--seed", str(self.seed.value), "--cpu", str(self.cpu.value)]
        if self.minimize.value:
            c += ["--minimize"]
        else:
            c += ["--autobox_ligand", self.autobox.value, "--autobox_add", str(self.autobox_add.value),
                  "--exhaustiveness", str(self.exhaustiveness.value),
                  "--num_modes", str(self.num_modes.value), "--energy_range", str(self.energy_range.value)]
            if self.flexible and self.flexres():
                c += ["--flexres", ",".join(self.flexres()), "--out_flex", str(self.flex_output)]
        if not self.scoring.value.startswith("vina"):
            c += ["--scoring", self.scoring.value]
        if self.extra.value.strip():
            c += shlex.split(self.extra.value)
        return c

    def widget(self):
        files = widgets.VBox([widgets.HTML("<b>Files</b>"), self.smina, self.receptor, self.autobox,
                              self.ligands, self.output])
        params = widgets.VBox([widgets.HTML("<b>Search parameters</b>"),
                               widgets.HBox([self.exhaustiveness, self.seed]),
                               widgets.HBox([self.cpu, self.cpu_label]),
                               widgets.HBox([self.autobox_add, self.num_modes]),
                               widgets.HBox([self.energy_range, self.scoring]),
                               self.minimize, self.minimize_note, self.extra, self.box_info])
        flex = widgets.VBox([widgets.HTML("<b>Receptor flexibility</b>"), self.mode, self.flex_box])
        return widgets.VBox([files, params, flex, widgets.HTML("<b>Command</b>"), self.preview,
                             self.run_button, self.log])

    def run(self):
        cmd = self.command()
        if not os.access(self.smina.value, os.X_OK):
            with self.log:
                print(f"smina executable not found or not executable: {self.smina.value}")
            return 1
        if self.flexible and not self.flexres():
            with self.log:
                print("Flexible mode selected but no residues chosen.")
            return 1
        out = Path(self.output.value)
        log_path = out.with_suffix(".log")
        lig_path = Path(self.ligands.value)
        input_titles = sdf_titles(lig_path) if lig_path.suffix.lower() == ".sdf" else None
        bad_records = []
        if input_titles is not None:
            RDLogger.DisableLog("rdApp.*")
            try:
                bad_records = [i for i, m in enumerate(Chem.SDMolSupplier(str(lig_path), removeHs=False,
                                                                          sanitize=False)) if m is None]
            finally:
                RDLogger.EnableLog("rdApp.*")
        if not self.flexible and self.flex_output.exists():
            self.flex_output.unlink()  # stale flexible residues from an earlier run with this name
        self.log.clear_output()
        self.run_button.disabled = True
        info = smina_run_info(cmd, self)
        info_path = out.with_name(out.stem + ".run.json")
        try:
            with self.log, open(log_path, "w") as logf:
                print("$ " + shlex.join(cmd))
                st = info["settings"]
                header = (f"# {info['smina']['version']}\n"
                          f"# mode {st['mode']}  seed {st['seed']}  cpu {st['cpu']}  "
                          f"exhaustiveness {st['exhaustiveness']}  scoring {st['scoring']}\n"
                          f"# platform {info['platform']['system']} {info['platform']['machine']}  "
                          f"started {info['created']}\n"
                          f"# run record: {info_path.name}\n")
                logf.write(header + "$ " + shlex.join(cmd) + "\n")
                print(header, end="")
                if info.get("box"):
                    bx = info["box"]
                    _, spans, sugg = box_check(self.autobox.value, self.autobox_add.value, self.ligands.value)
                    tight = spans[spans.status != "ok"]
                    line = (f"Box check (before docking): box {bx['size'][0]:.1f} x {bx['size'][1]:.1f} x "
                            f"{bx['size'][2]:.1f} A ({bx['volume']:,} A^3); longest ligand "
                            f"{spans.span.max():.1f} A, smallest box side {min(bx['size']):.1f} A. ")
                    line += ("May be limited: " + ", ".join(f"{r.name} {r.span:.1f} A ({r.status})" for r in tight.itertuples())
                             + f" - consider autobox_add >= {sugg:g} A." if len(tight) else "All ligands fit with >= 2 A to spare.")
                    print(("⚠ " if len(tight) else "✔ ") + line)
                    logf.write("# " + line + "\n")
                if bad_records:
                    i = bad_records[0]
                    print(f"⚠ Ligand file: record {i + 1} ('{input_titles[i]}') cannot be read"
                          f"{f' (and {len(bad_records) - 1} more)' if len(bad_records) > 1 else ''}. smina stops "
                          f"reading the file at an unreadable record and SKIPS ALL LATER LIGANDS without an error "
                          f"- fix or remove it. Coverage is checked after the run.")
                proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                        text=True, bufsize=1)
                for line in proc.stdout:
                    logf.write(line)
                    if line.strip() and not line.startswith(("*", "-----", "mode |")):
                        print(line, end="")
                rc = proc.wait()
                info["exit_code"] = rc
                info["finished"] = _now()
                info["output"] = {"path": str(out), "sha256": file_sha256(out) if out.exists() else None}
                if self.flexible and self.flex_output.exists():
                    info["output_flexres"] = {"path": str(self.flex_output),
                                              "sha256": file_sha256(self.flex_output)}
                import json
                info_path.write_text(json.dumps(info, indent=2))
                print(f"\nsmina finished with exit code {rc}. Output: {out}\nLog: {log_path}\n"
                      f"Run record: {info_path}")
                if self.flexible:
                    print(f"Flexible side chains (one model per pose): {self.flex_output}")
                if rc == 0 and out.exists() and input_titles is not None:
                    docked_names = set(sdf_titles(out))
                    wanted = list(dict.fromkeys(t for t in input_titles if t))
                    missing = [n for n in wanted if n not in docked_names]
                    info["ligands_in"] = len(wanted)
                    info["ligands_docked"] = len(wanted) - len(missing)
                    info["ligands_missing"] = missing
                    info_path.write_text(json.dumps(info, indent=2))
                    if missing:
                        print(f"\n⚠ COVERAGE: smina produced no poses for {len(missing)} of {len(wanted)} ligands: "
                              f"{', '.join(missing[:15])}{' ...' if len(missing) > 15 else ''}. "
                              f"smina exits normally when it skips ligands or stops at an unreadable record - "
                              f"check the ligand file and the log.")
                    else:
                        print(f"\n✔ Coverage: all {len(wanted)} ligands have poses.")
                if rc == 0 and info.get("box") and out.exists():
                    poses = load_poses(out)
                    margins = box_margins(poses, info["box"])
                    edge = [(m.GetProp("_Name"), g) for m, g in zip(poses, margins) if g < BOX_EDGE_MARGIN]
                    if edge:
                        names = sorted({n for n, _ in edge})
                        print(f"\n⚠ Box check: {len(edge)} of {len(poses)} poses have a heavy atom within "
                              f"{BOX_EDGE_MARGIN:g} Å of the box edge ({sum(g < 0 for _, g in edge)} outside it); "
                              f"ligands: {', '.join(names)}. The box may be limiting these ligands - "
                              f"consider a larger autobox_add.")
                    else:
                        print(f"\n✔ Box check: no pose within {BOX_EDGE_MARGIN:g} Å of the box edge.")
            if rc == 0:
                self.last_output = out
            return rc
        finally:
            self.run_button.disabled = False


# --------------------------------------------------------------------------
# Results
# --------------------------------------------------------------------------
def load_poses(docked_sdf):
    """Docked poses with a pose_id in file order (pose_0, pose_1, ...)."""
    poses = []
    RDLogger.DisableLog("rdApp.warning")  # smina SDFs are flagged 2D but carry 3D coordinates
    try:
        suppl = list(Chem.SDMolSupplier(str(docked_sdf), removeHs=False, sanitize=True))
    finally:
        RDLogger.EnableLog("rdApp.warning")
    for i, m in enumerate(suppl):
        if m is None:
            print(f"Warning: could not read pose {i}")
            continue
        m.SetProp("pose_id", f"pose_{i}")
        poses.append(m)
    return poses


def _pose_index(pose_id):
    return int(str(pose_id).rsplit("_", 1)[1])


def sdf_titles(path):
    """Title (first line) of every record in an SDF, readable or not, in file order."""
    titles, first = [], True
    with open(path) as f:
        for line in f:
            if first:
                titles.append(line.strip())
                first = False
            if line.startswith("$$$$"):
                first = True
    return titles


FLEX_CLASH = 1.5  # Å: a pose this close to its own flexed side chains cannot be a correct pairing


def pose_flex_models(poses, flex_pdb, docked_sdf=None, strict=True):
    """Map pose_id -> flexible-residue model (smina --out_flex).

    smina writes a ligand's poses and its side-chain models together, so the
    two files are normally in step. This is verified, not assumed:
    1. if the record titles of the docked SDF and the model names (COMPND)
       are identical, poses and models are paired by position;
    2. otherwise the two name sequences are aligned and only poses whose
       name matches are paired; the rest get no flexed receptor;
    3. every pair is checked geometrically - a pose overlapping its own
       flexed side chains (< 1.5 Å) is not a valid pairing.
    With strict=True an unresolved mismatch or a clash raises an error."""
    import difflib
    models = load_flex_models(flex_pdb)
    model_names = [m["name"] for m in models]
    by_index = {_pose_index(m.GetProp("pose_id")): m for m in poses}
    titles = sdf_titles(docked_sdf) if docked_sdf else [by_index[i].GetProp("_Name") if i in by_index else ""
                                                          for i in range(max(by_index) + 1)]
    if titles == model_names:
        pairs = {i: i for i in range(len(titles))}
    else:
        # align runs of consecutive same-name records (one run per ligand/conformer block);
        # a block is paired only if it has as many poses as models - otherwise positions
        # inside it cannot be trusted and the whole block is left unpaired
        def runs(names):
            out, start = [], 0
            for k in range(1, len(names) + 1):
                if k == len(names) or names[k] != names[start]:
                    out.append((names[start], start, k - start))
                    start = k
            return out
        ra, rb = runs(titles), runs(model_names)
        sm = difflib.SequenceMatcher(a=[r[0] for r in ra], b=[r[0] for r in rb], autojunk=False)
        pairs = {}
        for tag, a0, a1, b0, b1 in sm.get_opcodes():
            if tag == "equal":
                for (na, sa, ca), (nb, sb, cb) in zip(ra[a0:a1], rb[b0:b1]):
                    if ca == cb:
                        pairs.update({sa + k: sb + k for k in range(ca)})
        unpaired = [titles[i] for i in range(len(titles)) if i not in pairs]
        msg = (f"{len(titles)} poses but {len(models)} flexible-residue models in {Path(flex_pdb).name}; "
               f"{len(pairs)} paired by name, {len(unpaired)} without a flexed receptor "
               f"({', '.join(sorted(set(unpaired))) or '-'})")
        if strict and (unpaired or len(pairs) != len(models)):
            raise ValueError("Poses and flexible-residue models do not correspond: " + msg)
        print("Warning: " + msg)
    out, clashes = {}, []
    for i, j in pairs.items():
        m = by_index.get(i)
        if m is None:
            continue
        block = models[j]["block"]
        flex_xyz = np.array([[float(l[30:38]), float(l[38:46]), float(l[46:54])]
                             for l in block.splitlines()
                             if l.startswith(("ATOM", "HETATM")) and not _is_h(_pdb_element(l))])
        lig = m.GetConformer().GetPositions()[[a.GetAtomicNum() > 1 for a in m.GetAtoms()]]
        if len(flex_xyz) and np.linalg.norm(lig[:, None] - flex_xyz[None], axis=-1).min() < FLEX_CLASH:
            clashes.append(m.GetProp("pose_id"))
            continue
        out[m.GetProp("pose_id")] = block
    if clashes:
        msg = (f"{len(clashes)} pose(s) overlap their own flexed side chains (< {FLEX_CLASH:g} Å) - "
               f"pairing not valid: {', '.join(clashes[:10])}{' ...' if len(clashes) > 10 else ''}")
        if strict:
            raise ValueError(msg)
        print("Warning: " + msg)
    return out


def poses_table(poses):
    rows = []
    for m in poses:
        rows.append({"pose_id": m.GetProp("pose_id"), "name": m.GetProp("_Name"),
                     "minimizedAffinity": float(m.GetProp("minimizedAffinity"))
                     if m.HasProp("minimizedAffinity") else np.nan})
    return pd.DataFrame(rows)


def results_browser(poses, protein_pdb, reference_ligand=None, predictions=None,
                    sort_by="minimizedAffinity", flex_pdb=None):
    """Step through docked poses with a slider. `predictions` (optional) is a
    DataFrame with pose_id and AEV_PLIG_pK columns. With `flex_pdb` (smina
    --out_flex) each pose is shown with its own flexed side chains (magenta);
    the original side-chain positions can be overlaid in thin grey."""
    if not poses:
        print("No poses to show")
        return
    table = poses_table(poses)
    if predictions is not None and "AEV_PLIG_pK" in predictions:
        table = table.merge(predictions[["pose_id", "AEV_PLIG_pK"]], on="pose_id", how="left")
    ascending = sort_by != "AEV_PLIG_pK"
    table = table.sort_values(sort_by, ascending=ascending, kind="stable").reset_index(drop=True)
    by_id = {m.GetProp("pose_id"): m for m in poses}
    protein_block = Path(protein_pdb).read_text()
    ref_block = Path(reference_ligand).read_text() if reference_ligand else None
    ref_fmt = Path(reference_ligand).suffix.lstrip(".").lower() if reference_ligand else None
    ref_fmt = "sdf" if ref_fmt in ("sdf", "mol") else "pdb"
    flex_models = pose_flex_models(poses, flex_pdb, strict=False) if flex_pdb and Path(flex_pdb).exists() else {}
    prot_atoms = _protein_heavy_atoms(protein_block)

    slider, nav = _nav_slider(len(table), "Pose")
    show_site = widgets.Checkbox(value=True, description="Binding-site residues as sticks")
    show_orig = widgets.Checkbox(value=True, description="Original side chains (grey)")
    show_ref = widgets.Checkbox(value=True, description="Ligand from PDB (red)")
    toggles = widgets.HBox([show_site] + ([show_ref] if ref_block else []) + ([show_orig] if flex_models else []))
    out = widgets.Output()

    def show(i):
        row = table.iloc[i]
        m = by_id[row["pose_id"]]
        out.clear_output()
        with out:
            info = (f"<b>{i + 1}/{len(table)}</b> &nbsp; <b>{row['name']}</b> ({row['pose_id']}) &nbsp; "
                    f"smina affinity: <b>{row['minimizedAffinity']:.2f}</b> kcal/mol")
            if "AEV_PLIG_pK" in row and pd.notna(row.get("AEV_PLIG_pK")):
                info += f" &nbsp; AEV-PLIG pK: <b>{row['AEV_PLIG_pK']:.2f}</b>"
            flex_res = []
            prot = protein_block
            if flex_models and row["pose_id"] not in flex_models:
                info += ("<br><span style='color:#b00'>No verified flexed receptor for this pose - "
                         "shown with the rigid protein.</span>")
            if row["pose_id"] in flex_models:
                prot, flexed, unmatched = build_flexed_receptor(protein_block, flex_models[row["pose_id"]])
                flex_res = [(c, r) for _, c, r in flexed]
                info += "<br>Flexible residues (magenta): " + ", ".join(f"{n} {c}{r}" for n, c, r in flexed)
                if unmatched:
                    info += f" &nbsp; <span style='color:#b00'>not mapped: {', '.join(unmatched)}</span>"
            display(HTML(info + "<br>" + colour_legend_html()))
            v = py3Dmol.view(width=800, height=500)
            v.addModel(prot, "pdb")
            v.setStyle({"model": 0}, cartoon_style())
            model = 1
            if ref_block and show_ref.value:
                v.addModel(ref_block, ref_fmt)
                v.setStyle({"model": model}, {"stick": {"colorscheme": "redCarbon", "radius": 0.12}})
                model += 1
            v.addModel(Chem.MolToMolBlock(m), "sdf")
            lig_model = model
            v.setStyle({"model": lig_model}, {"stick": {"colorscheme": "greenCarbon", "radius": 0.2}})
            model += 1
            if show_site.value:  # residues within 5 Å of this pose (3Dmol 'within' cannot span models)
                lig_xyz = m.GetConformer().GetPositions()[[a.GetAtomicNum() > 1 for a in m.GetAtoms()]]
                for chain, resi in _residues_within(prot_atoms, lig_xyz, 5.0):
                    v.addStyle({"model": 0, "chain": chain, "resi": resi},
                               {"stick": {"radius": 0.12, "colorscheme": "cyanCarbon"}})
            for chain, resi in flex_res:
                v.setStyle({"model": 0, "chain": chain, "resi": resi},
                           {**cartoon_style(),
                            "stick": {"radius": 0.18, "colorscheme": "magentaCarbon"}})
            if flex_res and show_orig.value:
                v.addModel(protein_block, "pdb")
                v.setStyle({"model": model}, {})
                for chain, resi in flex_res:
                    v.setStyle({"model": model, "chain": chain, "resi": resi, "not": {"atom": ["N", "C", "O"]}},
                               {"stick": {"radius": 0.07, "color": "grey", "opacity": 0.6}})
            v.zoomTo({"model": lig_model})
            v.zoom(0.6)
            v.show()

    slider.observe(lambda c: show(c["new"]), names="value")
    show_site.observe(lambda c: show(slider.value), names="value")
    show_orig.observe(lambda c: show(slider.value), names="value")
    show_ref.observe(lambda c: show(slider.value), names="value")
    display(widgets.VBox([nav, toggles, out]))
    show(0)
    return table


# --------------------------------------------------------------------------
# AEV-PLIG
# --------------------------------------------------------------------------
def _ligand_pdb_block(mol, resname="LIG", chain="L", resnum=1):
    """PDB block (HETATM + CONECT) for a ligand with unique atom names."""
    m = Chem.Mol(mol)
    counts = {}
    for a in m.GetAtoms():
        el = a.GetSymbol()
        counts[el] = counts.get(el, 0) + 1
        name = f"{el}{counts[el]}"[:4]
        info = Chem.AtomPDBResidueInfo(f"{name:<4}" if len(el) == 2 or len(name) == 4 else f" {name:<3}",
                                       residueName=resname, chainId=chain, residueNumber=resnum,
                                       isHeteroAtom=True)
        a.SetMonomerInfo(info)
    return Chem.MolToPDBBlock(m)


def complex_pdb_text(receptor_pdb, mol, remarks=()):
    """Receptor + ligand (HETATM, chain L, resname LIG) as one PDB, with the
    ligand atoms numbered after the receptor's."""
    rec = [l for l in Path(receptor_pdb).read_text().splitlines()
           if l.startswith(("HELIX", "SHEET", "TURN", "ATOM", "HETATM", "TER"))]
    offset = max([int(l[6:11]) for l in rec if l.startswith(("ATOM", "HETATM"))] or [0])
    lig = []
    for l in _ligand_pdb_block(mol).splitlines():
        if l.startswith("HETATM"):
            lig.append(f"{l[:6]}{int(l[6:11]) + offset:5d}{l[11:]}")
        elif l.startswith("CONECT"):
            nums = [int(l[i:i + 5]) + offset for i in range(6, len(l), 5) if l[i:i + 5].strip()]
            lig.append("CONECT" + "".join(f"{n:5d}" for n in nums))
    head = [f"REMARK   1 {r}" for r in remarks]
    return "\n".join(head + rec + ["TER"] + lig + ["END"]) + "\n"


def export_best_poses_pdb(summary, poses, receptor_files, out_dir):
    """For the best AEV-PLIG pose of each ligand write <name>_<pose>_ligand.pdb
    and <name>_<pose>_complex.pdb (with that pose's receptor) to out_dir."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("*.pdb"):  # folder is per docked file: drop results of earlier runs
        old.unlink()
    by_id = {m.GetProp("pose_id"): m for m in poses}
    written = []
    for r in summary[summary.best_aevplig_for_ligand].itertuples():
        m = by_id[r.pose_id]
        safe = re.sub(r"[^\w.-]+", "_", str(r.name)).strip("_") or "ligand"
        remarks = [f"ligand {r.name}  {r.pose_id}",
                   f"smina minimizedAffinity {r.minimizedAffinity:.3f} kcal/mol",
                   f"AEV-PLIG pK {r.AEV_PLIG_pK:.3f} +/- {r.AEV_PLIG_pK_std:.3f}",
                   f"receptor {Path(receptor_files[r.pose_id]).name}"]
        lig_path = out_dir / f"{safe}_{r.pose_id}_ligand.pdb"
        lig_path.write_text("".join(f"REMARK   1 {x}\n" for x in remarks) + _ligand_pdb_block(m))
        cpx_path = out_dir / f"{safe}_{r.pose_id}_complex.pdb"
        cpx_path.write_text(complex_pdb_text(receptor_files[r.pose_id], m, remarks))
        written.append({"name": r.name, "pose_id": r.pose_id, "AEV_PLIG_pK": r.AEV_PLIG_pK,
                        "ligand_pdb": lig_path.name, "complex_pdb": cpx_path.name})
    return pd.DataFrame(written)


def smina_score_ligand(ligand_mol, receptor, out_dir, smina=DEFAULT_SMINA, scoring=None,
                       name="pdb_ligand"):
    """smina scores for a ligand in its given (e.g. crystal) pose:
    --score_only on the exact coordinates, and --minimize (local optimisation
    in the site; comparable with the docked poses' minimizedAffinity).
    Files: <out_dir>/<name>.sdf (input), <name>_minimized.sdf, <name>_smina.log."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    lig = out_dir / f"{name}.sdf"
    with Chem.SDWriter(str(lig)) as w:
        w.write(ligand_mol)
    base = [str(smina), "-r", str(receptor), "-l", str(lig)]
    if scoring:
        base += ["--scoring", scoring]
    res = {"score_only": np.nan, "minimizedAffinity": np.nan, "minimizedRMSD": np.nan}
    log = []
    r = subprocess.run(base + ["--score_only"], capture_output=True, text=True)
    log += ["$ " + shlex.join(base + ["--score_only"]), r.stdout, r.stderr]
    m = re.search(r"Affinity:\s+(-?\d+(?:\.\d+)?)", r.stdout)
    if m:
        res["score_only"] = float(m.group(1))
    minimized = out_dir / f"{name}_minimized.sdf"
    cmd = base + ["--minimize", "-o", str(minimized)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    log += ["$ " + shlex.join(cmd), r.stdout, r.stderr]
    if minimized.exists():
        mm = next(iter(Chem.SDMolSupplier(str(minimized), removeHs=False, sanitize=False)), None)
        for k in ("minimizedAffinity", "minimizedRMSD"):
            if mm is not None and mm.HasProp(k):
                res[k] = float(mm.GetProp(k))
    (out_dir / f"{name}_smina.log").write_text("\n".join(log))
    if np.isnan(res["minimizedAffinity"]):
        print(f"Warning: smina scoring of {name} failed; see {out_dir / (name + '_smina.log')}")
    else:
        print(f"PDB ligand smina: score_only {res['score_only']:.2f}, minimized "
              f"{res['minimizedAffinity']:.2f} kcal/mol (RMSD {res['minimizedRMSD']:.2f} Å)")
    return res


LIGAND_CARBON_COLOURS = ["green", "cyan", "yellow", "orange", "slate", "wheat", "lime", "teal",
                         "lightpink", "deepsalmon", "palecyan", "limon"]


def _moved_residues(rigid_pdb_text, pose_pdb_text, tol=0.1):
    """(chain, resnum) of residues whose heavy atoms differ between two PDBs."""
    def coords(text):
        d = {}
        for l in text.splitlines():
            if l.startswith("ATOM") and not _is_h(_pdb_element(l)):
                d[(l[21], l[22:26].strip(), l[12:16].strip())] = np.array(
                    [float(l[30:38]), float(l[38:46]), float(l[46:54])])
        return d
    a, b = coords(rigid_pdb_text), coords(pose_pdb_text)
    return sorted({k[:2] for k in a.keys() & b.keys() if np.linalg.norm(a[k] - b[k]) > tol},
                  key=lambda k: (k[0], int(re.sub(r"\D", "", k[1]) or 0)))


def export_pymol_session(docked_sdf, protein_pdb, reference_ligand=None, out_path=None,
                         site_cutoff=5.0, colouring=None, enable="top"):
    """PyMOL session (.pse) of the best AEV-PLIG pose of each ligand.

    Uses the AEV-PLIG results in <folder of docked_sdf>/predictions (run
    run_aev_plig first). Objects: 'receptor' (cartoon, RasMol colours by
    default; binding-site side chains as sticks), one group per ligand named
    <rank>_<ligand> holding the pose, its polar contacts and - after flexible
    docking - that pose's flexed side chains (magenta), and 'PDB_ligand'
    (red carbons). enable='top' shows only the best-ranked ligand initially,
    'all' shows every ligand.
    Saved as predictions/<stem>_best_poses.pse unless out_path is given.
    """
    import pymol2
    from pymol import util
    docked_sdf, protein_pdb = Path(docked_sdf).resolve(), Path(protein_pdb).resolve()
    stem = docked_sdf.stem
    pred_dir = docked_sdf.parent / "predictions"
    summary_csv = pred_dir / f"{stem}_summary.csv"
    if not summary_csv.exists():
        raise FileNotFoundError(f"{summary_csv} not found - run AEV-PLIG first")
    summary = pd.read_csv(summary_csv)
    best = summary[summary.best_aevplig_for_ligand.astype(bool)].sort_values(
        "AEV_PLIG_pK", ascending=False).reset_index(drop=True)
    by_id = {m.GetProp("pose_id"): m for m in load_poses(docked_sdf)}
    rigid_text = protein_pdb.read_text()
    out_path = Path(out_path or pred_dir / f"{stem}_best_poses.pse")

    with pymol2.PyMOL() as p:
        cmd = p.cmd
        cmd.load(str(protein_pdb), "receptor")
        cmd.remove("receptor and solvent")
        cmd.hide("everything")
        cmd.show("cartoon", "receptor")
        colour_protein_pymol(cmd, "receptor", colouring)
        cmd.set("cartoon_transparency", 0.2, "receptor")
        cmd.bg_color("white")
        cmd.set("valence", 1)
        cmd.set("stick_radius", 0.15)
        lig_names = []
        for rank, r in enumerate(best.itertuples(), start=1):
            m = by_id.get(r.pose_id)
            if m is None:
                continue
            safe = re.sub(r"[^A-Za-z0-9_]", "_", str(r.name))
            grp = f"{rank:02d}_{safe}"
            lig = f"{grp}_pose"
            m.SetProp("_Name", lig)
            cmd.load_raw(Chem.MolToMolBlock(m), "mol", lig)
            colour = LIGAND_CARBON_COLOURS[(rank - 1) % len(LIGAND_CARBON_COLOURS)]
            cmd.show("sticks", lig)
            cmd.set("stick_radius", 0.22, lig)
            cmd.color(colour, f"{lig} and elem C")
            util.cnc(lig, _self=cmd)
            cmd.hide("everything", f"{lig} and elem H and neighbor elem C")
            members = [lig]
            receptor = pred_dir / "poses" / stem / str(r.receptor_file)
            if receptor.exists() and receptor.resolve() != protein_pdb:
                pose_text = receptor.read_text()
                moved = _moved_residues(rigid_text, pose_text)
                if moved:
                    tmp = f"{grp}_tmp"
                    cmd.load_raw(pose_text, "pdb", tmp)
                    sel = " or ".join(f"(chain {c} and resi {n})" for c, n in moved)
                    flex = f"{grp}_flexed"
                    cmd.create(flex, f"{tmp} and ({sel}) and not name N+C+O")
                    cmd.delete(tmp)
                    cmd.show("sticks", flex)
                    cmd.hide("everything", f"{flex} and elem H and neighbor elem C")
                    cmd.color("magenta", f"{flex} and elem C")
                    util.cnc(flex, _self=cmd)
                    members.append(flex)
            hb = f"{grp}_hbonds"
            cmd.distance(hb, lig, "receptor" + (f" or {grp}_flexed" if len(members) > 1 else ""),
                         mode=2, cutoff=3.5)
            cmd.hide("labels", hb)
            cmd.color("orange", hb)
            members.append(hb)
            cmd.group(grp, " ".join(members))
            cmd.set_title(lig, 1, f"{r.name}  {r.pose_id}  pK {r.AEV_PLIG_pK:.2f}  smina {r.minimizedAffinity:.2f}")
            lig_names.append((grp, lig, members))
        if reference_ligand and Path(reference_ligand).exists():
            cmd.load(str(Path(reference_ligand).resolve()), "PDB_ligand")
            cmd.show("sticks", "PDB_ligand")
            cmd.set("stick_radius", 0.12, "PDB_ligand")
            cmd.color("red", "PDB_ligand and elem C")
            util.cnc("PDB_ligand", _self=cmd)
            cmd.hide("everything", "PDB_ligand and elem H and neighbor elem C")
        if lig_names:
            all_ligs = " or ".join(l for _, l, _ in lig_names)
            cmd.select("binding_site", f"byres (receptor within {site_cutoff} of ({all_ligs})) and not solvent")
            cmd.show("sticks", "binding_site and not name N+C+O and not (elem H and neighbor elem C)")
            cmd.color("lightblue", "binding_site and elem C")
            util.cnc("binding_site", _self=cmd)
            cmd.deselect()
            cmd.orient(all_ligs)
            cmd.zoom(all_ligs, 6)
            if enable == "top":
                for grp, _, members in lig_names[1:]:
                    for obj in members + [grp]:  # members too: a disabled group alone does not hide them
                        cmd.disable(obj)
        cmd.save(str(out_path))
    print(f"PyMOL session with {len(lig_names)} best poses written to {out_path}")
    return out_path


# 2.303 RT at 298.15 K (kcal/mol): converts pK to binding free energy, dG = -1.364 * pK
RT_LN10 = 0.0019872 * 298.15 * np.log(10)


def add_ligand_efficiency(df):
    """Add ligand efficiency columns (kcal/mol per heavy atom) to a table with
    heavy_atoms, AEV_PLIG_pK and minimizedAffinity columns:
      LE_AEV_PLIG = 1.364 * pK / heavy_atoms   (from the predicted affinity)
      LE_smina    = -minimizedAffinity / heavy_atoms"""
    df["LE_AEV_PLIG"] = RT_LN10 * df["AEV_PLIG_pK"] / df["heavy_atoms"]
    df["LE_smina"] = -df["minimizedAffinity"] / df["heavy_atoms"]
    return df


def run_aev_plig(docked_sdf, protein_pdb, predictions_dir=None, num_workers=None,
                 python=sys.executable, extra_files=(), add_hydrogens=True, flex_pdb=None,
                 reference_mol=None, reference_smina=None):
    """Predict binding affinity (pK) for every docked pose with AEV-PLIG.

    reference_mol: the ligand from the original PDB (RDKit mol in its crystal
    pose). It is scored in the same AEV-PLIG run against protein_pdb and added
    to <stem>_best_poses_table.csv (best pose per ligand + the PDB ligand).
    reference_smina: dict from smina_score_ligand() for that ligand; its
    minimized score fills the table's minimizedAffinity, and the score-only
    value and minimisation RMSD are added as extra columns.

    add_hydrogens: smina writes polar hydrogens only, but AEV-PLIG was trained on
    fully protonated ligands (its total_num_Hs atom feature counts explicit H
    neighbours), so all hydrogens are added with coordinates before scoring.
    The original notebook stripped all hydrogens (PandasTools), so its pK values
    differ (by up to ~1 pK unit for the Mpro example).

    flex_pdb: smina --out_flex file from flexible docking. Each pose is then
    scored against its own receptor (rigid receptor + that pose's flexed side
    chains), written as poses/<stem>/pose_<i>_receptor.pdb.

    Everything is exported to <folder of docked_sdf>/predictions/:
      poses/<stem>/pose_<i>.sdf, <stem>_aevplig_input.csv, <stem>_predictions.csv (raw),
      <stem>_summary.csv, <stem>_scored.sdf, <stem>_best_poses.sdf,
      <stem>_best_poses_table.csv, <stem>_run_info.json (docking + scoring settings,
      versions and input checksums), <stem>_smina_vs_aevplig.png, best_poses_pdb/<stem>/ (best pose per ligand as
      ligand-only and protein-ligand complex PDB files) and copies of the
      inputs in extra_files.
    """
    docked_sdf, protein_pdb = Path(docked_sdf).resolve(), Path(protein_pdb).resolve()
    pred_dir = Path(predictions_dir or docked_sdf.parent / "predictions").resolve()
    stem = docked_sdf.stem
    pose_dir = pred_dir / "poses" / stem  # per docked file, so runs never overwrite each other
    pose_dir.mkdir(parents=True, exist_ok=True)
    for old in list(pose_dir.glob("pose_*.sdf")) + list(pose_dir.glob("pose_*_receptor.pdb")):
        old.unlink()  # files from an earlier run of this docked file

    poses = load_poses(docked_sdf)
    flex_models = {}
    if flex_pdb:
        flex_pdb = Path(flex_pdb).resolve()
        flex_models = pose_flex_models(poses, flex_pdb, docked_sdf=docked_sdf)
        protein_text = protein_pdb.read_text()
        print(f"Flexible docking: building a receptor with flexed side chains for each pose")
    rows = []
    for m in poses:
        pid = m.GetProp("pose_id")
        f = pose_dir / f"{pid}.sdf"
        with Chem.SDWriter(str(f)) as w:
            w.write(Chem.AddHs(m, addCoords=True) if add_hydrogens else m)
        receptor = protein_pdb
        if flex_models and pid not in flex_models:
            continue  # flexible run but no verified flexed receptor for this pose: do not score it
        if pid in flex_models:
            text, _, unmatched = build_flexed_receptor(protein_text, flex_models[pid])
            if unmatched:
                print(f"Warning: {pid}: flexible residues not mapped: {', '.join(unmatched)}")
            receptor = pose_dir / f"{pid}_receptor.pdb"
            receptor.write_text(text)
        rows.append({"unique_id": pid, "name": m.GetProp("_Name"),
                     "minimizedAffinity": float(m.GetProp("minimizedAffinity"))
                     if m.HasProp("minimizedAffinity") else np.nan,
                     "sdf_file": str(f), "pdb_file": str(receptor)})
    ref_name = None
    if reference_mol is not None:
        ref_name = reference_mol.GetProp("_Name") if reference_mol.HasProp("_Name") else "PDB ligand"
        f = pose_dir / "pdb_ligand.sdf"
        with Chem.SDWriter(str(f)) as w:
            w.write(Chem.AddHs(reference_mol, addCoords=True) if add_hydrogens else reference_mol)
        rows.append({"unique_id": "pdb_ligand", "name": ref_name, "minimizedAffinity": np.nan,
                     "sdf_file": str(f), "pdb_file": str(protein_pdb)})
    inp = pd.DataFrame(rows)
    csv = pred_dir / f"{stem}_aevplig_input.csv"
    inp.to_csv(csv, index=False)

    missing = [i for i in range(10)
               if not (AEV_PLIG_DIR / "output" / "trained_models" / f"{AEV_PLIG_MODEL}_{i}.model").exists()]
    if missing or not (AEV_PLIG_DIR / "output" / "trained_models" / f"{AEV_PLIG_MODEL}.pickle").exists():
        raise FileNotFoundError(
            "AEV-PLIG trained models not found in aev_plig/output/trained_models/. "
            "Download them with:  python aev_plig/download_models.py")
    workers = num_workers or os.cpu_count()
    cmd = [python, "process_and_predict.py", f"--dataset_csv={csv}", f"--data_name={stem}",
           f"--trained_model_name={AEV_PLIG_MODEL}", f"--num_workers={workers}",
           f"--output_dir={pred_dir}"]
    print("$ " + shlex.join(cmd))
    log_path = pred_dir / f"{stem}_aevplig.log"
    with open(log_path, "w") as logf:
        rc = subprocess.run(cmd, cwd=AEV_PLIG_DIR, stdout=logf, stderr=subprocess.STDOUT).returncode
    if rc != 0:
        print(Path(log_path).read_text()[-3000:])
        raise RuntimeError(f"AEV-PLIG failed (exit {rc}); see {log_path}")
    raw = pd.read_csv(pred_dir / f"{stem}_predictions.csv")

    model_cols = [c for c in raw.columns if c.startswith("preds_")]
    summary = inp.drop(columns=["sdf_file"]).rename(columns={"unique_id": "pose_id", "pdb_file": "receptor_file"})
    summary["receptor_file"] = summary["receptor_file"].map(lambda f: Path(f).name)
    preds = raw[["unique_id"] + model_cols + ["preds"]].rename(
        columns={"unique_id": "pose_id", "preds": "AEV_PLIG_pK"})
    preds["AEV_PLIG_pK_std"] = raw[model_cols].std(axis=1).values
    summary = summary.merge(preds, on="pose_id", how="left")
    heavy = {m.GetProp("pose_id"): m.GetNumHeavyAtoms() for m in poses}
    if reference_mol is not None:
        heavy["pdb_ligand"] = reference_mol.GetNumHeavyAtoms()
    summary["heavy_atoms"] = summary.pose_id.map(heavy)
    summary = add_ligand_efficiency(summary)
    import json as _json
    _rec = docked_sdf.with_name(stem + ".run.json")
    _box = box_from_run_record(_json.loads(_rec.read_text())) if _rec.exists() else None
    if _box:
        _margin = dict(zip([m.GetProp("pose_id") for m in poses], box_margins(poses, _box)))
        if reference_mol is not None:
            _margin["pdb_ligand"] = box_margins([reference_mol], _box)[0]
        summary["box_margin"] = summary.pose_id.map(_margin)
        summary["at_box_edge"] = summary.box_margin < BOX_EDGE_MARGIN
    else:
        summary["box_margin"], summary["at_box_edge"] = np.nan, pd.NA
    reference = summary[summary.pose_id == "pdb_ligand"]
    summary = summary[summary.pose_id != "pdb_ligand"].reset_index(drop=True)
    dropped = summary["AEV_PLIG_pK"].isna().sum()
    if dropped:
        print(f"Note: {dropped} pose(s) were rejected by AEV-PLIG (unreadable or unsupported elements)")
    summary["best_smina_for_ligand"] = summary.groupby("name")["minimizedAffinity"].transform("min") == summary["minimizedAffinity"]
    summary["best_aevplig_for_ligand"] = summary.groupby("name")["AEV_PLIG_pK"].transform("max") == summary["AEV_PLIG_pK"]
    summary = summary.sort_values("AEV_PLIG_pK", ascending=False).reset_index(drop=True)
    summary_cols = ["pose_id", "name", "minimizedAffinity", "AEV_PLIG_pK", "AEV_PLIG_pK_std",
                    "heavy_atoms", "LE_AEV_PLIG", "LE_smina", "box_margin", "at_box_edge",
                    "best_smina_for_ligand", "best_aevplig_for_ligand", "receptor_file"] + model_cols
    summary = summary[summary_cols]
    summary.to_csv(pred_dir / f"{stem}_summary.csv", index=False)

    # Best pose per ligand + the ligand from the PDB (crystal pose)
    table_cols = ["name", "pose_id", "minimizedAffinity", "AEV_PLIG_pK", "AEV_PLIG_pK_std",
                  "heavy_atoms", "LE_AEV_PLIG", "LE_smina", "box_margin", "at_box_edge"]
    best_table = summary[summary.best_aevplig_for_ligand][table_cols].copy()
    best_table.insert(0, "source", "docked (best AEV-PLIG pose)")
    ref_pk = None
    if len(reference):
        ref = reference[table_cols].copy()
        ref["pose_id"] = "crystal pose"
        ref.insert(0, "source", "ligand from PDB")
        if reference_smina:
            ref["minimizedAffinity"] = reference_smina.get("minimizedAffinity", np.nan)
            ref["LE_smina"] = -ref["minimizedAffinity"] / ref["heavy_atoms"]
            ref["smina_score_only"] = reference_smina.get("score_only", np.nan)
            ref["minimizedRMSD"] = reference_smina.get("minimizedRMSD", np.nan)
        best_table = pd.concat([best_table, ref], ignore_index=True)
        ref_pk = reference["AEV_PLIG_pK"].iloc[0]
        if pd.isna(ref_pk):
            print("Note: AEV-PLIG rejected the PDB ligand (unreadable or unsupported elements)")
            ref_pk = None
        else:
            print(f"PDB ligand {ref_name}: AEV-PLIG pK {ref_pk:.2f} "
                  f"+/- {reference['AEV_PLIG_pK_std'].iloc[0]:.2f}")
    best_table.to_csv(pred_dir / f"{stem}_best_poses_table.csv", index=False)
    if _box:
        edge_best = summary[summary.best_aevplig_for_ligand & summary.at_box_edge.astype(bool)]
        if len(edge_best):
            print(f"⚠ Box check: best pose at the box edge for {', '.join(edge_best.name)} "
                  f"(margin < {BOX_EDGE_MARGIN:g} Å) - these ligands may be limited by the search box.")

    # Run record: docking settings (from smina's .run.json) + AEV-PLIG settings
    import json
    docking_info_path = docked_sdf.with_name(stem + ".run.json")
    run_info = {
        "created": _now(),
        "docked_sdf": {"path": str(docked_sdf), "sha256": file_sha256(docked_sdf)},
        "docking": json.loads(docking_info_path.read_text()) if docking_info_path.exists() else None,
        "aev_plig": {"model": AEV_PLIG_MODEL, "ensemble_size": len(model_cols),
                     "hydrogens_added": add_hydrogens, "flexible_receptors": bool(flex_models),
                     "receptor": {"path": str(protein_pdb), "sha256": file_sha256(protein_pdb)},
                     "poses_scored": int(summary["AEV_PLIG_pK"].notna().sum()),
                     "poses_rejected": int(dropped)},
        "reference_ligand": None if not len(reference) else {
            "name": ref_name, "AEV_PLIG_pK": None if ref_pk is None else round(float(ref_pk), 4),
            "smina": reference_smina},
        "software": software_versions(),
    }
    if run_info["docking"] is None:
        print(f"Note: no docking run record ({docking_info_path.name}) found; settings not recorded")
    (pred_dir / f"{stem}_run_info.json").write_text(json.dumps(run_info, indent=2, default=str))

    # Annotated SDFs
    lookup = summary.set_index("pose_id")
    with Chem.SDWriter(str(pred_dir / f"{stem}_scored.sdf")) as w_all, \
         Chem.SDWriter(str(pred_dir / f"{stem}_best_poses.sdf")) as w_best:
        for m in poses:
            pid = m.GetProp("pose_id")
            if pid not in lookup.index or pd.isna(lookup.at[pid, "AEV_PLIG_pK"]):
                continue
            m.SetProp("AEV_PLIG_pK", f"{lookup.at[pid, 'AEV_PLIG_pK']:.3f}")
            m.SetProp("AEV_PLIG_pK_std", f"{lookup.at[pid, 'AEV_PLIG_pK_std']:.3f}")
            w_all.write(m)
            if lookup.at[pid, "best_aevplig_for_ligand"]:
                w_best.write(m)

    # Best pose per ligand as PDB (ligand only + complex with that pose's receptor)
    receptor_files = dict(zip(inp["unique_id"], inp["pdb_file"]))
    best_pdb = export_best_poses_pdb(summary, poses, receptor_files, pred_dir / "best_poses_pdb" / stem)
    print(f"Best pose per ligand exported as PDB ({len(best_pdb)} ligands) to {pred_dir / 'best_poses_pdb' / stem}")

    # Plot
    import matplotlib
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for name, g in summary.groupby("name"):
        ax.scatter(g["minimizedAffinity"], g["AEV_PLIG_pK"], label=name, s=25, alpha=0.8)
    if ref_pk is not None:
        ax.axhline(ref_pk, ls="--", color="red", lw=1, label=f"{ref_name} (PDB ligand)")
    ax.set_xlabel("smina minimizedAffinity (kcal/mol)")
    ax.set_ylabel("AEV-PLIG predicted pK")
    ax.set_title("Docking score vs AEV-PLIG prediction")
    if summary["name"].nunique() <= 12:
        ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(pred_dir / f"{stem}_smina_vs_aevplig.png", dpi=150)
    plt.show()

    for f in [docked_sdf, docked_sdf.with_name(stem + ".run.json"), protein_pdb, flex_pdb, *extra_files]:
        if f and Path(f).exists() and Path(f).resolve().parent != pred_dir:
            shutil.copy2(f, pred_dir / Path(f).name)
    print(f"All results exported to {pred_dir}")
    return summary
