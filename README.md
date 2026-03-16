# Boltz2Score

Boltz2Score scores an existing complex with the Boltz-2 confidence head and can optionally run pose-conditioned flexible optimization.

The project now exposes one main entry with four modes:

- `score` (default): confidence scoring only, no diffusion resampling
- `pose`: optimize around the input pose with minimal drift
- `refine`: trade off pose retention and interface confidence
- `interface`: allow more interface adjustment to improve interface-focused metrics

## Code Layout

- `boltz2score.py`: main user-facing CLI and top-level orchestration entrypoint
- `core/cli.py`: CLI argument groups, validation, and execution-plan building
- `core/job.py`: per-record job orchestration for staging, scoring, postprocess, and optional affinity
- `core/pipeline.py`: high-level mode orchestration for flexible optimization
- `core/flexible_optimization.py`: built-in single-config optimization used by non-`score` modes
- `core/prepare_inputs.py`: internal structure/ligand preprocessing and manifest building
- `core/results.py`: chain-map writing, IPSAE postprocess, and diffusion-sample reranking
- `core/modes.py`: shared mode names and help text
- `utils/ligand_utils.py`: ligand file loading, atom-name normalization, and combined complex input assembly
- `utils/ligand_alignment.py`: ligand atom-confidence alignment helpers and per-atom confidence summaries
- `utils/affinity_utils.py`: optional affinity runner integration and ligand-alignment payload normalization
- `utils/score_diagnostics.py`: atom-coverage diagnostics and ligand confidence alignment postprocess
- `utils/structure_refinement.py`: anchored constraints and self-template setup
- `utils/result_utils.py`: result-discovery and artifact-resolution helpers
- `utils/writer_compat.py`: duplicate atom-id normalization and mmCIF writer safety checks
- `metrics/ligand_ipsae.py`: ligand-aware IPSAE calculation
- `core/inference.py`: low-level Boltz2 inference runner and writer integration
- `tools/collect_metrics.py`: optional CSV export utility

`score` remains the baseline mode.

## Install

```bash
cd /data/Boltz2Score
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

If you want the CUDA build of Boltz:

```bash
pip install --upgrade "boltz[cuda]"
```

Required cache assets under `BOLTZ_CACHE` or `~/.boltz`:

- `ccd.pkl`
- `mols/`
- `boltz2_conf.ckpt`

Optional for affinity:

- `boltz2_aff.ckpt`
- sibling `affinity/` module on `PYTHONPATH`

## Main Usage

### 1. Default: score only

```bash
python boltz2score.py \
  --protein_file data/cdk8/5hnb-chainA-prepared.pdb \
  --ligand_file data/cdk8/ligands.sdf \
  --output_dir results/cdk8_score \
  --compute_ipsae
```

This runs Boltz2 confidence scoring only. It does not run diffusion refinement.

### 2. Pose-preserving flexible optimization

```bash
python boltz2score.py \
  --mode pose \
  --protein_file data/cdk8/5hnb-chainA-prepared.pdb \
  --ligand_file data/cdk8/ligands.sdf \
  --output_dir results/cdk8_pose \
  --compute_ipsae
```

This runs anchored flexible optimization that tries to preserve the input ligand orientation.

### 3. Balanced flexible optimization

```bash
python boltz2score.py \
  --mode refine \
  --protein_file data/cdk8/5hnb-chainA-prepared.pdb \
  --ligand_file data/cdk8/ligands.sdf \
  --output_dir results/cdk8_refine \
  --compute_ipsae
```

### 4. Interface-confidence flexible optimization

```bash
python boltz2score.py \
  --mode interface \
  --protein_file data/cdk8/5hnb-chainA-prepared.pdb \
  --ligand_file data/cdk8/ligands.sdf \
  --output_dir results/cdk8_interface \
  --compute_ipsae
```

## Input Rules

- Use either `--input <complex.pdb/mmcif>` or `--protein_file + --ligand_file`.
- Flexible optimization modes currently require separate-input mode with an SDF ligand file.
- Multi-molecule SDF is supported. Each valid molecule becomes one record directory.
- Use `--ligand_indices 1,3-5` to score or optimize only selected 1-based entries from a multi-molecule ligand file. The default is all entries.
- If the user does not provide ligand SMILES, RDKit derives canonical SMILES from the ligand structure automatically.
- If `--compute_ipsae` is enabled, output is forced to `mmcif`.

### Optional ligand SMILES override

```bash
python boltz2score.py \
  --input /path/to/complex.pdb \
  --output_dir /path/to/out \
  --ligand_smiles_map '{"L:LIG":"CC1=CC=CC=C1"}'
```

Accepted keys:

- `chain`
- `chain:resname`

In separate-input mode the ligand structure remains the source of truth. RDKit-generated canonical SMILES override inconsistent manual SMILES so topology stays aligned with the uploaded ligand.

## Outputs

### `score`

Each record directory contains:

- `confidence_<record>_model_*.json`
- `confidence_<record>_model_*.cif` or `.pdb`
- `raw_ligand_atom_plddts_<record>_model_*.json/csv`: ligand heavy-atom pLDDT in Boltz2 writer/model order
- `ipsae_<record>_model_*.json` when `--compute_ipsae` is enabled
- `best_model.cif`
- `best_confidence.json`
- `best_ipsae.json` when IPSAE is enabled

Inside `confidence_<record>_model_*.json`, ligand atom confidence is now exposed in three explicit orders:

- `ligand_atom_plddts` / `ligand_atom_names`: uploaded ligand heavy-atom input order
- `ligand_atom_smiles_order_plddts` / `ligand_atom_smiles_order_names`: RDKit non-canonical SMILES traversal order
- `ligand_atom_model_order_plddts` / `ligand_atom_model_order_names`: Boltz2 writer/model order

### Flexible optimization modes

For `pose`, `refine`, and `interface`, record result directories are written directly under the top-level output directory, the same as `score`.

Each record directory also contains:

- `best_model.cif`
- `best_confidence.json`
- `best_ipsae.json` when available

## What The Three Optimization Modes Mean

### `pose`

- Best when you already trust the docking pose orientation
- Uses the gentlest local-refine preset (`sigma_max=0.25`, `sampling_steps=8`)
- Favors low ligand RMSD, low centroid shift, low protein drift
- Recommended default for small-molecule docking refinement

### `refine`

- Middle ground between geometry retention and interface metrics
- Uses a moderate local-refine preset (`sigma_max=0.35`, `sampling_steps=10`)
- Useful when you do not yet know which side matters more for a target family

### `interface`

- Uses the loosest local-refine preset (`sigma_max=0.45`, `sampling_steps=12`)
- Gives more weight to `ipTM`, ligand `pLDDT`, `IPSAE`, and interface stability
- More suitable when interface quality matters more than strict pose retention
- Often a better starting point for peptide-protein or antibody-antigen complexes

## Advanced Flags

Most users should stay with `--mode` and `--compute_ipsae`. The lower-level refinement controls still exist for direct experimentation:

- `--structure_refine`
- `--anchored_refine`
- `--reference_from_input`
- `--sampling_init_from_input`
- `--sigma_max`
- `--noise_scale`
- `--gamma_0`
- `--gamma_min`

Those flags are mainly for method development, not the default user path.

## Internal Layout

Normal usage only needs `boltz2score.py`.

- `core/`: internal orchestration and low-level inference modules
- `utils/`: shared helpers
- `metrics/`: scoring/metric calculation modules
- `tools/collect_metrics.py`: optional CSV export utility
- `data/`: analysis scripts and flexible-optimization helpers

## Notes

- Boltz2 confidence scoring is not the same thing as experimental affinity prediction.
- Flexible optimization is pose-conditioned engineering on top of Boltz2, not the original Boltz2 inference workflow.
- `NUMBA_CACHE_DIR=/tmp/numba_cache` is recommended if your environment has Numba cache permission issues.
