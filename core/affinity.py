from __future__ import annotations

import json
import pickle
from dataclasses import asdict, replace
from pathlib import Path
from types import MethodType

from pytorch_lightning import Trainer, seed_everything
from rdkit import Chem
from rdkit.Chem import Descriptors

from boltz.data import const
from boltz.data.module.inferencev2 import Boltz2InferenceDataModule
from boltz.data.types import AffinityInfo, Manifest, Record, StructureV2
from boltz.data.write.writer import BoltzAffinityWriter
from boltz.main import Boltz2DiffusionParams, BoltzSteeringParams, MSAModuleArgs, PairformerArgsV2
from boltz.model.models.boltz2 import Boltz2


def _chain_name_matches(candidate: str, requested: str) -> bool:
    cand = str(candidate or "").strip().upper()
    req = str(requested or "").strip().upper()
    if not cand or not req:
        return False
    return cand == req or cand.startswith(f"{req}X") or req.startswith(f"{cand}X")


def _load_manifest_record(processed_dir: Path, record_id: str) -> tuple[Manifest, Record]:
    manifest_path = processed_dir / "manifest.json"
    manifest = Manifest.load(manifest_path)
    for record in manifest.records:
        if record.id == record_id:
            return manifest, record
    raise KeyError(f"Record {record_id!r} not found in {manifest_path}")


def _select_affinity_ligand_chain(
    record: Record,
    requested_ligand_chain_id: str | None,
) -> object:
    ligand_chains = [
        chain
        for chain in record.chains
        if int(chain.mol_type) == const.chain_type_ids["NONPOLYMER"] and bool(chain.valid)
    ]
    if requested_ligand_chain_id:
        requested = str(requested_ligand_chain_id).strip()
        matches = [
            chain
            for chain in ligand_chains
            if _chain_name_matches(str(chain.chain_name).strip(), requested)
        ]
        if len(matches) == 1:
            return matches[0]
        available = [str(chain.chain_name) for chain in ligand_chains]
        raise ValueError(
            f"Requested ligand chain {requested!r} not found in record {record.id}. "
            f"Available ligand chains: {available or 'none'}."
        )
    if len(ligand_chains) != 1:
        available = [str(chain.chain_name) for chain in ligand_chains]
        raise ValueError(
            "Affinity prediction currently requires exactly one ligand chain when "
            "--ligand_chain is not provided. "
            f"Record={record.id}, available ligand chains={available or 'none'}."
        )
    return ligand_chains[0]


def _residue_names_for_chain(processed_dir: Path, record_id: str, chain_id: int) -> list[str]:
    structure = StructureV2.load(processed_dir / "structures" / f"{record_id}.npz")
    names: list[str] = []
    for chain in structure.chains:
        if int(chain["asym_id"]) != int(chain_id):
            continue
        res_start = int(chain["res_idx"])
        res_end = res_start + int(chain["res_num"])
        for residue in structure.residues[res_start:res_end]:
            name = str(residue["name"] or "").strip()
            if name:
                names.append(name)
        break
    return names


def _load_mol_from_processed_cache(processed_dir: Path, record_id: str, residue_names: list[str]) -> Chem.Mol | None:
    mols_path = processed_dir / "mols" / f"{record_id}.pkl"
    if not mols_path.exists():
        return None
    with mols_path.open("rb") as handle:
        payload = pickle.load(handle)  # noqa: S301
    if not isinstance(payload, dict):
        return None
    normalized_residue_names = {str(name or "").strip() for name in residue_names if str(name or "").strip()}
    for key, value in payload.items():
        key_name = str(key or "").strip()
        if normalized_residue_names and key_name not in normalized_residue_names:
            continue
        if isinstance(value, Chem.Mol):
            return Chem.Mol(value)
    for value in payload.values():
        if isinstance(value, Chem.Mol):
            return Chem.Mol(value)
    return None


def _load_mol_from_boltz_cache(cache_dir: Path, residue_names: list[str]) -> Chem.Mol | None:
    normalized_residue_names = [str(name or "").strip() for name in residue_names if str(name or "").strip()]
    for residue_name in normalized_residue_names:
        mol_path = cache_dir / "mols" / f"{residue_name}.pkl"
        if not mol_path.exists():
            continue
        with mol_path.open("rb") as handle:
            payload = pickle.load(handle)  # noqa: S301
        if isinstance(payload, Chem.Mol):
            return Chem.Mol(payload)
        if isinstance(payload, dict):
            for value in payload.values():
                if isinstance(value, Chem.Mol):
                    return Chem.Mol(value)
    return None


def _resolve_affinity_ligand_mw(
    processed_dir: Path,
    record_id: str,
    ligand_chain_id: int,
    cache_dir: Path,
    reference_ligand_mol: Chem.Mol | None,
) -> float:
    mol: Chem.Mol | None = None
    if reference_ligand_mol is not None:
        mol = Chem.Mol(reference_ligand_mol)
    if mol is None:
        residue_names = _residue_names_for_chain(processed_dir, record_id, ligand_chain_id)
        mol = _load_mol_from_processed_cache(processed_dir, record_id, residue_names)
        if mol is None:
            mol = _load_mol_from_boltz_cache(cache_dir, residue_names)
    if mol is None:
        raise RuntimeError(
            "Failed to resolve ligand molecule for affinity MW calculation. "
            f"record_id={record_id}, ligand_chain_id={ligand_chain_id}"
        )
    mol_no_h = Chem.RemoveHs(Chem.Mol(mol))
    return float(Descriptors.MolWt(mol_no_h))


def prepare_affinity_record(
    *,
    processed_dir: Path,
    cache_dir: Path,
    record_id: str,
    requested_ligand_chain_id: str | None,
    reference_ligand_mol: Chem.Mol | None,
) -> dict[str, object]:
    manifest, record = _load_manifest_record(processed_dir, record_id)
    ligand_chain = _select_affinity_ligand_chain(record, requested_ligand_chain_id)
    ligand_mw = _resolve_affinity_ligand_mw(
        processed_dir=processed_dir,
        record_id=record_id,
        ligand_chain_id=int(ligand_chain.chain_id),
        cache_dir=cache_dir,
        reference_ligand_mol=reference_ligand_mol,
    )
    affinity_info = AffinityInfo(
        chain_id=int(ligand_chain.chain_id),
        mw=float(ligand_mw),
    )
    updated_records = [
        replace(existing_record, affinity=affinity_info)
        if existing_record.id == record_id
        else existing_record
        for existing_record in manifest.records
    ]
    updated_manifest = Manifest(records=updated_records)
    updated_manifest.dump(processed_dir / "manifest.json")
    return {
        "record_id": record_id,
        "ligand_chain_name": str(ligand_chain.chain_name),
        "ligand_chain_id": int(ligand_chain.chain_id),
        "ligand_mw": float(ligand_mw),
    }


def _load_affinity_result_json(output_dir: Path, record_id: str) -> Path:
    result_path = output_dir / record_id / f"affinity_{record_id}.json"
    if not result_path.exists():
        raise FileNotFoundError(f"Expected affinity result not found: {result_path}")
    return result_path


def _augment_affinity_result(
    result: dict[str, object],
    ligand_alignment: dict[str, object] | None,
) -> dict[str, object]:
    if not isinstance(ligand_alignment, dict):
        return result

    aligned_smiles = str(ligand_alignment.get("ligand_smiles") or "").strip()
    model_chain = str(ligand_alignment.get("model_ligand_chain_id") or "").strip()
    requested_chain = str(ligand_alignment.get("requested_ligand_chain_id") or "").strip()

    normalized_map: dict[str, str] = {}
    raw_map = ligand_alignment.get("ligand_smiles_map")
    if isinstance(raw_map, dict):
        for key, value in raw_map.items():
            key_norm = str(key or "").strip()
            value_norm = str(value or "").strip()
            if key_norm and value_norm:
                normalized_map[key_norm] = value_norm

    if aligned_smiles and model_chain and model_chain not in normalized_map:
        normalized_map[model_chain] = aligned_smiles
    if aligned_smiles and requested_chain and requested_chain not in normalized_map:
        normalized_map[requested_chain] = aligned_smiles

    if aligned_smiles:
        result["ligand_smiles"] = aligned_smiles
    if model_chain:
        result["model_ligand_chain_id"] = model_chain
    if requested_chain:
        result["requested_ligand_chain_id"] = requested_chain
        result["requested_ligand_chain"] = requested_chain
        result["ligand_chain"] = requested_chain
    if normalized_map:
        result["ligand_smiles_map"] = normalized_map

    return result


def _expand_pre_affinity_coords(feats: dict, multiplicity: int) -> object:
    coords = feats["coords"]
    if len(coords.shape) == 4:
        if coords.shape[1] != 1:
            raise RuntimeError(
                "Affinity passthrough currently expects a single pre-affinity conformer. "
                f"Got coords shape={tuple(coords.shape)}."
            )
        coords = coords[:, 0]
    if len(coords.shape) != 3:
        raise RuntimeError(f"Unexpected affinity coords shape: {tuple(coords.shape)}")
    return coords.repeat_interleave(int(multiplicity), dim=0)


def _install_passthrough_structure_sampler(model_module: Boltz2) -> None:
    def _passthrough_sample(self, s_trunk, s_inputs, feats, num_sampling_steps, atom_mask, multiplicity, max_parallel_samples, steering_args, diffusion_conditioning):  # noqa: ANN001
        del self, s_trunk, s_inputs, num_sampling_steps, atom_mask, max_parallel_samples, steering_args, diffusion_conditioning
        return {
            "sample_atom_coords": _expand_pre_affinity_coords(feats, int(multiplicity)),
            "diff_token_repr": None,
        }

    model_module.structure_module.sample = MethodType(_passthrough_sample, model_module.structure_module)


def run_affinity_prediction(
    *,
    processed_dir: Path,
    output_dir: Path,
    cache_dir: Path,
    record_id: str,
    accelerator: str,
    devices: int,
    affinity_refine: bool = False,
    checkpoint: Path | None = None,
    seed: int | None = None,
    num_workers: int = 0,
    trainer_precision: int | str | None = None,
    ligand_alignment: dict[str, object] | None = None,
) -> dict | None:
    if seed is not None:
        seed_everything(seed)

    cache_dir = cache_dir.expanduser().resolve()
    affinity_ckpt = (checkpoint or (cache_dir / "boltz2_aff.ckpt")).expanduser().resolve()
    if not affinity_ckpt.exists():
        print(f"[Warning] Affinity checkpoint not found: {affinity_ckpt}. Skipping affinity.")
        return None

    manifest, record = _load_manifest_record(processed_dir, record_id)
    if record.affinity is None:
        raise RuntimeError(
            f"Affinity requested for {record_id}, but manifest affinity metadata was not prepared."
        )

    pre_affinity_path = output_dir / record_id / f"pre_affinity_{record_id}.npz"
    if not pre_affinity_path.exists():
        raise FileNotFoundError(
            f"Missing pre-affinity structure snapshot required by Boltz2 affinity: {pre_affinity_path}"
        )

    manifest_filtered = Manifest(records=[record])

    template_dir = processed_dir / "templates"
    if not template_dir.exists():
        template_dir = None
    constraints_dir = processed_dir / "constraints"
    if not constraints_dir.exists():
        constraints_dir = None
    extra_mols_dir = processed_dir / "mols"
    if not extra_mols_dir.exists():
        extra_mols_dir = None

    data_module = Boltz2InferenceDataModule(
        manifest=manifest_filtered,
        target_dir=output_dir,
        msa_dir=processed_dir / "msa",
        mol_dir=cache_dir / "mols",
        num_workers=num_workers,
        constraints_dir=constraints_dir,
        template_dir=template_dir,
        extra_mols_dir=extra_mols_dir,
        override_method="other",
        affinity=True,
    )

    predict_affinity_args = {
        "recycling_steps": 5,
        "sampling_steps": 200 if affinity_refine else 1,
        "diffusion_samples": 3 if affinity_refine else 1,
        "max_parallel_samples": 1,
        "write_confidence_summary": False,
        "write_full_pae": False,
        "write_full_pde": False,
    }

    diffusion_params = Boltz2DiffusionParams()
    pairformer_args = PairformerArgsV2()
    msa_args = MSAModuleArgs(
        subsample_msa=False,
        num_subsampled_msa=1024,
        use_paired_feature=True,
    )
    steering_args = BoltzSteeringParams()
    steering_args.fk_steering = False
    steering_args.physical_guidance_update = False
    steering_args.contact_guidance_update = False

    model_module = Boltz2.load_from_checkpoint(
        affinity_ckpt,
        strict=True,
        predict_args=predict_affinity_args,
        map_location="cpu",
        diffusion_process_args=asdict(diffusion_params),
        ema=False,
        pairformer_args=asdict(pairformer_args),
        msa_args=asdict(msa_args),
        steering_args=asdict(steering_args),
        affinity_mw_correction=True,
        use_kernels=False,
    )
    model_module.eval()
    _install_passthrough_structure_sampler(model_module)

    pred_writer = BoltzAffinityWriter(
        data_dir=str(processed_dir / "structures"),
        output_dir=str(output_dir),
    )

    resolved_precision: int | str
    if trainer_precision is not None:
        resolved_precision = 32 if str(trainer_precision).strip() == "32" else trainer_precision
    else:
        resolved_precision = 32

    trainer = Trainer(
        default_root_dir=output_dir / "affinity",
        callbacks=[pred_writer],
        accelerator=accelerator,
        devices=devices,
        precision=resolved_precision,
        logger=False,
        enable_checkpointing=False,
        inference_mode=True,
    )

    print(
        "[Info] Running official Boltz2 affinity head "
        f"on pre_affinity coordinates for {record_id}."
    )
    trainer.predict(model_module, datamodule=data_module, return_predictions=False)

    affinity_result_path = _load_affinity_result_json(output_dir, record_id)
    result = json.loads(affinity_result_path.read_text())
    result = _augment_affinity_result(result, ligand_alignment=ligand_alignment)
    affinity_result_path.write_text(json.dumps(result, indent=2) + "\n")
    return result
