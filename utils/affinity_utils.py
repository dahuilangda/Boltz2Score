from __future__ import annotations

import json
import os
from pathlib import Path


def run_affinity_prediction(
    complex_file: Path,
    output_dir: Path,
    cache_dir: Path,
    result_id: str,
    accelerator: str,
    devices: int,
    affinity_refine: bool = False,
    seed: int | None = None,
    work_dir: Path | None = None,
    ligand_alignment: dict[str, object] | None = None,
) -> dict | None:
    try:
        import sys

        repo_root = Path(__file__).resolve().parents[2]
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))
        import affinity.main as affinity_main

        if getattr(affinity_main, "_ccd_name_manager", None) is not None:
            affinity_main._ccd_name_manager.redis_client = None
        Boltzina = affinity_main.Boltzina
    except Exception as exc:  # noqa: BLE001
        print(f"[Warning] Failed to import affinity module: {exc}")
        return None

    affinity_ckpt = cache_dir / "boltz2_aff.ckpt"
    if not affinity_ckpt.exists():
        print(
            f"[Warning] Affinity checkpoint not found: {affinity_ckpt}. Skipping affinity."
        )
        return None

    os.environ["BOLTZ_CACHE"] = str(cache_dir)

    affinity_out = output_dir / "affinity"
    affinity_work = (work_dir / "affinity_work") if work_dir else (affinity_out / "work")
    affinity_out.mkdir(parents=True, exist_ok=True)
    affinity_work.mkdir(parents=True, exist_ok=True)

    boltzina = Boltzina(
        output_dir=str(affinity_out),
        work_dir=str(affinity_work),
        skip_run_structure=not affinity_refine,
        use_kernels=False,
        run_trunk_and_structure=True,
        accelerator=accelerator,
        devices=devices,
        num_workers=0,
        seed=seed,
    )
    print(
        f"[Info] Running affinity with "
        f"{'diffusion refinement' if affinity_refine else 'input-structure scoring'} "
        f"(seed={seed if seed is not None else 'none'})."
    )

    boltzina.predict([str(complex_file)])
    if not boltzina.results:
        return None

    result = dict(boltzina.results[0])
    if isinstance(ligand_alignment, dict):
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

        atom_name_keys = ligand_alignment.get("ligand_atom_name_keys")
        if isinstance(atom_name_keys, list):
            normalized_name_keys = [
                str(item or "").strip() for item in atom_name_keys if str(item or "").strip()
            ]
            if normalized_name_keys:
                result["ligand_atom_names"] = normalized_name_keys
                result["ligand_atom_name_keys"] = normalized_name_keys
                result["ligand_atom_input_order_names"] = normalized_name_keys

        atom_name_keys_by_chain = ligand_alignment.get("ligand_atom_name_keys_by_chain")
        if isinstance(atom_name_keys_by_chain, dict):
            normalized_name_keys_by_chain: dict[str, list[str]] = {}
            for key, value in atom_name_keys_by_chain.items():
                chain_key = str(key or "").strip()
                if not chain_key or not isinstance(value, list):
                    continue
                normalized_values = [
                    str(item or "").strip()
                    for item in value
                    if str(item or "").strip()
                ]
                if normalized_values:
                    normalized_name_keys_by_chain[chain_key] = normalized_values
            if normalized_name_keys_by_chain:
                result["ligand_atom_name_keys_by_chain"] = normalized_name_keys_by_chain
                result["ligand_atom_names_by_chain"] = normalized_name_keys_by_chain

        for list_key in (
            "ligand_atom_input_order_plddts",
            "ligand_atom_smiles_order_names",
            "ligand_atom_smiles_order_plddts",
            "ligand_atom_smiles_to_input_index",
        ):
            raw_values = ligand_alignment.get(list_key)
            if isinstance(raw_values, list):
                normalized_values: list[object] = []
                for item in raw_values:
                    if list_key.endswith("_names"):
                        text = str(item or "").strip()
                        if text:
                            normalized_values.append(text)
                    elif list_key.endswith("_index"):
                        try:
                            normalized_values.append(int(item))
                        except Exception:
                            continue
                    else:
                        try:
                            normalized_values.append(float(item))
                        except Exception:
                            continue
                if normalized_values:
                    result[list_key] = normalized_values

        for scalar_key in (
            "ligand_plddt_mean",
            "ligand_atom_plddt_min",
            "ligand_atom_plddt_p10",
            "ligand_atom_plddt_p25",
            "ligand_atom_plddt_median",
            "ligand_atom_plddt_p75",
            "ligand_atom_plddt_max",
            "ligand_atom_plddt_std",
            "ligand_atom_plddt_fraction_ge_50",
            "ligand_atom_plddt_fraction_ge_70",
        ):
            value = ligand_alignment.get(scalar_key)
            try:
                result[scalar_key] = float(value)
            except Exception:
                continue

        atom_plddt_by_name = ligand_alignment.get("ligand_atom_plddts_by_chain_and_name")
        if isinstance(atom_plddt_by_name, dict):
            normalized_plddt_by_name: dict[str, dict[str, float]] = {}
            for chain_key, atom_map in atom_plddt_by_name.items():
                chain = str(chain_key or "").strip()
                if not chain or not isinstance(atom_map, dict):
                    continue
                normalized_atom_map: dict[str, float] = {}
                for atom_key, atom_value in atom_map.items():
                    atom_name = str(atom_key or "").strip()
                    if not atom_name:
                        continue
                    try:
                        normalized_atom_map[atom_name] = float(atom_value)
                    except Exception:
                        continue
                if normalized_atom_map:
                    normalized_plddt_by_name[chain] = normalized_atom_map
            if normalized_plddt_by_name:
                result["ligand_atom_plddts_by_chain_and_name"] = normalized_plddt_by_name

    result["input_file"] = str(complex_file)
    affinity_json = output_dir / result_id / f"affinity_{result_id}.json"
    affinity_json.parent.mkdir(parents=True, exist_ok=True)
    affinity_json.write_text(json.dumps(result, indent=2))
    return result
