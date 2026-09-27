"""Versioned JSON API for running polymer generation jobs."""

from __future__ import annotations

import io
import json
import logging
import re
from typing import Any

from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

from .polymerizer import BuildConfig, build_polymer
from .push_off import push_off_chain

API_VERSION = "1.0"
_REQUEST_ID_RE = re.compile(r"^[A-Za-z0-9._:-]{1,128}$")


class APIRequestError(Exception):
    """A client error with a stable API error code."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def _molecule_from_input(value: Any, field: str) -> Chem.Mol:
    if not isinstance(value, dict):
        raise APIRequestError("INVALID_INPUT", f"{field} must be a molecule object")
    fmt, data = value.get("format"), value.get("data")
    if not isinstance(data, str) or not data.strip():
        raise APIRequestError("INVALID_INPUT", f"{field}.data must be a nonempty string")

    if fmt == "smiles":
        mol = Chem.MolFromSmiles(data)
        if mol is not None:
            mol = Chem.AddHs(mol)
    elif fmt == "mol":
        mol = Chem.MolFromMolBlock(data, removeHs=False, sanitize=True)
    elif fmt == "sdf":
        supplier = Chem.ForwardSDMolSupplier(
            io.BytesIO(data.encode("utf-8")), removeHs=False, sanitize=True
        )
        records = list(supplier)
        if len(records) != 1:
            raise APIRequestError("INVALID_INPUT", f"{field} SDF must contain exactly one molecule")
        mol = records[0]
    else:
        raise APIRequestError("INVALID_INPUT", f"{field}.format must be 'smiles', 'mol', or 'sdf'")

    if mol is None:
        raise APIRequestError("INVALID_MOLECULE", f"Could not parse {field} molecule")
    if mol.GetNumConformers() == 0:
        raise APIRequestError("INVALID_MOLECULE", f"{field} must include 3D coordinates")
    return mol


def _mol_output(mol: Chem.Mol) -> dict[str, str]:
    return {"format": "mol", "data": Chem.MolToMolBlock(mol)}


def _validate_keys(value: dict, allowed: set[str], field: str) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise APIRequestError("INVALID_INPUT", f"Unknown {field} field(s): {', '.join(unknown)}")


def _build(payload: dict) -> dict:
    _validate_keys(
        payload,
        {
            "smiles", "units", "target_atoms", "tacticity", "tacticity_center",
            "optimizer", "optimizer_options", "config", "seed",
        },
        "input",
    )
    if not isinstance(payload.get("smiles"), str):
        raise APIRequestError("INVALID_INPUT", "input.smiles is required and must be a string")
    kwargs = {key: payload[key] for key in (
        "units", "target_atoms", "tacticity", "tacticity_center", "optimizer",
        "optimizer_options", "seed",
    ) if key in payload}
    if "config" in payload:
        if not isinstance(payload["config"], dict):
            raise APIRequestError("INVALID_INPUT", "input.config must be an object")
        try:
            kwargs["config"] = BuildConfig(**payload["config"])
        except (TypeError, ValueError) as exc:
            raise APIRequestError("INVALID_INPUT", f"Invalid input.config: {exc}") from exc
    try:
        result = build_polymer(payload["smiles"], **kwargs)
    except (ValueError, TypeError) as exc:
        raise APIRequestError("INVALID_INPUT", str(exc)) from exc
    except ImportError as exc:
        raise APIRequestError("DEPENDENCY_UNAVAILABLE", str(exc)) from exc
    if not result.success:
        raise APIRequestError("BUILD_FAILED", result.failure_reason or "Polymerization failed")
    mol = result.molecule
    return {
        "molecule": _mol_output(mol),
        "properties": {
            "atoms": mol.GetNumAtoms(),
            "heavy_atoms": mol.GetNumHeavyAtoms(),
            "formula": rdMolDescriptors.CalcMolFormula(mol),
        },
        "diagnostics": {
            "attempts": result.attempts,
            "rollbacks": result.rollbacks,
            "elapsed_seconds": result.elapsed_seconds,
        },
    }


def _push_off(payload: dict) -> dict:
    allowed = {
        "molecule", "forcefield", "charge_method", "steps", "stages", "temperature",
        "timestep", "friction", "amplitude", "cutoff_scale", "seed", "platform",
        "soft_minimize", "soft_minimize_max_iters", "soft_minimize_tolerance",
        "minimize", "max_iters", "tolerance",
    }
    _validate_keys(payload, allowed, "input")
    mol = _molecule_from_input(payload.get("molecule"), "input.molecule")
    options = {key: value for key, value in payload.items() if key != "molecule"}
    try:
        relaxed = push_off_chain(mol, **options)
    except (ValueError, TypeError) as exc:
        raise APIRequestError("INVALID_INPUT", str(exc)) from exc
    except ImportError as exc:
        raise APIRequestError("DEPENDENCY_UNAVAILABLE", str(exc)) from exc
    return {
        "molecule": _mol_output(relaxed),
        "properties": {
            "atoms": relaxed.GetNumAtoms(),
            "heavy_atoms": relaxed.GetNumHeavyAtoms(),
            "formula": rdMolDescriptors.CalcMolFormula(relaxed),
        },
    }


def _pack(payload: dict) -> dict:
    _validate_keys(payload, {"molecules", "density", "max_attempts", "tolerance"}, "input")
    molecules = payload.get("molecules")
    if not isinstance(molecules, list) or not molecules:
        raise APIRequestError("INVALID_INPUT", "input.molecules must be a nonempty list")
    mols = [_molecule_from_input(item, f"input.molecules[{i}]") for i, item in enumerate(molecules)]
    options = {key: payload[key] for key in ("density", "max_attempts", "tolerance") if key in payload}
    try:
        from .packer import pack_chains

        topology = pack_chains(mols, **options)
        openmm_topology = topology.to_openmm()
        positions = topology.get_positions().to_openmm()
        from openmm.app import PDBFile

        pdb_file = io.StringIO()
        PDBFile.writeFile(openmm_topology, positions, pdb_file, keepIds=True)
        topology_json = topology.to_json()
        # Validate that this OpenFF version emitted JSON before putting it on the wire.
        json.loads(topology_json)
    except (ValueError, TypeError) as exc:
        raise APIRequestError("INVALID_INPUT", str(exc)) from exc
    except ImportError as exc:
        raise APIRequestError("DEPENDENCY_UNAVAILABLE", str(exc)) from exc
    return {
        "topology": {"format": "openff-topology-json", "data": topology_json},
        "structure": {"format": "pdb", "data": pdb_file.getvalue()},
        "properties": {"molecules": len(mols), "atoms": topology.n_atoms},
    }


def handle_request(request: Any) -> dict:
    """Validate and execute one API request, returning a JSON-compatible object."""
    request_id = request.get("request_id") if isinstance(request, dict) else None
    if not isinstance(request_id, str) or not _REQUEST_ID_RE.fullmatch(request_id):
        request_id = None
    base = {"api_version": API_VERSION, "request_id": request_id}
    try:
        if not isinstance(request, dict):
            raise APIRequestError("INVALID_REQUEST", "Request must be a JSON object")
        _validate_keys(request, {"api_version", "request_id", "action", "input"}, "request")
        if request.get("api_version") != API_VERSION:
            raise APIRequestError("UNSUPPORTED_VERSION", f"api_version must be '{API_VERSION}'")
        if "request_id" in request and request_id is None:
            raise APIRequestError("INVALID_REQUEST", "request_id must be 1-128 safe ASCII characters")
        action = request.get("action")
        payload = request.get("input")
        if not isinstance(payload, dict):
            raise APIRequestError("INVALID_REQUEST", "input must be a JSON object")
        handlers = {"build": _build, "push_off": _push_off, "pack": _pack}
        if action not in handlers:
            raise APIRequestError("UNKNOWN_ACTION", "action must be one of: build, push_off, pack")
        result = handlers[action](payload)
        return {**base, "ok": True, "result": result}
    except APIRequestError as exc:
        return {**base, "ok": False, "error": {"code": exc.code, "message": str(exc)}}
    except Exception as exc:  # Keep one failed job from terminating a JSONL runner.
        logging.getLogger(__name__).exception("Unhandled API failure")
        return {
            **base,
            "ok": False,
            "error": {"code": "INTERNAL_ERROR", "message": "Request failed; see runner stderr logs"},
        }
