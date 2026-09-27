"""Tests for the versioned JSON API and JSON Lines runner."""

import io
import json
from types import SimpleNamespace

from rdkit import Chem
from rdkit.Chem import AllChem

from polymer_lib import api, runner


def _request(action, payload, **extra):
    return {
        "api_version": "1.0",
        "action": action,
        "input": payload,
        **extra,
    }


def test_build_request_returns_mol_block_and_diagnostics(monkeypatch):
    mol = Chem.AddHs(Chem.MolFromSmiles("CC"))
    assert AllChem.EmbedMolecule(mol, randomSeed=11) == 0
    expected_kwargs = {
        "units": 2,
        "tacticity": "isotactic",
        "seed": 4,
    }

    def fake_build(smiles, **kwargs):
        assert smiles == "*CC*"
        assert kwargs == expected_kwargs
        return SimpleNamespace(
            success=True,
            molecule=mol,
            attempts=3,
            rollbacks=1,
            elapsed_seconds=0.25,
            failure_reason=None,
        )

    monkeypatch.setattr(api, "build_polymer", fake_build)
    response = api.handle_request(
        _request(
            "build",
            {"smiles": "*CC*", **expected_kwargs},
            request_id="test-1",
        )
    )

    assert response["ok"] is True
    assert response["request_id"] == "test-1"
    assert response["result"]["molecule"]["format"] == "mol"
    assert "V2000" in response["result"]["molecule"]["data"]
    assert response["result"]["diagnostics"]["attempts"] == 3


def test_invalid_contract_returns_stable_error():
    response = api.handle_request(_request("build", {"smiles": "*CC*", "units": 2, "target_atoms": 20}))
    assert response["ok"] is False
    assert response["error"]["code"] == "INVALID_INPUT"
    assert "mutually exclusive" in response["error"]["message"]


def test_unknown_input_keys_are_rejected():
    response = api.handle_request(_request("build", {"smiles": "*CC*", "units": 2, "unit": 2}))
    assert response["ok"] is False
    assert response["error"]["code"] == "INVALID_INPUT"
    assert "unit" in response["error"]["message"]


def test_composed_action_runs_build_push_off_and_pack(monkeypatch):
    calls = []

    def fake_build(payload):
        calls.append("build")
        seed = payload.get("seed")
        return {
            "molecule": {"format": "mol", "data": f"chain-{seed}"},
            "diagnostics": {"attempts": 1, "rollbacks": 0, "elapsed_seconds": 0.1},
        }

    def fake_push_off(payload):
        calls.append("push_off")
        return {"molecule": {"format": "mol", "data": f"relaxed-{payload['molecule']['data']}"}}

    def fake_pack(payload):
        calls.append("pack")
        assert len(payload["molecules"]) == 2
        return {"topology": {"format": "openff-topology-json", "data": "{}"}}

    monkeypatch.setattr(api, "_build", fake_build)
    monkeypatch.setattr(api, "_push_off", fake_push_off)
    monkeypatch.setattr(api, "_pack", fake_pack)
    response = api.handle_request(_request(
        "build_push_off_pack",
        {
            "build": {"smiles": "*CC*", "units": 2, "seed": 10},
            "chain_count": 2,
            "push_off": {"steps": 10},
            "pack": {"density": 0.3},
        },
    ))

    assert response["ok"] is True
    assert calls == ["build", "build", "push_off", "push_off", "pack"]
    assert response["result"]["build"]["molecules"][1]["data"] == "chain-11"
    assert response["result"]["push_off"]["molecules"][0]["data"] == "relaxed-chain-10"


def test_composed_action_names_failed_stage(monkeypatch):
    monkeypatch.setattr(api, "_build", lambda payload: {"molecule": {"format": "mol", "data": "chain"}, "diagnostics": {}})

    def fail_push_off(payload):
        raise api.APIRequestError("DEPENDENCY_UNAVAILABLE", "OpenMM is missing")

    monkeypatch.setattr(api, "_push_off", fail_push_off)
    response = api.handle_request(_request(
        "build_push_off_pack",
        {"build": {"smiles": "*CC*", "units": 1}},
    ))

    assert response["ok"] is False
    assert response["error"]["code"] == "PIPELINE_STAGE_FAILED"
    assert "push_off stage failed" in response["error"]["message"]


def test_runner_handles_multiple_jsonl_requests_and_malformed_json(monkeypatch, capsys):
    monkeypatch.setattr(
        "sys.stdin",
        io.StringIO(
            '{"api_version":"1.0","action":"future","input":{}}\n'
            'not-json\n'
            '\n'
            '{"api_version":"1.0","action":"future","input":{}}\n'
        ),
    )
    runner.main()
    responses = [json.loads(line) for line in capsys.readouterr().out.splitlines()]

    assert len(responses) == 3
    assert responses[0]["error"]["code"] == "UNKNOWN_ACTION"
    assert responses[1]["error"]["code"] == "INVALID_JSON"
    assert responses[2]["error"]["code"] == "UNKNOWN_ACTION"
