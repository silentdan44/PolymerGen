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
