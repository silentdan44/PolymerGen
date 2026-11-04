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
        "conformer_count": 3,
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


def test_composed_action_runs_one_build_with_push_off_options_per_chain(monkeypatch):
    calls = []
    def fake_build(payload):
        calls.append(payload)
        seed = payload['seed']
        return {"molecule": {"format": "mol", "data": f"relaxed-{seed}"},
                "unrelaxed_molecule": {"format": "mol", "data": f"raw-{seed}"},
                "diagnostics": {}}
    def fake_pack(payload):
        assert [m['data'] for m in payload['molecules']] == ['relaxed-10', 'relaxed-11']
        return {"topology": {"format": "openff-topology-json", "data": "{}"}}
    monkeypatch.setattr(api, '_build', fake_build)
    monkeypatch.setattr(api, '_pack', fake_pack)
    response = api.handle_request(_request('build_push_off_pack', {
        'build': {'smiles': '*CC*', 'units': 2, 'seed': 10}, 'chain_count': 2,
        'push_off': {'steps': 10, 'seed': 20}, 'pack': {'density': 0.3},
    }))
    assert response['ok']
    assert [c['push_off_options']['seed'] for c in calls] == [20, 21]
    assert response['result']['build']['molecules'][1]['data'] == 'raw-11'
    assert response['result']['push_off']['molecules'][0]['data'] == 'relaxed-10'


def test_composed_action_names_failed_stage(monkeypatch):
    def fail(payload):
        raise api.APIRequestError('DEPENDENCY_UNAVAILABLE', 'OpenMM is missing')
    monkeypatch.setattr(api, '_build', fail)
    response = api.handle_request(_request('build_push_off_pack',
                                           {'build': {'smiles': '*CC*', 'units': 1}}))
    assert response['error']['code'] == 'PIPELINE_STAGE_FAILED'
    assert 'build/push_off stage failed' in response['error']['message']


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


def test_pdb_output_normalizes_numeric_residue_identifiers():
    from openmm import unit
    from openmm.app import Topology, element

    topology = Topology()
    chain = topology.addChain('A')
    residue = topology.addResidue('UNK', chain, id='1')
    topology.addAtom('C', element.carbon, residue)
    residue.id = 1
    output = io.StringIO()
    api._write_pdb(topology, [[0, 0, 0]] * unit.nanometer, output)
    assert 'ATOM' in output.getvalue() or 'HETATM' in output.getvalue()
    assert residue.id == '1'


def test_pack_timeout_returns_actionable_error(monkeypatch):
    from polymer_lib.packer import PackmolPacker
    mol = Chem.AddHs(Chem.MolFromSmiles('CC'))
    assert AllChem.EmbedMolecule(mol, randomSeed=11) == 0
    def fail(self, chains):
        raise RuntimeError('Packmol timed out; try a longer timeout')
    monkeypatch.setattr(PackmolPacker, 'pack', fail)
    response = api.handle_request(_request('pack', {
        'molecules': [{'format': 'mol', 'data': Chem.MolToMolBlock(mol)}],
        'timeout': 2, 'seed': 1,
    }))
    assert response['error']['code'] == 'PACKING_FAILED'
    assert 'timed out' in response['error']['message']
