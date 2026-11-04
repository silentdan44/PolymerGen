"""Isolated OpenFF/Packmol attempt; launched by the packer with a timeout."""
import json
import sys
from pathlib import Path

from openff.interchange.components import _packmol
from openff.toolkit import Molecule
from openff.units import unit
from packaging.version import Version


def main(input_file, output_file):
    request = json.loads(Path(input_file).read_text())
    molecules = [Molecule.from_json(item) for item in request['molecules']]
    original = _packmol._build_input_file

    def build_input(*args, **kwargs):
        filename, output = original(*args, **kwargs)
        path = Path(filename)
        # Packmol's default random seed is fixed; make retries distinct while
        # keeping them reproducible when the caller supplies a seed.
        options = f"seed {request['seed']}\n"
        # When every input chain is its own Packmol structure, the move-bad
        # heuristic may move too much at once. Alternate it on later attempts.
        if request['disable_movebad'] and _packmol._get_packmol_version() >= Version('21.0.1'):
            options += 'disable_movebad\n'
        path.write_text(options + path.read_text())
        return filename, output

    _packmol._build_input_file = build_input
    topology = _packmol.pack_box(
        molecules=molecules, number_of_copies=[1] * len(molecules),
        target_density=request['density'] * unit.gram / unit.milliliter,
        tolerance=request['tolerance'] * unit.angstrom,
        box_shape=_packmol.UNIT_CUBE, working_directory=request['workdir'],
        retain_working_files=False,
    )
    Path(output_file).write_text(topology.to_json())


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
