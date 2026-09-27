# Polymer Generator

A Python library for building 3D polymer structures with automatic retry/rollback logic, force field optimization, and export to multiple simulation formats (OpenMM, LAMMPS, GROMACS).

<p align="center">
  <img src="assets/poly.png" alt="Polymer Generator" width="400"/>
</p>

## Features

- **Monomer Building**: Parse SMILES with connection points (`*`) and generate 3D coordinates
- **Polymerization**: Self-avoiding random walk with retry/rollback mechanism
- **Geometry Optimization**: MMFF94s or OpenFF with NAGL charges
- **Packing**: Pack multiple chains into periodic boxes using Packmol
- **Export**: Save to OpenMM XML, LAMMPS data files, GROMACS topology, PDB
- **Performance**: Numba-accelerated geometry calculations

## Installation

The recommended setup uses [micromamba](https://mamba.readthedocs.io/en/latest/user_guide/micromamba.html).
Install micromamba once, then run these commands from the repository root.

### Core package (RDKit and MMFF)

```bash
micromamba create -f environment.yml
micromamba activate polymer-generator
python -c "from polymer_lib import Monomer, Polymerizer; print('polymer_lib is ready')"
```

The environment file installs the project in editable mode, so code changes are immediately available without reinstalling.

### Install the package from GitHub

After replacing `<github-user>` and `<repository>` with the GitHub owner and repository name, users can install the latest version directly:

```bash
python -m pip install "git+https://github.com/<github-user>/<repository>.git"
```

For a clean environment with the required scientific dependencies, create the micromamba environment first, then install from GitHub instead of the local checkout:

```bash
micromamba create -n polymer-generator -c conda-forge python=3.11 pip rdkit numpy scipy numba tqdm
micromamba run -n polymer-generator python -m pip install "git+https://github.com/<github-user>/<repository>.git"
```

### Full environment (OpenFF, NAGL and Packmol)

```bash
micromamba create -f environment-openff.yml
micromamba activate polymer-generator-openff
python -c "from polymer_lib import Monomer, Polymerizer, OpenFFOptimizer; from polymer_lib.packer import PackmolPacker; print('full environment is ready')"
```

To use only the OpenFF optimizer from an existing core environment, install the optional Python extra with `python -m pip install -e ".[openff]"`; Packmol support is provided by the full environment.

To run the test suite after adding tests, use `python -m pytest`.

## Quick Start
### Build a chain in one call

`build_polymer` handles monomer creation, repeat expansion, and polymerization. The result carries the molecule and run diagnostics.

```python
from polymer_lib import build_polymer

result = build_polymer('*CC(c1ccccc1)*', units=20, seed=7)
if not result.success:
    raise RuntimeError(result.failure_reason)

polymer = result.molecule
print(f"built in {result.elapsed_seconds:.1f}s; attempts={result.attempts}")
```

### Configure a build

Use `BuildConfig` to group geometric and retry settings. Set `seed` to make random rotations reproducible.

```python
from polymer_lib import BuildConfig, Polymerizer

polymerizer = Polymerizer.from_smiles(
    '*CC(c1ccccc1)*',
    count=20,
    config=BuildConfig(dist_min=0.7, retry=100, rollback=5),
    optimizer='mmff',
    optimizer_options={'max_iters': 100},
    seed=7,
)
result = polymerizer.build()
if result.success:
    polymer = result.molecule
```

The advanced API still accepts a list of `Monomer` objects. `build()` now returns a `BuildResult`; use `build_mol()` if existing code needs the older molecule-or-`None` return value.

### Pack chains

For default packing settings, use the helper function:

```python
from polymer_lib.packer import pack_chains

topology = pack_chains(chains, density=0.3)
```

Use `PackmolPacker` directly when you need to configure retry count or tolerance.

### Advanced: build a single polymer chain
```python
from polymer_lib import Monomer, Polymerizer


smiles = '*CC(c1ccccc1)*'  # Polystyrene
monomer = Monomer(smiles)
monomers_array = [monomer.copy() for _ in range(20)]

polymerizer = Polymerizer(
    monomers=monomers_array,
    random_rot=True,
    dist_min=0.7,
    retry=100,
    rollback=5,
    retry_step=200,
    optimizer='mmff'  # or 'openff' for higher accuracy
)

result = polymerizer.build()
polymer = result.molecule if result.success else None
```
### 2. Pack Multiple Chains
```python
from polymer_lib.packer import PackmolPacker


chains = []
for i in range(5):
    monomers_array = [monomer.copy() for _ in range(20)]
    polymerizer = Polymerizer(
        monomers=monomers_array,
        optimizer='mmff'
    )
    result = polymerizer.build()
    if not result.success:
        raise RuntimeError(result.failure_reason)
    chains.append(result.molecule)

packer = PackmolPacker(
    density=0.3,        # g/cm³
    max_attempts=5,
    tolerance=2.0       # Å
)

topology = packer.pack(chains)
```

### 3. Create OpenMM System
```python
from openff.toolkit.typing.engines.smirnoff import ForceField
from openff.interchange import Interchange
import openmm


ff = ForceField("openff-2.1.0.offxml")
interchange = Interchange.from_smirnoff(
    force_field=ff,
    topology=topology
)


system = interchange.to_openmm_system()
topology_omm = interchange.to_openmm_topology()
positions = interchange.positions

interchange.to_pdb("system.pdb")

with open("system.xml", "w") as f:
    f.write(openmm.XmlSerializer.serialize(system))
```

### 4. Export to LAMMPS and GROMACS
```python
# LAMMPS
interchange.to_lammps(prefix="system", include_type_labels=True)
# Creates: system.lmp (data file)

# GROMACS
interchange.to_gromacs(prefix="system", monolithic=True)
# Creates: system.top, system.gro
```

## Architecture
```
polymer_lib/
├── __init__.py              # Main exports
├── calc.py                  # Geometry calculations (Numba-accelerated)
├── utils.py                 # RDKit utilities and logging
├── poly.py                  # Core polymerization logic
├── monomer.py               # Monomer class
├── polymerizer.py           # Polymerizer class
├── packer.py                # Packmol wrapper
└── optimizers/              # Structure optimizers
    ├── base.py              # BaseOptimizer ABC
    ├── mmff.py              # MMFF94s optimizer
    └── openff.py            # OpenFF + NAGL optimizer
```

## Optimizers
### MMFF94s (Fast)
```python
polymerizer = Polymerizer(
    monomers=monomers_array,
    optimizer='mmff'
)
```
### OpenFF with NAGL Charges (Accurate)
```python
from polymer_lib import OpenFFOptimizer

optimizer = OpenFFOptimizer(
    forcefield='openff-2.1.0.offxml',
    max_iters=50,
    platform='CPU',
    charge_method='openff-gnn-am1bcc-1.0.0.pt'
)

polymerizer = Polymerizer(
    monomers=monomers_array,
    optimizer=optimizer
)
```
