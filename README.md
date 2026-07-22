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

```bash
pip install rdkit numpy scipy numba
conda install -c conda-forge openff-toolkit openff-interchange openff-nagl openff-nagl-models openmm packmol
```
## Quick Start
### 1. Build a Single Polymer Chain
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

polymer = polymerizer.build()
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
    chain = polymerizer.build()
    chains.append(chain)

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