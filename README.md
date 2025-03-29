# Research data for "The structure and topology of an amorphous metal–organic framework"

<div align="center">
    
> **The structure and topology of an amorphous metal–organic framework**\
> _[Thomas C. Nicholas](https://tcnicholas.github.io/), Daniel F. Thomas du Toit, Louise A. M. Rosset, Davide M. Proserpio, [Andrew L. Goodwin](https://goodwingroupox.uk/), and [Volker L. Deringer](http://deringer.chem.ox.ac.uk)_

</div>

---

## Repository overview

This repository accompanies a manuscript to be submitted in due course, and 
contains data and structural models analysed and discussed therein.

### Data


- **MLIP model parameters**
  
  - Final ACE MLIP potential: [`zif-ace-24.yaml`](data/mlips/zif-ace-24.yaml).
  - Additional crystalline-upweighted ACE models (as described in Supplementary Note 1).
  - GAP MLIP parameter files exceed GitHub's file-size limitations and will be deposited in the accompanying Zenodo repository upon publication.


- **Structural models and analysis data**
  
  - Five independent $a$-ZIF models analysed in the manuscript are provided in [`data/structures/zif`](data/structures/zif). Files follow the naming convention:
    - `m<model-number>-zif_ace_24-anneal_1500K_5ns.data`

  - For each $a$-ZIF, $a$-Si, and $a$-SiO<sub>2</sub> model, additional topology files are included:
    - `<model-name>-topology.data`: Contains node–node and node–linker edge data.
    - `<model-name>-topology.dump`: Provides average ring-size data per node.

  - These topology files were used to create the structural visualisations presented in Figure 3 of the manuscript.

  - Crystallographic Geometry Definition (CGD) files are included for each model, providing node-only representations with explicit connectivity.

  - Raw local-properties data for each $a$-ZIF model, as well as the crystalline ZIF structures, are provided in [`data/raw`](data/raw).

  - Experimental and calculated total scattering functions are included in [`data/scattering`](data/scattering).


### src
  
  - **Simulation scripts**
    - Scripts used to perform HRMC refinements are provided in the [`pyhrmc`](src/pyhrmc) directory.

  - **Topology analysis scripts**
    - Topological analysis routines are provided in the [`amorphous_mof`](src/amorphous_mof) directory. These scripts interface with the Julia module [`RingStatistics.jl`](src/RingStatistics/src/RingStatistics.jl) *via* [PyCall](https://github.com/JuliaPy/PyCall.jl).
    - Internally, these routines utilise the [`CrystalNets.jl`](https://github.com/coudertlab/CrystalNets.jl) and [`PeriodicGraphs.jl`](https://github.com/Liozou/PeriodicGraphs.jl) packages.
    - To facilitate further development and usage of topology metrics, we have also released the standalone [`topo-metrics`](https://github.com/tcnicholas/topo-metrics) package.

---

## Working environment

```bash
# install Julia (if you haven't already)...
curl -fsSL https://install.julialang.org | sh

# install dependencies...
uv sync

# install `amorphous-mof` package.
uv pip install -e .
```

Please note, if you are using Juypter Notebooks in VSCode, we strongly reccomend
installing the Julia extension (vscode:extension/julialang.language-julia), 
which helps to configure paths to the Julia executable, etc.
