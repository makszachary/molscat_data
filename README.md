# molscat_data
**molscat_data** is a package for post-processing the output files from [molscat](https://github.com/molscat/molscat) – a package for quantum scattering calculations (Version 2020.0).

It allows for collecting the scattering matrices (S-matrices) from the molscat output files and organising them into multi-layer collections in order to extract higher-order scattering quantities such as cross sections, rate constants and probabilities of the given types of collisions.

## Installation

1. `git clone https://github.com/makszachary/molscat_data.git`

2. in the molscat_data directory:  
**Windows**  
`py -m venv venv`  
`venv/Scripts/activate`   
**Linux**  
`python -m venv venv`  
`source venv/bin/activate`  

3. **Windows**
`py -m pip install --upgrade pip`
`pip install -U wheel --no-cache-dir`
`pip install -U setuptools --no-cache-dir`
`pip install numpy==1.26.3`   
`cd venv/Lib/site-packages`
`git clone https://github.com/makszachary/py3nj`
`pip install .`
`cd ../../../../`
`pip install -r requirements_windows.txt`

**Linux**
`python -m pip install --upgrade pip`
`pip install -U wheel --no-cache-dir`
`pip install -U setuptools --no-cache-dir`
`pip install numpy==1.26.3`   
`pip install -r requirements.txt`