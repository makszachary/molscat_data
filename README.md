# molscat_data
**molscat_data** is a package for post-processing the output files from [molscat](https://github.com/molscat/molscat) – a package for quantum scattering calculations (Version 2020.0).

It allows for collecting the scattering matrices (S-matrices) from the molscat output files and organising them into multi-layer collections in order to extract higher-order scattering quantities such as cross sections, rate constants and probabilities of the given types of collisions.

## Installation

1. `git clone https://github.com/makszachary/molscat_data.git`

2. in the BARW directory:  
**Windows**  
`py -m venv venv`  
`venv/Scripts/activate`   
**Linux**  
`python -m venv venv`  
`source venv/bin/activate`  

3. `py -m pip install --upgrade pip`   
`pip install -r requirements.txt`
