# Sulfur equilibrium models

Equilibrium modelling for sulfur vapours, with data from *ab initio* calculations.
The data required for modelling is provided in JSON files under **data/**.
Raw data from *ab initio* calculations will be made available shortly.

A pre-print of the accompanying academic paper is available at arXiv.org: [arXiv:1509.00722](http://arxiv.org/abs/1509.00722)

## Usage
* The core functions are implemented as a Python library in the `sulfur` folder.

* Energy and frequency data from density functional theory (DFT) calculations is stored in ASE databases in JSON format in the `data` directory.

* Scripts in the `scripts` folder can be used to explore the model:
  * `scripts/plots.py` generates all the plots in the academic paper from the .json data files.
  * `frequency_table.py` writes a summary table of all the frequency data to standard output.
  * `scripts/gen_sulfur_db.py` is the script used to generate .json data files from the collected _ab initio_ data (to be made available separately with DOI:10.6084/m9.figshare.1566812)


## Requirements

* Python 2.7

* Atomic Simulation Environment (ASE). https://wiki.fysik.dtu.dk/ase/index.html
  * Tested with version 3.8.1
  * At the time of writing, the development version 3.9.x is not compatible due to problems with the JSON database features

## Citations

* The data files data/S2.dat and data/S8.dat contain data from Chase, M. W. J. NIST-JANAF Thermochemical Tables, Fourth Edition. J. Phys. Chem. Ref. Data, Monogr. 9, 1–1951 (1998).
  They are also used in http://github.com/WMD-Bath/CZTS-model
  
## License

Use at your own risk. This work is licensed under the GNU Public License, version 3.
