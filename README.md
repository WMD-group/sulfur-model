# Sulfur equilibrium models

Equilibrium modelling for sulfur vapours, with data from *ab initio* calculations.
The data required for modelling is provided in JSON files under **data/**.
Raw data from *ab initio* calculations will be made available shortly.

This is work in progress; no claims are made regarding correctness or accuracy at this stage.

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
