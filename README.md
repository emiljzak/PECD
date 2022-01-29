<div align="center">
  <img src="https://github.com/CFEL-CMI/PECD/blob/develop/temp_logo.png" height="200px"/>
</div>

# Quantum-mechanical calculations of photo-electron circular-dichroism (PECD)

[**Overview**](#overview)
| [**Quick Start**](#quick-start)
| [**Documentation**](https://readthedocs.org/projects/pecd-personal/latest/)


PECD is currently being developed and maintained by theory team of the Controlled Molecule Imaging group (https://www.controlled-molecule-imaging.org/), Center for Free-Electron Laser Science at Deutsches Elektronen-Synchrotron. This readme file is under construction with updates coming really soon.


## Quick start

To run the code type `python3 slurm_grid_run.py`. Choose `jobtype` 	= "local" or "maxwell" for a local execution or a job on SLURM engine, respectively. Provide name for the input file in `slurm_grid_run.py`.

Input file is given in xxx_input.py, where xxx is molecule name, e.g. n2, d2s, etc. Initially choose     `params['mode']      = 'propagate_grid' ` in input file to run TDSE propagation. Next re-run with `    params['mode']      = 'analyze_grid'` to analyse the created wavepackets. Finally run `ANALYZE_MPAD.py` to calculate PECD and other quantities.

## Documentation
