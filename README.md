# pyboo
A Python package to compute bond orientational order parameters as defined by Steinhardt Physical Review B (1983) doi:10.1103/PhysRevB.28.784.

Steinhardt's bond orientational order parameter is a popular method (>20k citations of the original paper) to identify local symmetries in an assembly of particles in 3D. It can be used in particle-based simulations (typically molecular dynamics, brownian dynamics, monte-carlo, etc.) or in particle-tracking experiments (colloids, granular materials) where the coordinates of all particles are known.

## Licence, citations, contact

This code is under GPL 3.0 licence. See LICENCE file.

Please cite Pyboo and it's author(s) in any scientific publication using this software.

@misc{
    pyboo, 
    title={Pyboo: A Python package to compute bond orientational order parameters},
    author={Mathieu Leocmach}, 
    year={2017}
}

Contact
    Mathieu LEOCMACH, Institut Lumière Matière, UMR-CNRS 5306, Lyon, France
    mathieu.leocmach AT univ-lyon1.fr
    

## Installation

Dependencies are numpy, scipy and numba. Tested with python 2.7 and python 3.5.

You can install with pip:
    pip install pyboo


## Input

The present library takes as input a (N,3) array of float coordinates named `pos` and a (M,2) array of integers named `bonds` that defines the bond network. If `bonds` contains the couple (10,55) it means that there is a bond between the particles which coordinates can be found at `pos[10]` and `pos[55]`. The bonds are supposed unique and bidirectional, therefore if the bond (10,55) is in `bonds`, the bond (55,10) exists *implicitely* and should not be part of `bonds`.

The library is agnostic on how the bonds were determined. Possible choices are (among others):
 - two particles closer than a maximum distance,
 - the k nearest neighbours of a particle,
 - Delaunay triangulation.
 
Other libraries have very efficient implementations of theses methods. See for example `scipy.spatial.KDTree` for fast spatial query or `scipy.spatial.Delaunay`.
