Pyboo
=====

A Python package to compute bond orientational order parameters as
defined by Steinhardt Physical Review B (1983)
doi:10.1103/PhysRevB.28.784.

Steinhardt's bond orientational order parameter is a popular method
(>20k citations of the original paper) to identify local symmetries in
an assembly of particles in 3D. It can be used in particle-based
simulations (typically molecular dynamics, brownian dynamics,
monte-carlo, etc.) or in particle-tracking experiments (colloids,
granular materials) where the coordinates of all particles are known.

Licence, citations, contact
---------------------------

This code is under GPL 3.0 licence. See LICENCE file.

Please cite Pyboo and it's author(s) in any scientific publication using
this software.

::

    @misc{
        pyboo, 
        title={Pyboo: A Python package to compute bond orientational order parameters},
        author={Mathieu Leocmach}, 
        year={2017},
        doi={10.5281/zenodo.1066568},
        url={https://github.com/MathieuLeocmach/pyboo}
    }

Contact Mathieu LEOCMACH, Institut Lumière Matière, UMR-CNRS 5306, Lyon,
France mathieu.leocmach AT univ-lyon1.fr

Installation
------------

Dependencies are numpy, scipy and numba. Tested with python 2.7 and
python 3.5.

You can install with pip: pip install pyboo

Documentation
-------------

Documentation is avaiable on Readthedocs: [http://pyboo.readthedocs.io]
