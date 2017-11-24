Introduction to Pyboo
=====================

A Python package to compute bond orientational order parameters as defined by Steinhardt Physical Review B (1983) doi:10.1103/PhysRevB.28.784.

Steinhardt's bond orientational order parameter is a popular method (>20k citations of the original paper) to identify local symmetries in an assembly of particles in 3D. It can be used in particle-based simulations (typically molecular dynamics, brownian dynamics, monte-carlo, etc.) or in particle-tracking experiments (colloids, granular materials) where the coordinates of all particles are known.

Licence, citations, contact
---------------------------

This code is under GPL 3.0 licence. See LICENCE file.

Please cite Pyboo and it's author(s) in any scientific publication using this software.

::

    @misc{
        pyboo, 
        title={Pyboo: A Python package to compute bond orientational order parameters},
        author={Mathieu Leocmach}, 
        year={2017}
    }

Contact
    Mathieu LEOCMACH, Institut Lumière Matière, UMR-CNRS 5306, Lyon, France
    mathieu.leocmach AT univ-lyon1.fr
    

Installation
------------

Dependencies are numpy, scipy and numba. Tested with python 2.7 and python 3.5.

You can install with pip: ::

    pip install pyboo


Input
-----

The present library takes as input a (N,3) array of float coordinates named ``pos`` and a (M,2) array of integers named ``bonds`` that defines the bond network. If ``bonds`` contains the couple (10,55) it means that there is a bond between the particles which coordinates can be found at ``pos[10]`` and ``pos[55]``. The bonds are supposed unique and bidirectional, therefore if the bond (10,55) is in ``bonds``, the bond (55,10) exists *implicitely* and should not be part of ``bonds``.

The library is agnostic on how the bonds were determined. Possible choices are (among others):
 - two particles closer than a maximum distance,
 - the k nearest neighbours of a particle,
 - Delaunay triangulation.
 
Other libraries have very efficient implementations of theses methods. See for example :class:`scipy.spatial.KDTree` for fast spatial query or :class:`scipy.spatial.Delaunay`.

Spherical harmonics
-------------------

The 3D orientation of each bond can be projected on a base of spherical harmonics :math:`Y_{\ell m}(\theta,\phi)`, where :math:`\ell \geq 0` indicates the order of symmetry, and :math:`m`, with :math:`-\ell \geq m \geq \ell`, indicates the orientation with respect to the referent set of orthonormal axes. Since our bonds are bidirectional :math:`\ell` is even. Averaging these coefficients over a set of bonds yields a measure of the symmetry of this set.

For example the whole system is characterised by the coefficients 

.. math:: \bar{q}_{\ell m} = \frac{1}{M} \sum_{i,j} Y_{\ell m}(\theta_{ij},\phi_{ij})

where the average is taken over all the bonds. More locally, each particle :math:`i` is characterised by the coefficients

.. math:: q_{\ell m}(i) = \frac{1}{N_i}\sum_{0}^{N_i} Y_{\ell m}(\theta_{ij},\phi_{ij})

where there are :math:`N_i` bonds starting from :math:`i`. We call the :math:`q_{\ell m}` coefficients the local tensorial bond orientational order parameter.

The function :meth:`~boo.boo.bonds2qlm` computes the :math:`q_{\ell m}` coefficients for the :math:`\ell`-fold symetry. The extra parameter `periods` allows to do so in periodic boundary conditions.

Invarients
----------

The tensorial :math:`q_{\ell m}` coefficients are dependent of the orientation of the reference axis. That is why we have to compute quantities that are rotationally invarients:
 - the second order invarient indicates the strength of the :math:`\ell`-fold symetry.
 
  .. math:: q_\ell = \sqrt{\frac{4\pi}{2l+1} \sum_{m=-\ell}^{\ell} |q_{\ell m}|^2 }

 - the third order invarient allows to discriminate different types of :math:`\ell`-fold symetric structures.

  .. math:: w_\ell = \sum_{m_1+m_2+m_3=0} 
			\left( \begin{array}{ccc}
				\ell & \ell & \ell \\
				m_1 & m_2 & m_3 
			\end{array} \right)
			q_{\ell m_1} q_{\ell m_2} q_{\ell m_3}

  where the term in brackets is the Wigner 3-j symbol. For example :math:`w_6` allows to disctiminate icosahedral structures, see Leocmach & Tanaka, Nature Com. (2012) doi: 10.1038/ncomms1974.

Invarients can be computed respectively by :meth:`~boo.boo.ql` and :meth:`~boo.boo.wl`.

Coarse-graining
---------------

It is possible to coarse-grain the tensorial bond orientational order to get more information about the second shell of neighbours and thus discriminate better crystal structures, see Lechner & Delago J. Chem. Phys. (2008) doi:10.1063/1.2977970:

.. math::  Q_{\ell m}(i) = \frac{1}{N_i+1}\left( q_{\ell m}(i) +  \sum_{j=0}^{N_i} q_{\ell m}(j)\right)

here the central particle is included in the sum.

Coarse-graining can be done with :meth:`~boo.boo.coarsegrain_qlm`. The parameter ``inside`` is a (N) array of booleans indicating particles where the original :math:`q_{\ell m}` coefficients were truthfully determined. Counter examples (where ``inside`` takes the value ``False``) are particles that were too close to one edge of the experimental window, so that some of their neighbours were not dectected, causing a incomplete bond set. Coarse-grained invariants :math:`Q_\ell` and :math:`W_\ell` can be computed in the same way by :meth:`~boo.boo.ql` and :meth:`~boo.boo.wl` respectively.

Cross product
-------------

The similarity between the symetry and the orientation of two neighbourhoods can be estimated by the normalized scalar product

.. math:: s_\ell(i,j) = \frac{4\pi}{2\ell + 1} \frac{\sum_{m=-\ell}^{\ell} q_{\ell m}(i) q_{\ell m}^{*}(j)}{q_\ell(i) q_\ell(j)}

This quantity is the result of :meth:`~boo.boo.product` divided by ``ql(qlm1) * ql(qlm2)``. The similarity between all neighbouring particles can be obtained from :meth:`~boo.boo.bond_normed_product`.

Typical use: when :math:`s_6(i,j)` is larger than a threshold value (typically 0.7) the bond can be considered crystalline. A particle is considered crystalline when it has at least 7 crystalline bonds. See Auer & Frenkel, J.Chem.Phys. (2004) doi: 10.1063/1.1638740. This procedure is implemented in :meth:`~boo.boo.x_particles`.

In a more continuous manner, the crystallinity parameter is defined as the average of the cross products over the neighbours, see Russo & Tanaka, Sci Rep. (2012) doi:10.1038/srep00505.

.. math:: C_\ell(i) = \frac{1}{N_i} \sum_{j=0}{N_i} s_\ell (i,j)

Crystallinity parameter is computed by :meth:`~boo.boo.crystallinity`.

Spatial correlation
-------------------

To know how spatially extended is the local symmetry and orientation, one can look at the average cross product at a certain distance.

.. math:: g_\ell(r) = \frac{\sum_{i,j} s_\ell(i,j)\delta(r_{ij}-r)}{\sum_{i,j} \delta(r_{ij}-r)}

where :math:`\delta` is a binning function equal to one between 0 and :math:`dr` and zero elsewhere.

The function :meth:`~boo.boo.gG_l` returns separately the numerator and denominator of the above expression to ease further averaging. ``maxdist`` is the maximum range to consider and ``Nbins`` the number of bins between 0 and ``maxdist``. ``qlms`` is a list of bond orientational order coefficients that can have different values of :math:`\ell`, some coarse-grained or not. ``is_center`` is a (N) array of boolean marking the particles that are further than maxdist from any edge of the experimental window in order to avoid edge effects.





