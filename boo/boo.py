"""Computation of bond orientational order"""
#
#    Copyright 2011 Mathieu Leocmach
#
#    This file is part of pyboo.
#
#    pyboo is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    pyboo is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with pyboo.  If not, see <http://www.gnu.org/licenses/>.
#
from math import floor
import numpy as np
from scipy.special import sph_harm
from scipy.spatial import cKDTree as KDTree
from numba import jit, vectorize, guvectorize


@vectorize#(['float64(complex128)', 'float32(complex64)'])
def abs2(x):
    """squared norm of a complex number"""
    return x.real**2 + x.imag**2

@vectorize(['float64(float64,float64,float64)'])
def periodify(u, v, period=-1.0):
    """
Given two arrays of points in a d-dimensional space
with periodic boundary conditions,
find the shortest vector between each pair.

Implemented using algorithm C4 of Deiters 2013 doi:10.1524/zpch.2013.0311

Parameters
----------
u,v : array_like float
period : float or an array of floats of length d.
    Negative periods indicate no periodicity in this dimension.

Returns
----------
array_like float
"""
    diff = v - u
    if period <= 0:
        return diff
    return diff - period * floor(diff /period + 0.5)


def cart2sph(cartesian):
    r"""
Convert Cartesian coordinates :math:`[[x,y,z],]`
to spherical coordinates :math:`[[r,\phi,\theta],]`.
:math:`\phi` is cologitudinal and :math:`\theta` azimutal
"""
    spherical = np.zeros_like(cartesian)
    #distances
    c2 = cartesian**2
    r2 = c2.sum(-1)
    spherical[:, 0] = np.sqrt(r2)
    #work only on non-zero, non purely z vectors
    sel = (r2 > c2[:, 0]) | (r2+1.0 > 1.0)
    x, y, z = cartesian[sel].T
    r = spherical[sel, 0]
    #colatitudinal phi [0, pi[
    spherical[sel, 1] = np.arccos(z/r)
    #azimutal (longitudinal) theta [0, 2pi[
    theta = np.arctan2(y, x)
    theta[theta < 0] += 2*np.pi
    spherical[sel, 2] = theta
    return spherical

def vect2Ylm(v, l):
    """Projects vectors v on the base of spherical harmonics of degree l."""
    spherical = cart2sph(v)
    return sph_harm(
        np.arange(l+1)[:, None], l,
        spherical[:, 2][None, :],
        spherical[:, 1][None, :]
        )

def single_pos2qlm(pos, i, ngb_indices, l=6):
    """Returns the qlm for a single position"""
    #vectors to neighbours
    vectors = pos[ngb_indices]-pos[i]
    return vect2Ylm(vectors, l).mean(-1)

def bonds2qlm(pos, bonds, l=6, periods=-1.0):
    """
Compute bond orientational order for each particle.

Parameters
----------
pos : (N, 3) array of floats
    Spatial coordinates
bonds : (M, 2) array of integers.
    Bonds are supposed unique and bidirectional.
l : int
    A positive even integer indicating the order of symmetry.
periods : float or (3) array of floats.
    Negative periods indicate no periodicity in this dimension.

Returns
----------
qlm : (N, 2*l+1) array of complex
    Tensorial order parameter of order l for each particle
"""
    qlm = np.zeros((len(pos), l+1), np.complex128)
    #spherical harmonic coefficients for each bond
    Ylm = vect2Ylm(
        periodify(
            pos[bonds[:, 0]],
            pos[bonds[:, 1]],
            periods
        ),
        l
    ).T
    #bin bond into each particle belonging to it
    np.add.at(qlm, bonds[:, 0], Ylm)
    np.add.at(qlm, bonds[:, 1], Ylm)
    #divide by the number of bonds each particle belongs to
    Nngb = np.zeros(len(pos), int)
    np.add.at(Nngb, bonds.ravel(), 1)
    return qlm / np.maximum(1, Nngb)[:, None]

def ngbs2qlm(pos, ngbs, l=6, periods=-1):
    """
Compute bond orientational order supposing at most M neigbours per particles.

Parameters
----------
pos : (N, 3) array of floats
    Spatial coordinates
ngbs : (N, k) array of int
    Neighbour indices.
    Negative indices correspond to invalid neighbours and give zero contribution.
l : int
    A positive even integer indicating the order of symmetry.
periods : float or (3) floats.
    Negative periods indicate no periodicity in this dimension.

Returns
----------
qlm : (N, 2*l+1) array of complex
    Tensorial order parameter of order l for each particle
"""
    assert len(pos) == len(ngbs)
    #eliminate neighbours with negative indices
    good = ngbs >= 0
    #spherical harmonic coefficients for each bond
    Ylm = vect2Ylm(
        periodify(
            pos[np.repeat(np.arange(ngbs.shape[0]), ngbs.shape[-1])[good.ravel()]],
            pos[ngbs[good].ravel()],
            periods
        ),
        l
    ).T
    Ylm2 = np.zeros([ngbs.shape[0], ngbs.shape[1], l+1], np.complex128)
    Ylm2.reshape((ngbs.shape[0]*ngbs.shape[1], l+1))[good.ravel()] = Ylm
    return Ylm2.mean(1)

def coarsegrain_qlm(qlm, bonds, inside):
    r"""
Coarse grain the bond orientational order on the neighbourhood of a particle

.. math:: Q_{\ell m}(i) = \frac{1}{N_i+1}\left(q_{\ell m}(i) + \sum_{j=0}^{N_i} q_{\ell m}(j)\right)

See Lechner & Delago J. Chem. Phys. (2008) doi:10.1063/1.2977970
Returns Qlm and the mask of the valid particles

Parameters
----------
qlm : (N, 2*l+1) array of complex
    Tensorial order parameter of order l for each particle
bonds : (M, 2) array of integers
    Bonds are supposed unique and bidirectional.
inside : (N) array of booleans
    Particles that are outside are not added and contaminate their neighbours.

Returns
----------
Qlm : (N, 2*l+1) array of complex
    Coarse-grained tensorial order parameter of order l for each particle
inside2 : (N) array of booleans
    Particles that are inside and have no neighbour outside
"""
    #Valid particles must be valid themselves have only valid neighbours
    inside2 = np.copy(inside)
    np.bitwise_and.at(inside2, bonds[:, 0], inside[bonds[:, 1]])
    np.bitwise_and.at(inside2, bonds[:, 1], inside[bonds[:, 0]])
    #number of neighbours
    Nngb = np.zeros(len(qlm), int)
    np.add.at(Nngb, bonds.ravel(), 1)
    #sum the boo coefficients of all the neighbours
    Qlm = np.zeros_like(qlm)
    np.add.at(Qlm, bonds[:, 0], qlm[bonds[:, 1]])
    np.add.at(Qlm, bonds[:, 1], qlm[bonds[:, 0]])
    Qlm[np.bitwise_not(inside2)] = 0
    return Qlm / np.maximum(1, Nngb)[:, None], inside2

def coarsegrain_qlm_ngbs(qlm, ngbs, inside):
    r"""
Coarse grain the bond orientational order on the neighbourhood of a particle

.. math:: Q_{\ell m}(i) = \frac{1}{N_i+1}\left(q_{\ell m}(i) + \sum_{j=0}^{N_i} q_{\ell m}(j)\right)

See Lechner & Delago J. Chem. Phys. (2008) doi:10.1063/1.2977970
Returns Qlm and the mask of the valid particles

Parameters
----------
qlm : (N, 2*l+1) array of complex
    Tensorial order parameter of order l for each particle
ngbs : (N, k) array of int
    Neighbour indices.
    Negative indices correspond to invalid neighbours and give zero contribution.
inside : (N) array of booleans
    Particles that are outside are not added and contaminate their neighbours.

Returns
----------
Qlm : (N, 2*l+1) array of complex
    Coarse-grained tensorial order parameter of order l for each particle
inside2 : (N) array of booleans
    Particles that are inside and have no neighbour outside
"""
    assert len(qlm) == len(ngbs)
    #eliminate neighbours with negative indices
    good = ngbs >= 0
    #Valid particles must be valid themselves have only valid neighbours
    inside2 = np.copy(inside)
    inside2 = inside & np.all(np.where(good, inside[ngbs], True), axis=-1)
    #sum
    Qlm = np.zeros((ngbs.shape[0], ngbs.shape[1], qlm.shape[-1]), qlm.dtype)
    flatshape = (ngbs.shape[0]*ngbs.shape[1], qlm.shape[-1])
    Qlm.reshape(flatshape)[good.ravel()] = qlm[ngbs].reshape(flatshape)[good.ravel()]
    return (Qlm.sum(1) + qlm) / (1 + ngbs.shape[1]), inside2


@guvectorize(
    ['void(complex128[:], complex128[:], float64[:])'],
    '(n),(n)->()',
    #nopython=True
)
def product(qlm1, qlm2, prod):
    r"""
Product between two qlm

.. math:: s_\ell (i,j) = \frac{4\pi}{2\ell + 1} \sum_{m=-\ell}{\ell} q_{\ell m}(i) q_{\ell m}(j)^*
"""
    l = qlm1.shape[0]-1
    prod[0] = (qlm1[0] * qlm2[0].conjugate()).real
    for i in range(1, len(qlm1)):
        prod[0] += 2 * (qlm1[i] * qlm2[i].conjugate()).real
    prod[0] *= 4*np.pi/(2*l+1)


@jit#(nopython=True)
def ql(qlm):
    r"""
Second order rotational invariant of the bond orientational order of l-fold symmetry

.. math::  q_\ell = \sqrt{\frac{4\pi}{2l+1} \sum_{m=-\ell}^{\ell} |q_{\ell m}|^2 }

Parameters
----------
qlm : (N, 2*l+1) array of complex
    Tensorial order parameter of order l for each particle

Returns
----------
(N) array of float
"""
    q = abs2(qlm[..., 0])
    for m in range(1, qlm.shape[-1]):
        q += 2 * abs2(qlm[..., m])
    l = qlm.shape[-1]-1
    return np.sqrt(4*np.pi / (2*l+1) * q)

@jit#(['complex128[:](complex128[:,:], int64)'], nopython=True)
def get_qlm(qlms, m):
    """qlm coefficients are redundant, negative m are obtained from positive m"""
    if m >= 0:
        return qlms[..., m]
    if (-m)%2 == 0:
        return np.conj(qlms[..., -m])
    return -np.conj(qlms[..., -m])


@jit#(['float64(int64, int64[:])'], nopython=True)
def get_w3j(l, ms):
    """Wigner 3j coefficients"""
    sm = np.sort(np.abs(ms))
    return _W3J_[l//2, _W3J_M1_OFFSET_[sm[-1]] + sm[0]]

@jit
def wl(qlm):
    r"""
Third order rotational invariant of the bond orientational order of l-fold symmetry

.. math::  w_\ell = \sum_{m_1+m_2+m_3=0}
		\left( \begin{array}{ccc}
			\ell & \ell & \ell \\
			m_1 & m_2 & m_3
		\end{array} \right)
		q_{\ell m_1} q_{\ell m_2} q_{\ell m_3}

Parameters
----------
qlm : (N, 2*l+1) array of complex
    Tensorial order parameter of order l for each particle

Returns
----------
(N) array of float
"""
    l = qlm.shape[-1]-1
    w = np.zeros(qlm.shape[:-1])
    for m1 in range(-l, l+1):
        for m2 in range(-l, l+1):
            m3 = -m1 - m2
            if -l <= m3 and m3 <= l:
                w += get_w3j(l, np.array([m1, m2, m3])) * (
                    get_qlm(qlm, m1) * get_qlm(qlm, m2) * get_qlm(qlm, m3)
                ).real
    return w

def bond_normed_product(qlm, bonds):
    r"""
Normalized cross product

.. math:: \hat{s}_\ell (i,j) = \frac{s_\ell (i,j)}{q_\ell(i) q_\ell(j)}

Parameters
----------
qlm : (N, 2*l+1) array of complex
    Tensorial order parameter of order l for each particle
bonds : (M, 2) array of integers
    Bonds are supposed unique and bidirectional.

Returns
----------
(M) array of float
"""
    q = ql(qlm)
    return product(qlm[bonds[:, 0]], qlm[bonds[:, 1]])/(
        q[bonds[:, 0]] * q[bonds[:, 1]]
    )

def x_bonds(qlm, bonds, threshold=0.7):
    """
Which bonds are crystalline?
If the normalized cross product of their qlm is larger than the threshold.

Parameters
----------
qlm : (N, 2*l+1) array of complex
    Tensorial order parameter of order l for each particle
bonds : (M, 2) array of integers
    Bonds are supposed unique and bidirectional.
threshold : float
    Lower bound of the normed product for a bond to be considered crystalline. Default 0.7.

Returns
----------
(P, 2) array of integers
    The bonds that are crystalline. P <= M
"""
    return bonds[bond_normed_product(qlm, bonds) > threshold]

def x_ngbs(qlm, ngbs, threshold=0.7):
    """
With which neighbours j does a particles i have a crystalline bond?
If the normalized cross product of their qlm is larger than the threshold.

Parameters
----------
qlm : (N, 2*l+1) array of complex
    Tensorial order parameter of order l for each particle
ngbs : (N, k) array of int
    Neighbour indices.
    Negative indices correspond to invalid neighbours and give zero contribution.
threshold : float
    Lower bound of the normed product for a bond to be considered crystalline. Default 0.7.

Returns
----------
(N, k) array of booleans
"""
    bonds = np.column_stack((
        np.repeat(np.arange(ngbs.shape[0]), ngbs.shape[1]),
        ngbs.ravel()
        ))
    good = ngbs >= 0
    xn = (bond_normed_product(qlm, bonds) > threshold).reshape(ngbs.shape)
    return xn & good

def x_particles(qlm, bonds, value_thr=0.7, nb_thr=7):
    """
Which particles are crystalline? If they have more than nb_thr crystalline bonds.

Parameters
----------
qlm : (N, 2*l+1) array of complex
    Tensorial order parameter of order l for each particle
bonds : (M, 2) array of integers.
    Bonds are supposed unique and bidirectional.
value_thr : float
    Lower bound of the normed product for a bond to be considered crystalline. Default 0.7.
nb_thr : int
    Minimum number of crystalline bonds for a particle to be considered crystalline.

Returns
----------
(N) array of booleans
"""
    xb = x_bonds(qlm, bonds, threshold=value_thr)
    nb = np.zeros(len(qlm), int)
    np.add.at(nb, xb.ravel(), 1)
    return nb >= nb_thr

def crystallinity(qlm, bonds):
    r"""
Crystallinity parameter, see Russo & Tanaka, Sci Rep. (2012) doi:10.1038/srep00505.

.. math:: C_\ell(i) = \frac{1}{N_i} \sum_{j=0}{N_i} \hat{s}_\ell (i,j)

Parameters
----------
qlm : (N, 2*l+1) array of complex
    Tensorial order parameter of order l for each particle
bonds : (M, 2) array of integers.
    Bonds are supposed unique and bidirectional.

Returns
----------
(N) array of floats
"""
    #cross product for all bonds
    bv = bond_normed_product(qlm, bonds)
    #count number or neighbours
    nb = np.zeros(len(qlm), int)
    np.add.at(nb, bonds.ravel(), 1)
    #sum cross product over bonds for each particle
    c = np.zeros(len(qlm))
    np.add.at(c, bonds[:, 0], bv)
    np.add.at(c, bonds[:, 1], bv)
    #mean
    return c/np.maximum(1, nb)

def gG_l(pos, qlms, is_center, Nbins, maxdist):
    """
Spatial correlation of the qlms (non normalized).

For each particle i tagged as is_center,
for each particle j closer than maxdist,
do the cross product between their qlm and count,
then bin each quantity with respect to distance.
The two first sums need to be normalised by the last one.

Periodic boundary conditions are not supported.

Parameters
----------
pos : (N, 3) array of floats
    Spatial coordinates
qlms : list
    A list of M (N, 2l+1) arrays of boo coordinates for l-fold symmetry.
    l can be different for each item.
is_center : (N) array of bool.
    For example all particles further away than maxdist from any edge of the box.
Nbins : int
    The number of bins along r
maxdist : float
    The maximum distance considered.

Returns
----------
hqQ : (Nbins, M) array of floats
    The sum of cross products for each distance and each qlm
g : (Nbins) array of ints
    The number of pairs for each distance
"""
    for qlm in qlms:
        assert len(pos) == len(qlm)
    assert len(is_center) == len(pos)
    #conversion factor between indices and bins
    l2r = Nbins/maxdist
    #result containers
    #an additional bin for the case where the distance is exactly equal to maxdist
    hqQ = np.zeros((Nbins+1, len(qlms)))
    g = np.zeros(Nbins+1, int)
    #compute ql for all particles
    qQ = np.array([ql(qlm) for qlm in qlms])
    nonzero = qQ.min(0) + 1.0 > 1.0
    #spatial indexing
    tree = KDTree(pos[nonzero], 12)
    centertree = KDTree(pos[is_center & nonzero], 12)
    #all pairs of points closer than maxdist with their distances in a record array
    query = centertree.sparse_distance_matrix(tree, maxdist, output_type='ndarray')
    #convert in original indices
    nonzeroindex = np.where(nonzero)[0]
    centerindex = np.where(is_center&nonzero)[0]
    query['i'] = centerindex[query['i']]
    query['j'] = nonzeroindex[query['j']]
    #keep only pairs where the points are distinct
    good = query['i'] != query['j']
    query = query[good]
    #binning of distances
    rs = (query['v'] * l2r).astype(int)
    np.add.at(g, rs, 1)
    #binning of boo cross products
    pqQs = np.empty((len(rs), len(qlms)))
    for it, qlm in enumerate(qlms):
        pqQs[:, it] = product(qlm[query['i']], qlm[query['j']])
        prodnorm = qQ[it, query['i']] * qQ[it, query['j']]
        pqQs[:, it] /= prodnorm
    np.add.at(hqQ, rs, pqQs)
    return hqQ[:-1], g[:-1]



# Define constants for Wigner 3j symbols

_W3J_ = np.zeros([6, 36])
_W3J_[0, 0] = 1
_W3J_[1, :4] = np.sqrt([2/35., 1/70., 2/35., 3/35.]) * [-1, 1, 1, -1]
_W3J_[2, :9] = np.sqrt([
    2/1001., 1/2002., 11/182., 5/1001.,
    7/286., 5/143., 14/143., 35/143., 5/143.,
]) * [
    3, -3, -1/3.0, 2, 1, -1/3.0, 1/3.0, -1/3.0, 1
]

_W3J_[3, :16] = np.sqrt([
    1/46189., 1/46189.,
    11/4199., 105/46189.,
    1/46189., 21/92378.,
    1/46189., 35/46189., 14/46189.,
    11/4199., 21/4199., 7/4199.,
    11/4199., 77/8398., 70/4199., 21/4199.
])*[
    -20, 10, 1, -2, -43/2.0, 3, 4, 2.5, -6, 2.5, -1.5, 1, 1, -1, 1, -2
]

_W3J_[4, :25] = np.sqrt([
    10/96577., 5/193154.,
    1/965770., 14/96577.,
    1/965770., 66/482885.,
    5/193154., 3/96577., 77/482885.,
    65/14858., 5/7429., 42/37145.,
    65/14858., 0.0, 3/7429., 66/37145.,
    13/74290., 78/7429., 26/37145., 33/37145.,
    26/37145., 13/37145., 273/37145., 429/37145., 11/7429.,
]) * [
    7, -7, -37, 6, 73, -3,
    -5, -8, 6, -1, 3, -1,
    1, 0, -3, 2, 7, -1, 3, -1,
    1, -3, 1, -1, 3
]

_W3J_[5, :36] = np.sqrt([
    7/33393355., 7/33393355.,
    7/33393355., 462/6678671.,
    7/33393355., 1001/6678671.,
    1/233753485., 77/6678671., 6006/6678671.,
    1/233753485., 55/46750697., 1155/13357342.,
    1/233753485., 2926/1757545., 33/46750697., 3003/6678671.,
    119/1964315., 22/2750041., 1914/474145., 429/5500082.,
    17/13750205., 561/2750041., 77/392863., 143/27500410., 2002/392863.,
    323/723695., 1309/20677., 374/144739., 143/144739., 1001/206770.,
    323/723695., 7106/723695., 561/723695., 2431/723695., 2002/103385., 1001/103385.
]) * [
    -126, 63, 196/3.0, -7, -259/2.0, 7/3.0,
    1097/3.0, 59/6.0, -2,
    4021/6.0, -113/2.0, 3,
    -914, 1/3.0, 48, -3,
    -7/3.0, 65/3.0, -1, 3,
    214/3.0, -3, -2/3.0, 71/3.0, -1,
    3, -1/3.0, 5/3.0, -2, 1/3.0,
    2/3.0, -1/3.0, 2, -4/3.0, 2/3.0, -1
]

_W3J_M1_OFFSET_ = np.array([0, 1, 2, 4, 6, 9, 12, 16, 20, 25, 30], int)
