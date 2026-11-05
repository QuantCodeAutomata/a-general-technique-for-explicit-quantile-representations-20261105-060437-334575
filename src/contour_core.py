"""Generic engine for the contour-integral quantile method.

Implements the two equivalent forms given in Schadner (2026), "A general
technique for explicit quantile representations":

  * the logarithmic form, Eq. (1):
        Q(p) = (1/pi) * Im int_I kappa'(x) * Log[(P(kappa(x)) - p) / (1 - p)] dx
  * the residue form, Eq. (4):
        Q(p) = -(1/pi) * Im int_I kappa(x) * kappa'(x) * g(kappa(x)) / (P(kappa(x)) - p) dx

with contour kappa(x) = x + i*h(x) (Eq. 2), where h is a p-independent height
function built from a phase or relative-variation bound on the working density
g (Section 2). The contour is built once per working law and reused for every
probability level p -- the point of the construction (Section 1).

Both forms share the same ingredients: a composite Gauss-Legendre mesh along
the real axis, the contour values kappa(x), kappa'(x), the density g(kappa(x))
and the complex CDF P(kappa(x)) accumulated cumulatively along the mesh. This
module implements those shared ingredients once; each distribution module in
this package supplies only g, h, the domain and (if used) a bounded observable
G (Eq. A.19/A.21 for Student's t) or a splitting into half-axis subdensities
(Theorem A.2 for the variance-gamma law).

The exact mesh construction (cell placement, growth) is not spelled out in the
paper beyond "cells lie in a logistic, logarithmic, or linear coordinate and
are no wider than the contour height in that coordinate" (Appendix D.1). Here
cells are built directly in the x-coordinate: width w(x) = h(x) near the
mesh's seed point, growing geometrically with distance from the seed so a
bounded number of cells covers the (possibly unbounded) support while keeping
cells no wider than the local contour height near the bulk of the mass. This
is an implementation choice, not a paper formula, recorded here and in
results/RESULTS.md.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional

import mpmath as mp

GL_N = 20  # composite 20-point Gauss-Legendre quadrature (Appendix D.1)

_GL_CACHE: dict = {}


def identity_observable(z):
    return z


def leggauss(n: int = GL_N):
    """Gauss-Legendre nodes/weights on [-1, 1] with n points at working precision.

    Roots of the degree-n Legendre polynomial, found by Newton's method at
    full working precision, seeded from NumPy's double-precision
    `numpy.polynomial.legendre.leggauss` nodes. A Tricomi-formula initial
    guess was tried first but converges to the wrong root for some indices
    near the interval ends (producing duplicate nodes); the accurate
    double-precision seed removes that failure mode.
    """
    key = (n, mp.mp.dps)
    if key in _GL_CACHE:
        return _GL_CACHE[key]
    import numpy as np
    seed_nodes, _ = np.polynomial.legendre.leggauss(n)
    nodes, weights = [], []
    for x0 in seed_nodes:
        def p_n(x, n=n):
            return mp.legendre(n, x)

        x = mp.findroot(p_n, mp.mpf(float(x0)))
        dpn = n * (x * mp.legendre(n, x) - mp.legendre(n - 1, x)) / (x * x - 1)
        w = 2 / ((1 - x * x) * dpn * dpn)
        nodes.append(x)
        weights.append(w)
    _GL_CACHE[key] = (nodes, weights)
    return nodes, weights


@dataclass
class Mesh:
    """A composite Gauss-Legendre mesh of points along the real axis.

    `xs` are ascending real quadrature nodes (mpf); `ws` are the matching
    quadrature weights, already including the cell Jacobian, so that
    sum(f(x_i) * w_i) approximates int f(x) dx over the covered range.
    `edges` are the n_cells + 1 cell boundaries and `gl_n` the number of
    nodes per cell (`len(xs) == gl_n * n_cells`), needed to accumulate the
    complex CDF cell by cell (see `accumulate`).
    """

    xs: List[mp.mpf]
    ws: List[mp.mpf]
    x_lo: mp.mpf
    x_hi: mp.mpf
    edges: List[mp.mpf]
    gl_n: int


_CUM_MATRIX_CACHE: dict = {}


def cumulative_matrix(n: int = GL_N):
    """The n x n "matching Gauss-Legendre integration matrix" M with

        (M @ f)[k] = \\int_{-1}^{t_k} p(t) dt,  k = 0, ..., n-1

    where p is the degree-(n-1) polynomial interpolating f at the n
    reference GL nodes t_0 < ... < t_{n-1}. Applied cell by cell (after
    scaling by the cell's half-width), this gives the cumulative integral of
    g(kappa(t)) kappa'(t) at every quadrature node within a cell -- the
    "matching Gauss-Legendre integration matrix" used to accumulate P(kappa)
    along the contour (Appendix D.1, and explicitly for the variance-gamma
    half-axis contours, Appendix D.1 / Theorem A.2 notes).

    A composite GL rule gives only full-cell integrals; a plain running sum
    of w_i * f_i stopping mid-cell is not a valid partial integral (GL
    weights are calibrated for the whole cell only), so this matrix is
    required to get P(kappa(x_i)) at interior nodes, not just cell edges.
    """
    key = (n, mp.mp.dps)
    if key in _CUM_MATRIX_CACHE:
        return _CUM_MATRIX_CACHE[key]
    nodes, _ = leggauss(n)
    V = mp.matrix(n, n)
    A = mp.matrix(n, n)
    for i in range(n):
        ti = nodes[i]
        for j in range(n):
            V[i, j] = ti ** j
            A[i, j] = (ti ** (j + 1) - (-1) ** (j + 1)) / (j + 1)
    M = A * (V ** -1)
    _CUM_MATRIX_CACHE[key] = M
    return M


def _one_sided_edges(h: Callable, seed: mp.mpf, x_stop: mp.mpf, direction: int,
                      width_factor: mp.mpf, growth: mp.mpf, max_cells: int):
    """Cell edges from `seed` towards `x_stop` (direction = +1 or -1).

    Each cell width is `width_factor * h(x)` evaluated at the *current*
    position (not merely grown geometrically from the seed), so cells stay
    no wider than the local contour height everywhere along the path -- as
    required when h is non-monotonic (e.g. the GIG height is O(x^2) near 0,
    rises, then settles to 1/b at infinity; a naive geometric grid seeded at
    a single point under-resolves the region where h is locally small, which
    is exactly where small/large probability-level quantiles can fall). The
    `growth` factor only bounds how fast an unboundedly growing local width
    (e.g. h(x) -> 1/b as x -> infinity) may further widen successive cells,
    to keep the cell count under `max_cells` for exponential domains.
    """
    edges = [seed]
    x = seed
    prev_width = None
    for _ in range(max_cells):
        local_width = width_factor * max(h(x), mp.mpf('1e-12'))
        if prev_width is not None:
            local_width = min(local_width, prev_width * growth)
        width = local_width
        prev_width = width
        nxt = x + direction * width
        if direction > 0 and nxt >= x_stop:
            edges.append(x_stop)
            break
        if direction < 0 and nxt <= x_stop:
            edges.append(x_stop)
            break
        edges.append(nxt)
        x = nxt
    else:
        edges.append(x_stop)
    return edges


def build_mesh(h: Callable[[mp.mpf], mp.mpf], center: mp.mpf, x_lo: mp.mpf, x_hi: mp.mpf,
               gl_n: int = GL_N, width_factor: float = 1.0, growth: float = 1.15,
               max_cells_per_side: int = 200) -> Mesh:
    """Composite Gauss-Legendre mesh covering the open interval (x_lo, x_hi).

    Cells are seeded at `center` (x_lo <= center <= x_hi) and grow
    geometrically towards both ends, so the narrowest cells (width set by the
    contour height h) sit near `center`, matching the guideline that cells be
    "no wider than the contour height" where the working density is largest.
    """
    width_factor = mp.mpf(width_factor)
    growth = mp.mpf(growth)
    right_edges = _one_sided_edges(h, center, x_hi, +1, width_factor, growth, max_cells_per_side)
    left_edges = _one_sided_edges(h, center, x_lo, -1, width_factor, growth, max_cells_per_side)
    edges = list(reversed(left_edges)) + right_edges[1:]
    edges = [e for i, e in enumerate(edges) if i == 0 or e != edges[i - 1]]
    nodes_unit, weights_unit = leggauss(gl_n)
    xs: List[mp.mpf] = []
    ws: List[mp.mpf] = []
    for a, b in zip(edges[:-1], edges[1:]):
        mid = (a + b) / 2
        half = (b - a) / 2
        for t, wt in zip(nodes_unit, weights_unit):
            xs.append(mid + half * t)
            ws.append(half * wt)
    return Mesh(xs=xs, ws=ws, x_lo=x_lo, x_hi=x_hi, edges=edges, gl_n=gl_n)


def find_right_truncation(order_fn: Callable[[mp.mpf], mp.mpf], x_start: mp.mpf,
                           target_log: float = 60.0, x_max_mult: float = 1e30) -> mp.mpf:
    """Smallest x >= x_start with -log(order_fn(x)) >= target_log (geometric search).

    Implements "each domain is truncated where the neglected integrand is of
    order e^{-60}" (Appendix D.1) for a generic tail-order function.
    """
    x = max(x_start, mp.mpf('1e-6'))
    target = mp.mpf(target_log)
    for _ in range(400):
        val = order_fn(x)
        if val <= 0 or -mp.log(val) >= target:
            return x
        x *= mp.mpf('1.5')
        if x > mp.mpf(x_max_mult):
            return x
    return x


def find_left_truncation(order_fn: Callable[[mp.mpf], mp.mpf], x_start: mp.mpf,
                          target_log: float = 60.0, x_min: mp.mpf = mp.mpf('1e-300')) -> mp.mpf:
    """Smallest-magnitude x <= x_start (x > 0) with -log(order_fn(x)) >= target_log."""
    x = x_start
    target = mp.mpf(target_log)
    for _ in range(400):
        val = order_fn(x)
        if val <= 0 or -mp.log(val) >= target:
            return x
        x /= mp.mpf('1.5')
        if x < x_min:
            return x
    return x


def numerical_derivative(f: Callable, x: mp.mpf, step: Optional[mp.mpf] = None) -> mp.mpc:
    """kappa'(x) by central differencing of f, used when an analytic derivative
    of the contour height is not hand-derived in the distribution module."""
    if step is None:
        step = mp.mpf(10) ** (-(mp.mp.dps // 2))
        scale = max(abs(x), mp.mpf(1))
        step = step * scale
    return (f(x + step) - f(x - step)) / (2 * step)


@dataclass
class ContourAccumulation:
    """Precomputed, p-independent contour data reused across probability levels."""

    mesh: Mesh
    kappa: List[mp.mpc]
    kappa_prime: List[mp.mpc]
    g_vals: List[mp.mpc]
    P_vals: List[mp.mpc]  # cumulative complex CDF evaluated at kappa(x_i)


def accumulate(mesh: Mesh, kappa_fn: Callable[[mp.mpf], mp.mpc],
               kappa_prime_fn: Callable[[mp.mpf], mp.mpc],
               g_fn: Callable[[mp.mpc], mp.mpc],
               p_left: mp.mpc = mp.mpf(0)) -> ContourAccumulation:
    """Build kappa, kappa', g and the cumulative complex CDF P(kappa(x)) along `mesh`.

    `p_left` is the (negligible) mass assumed to the left of mesh.x_lo. Full
    cells are accumulated by the standard composite Gauss-Legendre sum;
    inside each cell, the value at every interior node is obtained from the
    "matching Gauss-Legendre integration matrix" (`cumulative_matrix`), since
    a running sum of GL terms stopping mid-cell is not a valid partial
    integral (the weights are calibrated for the whole cell only).
    """
    n = mesh.gl_n
    M = cumulative_matrix(n)
    kappa_vals = [kappa_fn(x) for x in mesh.xs]
    kappa_prime_vals = [kappa_prime_fn(x) for x in mesh.xs]
    g_vals = [g_fn(k) for k in kappa_vals]
    integrand = [g_v * kp_v for g_v, kp_v in zip(g_vals, kappa_prime_vals)]
    n_cells = len(mesh.edges) - 1
    assert len(mesh.xs) == n * n_cells, "mesh nodes must be n_cells * gl_n"
    P_vals: List[mp.mpc] = [None] * len(mesh.xs)
    running = p_left
    for c in range(n_cells):
        a, b = mesh.edges[c], mesh.edges[c + 1]
        half = (b - a) / 2
        f_chunk = mp.matrix(integrand[c * n:(c + 1) * n])
        cum = M * f_chunk
        for k in range(n):
            P_vals[c * n + k] = running + half * cum[k]
        # mesh.ws already includes the cell Jacobian `half`, so the full-cell
        # integral is simply the standard GL weighted sum sum(w_i f_i).
        full_cell = sum(w * f for w, f in zip(mesh.ws[c * n:(c + 1) * n], integrand[c * n:(c + 1) * n]))
        running = running + full_cell
    return ContourAccumulation(mesh=mesh, kappa=kappa_vals, kappa_prime=kappa_prime_vals,
                                g_vals=g_vals, P_vals=P_vals)


def residue_quantile(acc: ContourAccumulation, p: mp.mpf,
                      observable: Callable = identity_observable) -> mp.mpf:
    """Eq. (4): Q(p) = -(1/pi) * Im sum[ G(kappa)*kappa'*g(kappa) / (P(kappa)-p) * w ]."""
    total = mp.mpc(0)
    for k, kp, g_v, P_v, w in zip(acc.kappa, acc.kappa_prime, acc.g_vals, acc.P_vals, acc.mesh.ws):
        total += observable(k) * kp * g_v / (P_v - p) * w
    return -mp.im(total) / mp.pi


def log_quantile(acc: ContourAccumulation, p: mp.mpf) -> mp.mpf:
    """Eq. (1): Q(p) = (1/pi) * Im sum[ kappa'(x) * Log((P(kappa)-p)/(1-p)) * w ]."""
    total = mp.mpc(0)
    one = mp.mpf(1)
    for kp, P_v, w in zip(acc.kappa_prime, acc.P_vals, acc.mesh.ws):
        ratio = (P_v - p) / (one - p)
        total += kp * mp.log(ratio) * w
    return mp.im(total) / mp.pi


def newton_quantile(cdf_real: Callable[[mp.mpf], mp.mpf], p: float, x0: float,
                     lo: Optional[float] = None, hi: Optional[float] = None,
                     dps: int = 30) -> mp.mpf:
    """Reference quantile solving P(x) = p by Newton's method (bisection
    safeguard) in `dps`-digit arithmetic, started from the double-precision
    guess `x0` (e.g. a SciPy `.ppf` value), as in Appendix D.1."""
    with mp.workdps(dps):
        p_m = mp.mpf(p)

        def F(x):
            return cdf_real(x) - p_m

        x = mp.mpf(x0)
        a, b = (mp.mpf(lo) if lo is not None else None), (mp.mpf(hi) if hi is not None else None)
        for _ in range(100):
            fx = F(x)
            if a is not None and fx < 0:
                a = x
            if b is not None and fx > 0:
                b = x
            step_h = mp.mpf(10) ** (-(dps // 2))
            deriv = (F(x + step_h) - F(x - step_h)) / (2 * step_h)
            if deriv == 0:
                break
            x_new = x - fx / deriv
            if a is not None and x_new <= a:
                x_new = (x + a) / 2
            if b is not None and x_new >= b:
                x_new = (x + b) / 2
            if abs(x_new - x) < mp.mpf(10) ** (-(dps - 4)):
                x = x_new
                break
            x = x_new
        return x


def relative_error(q_contour: mp.mpf, q_reference: mp.mpf) -> mp.mpf:
    if q_reference == 0:
        return abs(q_contour)
    return abs((q_contour - q_reference) / q_reference)


def build_contour(g: Callable[[mp.mpc], mp.mpc], h: Callable[[mp.mpf], mp.mpf],
                   center: mp.mpf, x_lo: mp.mpf, x_hi: mp.mpf,
                   kappa: Optional[Callable] = None, kappa_prime: Optional[Callable] = None,
                   p_left: mp.mpc = mp.mpf(0), mesh_kwargs: Optional[dict] = None
                   ) -> ContourAccumulation:
    """Build kappa(x) = x + i*h(x) (Eq. 2), mesh it, and accumulate P(kappa) once.

    The returned `ContourAccumulation` is independent of p (Section 1) and is
    reused by `residue_quantile`/`log_quantile` for every probability level.
    `kappa_prime` defaults to a numerical derivative of `kappa` (central
    differencing at full working precision), used whenever a distribution
    module does not hand-derive an analytic contour derivative.
    """
    if kappa is None:
        def kappa(x):
            return mp.mpc(x, h(x))
    if kappa_prime is None:
        def kappa_prime(x):
            return numerical_derivative(kappa, x)
    mesh = build_mesh(h, center, x_lo, x_hi, **(mesh_kwargs or {}))
    return accumulate(mesh, kappa, kappa_prime, g, p_left=p_left)
