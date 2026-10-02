# -*- coding: utf-8 -*-
"""
toolkit_2dstlm.py

Support functions for the 2D-STLM (2D Symmetric Trigonometric-Logistic
Map) dynamical analysis pipeline. This REPLACES the earlier pBCM-v2
(Mobius/Rational Coupled Map) toolkit with the map defined in the paper:

    x_{n+1} = ( r1*sin(pi*x_n)*cos(pi*y_n) + mu*y_n*(1 - y_n) )  mod 1
    y_{n+1} = ( r2*sin(pi*y_n)*cos(pi*x_n) + mu*x_n*(1 - x_n) )  mod 1

Parameters:
    r1, r2 : control parameters, range (0, 12]
    mu     : logistic cross-coupling strength, FIXED at 3.9 (edge-of-chaos
             regime of the classical logistic map)

Initial conditions:
    x0, y0 in (0, 1), with (x0, y0) != (0, 0)  [origin is a trivial fixed
    point of the system]

KEY STRUCTURAL DIFFERENCE FROM THE OLD pBCM-v2 MAP:
    pBCM-v2 was sequential (Gauss-Seidel style): y_{n+1} was computed
    using the ALREADY-UPDATED x_{n+1}. 2D-STLM is SIMULTANEOUS (Jacobi
    style): both x_{n+1} and y_{n+1} are computed directly from the same
    pair (x_n, y_n). This makes the Jacobian simpler (no chain-rule term
    through an intermediate x1), and there is no branch selector and no
    p-adic valuation switch -- it is a single smooth trigonometric map.

WHY NO SINGULARITY GUARD IS NEEDED:
    sin and cos are bounded in [-1, 1] and the logistic term y(1-y) /
    x(1-x) is bounded in [0, 0.25] for y, x in [0, 1). There is no
    denominator anywhere in this map, so unlike the old rational pBCM-v2
    map there is nothing that can blow up or require an epsilon-guard.

JACOBIAN (exact, analytic):

    Let A(x, y) = r1*sin(pi*x)*cos(pi*y) + mu*y*(1 - y)   [x_{n+1}, pre-mod]
        B(x, y) = r2*sin(pi*y)*cos(pi*x) + mu*x*(1 - x)   [y_{n+1}, pre-mod]

    Both A and B are evaluated at the SAME (x, y) -- no chain rule needed
    since the update is simultaneous, not sequential.

    dA/dx =  r1 * pi * cos(pi*x) * cos(pi*y)
    dA/dy = -r1 * pi * sin(pi*x) * sin(pi*y) + mu*(1 - 2*y)

    dB/dx = -r2 * pi * sin(pi*y) * sin(pi*x) + mu*(1 - 2*x)
    dB/dy =  r2 * pi * cos(pi*y) * cos(pi*x)

    (mod 1 has derivative 1 almost everywhere, so it does not enter the
    analytic Jacobian, exactly as in the previous toolkit's convention.)

VERIFIED HYPERCHAOTIC OPERATING POINT (see main_2dstlm.py Section 0):
    r1 = r2 = 12.0 (upper edge of the valid range), mu = 3.9:
    LE1 ~ 3.06, LE2 ~ 2.81 -- both clearly and stably positive across
    N = 5e3..1e5 and multiple initial conditions (genuinely hyperchaotic).

WHAT CARRIES OVER UNCHANGED FROM toolkit_2pbcm.py:
    - Byte extraction & randomness statistics (extract_bytes,
      shannon_entropy, chi2_test, serial_correlation) -- generic
      post-processing on the quantized (x_n, y_n) stream, independent of
      the map's internal structure.
    - Sample entropy (cKDTree-vectorized) and correlation dimension
      (Grassberger-Procaccia) -- generic time-series statistics.
    - The optional Stage 2 (2-adic T-function / ERPM byte-hardening
      layer) -- also generic, unrelated to the map's own dynamics.

Dependencies: numpy, scipy
"""

import os
import numpy as np
from scipy.spatial import cKDTree


# ---------------------------------------------------------------------------
# Default 2D-STLM parameters
# ---------------------------------------------------------------------------
DEFAULT_MU = 3.9  # fixed logistic cross-coupling strength (edge-of-chaos)

DEFAULT_PARAMS = dict(
    r1=12.0,  # control parameter, range (0, 12] -- verified hyperchaotic
    r2=12.0,  # control parameter, range (0, 12] -- verified hyperchaotic
    mu=DEFAULT_MU,
)


# ---------------------------------------------------------------------------
# Core map
# ---------------------------------------------------------------------------
def map_step(x, y, r1, r2, mu=DEFAULT_MU):
    """
    One iteration of 2D-STLM (simultaneous trigonometric-logistic map).
    Returns (x_next, y_next).
    """
    xn = np.mod(r1 * np.sin(np.pi * x) * np.cos(np.pi * y) + mu * y * (1.0 - y), 1.0)
    yn = np.mod(r2 * np.sin(np.pi * y) * np.cos(np.pi * x) + mu * x * (1.0 - x), 1.0)
    return xn, yn


def jacobian_step(x, y, r1, r2, mu=DEFAULT_MU):
    """
    2x2 analytic Jacobian of 2D-STLM at (x, y). See module docstring for
    the full derivation. Both partials are taken at the SAME (x, y)
    since the update is simultaneous (no chain rule needed).
    """
    dA_dx = r1 * np.pi * np.cos(np.pi * x) * np.cos(np.pi * y)
    dA_dy = -r1 * np.pi * np.sin(np.pi * x) * np.sin(np.pi * y) + mu * (1.0 - 2.0 * y)

    dB_dx = -r2 * np.pi * np.sin(np.pi * y) * np.sin(np.pi * x) + mu * (1.0 - 2.0 * x)
    dB_dy = r2 * np.pi * np.cos(np.pi * y) * np.cos(np.pi * x)

    return np.array([[dA_dx, dA_dy],
                      [dB_dx, dB_dy]])


def gen_seq(x0, y0, r1, r2, n_iters, mu=DEFAULT_MU):
    """Generate n_iters points of 2D-STLM starting from (x0, y0)."""
    X = np.zeros(n_iters)
    Y = np.zeros(n_iters)
    x, y = x0, y0
    for i in range(n_iters):
        X[i], Y[i] = x, y
        x, y = map_step(x, y, r1, r2, mu)
    return X, Y


# ---------------------------------------------------------------------------
# Lyapunov exponents (QR method) -- identical numerical method to the
# previous toolkit, just using the 2D-STLM Jacobian above.
# ---------------------------------------------------------------------------
def compute_LE(x0, y0, r1, r2, N, transient, mu=DEFAULT_MU):
    """
    Compute the two Lyapunov exponents of 2D-STLM via the standard
    QR-decomposition method. Returns (LE1, LE2, anisotropy).
    """
    x, y = x0, y0
    for _ in range(transient):
        x, y = map_step(x, y, r1, r2, mu)

    Jp = np.eye(2)
    s1 = s2 = 0.0
    for _ in range(N):
        J = jacobian_step(x, y, r1, r2, mu)
        Q, R = np.linalg.qr(J @ Jp)
        signs = np.sign(np.diag(R))
        signs[signs == 0] = 1.0
        Q = Q * signs
        R = (signs[:, None]) * R
        Jp = Q
        s1 += np.log(max(abs(R[0, 0]), 1e-15))
        s2 += np.log(max(abs(R[1, 1]), 1e-15))
        x, y = map_step(x, y, r1, r2, mu)

    LE1 = s1 / N
    LE2 = s2 / N
    aniso = abs(LE1 - LE2)
    return LE1, LE2, aniso


# ---------------------------------------------------------------------------
# Fixed points
# ---------------------------------------------------------------------------
def newton_fixed_point(x0, y0, r1, r2, max_iter, tol, mu=DEFAULT_MU):
    """
    Newton's method for a fixed point (x*, y*) = map(x*, y*) of 2D-STLM.
    Returns (fp, converged) where fp = np.array([x*, y*]).
    """
    x, y = x0, y0
    I2 = np.eye(2)
    for _ in range(max_iter):
        xn, yn = map_step(x, y, r1, r2, mu)
        r = np.array([xn - x, yn - y])
        if np.linalg.norm(r) < tol:
            return np.array([x, y]), True
        J = jacobian_step(x, y, r1, r2, mu) - I2
        try:
            delta = np.linalg.solve(J, -r)
        except np.linalg.LinAlgError:
            return np.array([x, y]), False
        x = np.mod(x + delta[0], 1.0)
        y = np.mod(y + delta[1], 1.0)
    xn, yn = map_step(x, y, r1, r2, mu)
    r = np.array([xn - x, yn - y])
    return np.array([x, y]), (np.linalg.norm(r) < tol)


# ---------------------------------------------------------------------------
# Byte extraction & randomness statistics (unchanged logic from the
# previous toolkit -- reused verbatim so results are comparable)
# ---------------------------------------------------------------------------
def extract_bytes(X, Y, rounds=4):
    """
    Extract a byte stream from the (x_n, y_n) sequence using the same
    validated 4-round XOR diffusion as before.
    """
    bx = np.clip(np.floor(256.0 * X), 0, 255).astype(np.uint8)
    by = np.clip(np.floor(256.0 * Y), 0, 255).astype(np.uint8)

    mixed = bx.copy() ^ by.copy()
    for k in range(1, rounds):
        mixed ^= np.roll(bx, -k) ^ np.roll(by, -k)

    return mixed.astype(np.uint8)


def shannon_entropy(byte_arr):
    """Shannon entropy (bits) of a byte stream, ideal = 8.0."""
    counts = np.bincount(byte_arr, minlength=256).astype(float)
    p = counts / counts.sum()
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))


def chi2_test(byte_arr):
    """Chi-square goodness-of-fit test against a uniform byte distribution."""
    counts = np.bincount(byte_arr, minlength=256).astype(float)
    expected = len(byte_arr) / 256.0
    return float(np.sum((counts - expected) ** 2 / expected))


def serial_correlation(byte_arr):
    """Serial correlation coefficient between consecutive bytes."""
    b = byte_arr.astype(float)
    n = len(b)
    b1 = b[:-1]
    b2 = b[1:]
    num = n * np.sum(b1 * b2) - np.sum(b1) * np.sum(b2)
    den = np.sqrt((n * np.sum(b1 ** 2) - np.sum(b1) ** 2) *
                  (n * np.sum(b2 ** 2) - np.sum(b2) ** 2))
    if den == 0:
        return 0.0
    return float(num / den)


# ---------------------------------------------------------------------------
# Sample entropy (identical vectorized cKDTree method as before)
# ---------------------------------------------------------------------------
def sample_entropy(series, m, r):
    """SampEn(m, r) of a 1D time series, via cKDTree (Chebyshev norm)."""
    series = np.asarray(series, dtype=float)
    N = len(series)

    def _phi(mlen):
        templates = np.array([series[i:i + mlen] for i in range(N - mlen + 1)])
        tree = cKDTree(templates)
        counts = tree.query_ball_point(templates, r=r, p=np.inf,
                                        return_length=True)
        return int(np.sum(counts)) - len(templates)

    B = _phi(m)
    A = _phi(m + 1)
    if B == 0 or A == 0:
        return 0.0
    return float(-np.log(A / B))


# ---------------------------------------------------------------------------
# Correlation dimension (Grassberger-Procaccia, identical method)
# ---------------------------------------------------------------------------
def correlation_dimension(X, Y, max_points=1500):
    """Estimate the correlation dimension D2 of the (x_n, y_n) attractor."""
    pts = np.column_stack([X, Y])
    n = len(pts)
    if n > max_points:
        idx = np.linspace(0, n - 1, max_points).astype(int)
        pts = pts[idx]
        n = len(pts)

    diff = pts[:, None, :] - pts[None, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=-1))
    iu = np.triu_indices(n, k=1)
    d = dist[iu]
    d = d[d > 0]
    if len(d) < 10:
        return np.nan

    r_min, r_max = np.percentile(d, [5, 50])
    if r_min <= 0 or r_max <= r_min:
        return np.nan
    radii = np.logspace(np.log10(r_min), np.log10(r_max), 12)

    Cr = np.array([np.mean(d <= r) for r in radii])
    valid = Cr > 0
    if np.sum(valid) < 4:
        return np.nan

    logr = np.log(radii[valid])
    logC = np.log(Cr[valid])
    slope, _ = np.polyfit(logr, logC, 1)
    return float(slope)


# ---------------------------------------------------------------------------
# Optional Stage 2: p-adic T-function (Klimov-Shamir), UNCHANGED from the
# previous toolkit. Its role is purely cryptographic (byte-stream
# full-cycle guarantee) and it is independent of the map structure above,
# so it carries over verbatim.
# ---------------------------------------------------------------------------
DEFAULT_L = 32
DEFAULT_C1 = 0xC2B2AE35
DEFAULT_C2 = 0x27D4EB2F

for _name, _c in (("C1", DEFAULT_C1), ("C2", DEFAULT_C2)):
    assert _c % 2 == 1, f"{_name} must be odd"
    assert _c % 8 in (5, 7), f"{_name} must be == 5 or 7 (mod 8) for a full 2^L cycle"


def quantize(x, y, L=DEFAULT_L):
    """Stage 2a: map real state (x, y) in [0,1)^2 to L-bit integers."""
    maxval = (1 << L) - 1
    X = np.clip(np.floor(np.asarray(x) * (1 << L)), 0, maxval).astype(np.uint64)
    Y = np.clip(np.floor(np.asarray(y) * (1 << L)), 0, maxval).astype(np.uint64)
    return X, Y


def t_function(X, C=DEFAULT_C1, L=DEFAULT_L):
    """Klimov-Shamir T-function: T(X) = X + (X^2 | C) mod 2^L."""
    mask = (1 << L) - 1
    X = np.asarray(X)
    if L <= 32:
        Xu = X.astype(np.uint64)
        return ((Xu + ((Xu * Xu) | np.uint64(C))) & np.uint64(mask)).astype(np.uint64)
    else:
        flat = X.astype(object).ravel()
        out = np.array([(int(v) + ((int(v) * int(v)) | C)) & mask for v in flat],
                        dtype=object).reshape(X.shape)
        return out


def apply_padic_layer(x, y, C1=DEFAULT_C1, C2=DEFAULT_C2, L=DEFAULT_L):
    """Full Stage 2 (Phi o Q): (Xh, Yh) = (T(Q_x(x); C1), T(Q_y(y); C2))."""
    X, Y = quantize(x, y, L)
    Xh = t_function(X, C1, L)
    Yh = t_function(Y, C2, L)
    return Xh, Yh


def verify_bijection(C, L):
    """Exhaustively verify bijection + full-cycle property at small L."""
    N = 1 << L
    mask = N - 1
    X = np.arange(N, dtype=np.uint64)
    T = ((X + ((X * X) | np.uint64(C))) & np.uint64(mask))

    is_bijection = len(np.unique(T)) == N

    seen = np.zeros(N, dtype=bool)
    cur = 0
    steps = 0
    while not seen[cur]:
        seen[cur] = True
        cur = int(T[cur])
        steps += 1
    is_full_cycle = bool(is_bijection and steps == N)

    return dict(is_bijection=bool(is_bijection), is_full_cycle=is_full_cycle,
                cycle_length=steps, L=L, C=C)


def extract_bytes_erpm(X, Y, C1=DEFAULT_C1, C2=DEFAULT_C2, L=DEFAULT_L, rounds=4):
    """
    Full 2D-STLM+Stage2 byte extraction: Stage-1 (x_n, y_n) sequences from
    the trigonometric-logistic map are passed through the optional Stage-2
    2-adic layer, then diffused with the same 4-round XOR scheme as
    extract_bytes().
    """
    Xh, Yh = apply_padic_layer(X, Y, C1, C2, L)
    bx = (Xh & np.uint64(0xFF)).astype(np.uint8)
    by = (Yh & np.uint64(0xFF)).astype(np.uint8)

    mixed = bx.copy() ^ by.copy()
    for k in range(1, rounds):
        mixed ^= np.roll(bx, -k) ^ np.roll(by, -k)

    return mixed.astype(np.uint8)


# ---------------------------------------------------------------------------
# Figure saving
# ---------------------------------------------------------------------------
def save_fig(fig, output_dir, name):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"{name}.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    return path