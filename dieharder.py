# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 09:40:19 2026

@author: USER
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
2D-STLM  ->  Dieharder input generator
=============================================================================
Map (both variables updated simultaneously from the same (x_n, y_n)):

  x_{n+1} = [ r1*sin(pi*x_n)*cos(pi*y_n) + mu*y_n*(1-y_n) ] mod 1
  y_{n+1} = [ r2*sin(pi*y_n)*cos(pi*x_n) + mu*x_n*(1-x_n) ] mod 1

Operating point: same as the dynamical analysis and the NIST test in the paper.

Bit extraction (no hashing / no whitening):
  From every iterate the 16 binary digits 17..32 of the state variable are taken,
      w(n) = floor(2^32 * x_n) mod 2^16
  (the same family of digits as the NIST test, which used digit 20).
  Two consecutive words form one 32-bit output value:
      U = w(n) * 2^16 + w(n+1)
  The X stream uses x_n only, the Y stream uses y_n only.

Two ways to use it:
  1) File mode (default): writes stlm_x_die.bin and stlm_y_die.bin
         dieharder -a -g 201 -f stlm_x_die.bin
         dieharder -a -g 201 -f stlm_y_die.bin
  2) Stream mode (recommended, no file rewinding, no disk space):
         python3 stlm_dieharder_generator.py --stream x | dieharder -a -g 200
         python3 stlm_dieharder_generator.py --stream y | dieharder -a -g 200
=============================================================================
"""

import math
import os
import sys
import numpy as np

# ============================================================
#  PARAMETERS (operating point of the paper)
# ============================================================
r1 = 12.0
r2 = 12.0
mu = 3.9
x0 = 0.32665264          # from 2D_STLM_DynamicalResults.mat
y0 = 0.21961687
WARMUP = 1000            # same transient as in the paper

TARGET_BYTES = 4 * 1024**3      # size of each output file in file mode (4 GB)
CHUNK_VALUES = 4_000_000        # uint32 values per chunk (8M iterations)

OUT_X = "stlm_x_die.bin"
OUT_Y = "stlm_y_die.bin"
# ============================================================

if not (0.0 < r1 <= 12.0 and 0.0 < r2 <= 12.0):
    sys.exit("r1 and r2 must lie in (0, 12]")
if not (0.0 < x0 < 1.0 and 0.0 < y0 < 1.0):
    sys.exit("x0 and y0 must lie in (0, 1)")
if x0 == y0 and r1 == r2:
    sys.exit("x0 = y0 with r1 = r2 keeps the orbit on the diagonal; choose x0 != y0")

PI = math.pi
TWO32 = 4294967296.0


def _block(x, y, n_iter, r1, r2, mu, wx, wy):
    """n_iter iterations; stores the 16-bit words of x and y."""
    for i in range(n_iter):
        sx = math.sin(PI * x); cx = math.cos(PI * x)
        sy = math.sin(PI * y); cy = math.cos(PI * y)
        xn = (r1 * sx * cy + mu * y * (1.0 - y)) % 1.0
        yn = (r2 * sy * cx + mu * x * (1.0 - x)) % 1.0
        x, y = xn, yn
        wx[i] = int(x * TWO32) & 0xFFFF
        wy[i] = int(y * TWO32) & 0xFFFF
    return x, y


try:
    from numba import njit
    _block = njit(cache=True)(_block)
    ENGINE = "numba"
except ImportError:
    ENGINE = "python (very slow for 4 GB: install numba with  pip install numba)"


def words_to_u32(w):
    """Pair consecutive 16-bit words into uint32 values."""
    w = w.astype(np.uint32)
    return (w[0::2] << np.uint32(16)) | w[1::2]


def run_file_mode():
    n_values = TARGET_BYTES // 4
    n_chunks = math.ceil(n_values / CHUNK_VALUES)
    print("=" * 62)
    print("  2D-STLM Dieharder generator (file mode)")
    print(f"  r1={r1}, r2={r2}, mu={mu}, x0={x0}, y0={y0}, warm-up={WARMUP}")
    print(f"  Extraction: digits 17-32 of x_n / y_n, 2 iterates -> 1 uint32")
    print(f"  Target: {TARGET_BYTES/1024**3:.2f} GB per file   Engine: {ENGINE}")
    print("=" * 62)

    wx = np.empty(2 * CHUNK_VALUES, dtype=np.uint16)
    wy = np.empty(2 * CHUNK_VALUES, dtype=np.uint16)
    x, y = _block(x0, y0, WARMUP, r1, r2, mu, wx, wy)

    written = 0
    with open(OUT_X, "wb") as fx, open(OUT_Y, "wb") as fy:
        for c in range(n_chunks):
            m = min(CHUNK_VALUES, n_values - written)
            x, y = _block(x, y, 2 * m, r1, r2, mu, wx, wy)
            fx.write(words_to_u32(wx[:2 * m]).tobytes())
            fy.write(words_to_u32(wy[:2 * m]).tobytes())
            written += m
            if c == 0 or (c + 1) % 25 == 0 or c + 1 == n_chunks:
                print(f"  chunk {c+1:4d}/{n_chunks}  ({100*written/n_values:6.2f} %)")

    for f in (OUT_X, OUT_Y):
        print(f"  [OK] {f}: {os.path.getsize(f):,} bytes")
    print("\n  Run:")
    print(f"    dieharder -a -g 201 -f {OUT_X}")
    print(f"    dieharder -a -g 201 -f {OUT_Y}")


def run_stream_mode(which):
    """Endless raw uint32 stream on stdout for:  ... | dieharder -a -g 200"""
    wx = np.empty(2 * CHUNK_VALUES, dtype=np.uint16)
    wy = np.empty(2 * CHUNK_VALUES, dtype=np.uint16)
    x, y = _block(x0, y0, WARMUP, r1, r2, mu, wx, wy)
    out = sys.stdout.buffer
    try:
        while True:
            x, y = _block(x, y, 2 * CHUNK_VALUES, r1, r2, mu, wx, wy)
            out.write(words_to_u32(wx if which == "x" else wy).tobytes())
    except (BrokenPipeError, KeyboardInterrupt):
        pass


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "--stream" and sys.argv[2] in ("x", "y"):
        run_stream_mode(sys.argv[2])
    else:
        run_file_mode()