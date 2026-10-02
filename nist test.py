# -*- coding: utf-8 -*-
"""
2D-STLM: Two-Dimensional Symmetric Trigonometric-Logistic Map
Binary file generator for the NIST SP 800-22 Statistical Test Suite (STS)

Map (both variables updated simultaneously from the same (x_n, y_n)):
  x_{n+1} = [ r1*sin(pi*x_n)*cos(pi*y_n) + mu*y_n*(1-y_n) ] mod 1
  y_{n+1} = [ r2*sin(pi*y_n)*cos(pi*x_n) + mu*x_n*(1-x_n) ] mod 1

Bit extraction (one bit per iteration per stream):
  b_x(n) = floor(x_n * 2^K) mod 2      -> K-th binary digit of x_n
  b_y(n) = floor(y_n * 2^K) mod 2      -> K-th binary digit of y_n

Output files (each: NUM_SEQUENCES x BITS_PER_SEQ bits):
  stlm_x.txt / .bin  - bits from the x-variable only
  stlm_y.txt / .bin  - bits from the y-variable only
"""

import math
import os
import numpy as np

# ─────────────────────────────────────────────
#  Parameters  (same operating point as the dynamical analysis in the paper)
# ─────────────────────────────────────────────
r1 = 12.0
r2 = 12.0
mu = 3.9
x0 = 0.32665264          # from 2D_STLM_DynamicalResults.mat
y0 = 0.21961687

NUM_SEQUENCES = 100
BITS_PER_SEQ = 1_000_000
WARMUP = 1000            # same transient as in the dynamical analysis
K = 20                   # binary digit extracted from each state variable

# OUTPUT_FORMAT:
#   "ascii"  -> one character '0'/'1' per bit : 100 x 10^6 bits = 100 MB per file
#              (in NIST STS choose  Input File Format = [0] ASCII)
#   "binary" -> 8 bits packed per byte        : 100 x 10^6 bits = 12.5 MB per file
#              (in NIST STS choose  Input File Format = [1] Binary)
OUTPUT_FORMAT = "ascii"

OUTPUT_DIR = "."
EXT = ".txt" if OUTPUT_FORMAT == "ascii" else ".bin"
FILES = {
    "x": os.path.join(OUTPUT_DIR, "stlm_x" + EXT),
    "y": os.path.join(OUTPUT_DIR, "stlm_y" + EXT),
}

if x0 == 0.0 and y0 == 0.0:
    raise ValueError("(x0, y0) must not be (0, 0): trivial fixed point of 2D-STLM")

PI = math.pi
SCALE = float(2 ** K)


# ─────────────────────────────────────────────
#  Core loop (numba if available, else plain Python with math)
# ─────────────────────────────────────────────
def _run_block(x, y, n, r1, r2, mu, scale, out_x, out_y):
    for i in range(n):
        sx = math.sin(PI * x); cx = math.cos(PI * x)
        sy = math.sin(PI * y); cy = math.cos(PI * y)
        xn = (r1 * sx * cy + mu * y * (1.0 - y)) % 1.0
        yn = (r2 * sy * cx + mu * x * (1.0 - x)) % 1.0
        x, y = xn, yn
        out_x[i] = int(x * scale) & 1
        out_y[i] = int(y * scale) & 1
    return x, y

try:
    from numba import njit
    _run_block = njit(cache=True)(_run_block)
    ENGINE = "numba"
except ImportError:
    ENGINE = "python (slow: install numba for ~50x speed-up)"


def warmup(x, y, n):
    dummy_x = np.empty(n, dtype=np.uint8)
    dummy_y = np.empty(n, dtype=np.uint8)
    return _run_block(x, y, n, r1, r2, mu, SCALE, dummy_x, dummy_y)


def generate():
    print("=" * 60)
    print("  2D-STLM Binary Generator for NIST SP 800-22")
    print("=" * 60)
    print(f"  r1={r1}, r2={r2}, mu={mu}, x0={x0}, y0={y0}")
    print(f"  Warm-up: {WARMUP}   Extracted digit K = {K}   Engine: {ENGINE}")
    print(f"  {NUM_SEQUENCES} sequences x {BITS_PER_SEQ:,} bits per file\n")

    handles = {k: open(v, "wb") for k, v in FILES.items()}
    x, y = warmup(x0, y0, WARMUP)

    bx = np.empty(BITS_PER_SEQ, dtype=np.uint8)
    by = np.empty(BITS_PER_SEQ, dtype=np.uint8)
    ones = {"x": 0, "y": 0}

    for seq in range(NUM_SEQUENCES):
        x, y = _run_block(x, y, BITS_PER_SEQ, r1, r2, mu, SCALE, bx, by)
        for key, bits in (("x", bx), ("y", by)):
            if OUTPUT_FORMAT == "ascii":
                handles[key].write((bits + 48).tobytes())          # 48 = ord('0')
            else:
                handles[key].write(np.packbits(bits, bitorder="big").tobytes())
            ones[key] += int(bits.sum())
        if seq == 0 or (seq + 1) % 10 == 0:
            print(f"  Seq {seq+1:3d}/{NUM_SEQUENCES} | balance  x={bx.mean():.4f}"
                  f"  y={by.mean():.4f}")

    for h in handles.values():
        h.close()

    total = NUM_SEQUENCES * BITS_PER_SEQ
    print()
    for key, path in FILES.items():
        print(f"  [OK] {path}  ({os.path.getsize(path)/1e6:.2f} MB)"
              f"  overall proportion of ones = {ones[key]/total:.5f}")
    print("\n  --- NIST STS settings -----------------------------")
    fmt = "0 (ASCII)" if OUTPUT_FORMAT == "ascii" else "1 (Binary)"
    print(f"  Command           : ./assess {BITS_PER_SEQ}")
    print(f"  Generator         : 0 (Input File)")
    print(f"  Input file format : {fmt}")
    print(f"  Stream length     : {BITS_PER_SEQ}")
    print(f"  Number of streams : {NUM_SEQUENCES}")
    print("  ---------------------------------------------------")


if __name__ == "__main__":
    generate()