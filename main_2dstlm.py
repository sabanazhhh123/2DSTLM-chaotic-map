# -*- coding: utf-8 -*-
"""
main_2dstlm.py  (corrected version)

Dynamical analysis pipeline for 2D-STLM (2D Symmetric Trigonometric-
Logistic Map):

    x_{n+1} = ( r1*sin(pi*x_n)*cos(pi*y_n) + mu*y_n*(1 - y_n) )  mod 1
    y_{n+1} = ( r2*sin(pi*y_n)*cos(pi*x_n) + mu*x_n*(1 - x_n) )  mod 1

CORRECTIONS IN THIS VERSION
---------------------------
1. Symmetric initial conditions removed.
   When r1 = r2 and x0 = y0, the map gives x_n = y_n for every n, i.e.
   the orbit stays on the invariant diagonal and the system behaves as a
   1D map. The old script used (0.5, 0.5) in Fig 2, Fig 5, Fig 9 and
   Fig 11, which is why:
     - Fig 5(c) showed the base orbit lying on the line x = y,
     - Fig 9 showed low-SE (yellow) cells on the diagonal r1 = r2.
   All of these now use asymmetric initial conditions (x0 != y0).

2. Figure 6 (entropy / chi-square):
     - chi-square now reported with its p-value and the 5% critical
       value (293.25 for 255 degrees of freedom), with PASS/FAIL.
     - chi-square additionally checked over 10 different initial
       conditions (pass rate reported), since a single run can fail by
       chance ~5% of the time even for an ideal generator.
     - KDE curve rescaled to the histogram's count scale and
       boundary-corrected (reflection), so it no longer sits at ~2000
       and no longer drops at byte values 0 and 255.
     - Histogram bins fixed to exact width 4 (64 bins over 0..255).

3. "Operating Point" / "OP" labels moved inside the axes (left of and
   below the star) so they are no longer hidden behind the colorbar.

Everything else (map, Jacobian, LE method, parameters) is unchanged.

Dependencies: numpy, scipy, matplotlib, toolkit_2dstlm.py (same folder)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import savemat
from scipy.stats import norm, chi2 as chi2_dist

from toolkit_2dstlm import (
    map_step, jacobian_step, gen_seq, compute_LE, newton_fixed_point,
    extract_bytes, shannon_entropy, chi2_test, serial_correlation,
    sample_entropy, correlation_dimension, save_fig,
    # Optional Stage 2 (2-adic T-function layer):
    quantize, t_function, apply_padic_layer, verify_bijection,
    extract_bytes_erpm, DEFAULT_L, DEFAULT_C1, DEFAULT_C2, DEFAULT_MU,
)

# ---------------------------------------------------------------------------
# Color palette (kept identical for visual consistency)
# ---------------------------------------------------------------------------
CP = {
    "blue":   (0.12, 0.35, 0.60),
    "red":    (0.75, 0.25, 0.20),
    "green":  (0.20, 0.55, 0.35),
    "orange": (0.85, 0.55, 0.15),
    "gray":   (0.40, 0.40, 0.40),
    "purple": (0.50, 0.20, 0.55),
    "gold":   (0.85, 0.65, 0.15),
    "teal":   (0.15, 0.55, 0.55),
}

plt.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 10,
    "axes.linewidth": 0.6,
    "figure.facecolor": "w",
})

OUTPUT_DIR = "./2D_STLM_Map"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Parameter range for 2D-STLM (paper: r1, r2 in (0, 12])
R_MIN, R_MAX = 0.1, 12.0

# Asymmetric initial conditions used for grid scans / bifurcation.
# IMPORTANT: never use x0 == y0 -- with r1 == r2 the orbit would stay on
# the invariant diagonal x = y and the map would collapse to 1D.
IC_A = (0.7, 0.3)
IC_LIST_SE = [(0.23, 0.61), (0.31, 0.67), (0.72, 0.18)]

CHI2_DOF = 255
CHI2_CRIT_5 = chi2_dist.ppf(0.95, CHI2_DOF)   # = 293.25


def mark_operating_point(ax, P, label="OP", fontsize=10):
    """Star at the operating point + label placed INSIDE the axes
    (left of and below the star), so it is never hidden behind the
    colorbar when the operating point sits on the plot edge."""
    ax.plot(P["r1"], P["r2"], marker="*", markersize=16, color="k",
            markerfacecolor="y", linewidth=1.5, clip_on=False, zorder=5)
    ax.annotate(label, xy=(P["r1"], P["r2"]),
                xytext=(-10, -14), textcoords="offset points",
                ha="right", va="top", fontsize=fontsize, color="k",
                fontweight="bold", zorder=6,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="w",
                          edgecolor="none", alpha=0.8))


def section0_parameters():
    """SECTION 0: PARAMETERS"""
    print("=================================================")
    print("  2D-STLM Analysis Script")
    print("  Symmetric Trigonometric-Logistic Map")
    print("=================================================\n")

    r1 = 12.0  # control parameter, range (0, 12]
    r2 = 12.0  # control parameter, range (0, 12]
    mu = DEFAULT_MU  # fixed logistic cross-coupling strength (= 3.9)

    # x0 != y0 (required: see note on the invariant diagonal above)
    x0 = 0.326652641700000
    y0 = 0.219616874200000

    W = 500
    total_iters = W + 256

    L2 = DEFAULT_L
    C1 = DEFAULT_C1
    C2 = DEFAULT_C2

    print("Map Parameters:")
    print(f"  r1={r1:.2f}, r2={r2:.2f}, mu={mu:.2f}")
    print(f"  IC: ({x0:.10f}, {y0:.10f})")
    print(f"  Optional Stage 2: L={L2}, C1={hex(C1)}, C2={hex(C2)}\n")

    return dict(r1=r1, r2=r2, mu=mu, x0=x0, y0=y0, W=W,
                total_iters=total_iters, L2=L2, C1=C1, C2=C2)


def section1_table1_LE_spectrum(P):
    """SECTION 1: TABLE 1 - Lyapunov Spectrum vs r1 (r2 held fixed)"""
    print("========================================")
    print("TABLE 1: Lyapunov Spectrum vs r1")
    print("========================================")

    r2, mu = P["r2"], P["mu"]
    r1_vals = [0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0]
    print(f"{'r1':<6} {'LE1':<8} {'LE2':<8} {'KE':<8} {'Anisotropy':<10}")
    print("-" * 45)
    for r1v in r1_vals:
        l1, l2, aniso = compute_LE(IC_A[0], IC_A[1], r1v, r2, 200000, 20000, mu)
        print(f"{r1v:<6} {l1:<8.4f} {l2:<8.4f} {l1+l2:<8.4f} {aniso:<10.4f}")
    print()

    print("High-precision LEs (N=1e6)...")
    LE1, LE2, ANISO = compute_LE(P["x0"], P["y0"], P["r1"], P["r2"],
                                  1000000, 20000, mu)
    KE_main = LE1 + LE2
    print(f"LE1={LE1:.4f}  LE2={LE2:.4f}  KE={KE_main:.4f}  A={ANISO:.4f}\n")

    return dict(LE1=LE1, LE2=LE2, KE_main=KE_main, ANISO=ANISO)


def section2_fig1_LE_convergence(P):
    """SECTION 2: FIGURE 1 - LE Convergence Study"""
    print("[Fig 1] LE Convergence...")
    r1, r2, mu = P["r1"], P["r2"], P["mu"]

    N_conv, N_step = 200000, 500
    n_pts = N_conv // N_step
    le1_c = np.zeros(n_pts)
    le2_c = np.zeros(n_pts)
    idx_c = 0

    x, y = P["x0"], P["y0"]
    for _ in range(20000):
        x, y = map_step(x, y, r1, r2, mu)

    Jp = np.eye(2)
    s1 = s2 = 0.0
    pt = 0
    for i in range(1, N_conv + 1):
        J = jacobian_step(x, y, r1, r2, mu)
        Q, R = np.linalg.qr(J @ Jp)
        signs = np.sign(np.diag(R))
        signs[signs == 0] = 1.0
        Q = Q * signs
        R = signs[:, None] * R
        Jp = Q
        s1 += np.log(max(abs(R[0, 0]), 1e-15))
        s2 += np.log(max(abs(R[1, 1]), 1e-15))
        x, y = map_step(x, y, r1, r2, mu)
        pt += 1
        if i % N_step == 0:
            le1_c[idx_c] = s1 / pt
            le2_c[idx_c] = s2 / pt
            idx_c += 1

    n_ax = np.arange(1, n_pts + 1) * N_step

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(n_ax, le1_c, color=CP["blue"], lw=1.5, label=r"$\lambda_1$")
    axes[0].plot(n_ax, le2_c, color=CP["green"], lw=1.5, label=r"$\lambda_2$")
    axes[0].axhline(0, color="k", ls="--", lw=0.8)
    axes[0].set_xlabel("Iteration N", fontweight="bold")
    axes[0].set_ylabel("Lyapunov Exponent", fontweight="bold")
    axes[0].set_title("(a) LE Convergence vs N", fontweight="bold")
    axes[0].legend(loc="best", fontsize=10)
    axes[0].grid(True)

    axes[1].plot(n_ax, le1_c - le1_c[-1], color=CP["blue"], lw=1.5, label=r"$\lambda_1$")
    axes[1].plot(n_ax, le2_c - le2_c[-1], color=CP["green"], lw=1.5, label=r"$\lambda_2$")
    axes[1].axhline(0, color="k", ls="--", lw=0.8)
    axes[1].set_xlabel("Iteration N", fontweight="bold")
    axes[1].set_ylabel("Deviation from Converged Value", fontweight="bold")
    axes[1].set_title("(b) Convergence Deviation", fontweight="bold")
    axes[1].legend(loc="best", fontsize=10)
    axes[1].grid(True)

    save_fig(fig, OUTPUT_DIR, "Figure1_LEConvergence")
    plt.close(fig)


def section3_fixed_points(P):
    """SECTION 3: FIXED POINT ANALYSIS"""
    print("\n========================================")
    print("  FIXED POINT ANALYSIS")
    print("========================================")

    r1, r2, mu = P["r1"], P["r2"], P["mu"]

    fp_list = []
    fp_eigs = []
    fp_types = []
    seeds = [0.15, 0.35, 0.55, 0.75, 0.95]
    types_lut = ["Stable", "Saddle", "Repeller-2"]

    for sx in seeds:
        for sy in seeds:
            fp, conv = newton_fixed_point(sx, sy, r1, r2, 200, 1e-12, mu)
            if conv:
                is_new = True
                for existing in fp_list:
                    if np.linalg.norm(fp - existing) < 1e-6:
                        is_new = False
                        break
                if is_new:
                    fp_list.append(fp)
                    Jfp = jacobian_step(fp[0], fp[1], r1, r2, mu)
                    ev = np.linalg.eigvals(Jfp)
                    fp_eigs.append(np.abs(ev))
                    n_uns = int(np.sum(np.abs(ev) > 1))
                    fp_types.append(types_lut[min(n_uns, 2)])

    fp_list = np.array(fp_list) if fp_list else np.zeros((0, 2))
    fp_eigs = np.array(fp_eigs) if fp_eigs else np.zeros((0, 2))

    print(f"Found {len(fp_list)} distinct fixed points:")
    print(f"{'#':<4} {'x*':<12} {'y*':<12} {'|eigenvalues|':<25} {'Type':<12}")
    print("-" * 65)
    for k in range(len(fp_list)):
        eig_str = f"[{fp_eigs[k,0]:<8.4f} {fp_eigs[k,1]:<8.4f}]"
        print(f"{k+1:<4} {fp_list[k,0]:<12.6f} {fp_list[k,1]:<12.6f}  "
              f"{eig_str:<25} {fp_types[k]:<12}")

    n_rep = fp_types.count("Repeller-2")
    n_sad = fp_types.count("Saddle")
    n_stab = fp_types.count("Stable")
    print(f"\nSummary: {n_rep} Repeller-2, {n_sad} Saddle, {n_stab} Stable")
    if n_stab == 0:
        print("INTERPRETATION: No stable fixed points found --- consistent")
        print("with the map being chaotic with no stable periodic windows.\n")

    return dict(fp_list=fp_list, fp_eigs=fp_eigs, fp_types=fp_types)


def section4_fig2_bifurcation(P):
    """SECTION 4: FIGURE 2 - Bifurcation Diagrams (vs r1)"""
    print("[Fig 2] Bifurcation Diagrams...")
    r2, mu = P["r2"], P["mu"]

    r1_bif = np.linspace(R_MIN, R_MAX, 400)
    n_tr, n_pl = 2000, 300
    Nr = len(r1_bif)

    bif_x = np.zeros(Nr * n_pl)
    bif_y = np.zeros(Nr * n_pl)
    par_r1 = np.zeros(Nr * n_pl)

    for i, r1v in enumerate(r1_bif):
        # FIX: asymmetric IC (was (0.5, 0.5), which collapses to the
        # diagonal at r1 = r2 = 12)
        Xb, Yb = gen_seq(IC_A[0], IC_A[1], r1v, r2, n_tr + n_pl, mu)
        idx_s = i * n_pl
        idx_e = (i + 1) * n_pl
        bif_x[idx_s:idx_e] = Xb[n_tr:]
        bif_y[idx_s:idx_e] = Yb[n_tr:]
        par_r1[idx_s:idx_e] = r1v
        if (i + 1) % 100 == 0:
            print(f"  Bif: {i+1}/{Nr}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].scatter(par_r1, bif_x, s=2, c=[CP["blue"]], edgecolors="none", alpha=0.2)
    axes[0].set_xlabel(r"$r_1$", fontweight="bold")
    axes[0].set_ylabel(r"$x_n$", fontweight="bold")
    axes[0].set_title(r"(a) $x_n$ vs $r_1$", fontweight="bold")
    axes[0].set_xlim(r1_bif[0], r1_bif[-1])
    axes[0].set_ylim(0, 1)
    axes[0].grid(True)

    axes[1].scatter(par_r1, bif_y, s=2, c=[CP["green"]], edgecolors="none", alpha=0.2)
    axes[1].set_xlabel(r"$r_1$", fontweight="bold")
    axes[1].set_ylabel(r"$y_n$", fontweight="bold")
    axes[1].set_title(r"(b) $y_n$ vs $r_1$", fontweight="bold")
    axes[1].set_xlim(r1_bif[0], r1_bif[-1])
    axes[1].set_ylim(0, 1)
    axes[1].grid(True)

    save_fig(fig, OUTPUT_DIR, "Figure2_Bifurcation")
    plt.close(fig)


def section5_fig3_LE_classification(P):
    """SECTION 5: FIGURE 3 - 2D Lyapunov Classification Chart (r1 x r2)"""
    print("[Fig 3] 2D Lyapunov Classification...")
    mu = P["mu"]

    grid_lc = 30
    r1_lc = np.linspace(R_MIN, R_MAX, grid_lc)
    r2_lc = np.linspace(R_MIN, R_MAX, grid_lc)
    class_map = np.zeros((grid_lc, grid_lc))

    for i in range(grid_lc):
        for j in range(grid_lc):
            l1, l2, _ = compute_LE(IC_A[0], IC_A[1], r1_lc[i], r2_lc[j], 40000, 10000, mu)
            thr = 0.02
            class_map[i, j] = sum(v > thr for v in (l1, l2))
        if (i + 1) % 10 == 0:
            print(f"  LE class: {i+1}/{grid_lc}")

    from matplotlib.colors import ListedColormap
    cmap_lc = ListedColormap([
        (0.15, 0.15, 0.15), (0.20, 0.40, 0.80), (0.90, 0.30, 0.30),
    ])

    fig, ax = plt.subplots(figsize=(7.5, 6.2))
    im = ax.imshow(class_map.T, origin="lower", cmap=cmap_lc, vmin=-0.5, vmax=2.5,
                   extent=[r1_lc[0], r1_lc[-1], r2_lc[0], r2_lc[-1]], aspect="auto")
    ax.set_xlabel(r"$r_1$", fontweight="bold")
    ax.set_ylabel(r"$r_2$", fontweight="bold")
    ax.set_title("2D Lyapunov Classification Chart", fontweight="bold")
    cb = fig.colorbar(im, ax=ax, ticks=[0, 1, 2])
    cb.ax.set_yticklabels(["Stable", r"Chaos (1$\lambda^+$)", r"Hyperchaos (2$\lambda^+$)"])
    mark_operating_point(ax, P, label="Operating Point", fontsize=9)

    save_fig(fig, OUTPUT_DIR, "Figure3_LEClassification")
    plt.close(fig)

    hyper_pct = 100.0 * np.mean(class_map == 2)
    print(f"  Hyperchaotic fraction of the (r1, r2) grid: {hyper_pct:.1f}%")
    return dict(class_map=class_map, hyper_pct=hyper_pct)


def section6_fig4_phase_space(P):
    """SECTION 6: FIGURE 4 - Phase Space + Time Series"""
    print("[Fig 4] Phase Space...")
    r1, r2, mu = P["r1"], P["r2"], P["mu"]

    n_transient = 3000
    n_plot = 30000

    x, y = P["x0"], P["y0"]
    for _ in range(n_transient):
        x, y = map_step(x, y, r1, r2, mu)

    Xp, Yp = gen_seq(x, y, r1, r2, n_plot, mu)
    colors = np.arange(n_plot)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))

    sc = axes[0].scatter(Xp, Yp, c=colors, s=3, cmap="viridis", alpha=0.25)
    axes[0].set_xlabel(r"$x_n$", fontweight="bold")
    axes[0].set_ylabel(r"$y_n$", fontweight="bold")
    axes[0].set_title(r"(a) Phase Space ($x_n$ vs $y_n$)", fontweight="bold")
    axes[0].set_aspect("equal", adjustable="box")
    cb = fig.colorbar(sc, ax=axes[0], shrink=0.8)
    cb.set_label("Iteration n (post burn-in)")
    axes[0].grid(True)

    n_show = 300
    it_ax = np.arange(n_show)
    axes[1].plot(it_ax, Xp[:n_show], color=CP["blue"], lw=1.2, label=r"$x_n$")
    axes[1].plot(it_ax, Yp[:n_show], color=CP["red"], lw=1.2, label=r"$y_n$")
    axes[1].set_xlabel("Iteration n", fontweight="bold")
    axes[1].set_ylabel("State value", fontweight="bold")
    axes[1].set_title("(b) Time Series (first 300 iterations)", fontweight="bold")
    axes[1].legend(loc="best", fontsize=9)
    axes[1].grid(True)

    save_fig(fig, OUTPUT_DIR, "Figure4_PhaseSpace")
    plt.close(fig)

    gx = np.clip((Xp * 20).astype(int), 0, 19)
    gy = np.clip((Yp * 20).astype(int), 0, 19)
    visited = len(set(zip(gx, gy)))
    print(f"  Phase-space grid coverage: {visited}/400 cells visited "
          f"({100*visited/400:.1f}%)")

    return dict(coverage_pct=100 * visited / 400)


def section7_fig5_sensitivity(P):
    """SECTION 7: FIGURE 5 - Sensitivity Analysis"""
    print("[Fig 5] Sensitivity...")
    r1, r2, mu = P["r1"], P["r2"], P["mu"]

    n_sens = 200
    dv = [1e-5, 2e-5]
    # FIX: base orbit now starts from the asymmetric operating IC
    # (was (0.5, 0.5), which kept the base orbit on the diagonal x = y)
    x0, y0 = P["x0"], P["y0"]
    Xb, Yb = gen_seq(x0, y0, r1, r2, n_sens, mu)
    Xd, Yd = gen_seq(x0 + dv[0], y0 + dv[1], r1, r2, n_sens, mu)
    dist5 = np.sqrt((Xb - Xd) ** 2 + (Yb - Yd) ** 2)
    iters5 = np.arange(1, n_sens + 1)

    fig = plt.figure(figsize=(12, 8))

    ax1 = fig.add_subplot(2, 2, 1)
    ax1.plot(iters5, Xb, color=CP["blue"], lw=1.2, label="Base")
    ax1.plot(iters5, Xd, color=CP["red"], lw=1.0, ls="--", label="Perturbed")
    ax1.set_ylabel(r"$x_n$", fontweight="bold")
    ax1.set_title(r"(a) $x_n$  ($\Delta x_0=10^{-5}$)", fontweight="bold")
    ax1.legend(loc="best", fontsize=9)
    ax1.grid(True)
    ax1.set_xlim(0, n_sens)

    ax2 = fig.add_subplot(2, 2, 2)
    ax2.plot(iters5, Yb, color=CP["blue"], lw=1.2, label="Base")
    ax2.plot(iters5, Yd, color=CP["red"], lw=1.0, ls="--", label="Perturbed")
    ax2.set_ylabel(r"$y_n$", fontweight="bold")
    ax2.set_title(r"(b) $y_n$  ($\Delta y_0=2\times10^{-5}$)", fontweight="bold")
    ax2.legend(loc="best", fontsize=9)
    ax2.grid(True)
    ax2.set_xlim(0, n_sens)

    ax3 = fig.add_subplot(2, 2, 3)
    ax3.scatter(Xb, Yb, s=3, color=CP["blue"], alpha=0.5, label="Base")
    ax3.scatter(Xd, Yd, s=3, color=CP["red"], alpha=0.5, label="Perturbed")
    ax3.set_xlabel(r"$x_n$", fontweight="bold")
    ax3.set_ylabel(r"$y_n$", fontweight="bold")
    ax3.set_title("(c) Phase Divergence", fontweight="bold")
    ax3.legend(loc="best", fontsize=9)
    ax3.grid(True)
    ax3.set_aspect("equal", adjustable="box")

    ax4 = fig.add_subplot(2, 2, 4)
    ax4.semilogy(iters5, dist5 + np.finfo(float).eps, color=CP["red"], lw=1.5)
    ax4.set_xlabel("Iteration n", fontweight="bold")
    ax4.set_ylabel(r"$||\Delta||_2$", fontweight="bold")
    ax4.set_title("(d) Euclidean Distance (log)", fontweight="bold")
    ax4.grid(True)
    ax4.set_xlim(0, n_sens)

    save_fig(fig, OUTPUT_DIR, "Figure5_Sensitivity")
    plt.close(fig)

    # iteration at which the two orbits have fully separated (> 0.1)
    sep = np.argmax(dist5 > 0.1) + 1 if np.any(dist5 > 0.1) else None
    print(f"  Orbits separated (||Delta|| > 0.1) after n = {sep} iterations")


def _reflected_kde_counts(byte_arr, bin_width):
    """Boundary-corrected (reflection) Gaussian KDE of a byte stream,
    returned on the SAME count scale as a histogram with the given bin
    width, so it overlays the bars correctly."""
    counts = np.bincount(byte_arr, minlength=256).astype(float)
    vals = np.arange(256, dtype=float)
    n = counts.sum()
    mean = np.sum(vals * counts) / n
    std = np.sqrt(np.sum(counts * (vals - mean) ** 2) / n)
    bw = 1.06 * std * n ** (-0.2)
    bw = max(bw, 1.0)

    kde_x = np.linspace(0, 255, 512)
    dens = np.zeros_like(kde_x)
    # reflect at both boundaries (-0.5 and 255.5) to remove edge drop
    for centers in (vals, -1.0 - vals, 511.0 - vals):
        dens += (counts[None, :] *
                 norm.pdf(kde_x[:, None], loc=centers[None, :], scale=bw)).sum(axis=1)
    dens /= n                     # probability density per byte value
    return kde_x, dens * n * bin_width


def section8_fig6_entropy(P):
    """SECTION 8: FIGURE 6 - Entropy Analysis (Byte-level randomness)"""
    print("[Fig 6] Entropy Analysis...")
    r1, r2, mu = P["r1"], P["r2"], P["mu"]

    N_ent, W_ent = 500000, 10000
    Xe, Ye = gen_seq(P["x0"], P["y0"], r1, r2, W_ent + N_ent, mu)
    bytes_ent = extract_bytes(Xe[W_ent:], Ye[W_ent:])

    H_val = shannon_entropy(bytes_ent)
    chi2 = chi2_test(bytes_ent)
    chi2_p = float(chi2_dist.sf(chi2, CHI2_DOF))
    chi2_pass = chi2 < CHI2_CRIT_5
    sc_val = serial_correlation(bytes_ent)
    uq = len(np.unique(bytes_ent))
    print(f"  H={H_val:.6f}  Chi2={chi2:.2f} (p={chi2_p:.4f}, "
          f"crit={CHI2_CRIT_5:.2f}, {'PASS' if chi2_pass else 'FAIL'})  "
          f"SC={sc_val:.2e}  Unique={uq}")

    # --- chi-square robustness over multiple initial conditions -------
    # A single chi-square run fails by chance ~5% of the time even for a
    # perfect generator, so report the pass rate over several ICs.
    print("  Chi-square robustness over 10 initial conditions...")
    rng = np.random.default_rng(2026)
    chi2_multi, p_multi = [], []
    for _ in range(10):
        ax0, ay0 = rng.uniform(0.01, 0.99, 2)
        Xm, Ym = gen_seq(ax0, ay0, r1, r2, W_ent + N_ent, mu)
        bm = extract_bytes(Xm[W_ent:], Ym[W_ent:])
        c = chi2_test(bm)
        chi2_multi.append(c)
        p_multi.append(float(chi2_dist.sf(c, CHI2_DOF)))
    chi2_multi = np.array(chi2_multi)
    p_multi = np.array(p_multi)
    pass_rate = 100.0 * np.mean(chi2_multi < CHI2_CRIT_5)
    print(f"    mean Chi2={chi2_multi.mean():.2f} +/- {chi2_multi.std():.2f}, "
          f"pass rate={pass_rate:.0f}% (expected ~95% for an ideal source)")

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    bdata = bytes_ent.astype(float)

    # (a) histogram with exact width-4 bins + KDE on the same scale
    n_bins = 64
    bin_width = 256 / n_bins
    edges = np.arange(0, 256 + bin_width, bin_width) - 0.5
    axes[0].hist(bdata, bins=edges, color=CP["green"], edgecolor="none",
                 alpha=0.75, label="Observed")
    exp_c = len(bytes_ent) / n_bins
    axes[0].plot([0, 255], [exp_c, exp_c], "k--", lw=1.5, label="Ideal uniform")
    kde_x, kde_y = _reflected_kde_counts(bytes_ent, bin_width)
    axes[0].plot(kde_x, kde_y, "-", color=CP["red"], lw=2, label="KDE")
    axes[0].set_xlim(-0.5, 255.5)
    axes[0].set_ylim(0, exp_c * 1.25)
    axes[0].set_xlabel("Byte Value", fontweight="bold")
    axes[0].set_ylabel("Frequency", fontweight="bold")
    axes[0].set_title("(a) Distribution + KDE", fontweight="bold")
    axes[0].legend(loc="lower center", fontsize=9)
    axes[0].grid(True)

    bsort = np.sort(bdata)
    ecdf_y = np.arange(1, len(bsort) + 1) / len(bsort)
    axes[1].plot(bsort, ecdf_y, "-", color=CP["blue"], lw=1.5, label="Observed")
    axes[1].plot([0, 255], [0, 1], "k--", lw=1.5, label="Ideal")
    axes[1].set_xlabel("Byte Value", fontweight="bold")
    axes[1].set_ylabel("Empirical CDF", fontweight="bold")
    axes[1].set_title("(b) Cumulative Distribution", fontweight="bold")
    axes[1].legend(loc="best", fontsize=10)
    axes[1].grid(True)

    max_lag = 300
    acorr = np.zeros(max_lag)
    for lag in range(1, max_lag + 1):
        x1a = bdata[: len(bdata) - lag]
        x2a = bdata[lag:]
        acorr[lag - 1] = (np.sum((x1a - x1a.mean()) * (x2a - x2a.mean())) /
                           np.sqrt(np.sum((x1a - x1a.mean()) ** 2) * np.sum((x2a - x2a.mean()) ** 2)))
    axes[2].stem(np.arange(1, max_lag + 1), acorr, markerfmt=".", basefmt=" ",
                 linefmt=CP["blue"])
    axes[2].axhline(0, color="k", lw=0.8)
    ci_lim = 1.96 / np.sqrt(len(bytes_ent))
    axes[2].axhline(ci_lim, color="r", ls="--", lw=1.0, label="95% CI")
    axes[2].axhline(-ci_lim, color="r", ls="--", lw=1.0)
    axes[2].set_xlabel("Lag", fontweight="bold")
    axes[2].set_ylabel("Autocorrelation", fontweight="bold")
    axes[2].set_title("(c) Autocorrelation", fontweight="bold")
    axes[2].legend(loc="upper right", fontsize=9)
    axes[2].grid(True)
    axes[2].set_xlim(0, max_lag)
    frac_out = 100.0 * np.mean(np.abs(acorr) > ci_lim)
    print(f"  Autocorrelation: {frac_out:.1f}% of lags outside 95% CI "
          f"(expected ~5% for an ideal source)")

    fig.subplots_adjust(right=0.88)
    fig.text(0.895, 0.7,
             f"H = {H_val:.4f}\n"
             f"$\\chi^2$ = {chi2:.1f} (crit. {CHI2_CRIT_5:.1f})\n"
             f"p = {chi2_p:.3f}\n"
             f"SC = {sc_val:.2e}\n"
             f"N = {len(bytes_ent)//1000}k",
             fontsize=10, ha="left", va="top",
             bbox=dict(boxstyle="round", edgecolor=(0.3, 0.3, 0.3), facecolor=(0.97, 0.97, 0.97)))

    save_fig(fig, OUTPUT_DIR, "Figure6_Entropy")
    plt.close(fig)

    return dict(bytes_ent=bytes_ent, H_val=H_val, chi2=chi2, chi2_p=chi2_p,
                chi2_pass=chi2_pass, chi2_multi=chi2_multi, p_multi=p_multi,
                chi2_pass_rate=pass_rate, sc_val=sc_val, acorr_out_pct=frac_out)


def section13_fig9_sample_entropy(P):
    """SECTION 13: FIGURE 9 - Sample Entropy (SE) Surface Analysis (r1 x r2)"""
    print("\n[Fig 9] Sample Entropy Surface Analysis...")
    print("  Computing SE over r1-r2 parameter grid (multi-IC average)...")
    mu = P["mu"]

    grid_se = 25
    r1_se = np.linspace(R_MIN, R_MAX, grid_se)
    r2_se = np.linspace(R_MIN, R_MAX, grid_se)
    SE_map = np.zeros((grid_se, grid_se))

    N_se, W_se = 4000, 1000
    # FIX: (0.5, 0.5) removed -- on the diagonal r1 = r2 it collapsed the
    # orbit to x = y and produced the artificial low-SE diagonal cells.
    ic_list = IC_LIST_SE

    for i in range(grid_se):
        for j in range(grid_se):
            vals = []
            for icx, icy in ic_list:
                Xse, Yse = gen_seq(icx, icy, r1_se[i], r2_se[j], N_se, mu)
                vals.append(sample_entropy(Xse[W_se:], 2, 0.2))
            SE_map[i, j] = np.mean(vals)
        if (i + 1) % 5 == 0:
            print(f"    SE grid: {i+1}/{grid_se}")

    fig = plt.figure(figsize=(14, 5.5))
    R1, R2 = np.meshgrid(r1_se, r2_se, indexing="ij")

    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax1.plot_surface(R1, R2, SE_map, cmap="jet")
    ax1.set_xlabel(r"$r_1$", fontweight="bold")
    ax1.set_ylabel(r"$r_2$", fontweight="bold")
    ax1.set_zlabel("Sample Entropy", fontweight="bold")
    ax1.set_title(r"(a) SE Surface ($r_1 \times r_2$)", fontweight="bold")
    ax1.view_init(elev=30, azim=-30)

    ax2 = fig.add_subplot(1, 2, 2)
    im = ax2.imshow(SE_map.T, origin="lower", cmap="jet",
                     extent=[r1_se[0], r1_se[-1], r2_se[0], r2_se[-1]], aspect="auto")
    fig.colorbar(im, ax=ax2, label="SE")
    ax2.set_xlabel(r"$r_1$", fontweight="bold")
    ax2.set_ylabel(r"$r_2$", fontweight="bold")
    ax2.set_title(r"(b) SE Heatmap ($r_1 \times r_2$)", fontweight="bold")
    mark_operating_point(ax2, P, label="OP")

    save_fig(fig, OUTPUT_DIR, "Figure9_SampleEntropy_Surface")
    plt.close(fig)

    Xse_op, Yse_op = gen_seq(P["x0"], P["y0"], P["r1"], P["r2"], 5000, mu)
    SE_x = sample_entropy(Xse_op[1000:], 2, 0.2)
    SE_y = sample_entropy(Yse_op[1000:], 2, 0.2)
    print(f"  SE at operating point: SE_x={SE_x:.4f}, SE_y={SE_y:.4f}")

    return dict(SE_map=SE_map, SE_x=SE_x, SE_y=SE_y)


def section14_fig10_kolmogorov_entropy(P):
    """SECTION 14: FIGURE 10 - Kolmogorov Entropy (KE) Analysis (r1 x r2)"""
    print("\n[Fig 10] Kolmogorov Entropy Analysis...")
    print("  For 2D-STLM: KE = LE1 + LE2")

    r2_fixed, mu = P["r2"], P["mu"]

    r1_ke = np.linspace(R_MIN, R_MAX, 60)
    KE_curve = np.zeros(len(r1_ke))
    LE1_ke = np.zeros(len(r1_ke))
    LE2_ke = np.zeros(len(r1_ke))

    for i, r1v in enumerate(r1_ke):
        l1, l2, _ = compute_LE(IC_A[0], IC_A[1], r1v, r2_fixed, 60000, 10000, mu)
        LE1_ke[i], LE2_ke[i] = l1, l2
        KE_curve[i] = l1 + l2
        if (i + 1) % 15 == 0:
            print(f"    KE curve: {i+1}/{len(r1_ke)}")

    fig = plt.figure(figsize=(14, 5.5))
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.plot(r1_ke, KE_curve, color=CP["purple"], lw=2.5, label=r"KE = $\Sigma\lambda$")
    ax1.plot(r1_ke, LE1_ke, "--", color=CP["blue"], lw=1.2, label=r"$\lambda_1$")
    ax1.plot(r1_ke, LE2_ke, "--", color=CP["green"], lw=1.2, label=r"$\lambda_2$")
    ax1.axhline(0, color="k", ls=":", lw=0.8)
    ax1.set_xlabel(r"$r_1$", fontweight="bold")
    ax1.set_ylabel("Entropy / LE", fontweight="bold")
    ax1.set_title(r"(a) Kolmogorov Entropy vs $r_1$", fontweight="bold")
    ax1.legend(loc="best", fontsize=9)
    ax1.grid(True)
    ax1.set_xlim(r1_ke[0], r1_ke[-1])

    print("  Computing KE 2D surface (r1 x r2)...")
    grid_ke = 20
    r1_ke2 = np.linspace(R_MIN, R_MAX, grid_ke)
    r2_ke = np.linspace(R_MIN, R_MAX, grid_ke)
    KE_surf = np.zeros((grid_ke, grid_ke))

    for i in range(grid_ke):
        for j in range(grid_ke):
            l1, l2, _ = compute_LE(IC_A[0], IC_A[1], r1_ke2[i], r2_ke[j], 40000, 8000, mu)
            KE_surf[i, j] = l1 + l2
        if (i + 1) % 5 == 0:
            print(f"    KE surface: {i+1}/{grid_ke}")

    ax2 = fig.add_subplot(1, 2, 2)
    im = ax2.contourf(r1_ke2, r2_ke, KE_surf.T, levels=20, cmap="hot")
    fig.colorbar(im, ax=ax2, label="KE")
    ax2.set_xlabel(r"$r_1$", fontweight="bold")
    ax2.set_ylabel(r"$r_2$", fontweight="bold")
    ax2.set_title(r"(b) KE Contour ($r_1 \times r_2$)", fontweight="bold")
    mark_operating_point(ax2, P, label="OP")
    ax2.grid(True)

    save_fig(fig, OUTPUT_DIR, "Figure10_KolmogorovEntropy")
    plt.close(fig)

    return dict(KE_curve=KE_curve, KE_surf=KE_surf)


def section15_fig11_correlation_dimension(P):
    """SECTION 15: FIGURE 11 - Correlation Dimension (CD) Analysis"""
    print("\n[Fig 11] Correlation Dimension Analysis...")
    r1, r2, mu = P["r1"], P["r2"], P["mu"]

    N_cd_conv = [500, 1000, 2000, 3000, 5000, 8000, 12000, 18000, 25000]
    CD_conv = np.zeros(len(N_cd_conv))

    Xcd, Ycd = gen_seq(P["x0"], P["y0"], r1, r2, 30000, mu)

    for i, n in enumerate(N_cd_conv):
        CD_conv[i] = correlation_dimension(Xcd[:n], Ycd[:n])
        print(f"    CD convergence: N={n}, CD={CD_conv[i]:.4f}")

    print("  Computing CD parameter surface (r1 x r2)...")
    grid_cd = 15
    r1_cd = np.linspace(R_MIN, R_MAX, grid_cd)
    r2_cd = np.linspace(R_MIN, R_MAX, grid_cd)
    CD_surf = np.zeros((grid_cd, grid_cd))

    for i in range(grid_cd):
        for j in range(grid_cd):
            # FIX: asymmetric IC (was (0.5, 0.5) -> 1D orbit on r1 = r2)
            Xcdp, Ycdp = gen_seq(IC_A[0], IC_A[1], r1_cd[i], r2_cd[j], 3000, mu)
            CD_surf[i, j] = correlation_dimension(Xcdp[500:], Ycdp[500:])
        if (i + 1) % 3 == 0:
            print(f"    CD surface: {i+1}/{grid_cd}")

    fig = plt.figure(figsize=(14, 5.5))
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.plot(N_cd_conv, CD_conv, "-o", color=CP["teal"], lw=2, markerfacecolor=CP["teal"], markersize=8)
    ax1.axhline(2, color="k", ls="--", lw=1.5, label="Embedding dim = 2")
    ax1.axhline(1, color="r", ls=":", lw=1.2, label="Lower bound")
    ax1.set_xlabel("Number of Points N", fontweight="bold")
    ax1.set_ylabel(r"Correlation Dimension $D_2$", fontweight="bold")
    ax1.set_title("(a) CD Convergence", fontweight="bold")
    ax1.legend(loc="best", fontsize=9)
    ax1.grid(True)
    ax1.set_xlim(0, max(N_cd_conv) * 1.05)

    ax2 = fig.add_subplot(1, 2, 2)
    im = ax2.contourf(r1_cd, r2_cd, CD_surf.T, levels=15, cmap="viridis")
    fig.colorbar(im, ax=ax2, label=r"$D_2$")
    ax2.set_xlabel(r"$r_1$", fontweight="bold")
    ax2.set_ylabel(r"$r_2$", fontweight="bold")
    ax2.set_title(r"(b) CD Contour ($r_1 \times r_2$)", fontweight="bold")
    mark_operating_point(ax2, P, label="OP")
    ax2.grid(True)

    save_fig(fig, OUTPUT_DIR, "Figure11_CorrelationDimension")
    plt.close(fig)

    return dict(CD_conv=CD_conv, CD_surf=CD_surf)


def section16_stage2_verification(P):
    """SECTION 16: Optional Stage 2 - 2-adic T-function Layer Verification"""
    print("\n========================================")
    print("SECTION 16: Optional Stage 2 (2-adic layer) Verification")
    print("========================================")

    r1, r2, mu = P["r1"], P["r2"], P["mu"]
    L2, C1, C2 = P["L2"], P["C1"], P["C2"]

    L_test = 16
    print(f"  Exhaustive check at L={L_test} (2^{L_test}={1<<L_test} states)...")
    res1 = verify_bijection(C1, L_test)
    res2 = verify_bijection(C2, L_test)
    for name, res in (("C1", res1), ("C2", res2)):
        status = "OK: bijection + full single cycle" if res["is_full_cycle"] else "FAILED"
        print(f"    {name}={hex(res['C'])}: bijection={res['is_bijection']}, "
              f"cycle_length={res['cycle_length']}/{1<<L_test}  -> {status}")

    N_ent, W_ent = 500000, 10000
    Xe, Ye = gen_seq(P["x0"], P["y0"], r1, r2, W_ent + N_ent, mu)
    Xop, Yop = Xe[W_ent:], Ye[W_ent:]

    bytes_stage1 = extract_bytes(Xop, Yop)
    bytes_stage2 = extract_bytes_erpm(Xop, Yop, C1, C2, L2)

    H1, chi2_1 = shannon_entropy(bytes_stage1), chi2_test(bytes_stage1)
    H2, chi2_2 = shannon_entropy(bytes_stage2), chi2_test(bytes_stage2)
    p1 = float(chi2_dist.sf(chi2_1, CHI2_DOF))
    p2 = float(chi2_dist.sf(chi2_2, CHI2_DOF))

    print(f"\n  Byte-stream comparison (N={N_ent}, chi2 crit={CHI2_CRIT_5:.2f}):")
    print(f"    2D-STLM only (no Stage 2)   : H={H1:.6f}  Chi2={chi2_1:.2f}  p={p1:.4f}")
    print(f"    2D-STLM + optional Stage 2  : H={H2:.6f}  Chi2={chi2_2:.2f}  p={p2:.4f}")

    return dict(bijection_C1=res1, bijection_C2=res2,
                H_stage1=H1, chi2_stage1=chi2_1, p_stage1=p1,
                H_stage2=H2, chi2_stage2=chi2_2, p_stage2=p2)


def section18_final_summary(P, LE, entropy_data, se_data, ke_data, cd_data,
                             phase_data, class_data, stage2_data=None):
    """SECTION 18: FINAL SUMMARY + save results"""
    print("\n=================================================")
    print("  SUMMARY: 2D-STLM Complete")
    print("=================================================")
    print("DYNAMICAL:")
    print(f"  LE1={LE['LE1']:.4f} LE2={LE['LE2']:.4f} "
          f"KE={LE['KE_main']:.4f} A={LE['ANISO']:.4f}")
    if LE["LE1"] > 0.02 and LE["LE2"] > 0.02:
        print("  -> Both LE clearly positive: hyperchaotic at this operating point.")
    else:
        print("  -> WARNING: at least one LE is NOT clearly positive here.")
    print(f"  Hyperchaotic fraction of (r1, r2) grid = {class_data['hyper_pct']:.1f}%")
    print(f"  SE_x={se_data['SE_x']:.4f} SE_y={se_data['SE_y']:.4f}")
    print(f"  CD_converged={np.nanmean(cd_data['CD_conv'][-3:]):.3f}")
    print(f"  Phase-space coverage={phase_data['coverage_pct']:.1f}%")
    print("RANDOMNESS:")
    print(f"  H={entropy_data['H_val']:.6f}  Chi2={entropy_data['chi2']:.2f} "
          f"(p={entropy_data['chi2_p']:.4f}, "
          f"{'PASS' if entropy_data['chi2_pass'] else 'FAIL'} at 5%)  "
          f"SC={entropy_data['sc_val']:.2e}")
    print(f"  Chi2 over 10 ICs: mean={entropy_data['chi2_multi'].mean():.2f}, "
          f"pass rate={entropy_data['chi2_pass_rate']:.0f}%")
    print(f"  Autocorrelation lags outside 95% CI: {entropy_data['acorr_out_pct']:.1f}%")
    if stage2_data is not None:
        print("OPTIONAL STAGE 2 (2-adic layer):")
        print(f"  C1 full-cycle: {stage2_data['bijection_C1']['is_full_cycle']}  "
              f"C2 full-cycle: {stage2_data['bijection_C2']['is_full_cycle']}")
        print(f"  2D-STLM-only H={stage2_data['H_stage1']:.6f} Chi2={stage2_data['chi2_stage1']:.2f}")
        print(f"  +Stage 2     H={stage2_data['H_stage2']:.6f} Chi2={stage2_data['chi2_stage2']:.2f}")
    print("FIGURES: 9 PNG files saved (Fig 1,2,3,4,5,6,9,10,11)")
    print("=================================================")

    mat_dict = {
        "r1": P["r1"], "r2": P["r2"], "mu": P["mu"],
        "x0": P["x0"], "y0": P["y0"],
        "LE1": LE["LE1"], "LE2": LE["LE2"], "KE_main": LE["KE_main"], "ANISO": LE["ANISO"],
        "hyper_pct": class_data["hyper_pct"], "class_map": class_data["class_map"],
        "H_val": entropy_data["H_val"], "chi2": entropy_data["chi2"],
        "chi2_p": entropy_data["chi2_p"], "chi2_multi": entropy_data["chi2_multi"],
        "chi2_pass_rate": entropy_data["chi2_pass_rate"],
        "sc_val": entropy_data["sc_val"],
        "bytes_ent": entropy_data["bytes_ent"],
        "SE_map": se_data["SE_map"], "SE_x": se_data["SE_x"], "SE_y": se_data["SE_y"],
        "KE_curve": ke_data["KE_curve"], "KE_surf": ke_data["KE_surf"],
        "CD_conv": cd_data["CD_conv"], "CD_surf": cd_data["CD_surf"],
        "coverage_pct": phase_data["coverage_pct"],
    }
    if stage2_data is not None:
        mat_dict.update({
            "stage2_L": P["L2"], "stage2_C1": P["C1"], "stage2_C2": P["C2"],
            "stage2_C1_full_cycle": stage2_data["bijection_C1"]["is_full_cycle"],
            "stage2_C2_full_cycle": stage2_data["bijection_C2"]["is_full_cycle"],
            "stage2_H": stage2_data["H_stage2"], "stage2_chi2": stage2_data["chi2_stage2"],
        })

    mat_path = os.path.join(OUTPUT_DIR, "2D_STLM_DynamicalResults.mat")
    savemat(mat_path, mat_dict)
    print(f"\nSaved: {mat_path}")


def main():
    P = section0_parameters()
    LE = section1_table1_LE_spectrum(P)
    section2_fig1_LE_convergence(P)
    section3_fixed_points(P)
    section4_fig2_bifurcation(P)
    class_data = section5_fig3_LE_classification(P)
    phase_data = section6_fig4_phase_space(P)
    section7_fig5_sensitivity(P)
    entropy_data = section8_fig6_entropy(P)
    se_data = section13_fig9_sample_entropy(P)
    ke_data = section14_fig10_kolmogorov_entropy(P)
    cd_data = section15_fig11_correlation_dimension(P)
    stage2_data = section16_stage2_verification(P)
    section18_final_summary(P, LE, entropy_data, se_data, ke_data, cd_data,
                             phase_data, class_data, stage2_data)


if __name__ == "__main__":
    main()