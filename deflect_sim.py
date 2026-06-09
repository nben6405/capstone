"""
muography_sim.py
-------------------
Simulates two detector configurations for a binary muon shadow imager,
and compares how multiple Coulomb scattering in different materials smears
the shadow boundary.

  Config A: 2 large single-plane trigger paddles (top) + imaging grid (bottom)
  Config B: Imaging grid on top AND bottom (tiled both layers)

Physical units throughout are cm.
Muon tracks follow a realistic cos^2(theta) zenith distribution.

Usage:
  python3 muography_sim.py

Edit the constants block below to change geometry.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle

# ── Physical geometry (all units in cm) ──────────────────────────────────────
#
# Geometry is chosen so that Highland scatter halos are spatially resolved
# and statistically separable across materials.
#
# Key constraint: lateral displacement at bottom = (BLOCK_Z) × tan(θ_Highland)
#   Iron   10 cm slab, 4 GeV/c: θ₀ ≈ 0.009 rad → ~1.1 cm at 120 cm
#   Lead                        θ₀ ≈ 0.016 rad → ~1.9 cm
#   Tungsten                    θ₀ ≈ 0.021 rad → ~2.5 cm
#   Uranium                     θ₀ ≈ 0.022 rad → ~2.6 cm
# With 1.0 cm bins (FINE_BINS=50 over 50 cm) these span 1–3 bins: clearly
# distinguishable halos while differences between W and U are honestly small.
#
SCINT_WIDTH  = 50.0   # cm  (was 20 — scaled up to retain near-vertical muons
SCINT_HEIGHT = 50.0   #      at 150 cm total height)

GRID_COLS = 10
GRID_ROWS = 10

TILE_W = SCINT_WIDTH  / GRID_COLS
TILE_H = SCINT_HEIGHT / GRID_ROWS

# Layer heights — all Z values are heights above the bottom imaging grid (Z=0).
# Muons travel downward, so Z decreases along the muon path.
#
#   Z_TOP_PADDLE  ──── 100 cm   upper trigger paddle (muons enter here)
#   Z_BOT_PADDLE  ────  90 cm   lower trigger paddle (coincidence = valid track)
#                 ────  85 cm   top face of block
#   BLOCK_Z       ────  75 cm   bottom face of block (muons exit here, scatter applied)
#                 ──────────    75 cm of free drift to bottom detector
#   Z_IMAGING_BOT ────   0 cm   bottom imaging detector
#
Z_TOP_PADDLE  = 100.0  # cm
Z_BOT_PADDLE  =  90.0  # cm  above block top face (85 cm)
Z_IMAGING_BOT =   0.0  # cm  reference plane

# Dense block — centred in the active area
BLOCK_X     = 15.0    # cm  left edge   (centre at 25 cm = (50-20)/2)
BLOCK_Y     = 15.0    # cm  bottom edge
BLOCK_W     = 10.0    # cm  width
BLOCK_H     = 10.0    # cm  height
BLOCK_Z     = 75.0    # cm  Z of block's bottom exit face; top face at BLOCK_Z+BLOCK_THICK
BLOCK_THICK = 5.0    # cm  vertical depth (block occupies 75–85 cm)

# Simulation — more muons to compensate reduced solid-angle acceptance
# at larger detector spacing
N_MUONS = 300_000

# ── Material properties ───────────────────────────────────────────────────────
#
# X0   : radiation length (cm) — governs Highland MCS scatter angle
# lambda_I : nuclear interaction length (cm) — governs muon absorption
#            Attenuation through thickness t = 1 - exp(-t / lambda_I)
#
# Sources: PDG Review of Particle Physics, Tables 34.1 & 34.2 (2022)
#   Material   X0 (cm)   rho (g/cm3)  lambda_I (g/cm2)  lambda_I (cm)
#   water      36.08     1.00         83.3               83.3
#   concrete   10.00     2.30         99.9               43.4
#   aluminium   8.897    2.70        107.2               39.7
#   iron        1.757    7.87        131.9               16.8
#   lead        0.5612  11.35        194.0               17.1
#   tungsten    0.3504  19.30        185.0                9.6
#   uranium     0.3166  19.10        199.0               10.4
#
MATERIALS = {
    'vacuum':   {'X0': 1e12,    'lambda_I': 1e12,  'label': 'No scatter (vacuum)', 'color': '#888888'},
    'water':    {'X0':  36.08,  'lambda_I':  83.3, 'label': 'Water',               'color': '#4477aa'},
    'concrete': {'X0':  10.00,  'lambda_I':  43.4, 'label': 'Concrete',            'color': '#998855'},
    'aluminum': {'X0':   8.897, 'lambda_I':  39.7, 'label': 'Aluminium',           'color': '#44aacc'},
    'iron':     {'X0':   1.757, 'lambda_I':  16.8, 'label': 'Iron',                'color': '#cc6622'},
    'lead':     {'X0':   0.5612,'lambda_I':  17.1, 'label': 'Lead',                'color': '#7733aa'},
    'tungsten': {'X0':   0.3504,'lambda_I':   9.6, 'label': 'Tungsten',            'color': '#dd2255'},
    'uranium':  {'X0':   0.3166,'lambda_I':  10.4, 'label': 'Uranium',             'color': '#33aa55'},
}

# Default material for the Config A / B comparison
BLOCK_MATERIAL = 'iron'

# Materials included in the deflection comparison figure
MATERIALS_TO_COMPARE = ['vacuum', 'concrete', 'iron', 'lead', 'tungsten', 'uranium']

# Resolution of the continuous hit-density maps used in Figure 2.
# FINE_BINS=50 over a 50 cm detector gives 1.0 cm/bin, resolving
# Fe halos (~1 cm) and clearly separating them from Pb/W/U (~2–3 cm).
FINE_BINS = 50

# Maximum connection lines drawn per material panel in Figure 3.
# All crossed events are used for the reconstruction map and histogram;
# only a sample is drawn as lines to keep the plot legible.
CROSS_MAX_ARROWS = 600

# Typical sea-level muon momentum used in Highland formula.
# For relativistic muons beta~1, so beta*c*p ~ p = 4 GeV/c.
MUON_BCP_MEV = 1_000.0   # MeV  (1 GeV/c — maximises scatter separation between materials)

# ── Track generation ─────────────────────────────────────────────────────────

def generate_muons(n, rng):
    """
    Generate muon tracks with realistic angular distribution.

    (x0, y0) : entry point at top paddle (cm)
    theta    : zenith angle from vertical (radians)
    phi      : azimuth angle (radians)

    Zenith distribution follows cos^2(theta); hard cutoff at 70 degrees.
    """
    x0 = rng.uniform(0, SCINT_WIDTH,  n)
    y0 = rng.uniform(0, SCINT_HEIGHT, n)

    theta = []
    while len(theta) < n:
        candidates = rng.uniform(0, np.radians(70), n)
        accept_prob = np.cos(candidates) ** 2
        mask = rng.uniform(0, 1, n) < accept_prob
        theta.extend(candidates[mask].tolist())
    theta = np.array(theta[:n])

    phi = rng.uniform(0, 2 * np.pi, n)
    return x0, y0, theta, phi


def track_position_at_z(x0, y0, theta, phi, z_start, z_target):
    """Return (x, y) at z_target given entry (x0,y0) at z_start."""
    dz    = z_start - z_target
    drift = dz * np.tan(theta)
    return x0 + drift * np.cos(phi), y0 + drift * np.sin(phi)

# ── Highland multiple Coulomb scattering ──────────────────────────────────────

def apply_highland_scatter(theta, phi, path_cm, X0_cm, rng):
    """
    Deflect muon directions using the Highland approximation for multiple
    Coulomb scattering through a material slab.

    theta0 = (13.6 MeV / beta*c*p) * sqrt(x/X0) * [1 + 0.038 * ln(x/X0)]

    Scattering is sampled in two independent projected planes defined by the
    local track frame (d/dtheta and d/dphi directions).

    Returns (theta_new, phi_new, delta_theta) where delta_theta is the
    total 3-D scattering angle (useful for diagnostics / histograms).
    """
    x_over_X0 = np.maximum(path_cm / X0_cm, 1e-12)
    theta0 = (13.6 / MUON_BCP_MEV) * np.sqrt(x_over_X0) * (
        1.0 + 0.038 * np.log(x_over_X0)
    )

    n = len(theta)
    # Two projected Gaussian scatter angles in the local track frame
    alpha = rng.normal(0.0, theta0, n)   # in the (d, d_theta) plane
    beta  = rng.normal(0.0, theta0, n)   # in the (d, d_phi) plane

    # Track direction vector (muon travels downward, z is up)
    sin_t = np.sin(theta)
    cos_t = np.cos(theta)
    dx = sin_t * np.cos(phi)
    dy = sin_t * np.sin(phi)
    dz = -cos_t

    # Local perpendicular basis vectors
    # u = partial d/d(theta) = (cos_t*cos_phi, cos_t*sin_phi, sin_t)
    ux =  cos_t * np.cos(phi)
    uy =  cos_t * np.sin(phi)
    uz =  sin_t
    # v = partial d/d(phi) normalised = (-sin_phi, cos_phi, 0)
    vx = -np.sin(phi)
    vy =  np.cos(phi)

    # Scattered direction (small angle; normalise to unit vector)
    dx_s = dx + alpha * ux + beta * vx
    dy_s = dy + alpha * uy + beta * vy
    dz_s = dz + alpha * uz              # vz = 0

    mag = np.sqrt(dx_s**2 + dy_s**2 + dz_s**2)
    dx_s /= mag
    dy_s /= mag
    dz_s /= mag

    # Reconstruct spherical angles; keep muon going downward (dz <= 0)
    theta_new = np.arctan2(np.hypot(dx_s, dy_s), -dz_s)
    phi_new   = np.arctan2(dy_s, dx_s)

    # Total 3-D scatter angle for diagnostics
    dot = np.clip(dx * dx_s + dy * dy_s + dz * dz_s, -1.0, 1.0)
    delta_theta = np.degrees(np.arccos(dot))

    return theta_new, phi_new, delta_theta

# ── Block intersection + scattering ──────────────────────────────────────────

def process_block(x0, y0, theta, phi, material, rng):
    """
    Propagate muons to BLOCK_Z, apply probabilistic attenuation, and
    scatter surviving muons that traversed the block (Highland formula).

    Returns
    -------
    survived    : bool mask
    x_eff       : x origin for continued propagation (at BLOCK_Z if in block,
                  at Z_TOP_PADDLE otherwise)
    y_eff       : same for y
    z_eff       : corresponding z height
    theta_eff   : post-scatter zenith angle
    phi_eff     : post-scatter azimuth
    delta_theta : 3-D scatter angle in degrees (0 for non-block muons)
    """
    x_blk, y_blk = track_position_at_z(
        x0, y0, theta, phi, Z_TOP_PADDLE, BLOCK_Z
    )

    in_block = (
        (x_blk >= BLOCK_X) & (x_blk <= BLOCK_X + BLOCK_W) &
        (y_blk >= BLOCK_Y) & (y_blk <= BLOCK_Y + BLOCK_H)
    )

    # Attenuation: fraction absorbed = 1 - exp(-path / lambda_I).
    # Vacuum has lambda_I = 1e12 so atten -> 0 (no absorption).
    # Slant path through block scales with 1/cos(theta).
    lambda_I  = MATERIALS[material]['lambda_I']
    slant     = BLOCK_THICK / np.cos(theta)
    atten     = 1.0 - np.exp(-slant / lambda_I)
    absorbed  = in_block & (rng.random(len(x0)) < atten)
    survived  = ~absorbed

    theta_eff   = theta.copy()
    phi_eff     = phi.copy()
    delta_theta = np.zeros(len(x0))

    # Apply scatter to survivors that passed through the block
    scatter_idx = survived & in_block
    if scatter_idx.any() and material != 'vacuum':
        X0   = MATERIALS[material]['X0']
        path = BLOCK_THICK / np.cos(theta[scatter_idx])   # slant path length
        t_sc, p_sc, dt = apply_highland_scatter(
            theta[scatter_idx], phi[scatter_idx], path, X0, rng
        )
        theta_eff[scatter_idx]   = t_sc
        phi_eff[scatter_idx]     = p_sc
        delta_theta[scatter_idx] = dt

    # Effective propagation origin: block level for muons that hit it
    x_eff = np.where(in_block, x_blk, x0)
    y_eff = np.where(in_block, y_blk, y0)
    z_eff = np.where(in_block, BLOCK_Z, Z_TOP_PADDLE)

    # Muons that both survived AND passed through the block — these are the
    # only ones that carry deflection information.
    traversed = survived & in_block

    return survived, x_eff, y_eff, z_eff, theta_eff, phi_eff, delta_theta, traversed


def pos_to_tile(x, y):
    """Convert (x, y) in cm to (row, col) tile index; -1 if out of bounds."""
    col = (x / TILE_W).astype(int)
    row = (y / TILE_H).astype(int)
    in_bounds = (
        (x >= 0) & (x < SCINT_WIDTH) &
        (y >= 0) & (y < SCINT_HEIGHT)
    )
    col = np.clip(col, 0, GRID_COLS - 1)
    row = np.clip(row, 0, GRID_ROWS - 1)
    col = np.where(in_bounds, col, -1)
    row = np.where(in_bounds, row, -1)
    return row, col

# ── Config simulations ────────────────────────────────────────────────────────

def simulate_A(x0_orig, y0_orig, x_eff, y_eff, z_eff, theta_eff, phi_eff, survived):
    """
    Config A: trigger paddles on top, imaging on bottom.
    Deflected muons continue from their post-block position / angle.
    x0_orig / y0_orig are unused here (kept for uniform call signature).
    """
    hit_map = np.zeros((GRID_ROWS, GRID_COLS), dtype=int)
    total_events = 0

    xs  = x_eff[survived];  ys  = y_eff[survived]
    zs  = z_eff[survived]
    ths = theta_eff[survived]; phs = phi_eff[survived]

    dz    = zs - Z_IMAGING_BOT
    x_bot = xs + dz * np.tan(ths) * np.cos(phs)
    y_bot = ys + dz * np.tan(ths) * np.sin(phs)

    row_bot, col_bot = pos_to_tile(x_bot, y_bot)
    in_bounds = (row_bot >= 0) & (col_bot >= 0)

    for r, c in zip(row_bot[in_bounds], col_bot[in_bounds]):
        hit_map[r, c] += 1
        total_events  += 1

    return hit_map, total_events


def simulate_B(x0_orig, y0_orig, x_eff, y_eff, z_eff, theta_eff, phi_eff, survived):
    """
    Config B: imaging grids on top AND bottom.
    Top tile: original entry (x0, y0) — fires before the block.
    Bottom tile: propagated from post-scatter position.
    Deflected muons that cross a tile boundary are rejected.
    """
    hit_map = np.zeros((GRID_ROWS, GRID_COLS), dtype=int)
    total_events = 0

    row_top, col_top = pos_to_tile(x0_orig[survived], y0_orig[survived])

    xs  = x_eff[survived];  ys  = y_eff[survived]
    zs  = z_eff[survived]
    ths = theta_eff[survived]; phs = phi_eff[survived]

    dz    = zs - Z_IMAGING_BOT
    x_bot = xs + dz * np.tan(ths) * np.cos(phs)
    y_bot = ys + dz * np.tan(ths) * np.sin(phs)

    row_bot, col_bot = pos_to_tile(x_bot, y_bot)

    in_bounds = ((row_top >= 0) & (col_top >= 0) &
                 (row_bot >= 0) & (col_bot >= 0))
    same_tile = (row_top == row_bot) & (col_top == col_bot)
    valid = in_bounds & same_tile

    for r, c in zip(row_bot[valid], col_bot[valid]):
        hit_map[r, c] += 1
        total_events  += 1

    return hit_map, total_events

# ── SNR accumulation ──────────────────────────────────────────────────────────

def run_snr_test(config_fn, x0, y0, theta, phi, material, rng,
                 target_snr=3.0, step=500):
    """
    Incrementally accumulate events and track when SNR first reaches target.

    SNR = (mean_open - mean_shadow) / sqrt(mean_open)
    Poisson noise floor: sqrt(N) for counting statistics.
    """
    hit_map  = np.zeros((GRID_ROWS, GRID_COLS), dtype=int)
    # fine_map_bot: where surviving muons land on the BOTTOM detector (post-scatter).
    # fine_map_top: where those same muons entered at the TOP plane (pre-scatter).
    # The signed difference (bot - top) directly shows deflection: positive bins
    # mark muons that arrived somewhere different from where they entered.
    fine_map_bot = np.zeros((FINE_BINS, FINE_BINS), dtype=float)
    fine_map_top = np.zeros((FINE_BINS, FINE_BINS), dtype=float)
    total_events   = 0
    snr_history    = []
    muon_history   = []
    scatter_angles = []
    muons_to_detect = None

    shadow_mask = np.zeros((GRID_ROWS, GRID_COLS), dtype=bool)
    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            tx = (c + 0.5) * TILE_W
            ty = (r + 0.5) * TILE_H
            if (BLOCK_X <= tx <= BLOCK_X + BLOCK_W and
                    BLOCK_Y <= ty <= BLOCK_Y + BLOCK_H):
                shadow_mask[r, c] = True

    n = len(x0)
    for i in range(0, n, step):
        bx  = x0[i:i+step];  by  = y0[i:i+step]
        bth = theta[i:i+step]; bph = phi[i:i+step]

        survived, x_eff, y_eff, z_eff, th_eff, ph_eff, dt, traversed = \
            process_block(bx, by, bth, bph, material, rng)

        batch_map, batch_events = config_fn(
            bx, by, x_eff, y_eff, z_eff, th_eff, ph_eff, survived
        )

        # ── Fine maps ────────────────────────────────────────────────────────
        # TOP map  = ALL incoming muons projected straight to Z_IMAGING_BOT.
        # This is what the top detector physically measures: every muon before
        # any block interaction.  The distribution is uniform (no shadow) and
        # identical across all materials.
        dz_straight = Z_TOP_PADDLE - Z_IMAGING_BOT
        x_top_proj = bx + dz_straight * np.tan(bth) * np.cos(bph)
        y_top_proj = by + dz_straight * np.tan(bth) * np.sin(bph)
        H_top, _, _ = np.histogram2d(
            x_top_proj, y_top_proj,
            bins=FINE_BINS,
            range=[[0, SCINT_WIDTH], [0, SCINT_HEIGHT]]
        )
        fine_map_top += H_top.T

        xs  = x_eff[survived]; ys  = y_eff[survived]
        zs  = z_eff[survived]
        ths = th_eff[survived]; phs = ph_eff[survived]
        dz  = zs - Z_IMAGING_BOT
        x_bot_fine = xs + dz * np.tan(ths) * np.cos(phs)
        y_bot_fine = ys + dz * np.tan(ths) * np.sin(phs)
        H_bot, _, _ = np.histogram2d(
            x_bot_fine, y_bot_fine,
            bins=FINE_BINS,
            range=[[0, SCINT_WIDTH], [0, SCINT_HEIGHT]]
        )
        fine_map_bot += H_bot.T   # row=y, col=x to match imshow origin='lower'

        # Collect non-zero scatter angles from this batch
        scatter_angles.extend(dt[dt > 0].tolist())

        hit_map      += batch_map
        total_events += batch_events

        if total_events < 20:
            continue

        open_counts   = hit_map[~shadow_mask].astype(float)
        shadow_counts = hit_map[ shadow_mask].astype(float)

        if len(shadow_counts) == 0 or open_counts.mean() == 0:
            continue

        mean_open   = open_counts.mean()
        mean_shadow = shadow_counts.mean()
        snr = (mean_open - mean_shadow) / np.sqrt(mean_open)

        snr_history.append(float(snr))
        muon_history.append(i + step)

        if snr >= target_snr and muons_to_detect is None:
            muons_to_detect = i + step

    if muons_to_detect is None:
        muons_to_detect = n

    # Normalise both fine maps to [0, 1] relative hit rate
    bot_max = fine_map_bot.max()
    top_max = fine_map_top.max()
    if bot_max > 0: fine_map_bot /= bot_max
    if top_max > 0: fine_map_top /= top_max

    return (hit_map, total_events, muons_to_detect,
            snr_history, muon_history, np.array(scatter_angles),
            fine_map_bot, fine_map_top)

# ── Plotting helpers ──────────────────────────────────────────────────────────

def _academic_rcparams():
    plt.rcParams.update({
        'font.family':       'serif',
        'font.serif':        ['DejaVu Serif', 'Times New Roman', 'Times', 'serif'],
        'font.size':          8,
        'axes.labelsize':     8,
        'axes.titlesize':     8,
        'xtick.labelsize':    7,
        'ytick.labelsize':    7,
        'legend.fontsize':    7,
        'axes.linewidth':     0.6,
        'xtick.major.width':  0.5,
        'ytick.major.width':  0.5,
        'xtick.major.size':   3,
        'ytick.major.size':   3,
        'xtick.direction':    'in',
        'ytick.direction':    'in',
        'xtick.top':          True,
        'ytick.right':        True,
        'axes.grid':          False,
        'figure.dpi':         300,
    })


def _draw_block(ax):
    """Block outline on the coarse 4×4 tile grid (tile-coordinate units)."""
    rx = BLOCK_X / TILE_W - 0.5
    ry = BLOCK_Y / TILE_H - 0.5
    ax.add_patch(Rectangle(
        (rx, ry), BLOCK_W / TILE_W, BLOCK_H / TILE_H,
        linewidth=0.9, edgecolor='white',
        facecolor='none', linestyle='--', zorder=5
    ))


def _draw_block_cm(ax):
    """Block outline on a cm-scale axis (fine-map or physical coordinates)."""
    ax.add_patch(Rectangle(
        (BLOCK_X, BLOCK_Y), BLOCK_W, BLOCK_H,
        linewidth=0.9, edgecolor='white',
        facecolor='none', linestyle='--', zorder=5
    ))


def _style_heatmap(ax, panel, subtitle):
    """Label a tile-grid heatmap."""
    ax.set_xlabel('Tile column', labelpad=2)
    ax.set_ylabel('Tile row',    labelpad=2)
    ax.set_xticks(range(GRID_COLS))
    ax.set_yticks(range(GRID_ROWS))
    ax.text(0.03, 0.97, panel, transform=ax.transAxes,
            fontsize=8, fontweight='bold', va='top', ha='left', color='white',
            bbox=dict(boxstyle='round,pad=0.12', facecolor='black',
                      alpha=0.5, edgecolor='none'))
    ax.set_title(subtitle, fontsize=7.5, pad=3, loc='left', style='italic')


def _style_fine_map(ax, panel, subtitle):
    """Label a cm-scale fine-resolution density map."""
    ax.set_xlabel('x (cm)', labelpad=2)
    ax.set_ylabel('y (cm)', labelpad=2)
    # Light tick grid at tile boundaries
    ax.set_xticks(np.arange(0, SCINT_WIDTH + TILE_W, TILE_W), minor=False)
    ax.set_yticks(np.arange(0, SCINT_HEIGHT + TILE_H, TILE_H), minor=False)
    ax.tick_params(which='major', length=2)
    ax.text(0.03, 0.97, panel, transform=ax.transAxes,
            fontsize=8, fontweight='bold', va='top', ha='left', color='white',
            bbox=dict(boxstyle='round,pad=0.12', facecolor='black',
                      alpha=0.5, edgecolor='none'))
    ax.set_title(subtitle, fontsize=7.5, pad=3, loc='left', style='italic')

# ── Figure 1: Config A vs B ───────────────────────────────────────────────────

def plot_results(res_a, res_b):
    import os
    hit_a, ev_a, mu_a, snr_a, mh_a, _, fine_bot_a, fine_top_a = res_a
    hit_b, ev_b, mu_b, snr_b, mh_b, _, fine_bot_b, fine_top_b = res_b
    fine_a = fine_bot_a
    fine_b = fine_bot_b

    FLUX   = 1.0
    area   = SCINT_WIDTH * SCINT_HEIGHT
    time_a = mu_a / (FLUX * area)
    time_b = mu_b / (FLUX * area)

    _academic_rcparams()

    fig = plt.figure(figsize=(10.0, 7.8), facecolor='white')
    outer = gridspec.GridSpec(3, 1, figure=fig,
                              height_ratios=[2.6, 2.6, 0.8], hspace=0.38,
                              left=0.06, right=0.98, top=0.95, bottom=0.02)
    gs_top = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=outer[0],
                                              wspace=0.55)
    gs_mid = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[1],
                                              wspace=0.45, width_ratios=[1.8, 1.2])

    cmap = plt.cm.viridis
    ext  = [0, SCINT_WIDTH, 0, SCINT_HEIGHT]   # cm extent for fine maps
    vmax = max(fine_a.max(), fine_b.max())

    # (a) Config A — fine hit-density map in cm
    ax_a = fig.add_subplot(gs_top[0, 0])
    im_a = ax_a.imshow(fine_a, cmap=cmap, vmin=0, vmax=vmax,
                       origin='lower', aspect='equal', extent=ext)
    _draw_block_cm(ax_a)
    _style_fine_map(ax_a, '(a)', 'Config A: paddles + bottom grid')
    cb = plt.colorbar(im_a, ax=ax_a, fraction=0.046, pad=0.04)
    cb.set_label('Rel. hit rate', fontsize=7); cb.ax.tick_params(labelsize=6.5)

    # (b) Config B — fine hit-density map in cm
    ax_b = fig.add_subplot(gs_top[0, 1])
    im_b = ax_b.imshow(fine_b, cmap=cmap, vmin=0, vmax=vmax,
                       origin='lower', aspect='equal', extent=ext)
    _draw_block_cm(ax_b)
    _style_fine_map(ax_b, '(b)', 'Config B: dual imaging grids')
    cb = plt.colorbar(im_b, ax=ax_b, fraction=0.046, pad=0.04)
    cb.set_label('Rel. hit rate', fontsize=7); cb.ax.tick_params(labelsize=6.5)

    # (c) Difference fine map
    ax_d = fig.add_subplot(gs_top[0, 2])
    diff = fine_a - fine_b
    lim  = np.abs(diff).max() or 1.0
    im_d = ax_d.imshow(diff, cmap='RdBu_r', vmin=-lim, vmax=lim,
                       origin='lower', aspect='equal', extent=ext)
    _draw_block_cm(ax_d)
    _style_fine_map(ax_d, '(c)', 'Difference map (A−B)')
    cb = plt.colorbar(im_d, ax=ax_d, fraction=0.046, pad=0.04)
    cb.set_label('Δ hit rate', fontsize=7); cb.ax.tick_params(labelsize=6.5)

    # (d) SNR curve
    ax_snr = fig.add_subplot(gs_mid[0, 0])
    ax_snr.plot(mh_a, snr_a, color='#2166ac', linewidth=1.1,
                label='Config A (paddles + grid)')
    ax_snr.plot(mh_b, snr_b, color='#d6604d', linewidth=1.1,
                label='Config B (dual grid)')
    ax_snr.axhline(3.0, color='black', linewidth=0.65, linestyle='--',
                   label='SNR = 3')
    if mu_a < N_MUONS:
        ax_snr.axvline(mu_a, color='#2166ac', linewidth=0.65, linestyle=':')
    if mu_b < N_MUONS:
        ax_snr.axvline(mu_b, color='#d6604d', linewidth=0.65, linestyle=':')
    ax_snr.set_xlabel('Simulated muon tracks', labelpad=2)
    ax_snr.set_ylabel('Detection SNR', labelpad=2)
    ax_snr.set_title('(d)', fontsize=8, fontweight='bold', loc='left', pad=3)
    ax_snr.legend(frameon=True, framealpha=0.9, edgecolor='#bbbbbb',
                  fontsize=7, loc='upper left')
    ax_snr.set_xlim(left=0); ax_snr.set_ylim(bottom=0)

    # (e) Parameter table
    ax_t = fig.add_subplot(gs_mid[0, 1])
    ax_t.axis('off')
    ax_t.set_title('(e)', fontsize=8, fontweight='bold', loc='left', pad=3)
    tbl_data = [
        ['Active area',         f'{SCINT_WIDTH:.0f}×{SCINT_HEIGHT:.0f} cm'],
        ['Grid',                f'{GRID_ROWS}×{GRID_COLS} tiles'],
        ['Tile size',           f'{TILE_W:.1f}×{TILE_H:.1f} cm'],
        ['Layer spacing',       f'{Z_TOP_PADDLE:.0f} cm'],
        ['Block size',          f'{BLOCK_W:.0f}×{BLOCK_H:.0f}×{BLOCK_THICK:.0f} cm'],
        ['Block material',      MATERIALS[BLOCK_MATERIAL]['label']],
        ['X₀ (material)',  f'{MATERIALS[BLOCK_MATERIAL]["X0"]:.2f} cm'],
        ['Attenuation',         f'{(1-np.exp(-BLOCK_THICK/MATERIALS[BLOCK_MATERIAL]["lambda_I"]))*100:.0f}%'],
        ['A valid events',      f'{ev_a:,}'],
        ['B valid events',      f'{ev_b:,}'],
        ['A → SNR>=3', f'{mu_a:,} μ'],
        ['B → SNR>=3', f'{mu_b:,} μ'],
        ['A image time',        f'~{time_a:.1f} min'],
        ['B image time',        f'~{time_b:.1f} min'],
    ]
    tbl = ax_t.table(cellText=tbl_data, colLabels=['Parameter', 'Value'],
                     loc='upper center', cellLoc='left',
                     bbox=[0.0, 0.0, 1.0, 1.0])
    tbl.auto_set_font_size(False); tbl.set_fontsize(6.5)
    for row in range(len(tbl_data) + 1):
        tbl[row, 0].set_width(0.60); tbl[row, 1].set_width(0.40)
    for col in range(2):
        tbl[0, col].set_facecolor('#d0d0d0')
        tbl[0, col].set_text_props(fontweight='bold')
        tbl[0, col].set_edgecolor('#999999')
    for row in range(1, len(tbl_data) + 1):
        for col in range(2):
            tbl[row, col].set_facecolor('white' if row % 2 == 1 else '#f4f4f4')
            tbl[row, col].set_edgecolor('#cccccc')

    # Caption
    ax_cap = fig.add_subplot(outer[2]); ax_cap.axis('off')
    mat_label = MATERIALS[BLOCK_MATERIAL]['label']
    lines = [
        (f'Figure 1.  Simulated muon hit-rate maps and detection statistics for two detector '
         f'configurations ({GRID_ROWS}×{GRID_COLS} grid, {TILE_W:.1f}×{TILE_H:.1f} cm tiles, '
         f'{SCINT_WIDTH:.0f}×{SCINT_HEIGHT:.0f} cm active area, layer spacing {Z_TOP_PADDLE:.0f} cm).'),
        (f'Dense {mat_label} block {BLOCK_W:.0f}×{BLOCK_H:.0f}×{BLOCK_THICK:.0f} cm '
         f'(X₀ = {MATERIALS[BLOCK_MATERIAL]["X0"]:.2f} cm) at {(1-__import__("numpy").exp(-BLOCK_THICK/MATERIALS[BLOCK_MATERIAL]["lambda_I"]))*100:.0f}% attenuation; '
         f'Highland MCS applied to transmitted muons. Tracks from cos²(θ) distribution (θ < 70°).'),
        (f'Dashed rectangles show projected block boundary. '
         f'Dotted vertical lines in (d) mark the muon count at which SNR = 3 is first reached.'),
    ]
    for i, line in enumerate(lines):
        ax_cap.text(0.5, 0.92 - i * 0.32, line, ha='center', va='top',
                    fontsize=6.8, style='italic', color='#222222',
                    transform=ax_cap.transAxes)

    import os
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       'muography_sim_results.png')
    plt.savefig(out, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved: {out}")
    plt.show()

# ── Figure 2: material deflection comparison ──────────────────────────────────

def run_deflection_comparison(x0, y0, theta, phi, rng):
    """Run Config A for each material in MATERIALS_TO_COMPARE."""
    results = {}
    for mat in MATERIALS_TO_COMPARE:
        label = MATERIALS[mat]['label']
        print(f"  Running Config A — {label}...")
        res = run_snr_test(simulate_A, x0, y0, theta, phi, mat, rng)
        results[mat] = res
    return results


def plot_deflection_results(deflection_results):
    """
    Figure 2 layout (5 rows):
      Row 0: bottom-detector fine hit-density, one panel per material.
      Row 1: top-detector fine hit-density (entry positions of same survivors).
      Row 2: signed difference (bottom - top).
             Red  = muons deflected INTO this location (arrived here but entered elsewhere).
             Blue = muons that entered here but were deflected/absorbed before reaching bottom.
             Halo width and intensity directly encode deflection severity.
      Row 3: scatter angle histogram + lateral-displacement axis (left),
             SNR curves (right).
      Row 4: caption.
    """
    import os
    _academic_rcparams()

    n_mat = len(MATERIALS_TO_COMPARE)
    col_w = 2.85
    fig = plt.figure(figsize=(col_w * n_mat + 0.5, 14.0), facecolor='white')

    outer = gridspec.GridSpec(
        5, 1, figure=fig,
        height_ratios=[2.2, 2.2, 2.2, 2.0, 0.50],
        hspace=0.46,
        left=0.06, right=0.98, top=0.97, bottom=0.01
    )
    gs_bot  = gridspec.GridSpecFromSubplotSpec(1, n_mat, subplot_spec=outer[0], wspace=0.55)
    gs_top  = gridspec.GridSpecFromSubplotSpec(1, n_mat, subplot_spec=outer[1], wspace=0.55)
    gs_diff = gridspec.GridSpecFromSubplotSpec(1, n_mat, subplot_spec=outer[2], wspace=0.55)
    gs_anal = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[3],
                                               wspace=0.42, width_ratios=[1, 1])
    ax_hist = fig.add_subplot(gs_anal[0, 0])
    ax_snr  = fig.add_subplot(gs_anal[0, 1])
    ax_cap  = fig.add_subplot(outer[4]); ax_cap.axis('off')

    ext = [0, SCINT_WIDTH, 0, SCINT_HEIGHT]

    fine_bots = {m: deflection_results[m][6] for m in MATERIALS_TO_COMPARE}
    fine_tops = {m: deflection_results[m][7] for m in MATERIALS_TO_COMPARE}
    # Shared colour scale so rows 0 and 1 are directly comparable
    vmax_both = max(
        max(fm.max() for fm in fine_bots.values()),
        max(fm.max() for fm in fine_tops.values())
    )

    # Difference = material_bottom - vacuum_bottom.
    # Using the vacuum bottom as the geometric baseline ensures that vacuum's
    # own difference is identically zero, and any non-zero value for denser
    # materials is purely due to scatter (not geometric drift or projection
    # artefacts).
    vac_bot   = fine_bots['vacuum']
    diff_maps = {m: fine_bots[m] - vac_bot for m in MATERIALS_TO_COMPARE}
    diff_lim  = max(np.abs(d).max() for d in diff_maps.values()) or 1.0

    cmap_density = plt.cm.inferno
    cmap_diff    = 'RdBu_r'

    p_bot  = ['({})'.format(chr(ord('a') + i))           for i in range(n_mat)]
    p_top  = ['({})'.format(chr(ord('a') + n_mat + i))   for i in range(n_mat)]
    p_diff = ['({})'.format(chr(ord('a') + 2*n_mat + i)) for i in range(n_mat)]

    for idx, mat in enumerate(MATERIALS_TO_COMPARE):
        res   = deflection_results[mat]
        color = MATERIALS[mat]['color']
        label = MATERIALS[mat]['label']
        X0    = MATERIALS[mat]['X0']
        mu    = res[2]
        fbot  = fine_bots[mat]
        ftop  = fine_tops[mat]
        diff  = diff_maps[mat]
        x0_lbl = 'X0={:.4f} cm'.format(X0) if X0 < 1000 else 'no scatter'

        # Row 0: bottom detector hit density
        ax_b = fig.add_subplot(gs_bot[0, idx])
        im_b = ax_b.imshow(fbot, cmap=cmap_density, vmin=0, vmax=vmax_both,
                           origin='lower', aspect='equal', extent=ext)
        _draw_block_cm(ax_b)
        subtitle_b = '{} bottom\n({})'.format(label, x0_lbl)
        _style_fine_map(ax_b, p_bot[idx], subtitle_b)
        cb = plt.colorbar(im_b, ax=ax_b, fraction=0.046, pad=0.04)
        cb.set_label('Rel. hits', fontsize=6); cb.ax.tick_params(labelsize=5.5)

        # Row 1: top detector hit density (entry positions)
        ax_t = fig.add_subplot(gs_top[0, idx])
        im_t = ax_t.imshow(ftop, cmap=cmap_density, vmin=0, vmax=vmax_both,
                           origin='lower', aspect='equal', extent=ext)
        _draw_block_cm(ax_t)
        _style_fine_map(ax_t, p_top[idx], 'Incoming flux\n(all muons, pre-block)')
        cb = plt.colorbar(im_t, ax=ax_t, fraction=0.046, pad=0.04)
        cb.set_label('Rel. hits', fontsize=6); cb.ax.tick_params(labelsize=5.5)

        # Row 2: scatter signal = material_bottom - vacuum_bottom
        ax_d = fig.add_subplot(gs_diff[0, idx])
        im_d = ax_d.imshow(diff, cmap=cmap_diff,
                           vmin=-diff_lim, vmax=diff_lim,
                           origin='lower', aspect='equal', extent=ext)
        _draw_block_cm(ax_d)
        _style_fine_map(ax_d, p_diff[idx], 'vs vacuum\n(scatter signal)')
        cb = plt.colorbar(im_d, ax=ax_d, fraction=0.046, pad=0.04)
        cb.set_label('Delta hits', fontsize=6); cb.ax.tick_params(labelsize=5.5)

        # Scatter angle histogram
        angles = res[5]
        if len(angles) > 0 and mat != 'vacuum':
            angle_cap = min(float(np.percentile(angles, 99.5)), 30.0)
            ax_hist.hist(angles, bins=80, range=(0, angle_cap),
                         density=True, histtype='step',
                         linewidth=1.1, color=color, label=label)

        # SNR curve
        ax_snr.plot(res[4], res[3], color=color, linewidth=1.1, label=label)
        if mu < N_MUONS:
            ax_snr.axvline(mu, color=color, linewidth=0.55, linestyle=':')

    prop_dist = BLOCK_Z - Z_IMAGING_BOT
    lbl_hist  = '({})'.format(chr(ord('a') + 3 * n_mat))
    lbl_snr   = '({})'.format(chr(ord('a') + 3 * n_mat + 1))

    ax_hist.set_xlabel('3-D scatter angle (degrees)', labelpad=2)
    ax_hist.set_ylabel('Probability density', labelpad=2)
    ax_hist.set_title(lbl_hist, fontsize=8, fontweight='bold', loc='left', pad=3)
    ax_hist.legend(frameon=True, framealpha=0.9, edgecolor='#bbbbbb',
                   fontsize=6.5, loc='upper right')
    ax_hist.set_xlim(left=0)

    # Secondary x-axis: lateral displacement at bottom detector
    xlim = ax_hist.get_xlim()
    ax2  = ax_hist.twiny()
    ax2.set_xlim(
        np.tan(np.radians(xlim[0])) * prop_dist,
        np.tan(np.radians(max(xlim[1], 0.01))) * prop_dist
    )
    ax2.set_xlabel(
        'Lateral displacement at detector  (cm,  d = {:.0f} cm)'.format(prop_dist),
        fontsize=6.5, labelpad=3
    )
    ax2.tick_params(labelsize=6)

    ax_snr.axhline(3.0, color='black', linewidth=0.65, linestyle='--', label='SNR = 3')
    ax_snr.set_xlabel('Simulated muon tracks', labelpad=2)
    ax_snr.set_ylabel('Detection SNR', labelpad=2)
    ax_snr.set_title(lbl_snr, fontsize=8, fontweight='bold', loc='left', pad=3)
    ax_snr.legend(frameon=True, framealpha=0.9, edgecolor='#bbbbbb',
                  fontsize=6.5, loc='upper left', ncol=2)
    ax_snr.set_xlim(left=0); ax_snr.set_ylim(bottom=0)

    snr_lines = ', '.join(
        '{}: {:,} mu'.format(MATERIALS[m]['label'], deflection_results[m][2])
        for m in MATERIALS_TO_COMPARE
    )
    cap = (
        'Figure 2.  Top-vs-bottom hit-density comparison across materials '
        '(Config A, {:.0f}x{:.0f}x{:.0f} cm block, material-dependent attenuation, '
        'block-to-detector distance {:.0f} cm).  '
        'Rows 1-2: fine density maps ({:d}x{:d} bins, {:.1f} cm/bin). '
        'Row 1 (bottom detector): surviving muons actual landing positions. '
        'Row 2 (incoming flux): ALL muons projected straight to detector plane '
        '— uniform for every material, no block shadow. '
        'Row 3: scatter signal (material bottom minus vacuum bottom) -- '
        'red = more muons than vacuum baseline (halo of scattered muons); '
        'blue = fewer muons (scatter deplets straight-line shadow). '
        'Vacuum panel is identically zero by construction; halo width grows with decreasing X0. '
        'Muons to SNR>=3: {}.'
    ).format(
        BLOCK_W, BLOCK_H, BLOCK_THICK, prop_dist,
        FINE_BINS, FINE_BINS, SCINT_WIDTH / FINE_BINS, snr_lines
    )
    ax_cap.text(0.5, 0.95, cap, ha='center', va='top', fontsize=6.3,
                style='italic', color='#222222', transform=ax_cap.transAxes)

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       'muography_deflection_results.png')
    plt.savefig(out, dpi=300, bbox_inches='tight', facecolor='white')
    print('Saved: {}'.format(out))
    plt.show()


# ── Crossing / reconstruction analysis ───────────────────────────────────────

def run_crossing_analysis(x0, y0, theta, phi, material, rng):
    """
    Collect block-traversing muons whose bottom fine-bin (1 cm) differs from
    their top fine-bin — i.e. deflected by at least one bin width.

    These events carry direct scattering information: the entry position
    reconstructs the object footprint; the bottom position shows where the
    deflected muon landed.

    Returns
    -------
    x_top, y_top   : entry positions at top detector  (cm)
    x_bot, y_bot   : landing positions at bottom detector  (cm)
    recon_map      : FINE_BINS x FINE_BINS normalised density of entry positions
    disp_cm        : 2-D lateral displacement magnitude  (cm)
    n_traversed    : total traversed muons (denominator for crossing fraction)
    """
    bin_w = SCINT_WIDTH  / FINE_BINS
    bin_h = SCINT_HEIGHT / FINE_BINS

    x_top_list = []; y_top_list = []
    x_bot_list = []; y_bot_list = []
    dt_list    = []
    n_traversed = 0

    step = 2000
    n    = len(x0)

    for i in range(0, n, step):
        bx  = x0[i:i+step]; by  = y0[i:i+step]
        bth = theta[i:i+step]; bph = phi[i:i+step]

        survived, x_eff, y_eff, z_eff, th_eff, ph_eff, dt, traversed = \
            process_block(bx, by, bth, bph, material, rng)

        if not traversed.any():
            continue

        n_traversed += int(traversed.sum())

        # Use only block-traversing muons so the reconstruction maps the block
        xt = bx[traversed]; yt = by[traversed]
        xs = x_eff[traversed]; ys = y_eff[traversed]
        zs = z_eff[traversed]
        ths = th_eff[traversed]; phs = ph_eff[traversed]
        dts = dt[traversed]

        dz = zs - Z_IMAGING_BOT
        xb = xs + dz * np.tan(ths) * np.cos(phs)
        yb = ys + dz * np.tan(ths) * np.sin(phs)

        # Fine-bin indices for top and bottom
        ixt = np.clip((xt / bin_w).astype(int), 0, FINE_BINS - 1)
        iyt = np.clip((yt / bin_h).astype(int), 0, FINE_BINS - 1)
        ixb = (xb / bin_w).astype(int)
        iyb = (yb / bin_h).astype(int)

        in_bounds = ((xb >= 0) & (xb < SCINT_WIDTH) &
                     (yb >= 0) & (yb < SCINT_HEIGHT))
        crossed   = in_bounds & ((ixt != ixb) | (iyt != iyb))

        x_top_list.append(xt[crossed]); y_top_list.append(yt[crossed])
        x_bot_list.append(xb[crossed]); y_bot_list.append(yb[crossed])
        dt_list.append(dts[crossed])

    x_top   = np.concatenate(x_top_list) if x_top_list else np.array([])
    y_top   = np.concatenate(y_top_list) if y_top_list else np.array([])
    x_bot   = np.concatenate(x_bot_list) if x_bot_list else np.array([])
    y_bot   = np.concatenate(y_bot_list) if y_bot_list else np.array([])
    scat_ang = np.concatenate(dt_list)   if dt_list    else np.array([])
    disp    = np.hypot(x_bot - x_top, y_bot - y_top) if len(x_top) else np.array([])

    if len(x_top) > 0:
        recon_map, _, _ = np.histogram2d(
            x_top, y_top,
            bins=FINE_BINS,
            range=[[0, SCINT_WIDTH], [0, SCINT_HEIGHT]]
        )
        recon_map = recon_map.T            # row = y, col = x
        if recon_map.max() > 0:
            recon_map = recon_map / recon_map.max()
    else:
        recon_map = np.zeros((FINE_BINS, FINE_BINS))

    return x_top, y_top, x_bot, y_bot, recon_map, disp, scat_ang, n_traversed


def run_crossing_comparison(x0, y0, theta, phi, rng):
    results = {}
    for mat in MATERIALS_TO_COMPARE:
        print('  Crossing analysis -- {}...'.format(MATERIALS[mat]['label']))
        results[mat] = run_crossing_analysis(x0, y0, theta, phi, mat, rng)
    return results


def plot_crossing_reconstruction(crossing_results):
    """
    Figure 3 layout (4 rows):
      Row 0: Reconstruction maps — density of entry (top) positions of
             crossed events. Bright region traces the scattering object.
      Row 1: Connection plots — white dot = incident tick mark at entry;
             coloured line connects entry to bottom landing position.
             Line colour encodes lateral displacement magnitude.
      Row 2: Lateral displacement histogram (all materials overlaid).
      Row 3: Caption.
    """
    import os
    _academic_rcparams()

    n_mat = len(MATERIALS_TO_COMPARE)
    col_w = 2.85
    fig = plt.figure(figsize=(col_w * n_mat + 0.5, 11.5), facecolor='white')

    outer = gridspec.GridSpec(
        4, 1, figure=fig,
        height_ratios=[2.4, 3.0, 2.0, 0.50],
        hspace=0.44,
        left=0.06, right=0.98, top=0.97, bottom=0.01
    )
    gs_recon = gridspec.GridSpecFromSubplotSpec(
        1, n_mat, subplot_spec=outer[0], wspace=0.55)
    gs_conn  = gridspec.GridSpecFromSubplotSpec(
        1, n_mat, subplot_spec=outer[1], wspace=0.55)
    ax_hist  = fig.add_subplot(outer[2])
    ax_cap   = fig.add_subplot(outer[3]); ax_cap.axis('off')

    ext = [0, SCINT_WIDTH, 0, SCINT_HEIGHT]

    # Shared colour scales
    recon_maps = {m: crossing_results[m][4] for m in MATERIALS_TO_COMPARE}
    recon_vmax = max(rm.max() for rm in recon_maps.values()) or 1.0

    all_sangs = np.concatenate([crossing_results[m][6]
                                 for m in MATERIALS_TO_COMPARE
                                 if len(crossing_results[m][6]) > 0])
    sang_vmax = float(np.percentile(all_sangs, 98)) if len(all_sangs) else 5.0

    cmap_conn = plt.cm.plasma
    norm_conn = plt.Normalize(0, sang_vmax)
    rng_sample = np.random.default_rng(7)   # fixed seed for reproducible sample

    p_recon = ['({})'.format(chr(ord('a') + i))         for i in range(n_mat)]
    p_conn  = ['({})'.format(chr(ord('a') + n_mat + i)) for i in range(n_mat)]

    for idx, mat in enumerate(MATERIALS_TO_COMPARE):
        x_top, y_top, x_bot, y_bot, recon_map, disp_cm, scat_ang, n_trav = \
            crossing_results[mat]
        n_cross = len(x_top)
        color   = MATERIALS[mat]['color']
        label   = MATERIALS[mat]['label']
        X0      = MATERIALS[mat]['X0']
        x0_lbl  = 'X0={:.4f} cm'.format(X0) if X0 < 1000 else 'no scatter'
        frac    = (n_cross / n_trav * 100) if n_trav > 0 else 0.0

        # Row 0: reconstruction map
        ax_r = fig.add_subplot(gs_recon[0, idx])
        im_r = ax_r.imshow(recon_map, cmap='hot', vmin=0, vmax=recon_vmax,
                           origin='lower', aspect='equal', extent=ext)
        _draw_block_cm(ax_r)
        _style_fine_map(ax_r, p_recon[idx],
                        '{} ({})\nn={:,}  ({:.1f}% crossed)'.format(
                            label, x0_lbl, n_cross, frac))
        cb = plt.colorbar(im_r, ax=ax_r, fraction=0.046, pad=0.04)
        cb.set_label('Norm. density', fontsize=6)
        cb.ax.tick_params(labelsize=5.5)

        # Row 1: connection plot
        ax_c = fig.add_subplot(gs_conn[0, idx])
        ax_c.set_facecolor('#111111')           # dark background — lines pop
        ax_c.set_xlim(0, SCINT_WIDTH)
        ax_c.set_ylim(0, SCINT_HEIGHT)
        ax_c.set_aspect('equal')
        _draw_block_cm(ax_c)
        _style_fine_map(ax_c, p_conn[idx], '{} connections'.format(label))
        # Override tick/label colour for dark bg
        ax_c.tick_params(colors='#cccccc', labelsize=6)
        ax_c.xaxis.label.set_color('#cccccc')
        ax_c.yaxis.label.set_color('#cccccc')
        ax_c.title.set_color('white')

        if n_cross > 0:
            # Sample events to draw
            draw_n   = min(CROSS_MAX_ARROWS, n_cross)
            sel      = rng_sample.choice(n_cross, size=draw_n, replace=False)
            xs_t = x_top[sel]; ys_t = y_top[sel]
            xs_b = x_bot[sel]; ys_b = y_bot[sel]
            d_s  = scat_ang[sel]

            for xt, yt, xb, yb, d in zip(xs_t, ys_t, xs_b, ys_b, d_s):
                lc = cmap_conn(norm_conn(d))
                ax_c.plot([xt, xb], [yt, yb], '-',
                          color=lc, linewidth=0.55, alpha=0.55, zorder=2)
                # Incident tick mark: small bright dot at entry position
                ax_c.plot(xt, yt, 'o', color='white',
                          markersize=2.0, markeredgewidth=0, zorder=3)

            # Shared displacement colour bar
            sm = plt.cm.ScalarMappable(cmap=cmap_conn, norm=norm_conn)
            sm.set_array([])
            cb_c = plt.colorbar(sm, ax=ax_c, fraction=0.046, pad=0.04)
            cb_c.set_label('Scatter angle (°)', fontsize=6)
            cb_c.ax.tick_params(labelsize=5.5)

        # Scatter angle histogram
        if n_cross > 0 and mat != 'vacuum':
            a_cap = min(float(np.percentile(scat_ang, 99)), 20.0)
            ax_hist.hist(scat_ang, bins=70, range=(0, a_cap),
                         density=True, histtype='step', linewidth=1.1,
                         color=color,
                         label='{} (n={:,}, {:.0f}%)'.format(
                             label, n_cross, frac))

    lbl_hist = '({})'.format(chr(ord('a') + 2 * n_mat))
    ax_hist.set_xlabel('Highland 3-D scatter angle  (degrees)', labelpad=2)
    ax_hist.set_ylabel('Probability density', labelpad=2)
    ax_hist.set_title(lbl_hist, fontsize=8, fontweight='bold', loc='left', pad=3)
    ax_hist.legend(frameon=True, framealpha=0.9, edgecolor='#bbbbbb',
                   fontsize=6.5, loc='upper right')
    ax_hist.set_xlim(left=0)

    prop_dist = BLOCK_Z - Z_IMAGING_BOT
    cap = (
        'Figure 3.  Muon deflection reconstruction across materials. '
        'Only block-traversing muons whose bottom 1 cm bin differs from '
        'their top 1 cm bin are included. '
        'Row 1: entry-position density of these crossed events -- the bright '
        'region directly reconstructs the scattering object footprint; '
        'percentage shows crossing rate relative to all traversed muons. '
        'Row 2: connection plot on dark background; white dots = incident '
        'tick marks at top-detector entry; coloured lines connect to the '
        'bottom landing position (colour = Highland scatter angle). '
        'Denser materials (smaller X0) produce larger scatter angles. '
        'Row 3: scatter angle distributions -- heavier materials shift the '
        'tail to higher angles, isolating material from geometric drift. '
        'Geometry: {:.0f}x{:.0f} cm detector, {:.0f} cm layer separation, '
        '{:.0f} cm block-to-detector, {:.0f}x{:.0f}x{:.0f} cm block, '
        'material-dependent attenuation (1-exp(-t/lambda_I)).'
    ).format(
        SCINT_WIDTH, SCINT_HEIGHT, Z_TOP_PADDLE, prop_dist,
        BLOCK_W, BLOCK_H, BLOCK_THICK
    )
    ax_cap.text(0.5, 0.95, cap, ha='center', va='top', fontsize=6.3,
                style='italic', color='#222222', transform=ax_cap.transAxes)

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       'muography_crossing_reconstruction.png')
    plt.savefig(out, dpi=300, bbox_inches='tight', facecolor='white')
    print('Saved: {}'.format(out))
    plt.show()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    rng = np.random.default_rng(42)

    print(f"Generating {N_MUONS:,} muon tracks...")
    x0, y0, theta, phi = generate_muons(N_MUONS, rng)

    # ── Figure 1: Config A vs Config B ───────────────────────────────────────
    print(f"\nFigure 1 — Config A vs B  (material: {MATERIALS[BLOCK_MATERIAL]['label']})")
    print("Running Config A...")
    res_a = run_snr_test(simulate_A, x0, y0, theta, phi, BLOCK_MATERIAL, rng)
    print("Running Config B...")
    res_b = run_snr_test(simulate_B, x0, y0, theta, phi, BLOCK_MATERIAL, rng)

    print(f"  Config A: {res_a[1]:,} valid events, needs ~{res_a[2]:,} muons for SNR>=3")
    print(f"  Config B: {res_b[1]:,} valid events, needs ~{res_b[2]:,} muons for SNR>=3")
    plot_results(res_a, res_b)

    # ── Figure 2: deflection comparison across materials ──────────────────────
    print(f"\nFigure 2 — material deflection comparison (Config A)")
    defl = run_deflection_comparison(x0, y0, theta, phi, rng)
    plot_deflection_results(defl)

    # ── Figure 3: crossing-event reconstruction ───────────────────────────────
    print(f"\nFigure 3 — crossing reconstruction (all materials)")
    cross = run_crossing_comparison(x0, y0, theta, phi, rng)
    plot_crossing_reconstruction(cross)


if __name__ == '__main__':
    main()
