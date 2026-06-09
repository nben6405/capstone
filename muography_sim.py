"""
muography_sim.py
-------------------
Simulates two detector configurations for a binary muon shadow imager.

  Config A: 2 large single-plane trigger paddles (top) + imaging grid (bottom)
  Config B: 3 imaging grids (top 2 + bottom 1), same-tile coincidence required

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
import os

# ── Physical geometry (all units in cm) ──────────────────────────────────────

# Scintillator active area
SCINT_WIDTH  = 40.0
SCINT_HEIGHT = 40.0

# Grid resolution — set directly
GRID_COLS = 4
GRID_ROWS = 4

# Tile size derived automatically
TILE_W = SCINT_WIDTH  / GRID_COLS
TILE_H = SCINT_HEIGHT / GRID_ROWS

# Layer heights (cm above bottom imaging grid)
Z_TOP_PADDLE  = 35.0  # top trigger paddle (Config A) / top grid (Config B)
Z_BOT_PADDLE  = 25.0  # second trigger paddle (Config A only)
Z_MID_GRID    = 25.0  # second imaging grid (Config B only)
Z_IMAGING_BOT =  0.0  # bottom imaging grid — reference plane

# Dense block — 20x20cm centred on 40x40cm area, covers 2x2 tiles cleanly
BLOCK_X     = 10.0    # cm  left edge
BLOCK_Y     = 10.0    # cm  bottom edge
BLOCK_W     = 20.0    # cm  width  (2 tiles)
BLOCK_H     = 20.0    # cm  height (2 tiles)
BLOCK_Z     = 10.0    # cm  height above bottom grid (block occupies BLOCK_Z to BLOCK_Z+BLOCK_THICK)
BLOCK_THICK = 10.0    # cm  vertical depth of block
# BLOCK_ATTEN removed — each material now has its own nuclear interaction length (see MATERIALS)

# Material under test for Config A / B shadow comparison (Figure 1)
BLOCK_MATERIAL = 'iron'

# Materials shown in the deflection comparison (Figure 2)
MATERIALS_TO_COMPARE = ['vacuum', 'water', 'concrete', 'iron', 'lead', 'tungsten']

# ── Material properties ───────────────────────────────────────────────────────
# X0       : radiation length (cm)        — Highland MCS scatter angle ∝ 1/√X0
# lambda_I : nuclear interaction length (cm) — muon absorption ∝ 1-exp(-t/λ_I)
# Sources  : PDG Review of Particle Physics, Tables 34.1 & 34.2 (2022)
#
MATERIALS = {
    'vacuum':   {'X0': 1e12,    'lambda_I': 1e12,  'label': 'Vacuum (no interaction)', 'color': '#888888'},
    'water':    {'X0':  36.08,  'lambda_I':  83.3, 'label': 'Water',                   'color': '#4477aa'},
    'concrete': {'X0':  10.00,  'lambda_I':  43.4, 'label': 'Concrete',                'color': '#998855'},
    'iron':     {'X0':   1.757, 'lambda_I':  16.8, 'label': 'Iron',                    'color': '#cc6622'},
    'lead':     {'X0':   0.5612,'lambda_I':  17.1, 'label': 'Lead',                    'color': '#7733aa'},
    'tungsten': {'X0':   0.3504,'lambda_I':   9.6, 'label': 'Tungsten',                'color': '#dd2255'},
    'uranium':  {'X0':   0.3166,'lambda_I':  10.4, 'label': 'Uranium',                 'color': '#33aa55'},
}

# Muon momentum for Highland formula (sea-level median ≈ 3–4 GeV/c;
# 1 GeV/c used here to represent the softer part of the spectrum that
# scatters most and is most relevant for short-range imaging)
MUON_BCP_MEV = 1_000.0   # MeV  (β≈1 so βcp ≈ p)

# Simulation — higher count needed because muons are generated over a padded
# area larger than the active detector, so many miss the bottom grid
N_MUONS = 1_000_000

# ── Track generation ─────────────────────────────────────────────────────────

def generate_muons(n, rng):
    """
    Generate muon tracks with realistic angular distribution.

    Each muon defined by:
      (x0, y0) : entry point at top paddle (cm)
      theta    : zenith angle from vertical (radians)
      phi      : azimuth angle (radians)

    Zenith distribution follows cos^2(theta).
    Hard cutoff at 70 degrees — flux negligible beyond that.

    Entry points are sampled over a padded area larger than the active
    detector — angled muons entering outside the active area can still
    drift onto the bottom grid. Without this, edge tiles receive
    artificially fewer counts than central tiles, masking the shadow.
    """
    # Max horizontal drift at 70 degrees over full layer height
    margin = Z_TOP_PADDLE * np.tan(np.radians(70))

    x0 = rng.uniform(-margin, SCINT_WIDTH  + margin, n)
    y0 = rng.uniform(-margin, SCINT_HEIGHT + margin, n)

    # Rejection sampling for cos^2(theta) distribution
    theta = []
    while len(theta) < n:
        candidates = rng.uniform(0, np.radians(70), n)
        accept_prob = np.cos(candidates) ** 2
        mask = rng.uniform(0, 1, n) < accept_prob
        theta.extend(candidates[mask].tolist())
    theta = np.array(theta[:n])

    # Azimuth uniform — no preferred horizontal direction
    phi = rng.uniform(0, 2 * np.pi, n)
    return x0, y0, theta, phi


def track_position_at_z(x0, y0, theta, phi, z_start, z_target):
    dz    = z_start - z_target
    drift = dz * np.tan(theta)
    x     = x0 + drift * np.cos(phi)
    y     = y0 + drift * np.sin(phi)
    return x, y

# ── Highland multiple Coulomb scattering ──────────────────────────────────────

def apply_highland_scatter(theta, phi, path_cm, X0_cm, rng):
    """
    Deflect muon directions using the Highland approximation:
      theta0 = (13.6 MeV / beta*c*p) * sqrt(x/X0) * [1 + 0.038*ln(x/X0)]
    Scatter sampled in two independent projected planes of the local track frame.
    Returns (theta_new, phi_new, delta_theta_deg).
    """
    x_over_X0 = np.maximum(path_cm / X0_cm, 1e-12)
    theta0 = (13.6 / MUON_BCP_MEV) * np.sqrt(x_over_X0) * (
        1.0 + 0.038 * np.log(x_over_X0)
    )
    n = len(theta)
    alpha = rng.normal(0.0, theta0, n)
    beta  = rng.normal(0.0, theta0, n)

    sin_t = np.sin(theta);  cos_t = np.cos(theta)
    dx = sin_t * np.cos(phi);  dy = sin_t * np.sin(phi);  dz = -cos_t
    ux =  cos_t * np.cos(phi); uy =  cos_t * np.sin(phi); uz =  sin_t
    vx = -np.sin(phi);         vy =  np.cos(phi)

    dx_s = dx + alpha*ux + beta*vx
    dy_s = dy + alpha*uy + beta*vy
    dz_s = dz + alpha*uz
    mag  = np.sqrt(dx_s**2 + dy_s**2 + dz_s**2)
    dx_s /= mag;  dy_s /= mag;  dz_s /= mag

    theta_new = np.arctan2(np.hypot(dx_s, dy_s), -dz_s)
    phi_new   = np.arctan2(dy_s, dx_s)
    dot       = np.clip(dx*dx_s + dy*dy_s + dz*dz_s, -1.0, 1.0)
    delta_deg = np.degrees(np.arccos(dot))
    return theta_new, phi_new, delta_deg


# ── Block intersection ────────────────────────────────────────────────────────

def process_block(x0, y0, theta, phi, material, rng):
    """
    Propagate muons to the block, apply per-material absorption (from nuclear
    interaction length lambda_I) and Highland scatter to survivors.

    Returns
    -------
    survived    : bool mask — muons not absorbed
    theta_eff   : post-scatter zenith angles  (unchanged for non-block muons)
    phi_eff     : post-scatter azimuth angles
    delta_theta : 3-D scatter angle in degrees (0 for non-block / vacuum)
    traversed   : bool mask — survived AND passed through block
    """
    x_blk, y_blk = track_position_at_z(
        x0, y0, theta, phi, Z_TOP_PADDLE, BLOCK_Z
    )
    in_block = (
        (x_blk >= BLOCK_X) & (x_blk <= BLOCK_X + BLOCK_W) &
        (y_blk >= BLOCK_Y) & (y_blk <= BLOCK_Y + BLOCK_H)
    )

    # Per-material absorption: 1 - exp(-slant_path / lambda_I)
    lam   = MATERIALS[material]['lambda_I']
    slant = BLOCK_THICK / np.cos(theta)
    atten = 1.0 - np.exp(-slant / lam)
    absorbed = in_block & (rng.random(len(x0)) < atten)
    survived = ~absorbed

    theta_eff   = theta.copy()
    phi_eff     = phi.copy()
    delta_theta = np.zeros(len(x0))

    scatter_idx = survived & in_block
    if scatter_idx.any() and material != 'vacuum':
        X0   = MATERIALS[material]['X0']
        path = BLOCK_THICK / np.cos(theta[scatter_idx])
        t_sc, p_sc, dt = apply_highland_scatter(
            theta[scatter_idx], phi[scatter_idx], path, X0, rng
        )
        theta_eff[scatter_idx]   = t_sc
        phi_eff[scatter_idx]     = p_sc
        delta_theta[scatter_idx] = dt

    traversed = survived & in_block
    return survived, theta_eff, phi_eff, delta_theta, traversed


def check_block(x0, y0, theta, phi, rng):
    """Thin wrapper used by Figure 1 — applies BLOCK_MATERIAL attenuation."""
    survived, _, _, _, _ = process_block(
        x0, y0, theta, phi, BLOCK_MATERIAL, rng
    )
    return survived


def pos_to_tile(x, y):
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

def simulate_A(x0, y0, theta, phi, survived, rng):
    """
    Config A: 2 large paddles on top, imaging grid on bottom.
    Valid event = survived block + lands in active area of bottom grid.
    """
    hit_map = np.zeros((GRID_ROWS, GRID_COLS), dtype=int)
    total_events = 0

    x_bot, y_bot = track_position_at_z(
        x0[survived], y0[survived], theta[survived], phi[survived],
        z_start=Z_TOP_PADDLE, z_target=Z_IMAGING_BOT
    )
    row_bot, col_bot = pos_to_tile(x_bot, y_bot)
    in_bounds = (row_bot >= 0) & (col_bot >= 0)

    for r, c in zip(row_bot[in_bounds], col_bot[in_bounds]):
        hit_map[r, c] += 1
        total_events  += 1

    return hit_map, total_events


def simulate_B(x0, y0, theta, phi, survived, rng):
    """
    Config B: 3 imaging grids (top 2 + bottom 1).
    Valid event = survived block + same tile fires on top grid AND bottom grid.
    Angled muons that drift across a tile boundary are rejected.
    """
    hit_map = np.zeros((GRID_ROWS, GRID_COLS), dtype=int)
    total_events = 0

    # Top grid — entry position, no drift yet
    x_top = x0[survived].copy()
    y_top = y0[survived].copy()

    # Bottom grid — muon has drifted over full layer spacing
    x_bot, y_bot = track_position_at_z(
        x0[survived], y0[survived], theta[survived], phi[survived],
        z_start=Z_TOP_PADDLE, z_target=Z_IMAGING_BOT
    )

    row_top, col_top = pos_to_tile(x_top, y_top)
    row_bot, col_bot = pos_to_tile(x_bot, y_bot)

    in_bounds = (row_top >= 0) & (col_top >= 0) & \
                (row_bot >= 0) & (col_bot >= 0)
    same_tile = (row_top == row_bot) & (col_top == col_bot)
    valid     = in_bounds & same_tile

    for r, c in zip(row_bot[valid], col_bot[valid]):
        hit_map[r, c] += 1
        total_events  += 1

    return hit_map, total_events

# ── SNR test ──────────────────────────────────────────────────────────────────

def run_snr_test(config_fn, x0, y0, theta, phi, rng,
                 target_snr=3.0, step=500):
    """
    Poisson SNR = (mean open count - mean shadow count) / sqrt(mean open count)
    Stable for uniform hit maps, physically grounded in counting statistics.
    """
    hit_map = np.zeros((GRID_ROWS, GRID_COLS), dtype=int)
    total_events    = 0
    snr_history     = []
    muon_history    = []
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

        survived = check_block(bx, by, bth, bph, rng)
        batch_map, batch_events = config_fn(bx, by, bth, bph, survived, rng)

        hit_map      += batch_map
        total_events += batch_events

        if total_events < 20:
            continue

        open_counts   = hit_map[~shadow_mask].astype(float)
        shadow_counts = hit_map[shadow_mask].astype(float)

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

    return hit_map, total_events, muons_to_detect, snr_history, muon_history

# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_results(res_a, res_b):
    hit_a, ev_a, mu_a, snr_a, mh_a = res_a
    hit_b, ev_b, mu_b, snr_b, mh_b = res_b

    rate_a = hit_a / ev_a if ev_a > 0 else hit_a.astype(float)
    rate_b = hit_b / ev_b if ev_b > 0 else hit_b.astype(float)

    FLUX   = 1.0
    area   = SCINT_WIDTH * SCINT_HEIGHT
    rate   = FLUX * area
    time_a = mu_a / rate
    time_b = mu_b / rate

    plt.rcParams.update({
        'font.family':        'serif',
        'font.serif':         ['DejaVu Serif', 'Times New Roman', 'Times', 'serif'],
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

    fig = plt.figure(figsize=(10.0, 7.8), facecolor='white')

    outer = gridspec.GridSpec(
        3, 1, figure=fig,
        height_ratios=[2.6, 2.6, 0.8],
        hspace=0.38,
        left=0.06, right=0.98,
        top=0.95, bottom=0.02
    )
    gs_top = gridspec.GridSpecFromSubplotSpec(
        1, 3, subplot_spec=outer[0], wspace=0.55
    )
    gs_mid = gridspec.GridSpecFromSubplotSpec(
        1, 2, subplot_spec=outer[1],
        wspace=0.45, width_ratios=[1.8, 1.2]
    )

    # ── Boosted contrast colormap ─────────────────────────────────────────────
    # Normalize each map independently so shadow tiles always appear dark
    # relative to open tiles, regardless of absolute count levels
    def make_norm(rate_map):
        """Stretch colormap so vmin=shadow mean, vmax=open mean * 1.1"""
        shadow_mask_local = np.zeros((GRID_ROWS, GRID_COLS), dtype=bool)
        for r in range(GRID_ROWS):
            for c in range(GRID_COLS):
                tx = (c + 0.5) * TILE_W
                ty = (r + 0.5) * TILE_H
                if (BLOCK_X <= tx <= BLOCK_X + BLOCK_W and
                        BLOCK_Y <= ty <= BLOCK_Y + BLOCK_H):
                    shadow_mask_local[r, c] = True
        open_mean   = rate_map[~shadow_mask_local].mean()
        shadow_mean = rate_map[shadow_mask_local].mean()
        vmin = shadow_mean * 0.85   # push shadow tiles toward bottom of scale
        vmax = open_mean   * 1.05   # push open tiles toward top of scale
        return vmin, vmax

    cmap = plt.cm.RdYlGn   # red=low (shadow), green=high (open) — high contrast
    vmin_a, vmax_a = make_norm(rate_a)
    vmin_b, vmax_b = make_norm(rate_b)
    # Use shared scale across A and B so they're comparable
    vmin = min(vmin_a, vmin_b)
    vmax = max(vmax_a, vmax_b)

    def draw_block(ax):
        rx = BLOCK_X / TILE_W - 0.5
        ry = BLOCK_Y / TILE_H - 0.5
        rw = BLOCK_W / TILE_W
        rh = BLOCK_H / TILE_H
        ax.add_patch(Rectangle(
            (rx, ry), rw, rh,
            linewidth=1.2, edgecolor='black',
            facecolor='none', linestyle='--', zorder=5
        ))

    def style_heatmap(ax, panel, subtitle):
        ax.set_xlabel('Tile column', labelpad=2)
        ax.set_ylabel('Tile row',    labelpad=2)
        ax.set_xticks(range(GRID_COLS))
        ax.set_yticks(range(GRID_ROWS))
        ax.text(0.03, 0.97, panel, transform=ax.transAxes,
                fontsize=8, fontweight='bold', va='top', ha='left',
                color='black',
                bbox=dict(boxstyle='round,pad=0.12', facecolor='white',
                          alpha=0.7, edgecolor='none'))
        ax.set_title(subtitle, fontsize=7.5, pad=3, loc='left', style='italic')

    # (a) Config A
    ax_a = fig.add_subplot(gs_top[0, 0])
    im_a = ax_a.imshow(rate_a, cmap=cmap, vmin=vmin, vmax=vmax,
                       origin='lower', aspect='equal',
                       extent=[-0.5, GRID_COLS-0.5, -0.5, GRID_ROWS-0.5])
    draw_block(ax_a)
    style_heatmap(ax_a, '(a)', 'Config A: paddles + bottom grid')
    cb = plt.colorbar(im_a, ax=ax_a, fraction=0.046, pad=0.04)
    cb.set_label('Rel. hit rate', fontsize=7)
    cb.ax.tick_params(labelsize=6.5)

    # (b) Config B
    ax_b = fig.add_subplot(gs_top[0, 1])
    im_b = ax_b.imshow(rate_b, cmap=cmap, vmin=vmin, vmax=vmax,
                       origin='lower', aspect='equal',
                       extent=[-0.5, GRID_COLS-0.5, -0.5, GRID_ROWS-0.5])
    draw_block(ax_b)
    style_heatmap(ax_b, '(b)', 'Config B: dual imaging grids')
    cb = plt.colorbar(im_b, ax=ax_b, fraction=0.046, pad=0.04)
    cb.set_label('Rel. hit rate', fontsize=7)
    cb.ax.tick_params(labelsize=6.5)

    # (c) Difference
    ax_d = fig.add_subplot(gs_top[0, 2])
    diff = rate_a - rate_b
    lim  = np.abs(diff).max() if np.abs(diff).max() > 0 else 1
    im_d = ax_d.imshow(diff, cmap='RdBu_r', vmin=-lim, vmax=lim,
                       origin='lower', aspect='equal',
                       extent=[-0.5, GRID_COLS-0.5, -0.5, GRID_ROWS-0.5])
    draw_block(ax_d)
    style_heatmap(ax_d, '(c)', 'Difference map (A\u2212B)')
    cb = plt.colorbar(im_d, ax=ax_d, fraction=0.046, pad=0.04)
    cb.set_label('\u0394 hit rate', fontsize=7)
    cb.ax.tick_params(labelsize=6.5)

    # (d) SNR curves
    ax_snr = fig.add_subplot(gs_mid[0, 0])
    ax_snr.plot(mh_a, snr_a, color='#2166ac', linewidth=1.1,
                label='Config A (paddles + grid)')
    ax_snr.plot(mh_b, snr_b, color='#d6604d', linewidth=1.1,
                label='Config B (dual grid)')
    ax_snr.axhline(3.0, color='black', linewidth=0.65,
                   linestyle='--', label='SNR\u2009=\u20093')
    if mu_a < N_MUONS:
        ax_snr.axvline(mu_a, color='#2166ac', linewidth=0.65, linestyle=':')
    if mu_b < N_MUONS:
        ax_snr.axvline(mu_b, color='#d6604d', linewidth=0.65, linestyle=':')
    ax_snr.set_xlabel('Simulated muon tracks', labelpad=2)
    ax_snr.set_ylabel('Detection SNR', labelpad=2)
    ax_snr.set_title('(d)', fontsize=8, fontweight='bold', loc='left', pad=3)
    ax_snr.legend(frameon=True, framealpha=0.9, edgecolor='#bbbbbb',
                  fontsize=7, loc='upper left')
    ax_snr.set_xlim(left=0)
    ax_snr.set_ylim(bottom=0)

    # (e) Parameter table
    ax_t = fig.add_subplot(gs_mid[0, 1])
    ax_t.axis('off')
    ax_t.set_title('(e)', fontsize=8, fontweight='bold', loc='left', pad=3)

    tbl_data = [
        ['Active area',         f'{SCINT_WIDTH:.0f}\u00d7{SCINT_HEIGHT:.0f} cm'],
        ['Grid',                f'{GRID_ROWS}\u00d7{GRID_COLS} tiles'],
        ['Tile size',           f'{TILE_W:.1f}\u00d7{TILE_H:.1f} cm'],
        ['Layer spacing',       f'{Z_TOP_PADDLE:.0f} cm'],
        ['Block size',          f'{BLOCK_W:.0f}\u00d7{BLOCK_H:.0f} cm'],
        ['Attenuation',         f'{(1-__import__("numpy").exp(-BLOCK_THICK/MATERIALS[BLOCK_MATERIAL]["lambda_I"]))*100:.0f}%'],
        ['Sea-level flux',      '1 cm\u207b\u00b2 min\u207b\u00b9'],
        ['A valid events',      f'{ev_a:,}'],
        ['B valid events',      f'{ev_b:,}'],
        ['A \u2192 SNR\u22653', f'{mu_a:,} \u03bc'],
        ['B \u2192 SNR\u22653', f'{mu_b:,} \u03bc'],
        ['A image time',        f'~{time_a:.1f} min'],
        ['B image time',        f'~{time_b:.1f} min'],
    ]

    tbl = ax_t.table(
        cellText=tbl_data,
        colLabels=['Parameter', 'Value'],
        loc='upper center', cellLoc='left',
        bbox=[0.0, 0.0, 1.0, 1.0]
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(6.5)
    for row in range(len(tbl_data) + 1):
        tbl[row, 0].set_width(0.58)
        tbl[row, 1].set_width(0.42)
    for col in range(2):
        cell = tbl[0, col]
        cell.set_facecolor('#d0d0d0')
        cell.set_text_props(fontweight='bold')
        cell.set_edgecolor('#999999')
    for row in range(1, len(tbl_data) + 1):
        for col in range(2):
            cell = tbl[row, col]
            cell.set_facecolor('white' if row % 2 == 1 else '#f4f4f4')
            cell.set_edgecolor('#cccccc')

    # Caption
    ax_cap = fig.add_subplot(outer[2])
    ax_cap.axis('off')
    line1 = (
        f'Figure 1.  Simulated muon hit-rate maps and detection statistics for two detector '
        f'configurations ({GRID_ROWS}\u00d7{GRID_COLS} grid, {TILE_W:.1f}\u00d7{TILE_H:.1f}\u2009cm tiles, '
        f'{SCINT_WIDTH:.0f}\u00d7{SCINT_HEIGHT:.0f}\u2009cm active area, layer spacing {Z_TOP_PADDLE:.0f}\u2009cm).'
    )
    line2 = (
        'Dense block {:.0f}\u00d7{:.0f}\u2009cm ({mat}, \u03bb_I={lam:.0f} cm) '.format(
            BLOCK_W, BLOCK_H,
            mat=BLOCK_MATERIAL,
            lam=MATERIALS[BLOCK_MATERIAL]['lambda_I']) +
        f'centred at ({BLOCK_X+BLOCK_W/2:.0f},\u2009{BLOCK_Y+BLOCK_H/2:.0f})\u2009cm. '
        f'Tracks sampled from cos\u00b2(\u03b8) zenith distribution (\u03b8\u2009<\u200970\u00b0); '
        f'sea-level flux 1\u2009cm\u207b\u00b2\u2009min\u207b\u00b9.'
    )
    line3 = (
        f'Dashed rectangles in (a)\u2013(c) show the projected block boundary. '
        f'Dotted vertical lines in (d) mark the muon count at which SNR\u2009=\u20093 is first reached.'
    )
    for i, line in enumerate([line1, line2, line3]):
        ax_cap.text(0.5, 0.92 - i * 0.32, line,
                    ha='center', va='top', fontsize=6.8,
                    style='italic', color='#222222',
                    transform=ax_cap.transAxes)

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       'muography_sim_results.png')
    plt.savefig(out, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved: {out}")
    plt.show()

# ── Material deflection comparison ───────────────────────────────────────────

def run_material_comparison(x0, y0, theta, phi, rng, step=500):
    """
    Run simulate_A for every material in MATERIALS_TO_COMPARE and collect:
      - tile hit map  (same resolution as Figure 1)
      - scatter angle distribution
      - SNR curve
    """
    results = {}
    n = len(x0)

    for material in MATERIALS_TO_COMPARE:
        print(f'  Material: {MATERIALS[material]["label"]}...')
        hit_map      = np.zeros((GRID_ROWS, GRID_COLS), dtype=int)
        total_events = 0
        scatter_angles = []
        snr_history  = []
        muon_history = []
        muons_to_snr3 = None

        shadow_mask = np.zeros((GRID_ROWS, GRID_COLS), dtype=bool)
        for r in range(GRID_ROWS):
            for c in range(GRID_COLS):
                tx = (c + 0.5) * TILE_W
                ty = (r + 0.5) * TILE_H
                if (BLOCK_X <= tx <= BLOCK_X + BLOCK_W and
                        BLOCK_Y <= ty <= BLOCK_Y + BLOCK_H):
                    shadow_mask[r, c] = True

        for i in range(0, n, step):
            bx  = x0[i:i+step];  by  = y0[i:i+step]
            bth = theta[i:i+step]; bph = phi[i:i+step]

            survived, th_eff, ph_eff, dt, traversed = \
                process_block(bx, by, bth, bph, material, rng)

            # Collect non-zero scatter angles (block-traversing muons only)
            scatter_angles.extend(dt[dt > 0].tolist())

            # Tile hit map — use post-scatter angles so halos appear in edge tiles
            batch_map, batch_events = simulate_A(
                bx, by, th_eff, ph_eff, survived, rng
            )
            hit_map      += batch_map
            total_events += batch_events

            if total_events < 20:
                continue
            open_counts   = hit_map[~shadow_mask].astype(float)
            shadow_counts = hit_map[shadow_mask].astype(float)
            if len(shadow_counts) == 0 or open_counts.mean() == 0:
                continue
            snr = (open_counts.mean() - shadow_counts.mean()) / np.sqrt(open_counts.mean())
            snr_history.append(float(snr))
            muon_history.append(i + step)
            if snr >= 3.0 and muons_to_snr3 is None:
                muons_to_snr3 = i + step

        if muons_to_snr3 is None:
            muons_to_snr3 = n

        rate = hit_map / total_events if total_events > 0 else hit_map.astype(float)
        results[material] = {
            'rate':          rate,
            'hit_map':       hit_map,
            'total_events':  total_events,
            'scatter_angles': np.array(scatter_angles),
            'snr_history':   snr_history,
            'muon_history':  muon_history,
            'muons_to_snr3': muons_to_snr3,
        }

    return results


def plot_material_comparison(results):
    """
    Figure 2 — material deflection comparison at TILE resolution.

    Row 0 : Tile-level shadow maps for each material (what the device sees).
    Row 1 : Scatter angle histograms (material discrimination in angle space).
    Row 2 : SNR vs muon count per material.
    Row 3 : Caption.
    """
    import matplotlib.ticker as ticker

    mats  = MATERIALS_TO_COMPARE
    n_mat = len(mats)

    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 8, 'axes.labelsize': 8, 'axes.titlesize': 8,
        'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 6.5,
        'axes.linewidth': 0.6, 'xtick.major.width': 0.5, 'ytick.major.width': 0.5,
        'xtick.major.size': 3, 'ytick.major.size': 3,
        'xtick.direction': 'in', 'ytick.direction': 'in',
        'xtick.top': True, 'ytick.right': True,
        'axes.grid': False, 'figure.dpi': 300,
    })

    col_w = 2.2
    fig_w = n_mat * col_w + 1.0
    fig   = plt.figure(figsize=(fig_w, 10.0), facecolor='white')

    gs = gridspec.GridSpec(
        4, 1, figure=fig,
        height_ratios=[2.2, 2.0, 2.0, 0.9],
        hspace=0.55, left=0.07, right=0.97, top=0.96, bottom=0.02
    )
    gs_maps = gridspec.GridSpecFromSubplotSpec(1, n_mat, subplot_spec=gs[0], wspace=0.45)
    gs_bot  = gridspec.GridSpecFromSubplotSpec(1, 2,     subplot_spec=gs[1], wspace=0.38)
    gs_cap  = gs[3]

    ax_hist = fig.add_subplot(gs_bot[0, 0])
    ax_snr  = fig.add_subplot(gs_bot[0, 1])
    ax_cap  = fig.add_subplot(gs_cap)
    ax_cap.axis('off')

    # Shared colour scale across all tile maps
    all_rates = np.stack([results[m]['rate'] for m in mats])
    vmin = all_rates.min()
    vmax = all_rates.max()
    cmap = plt.cm.RdYlGn

    shadow_mask = np.zeros((GRID_ROWS, GRID_COLS), dtype=bool)
    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            tx = (c + 0.5) * TILE_W;  ty = (r + 0.5) * TILE_H
            if (BLOCK_X <= tx <= BLOCK_X + BLOCK_W and
                    BLOCK_Y <= ty <= BLOCK_Y + BLOCK_H):
                shadow_mask[r, c] = True

    panel_labels = ['({})'.format(chr(ord('a') + i)) for i in range(n_mat)]

    for idx, mat in enumerate(mats):
        res   = results[mat]
        color = MATERIALS[mat]['color']
        label = MATERIALS[mat]['label']
        lam   = MATERIALS[mat]['lambda_I']
        atten_pct = (1 - np.exp(-BLOCK_THICK / lam)) * 100

        # ── Row 0: tile shadow map ──────────────────────────────────────────
        ax_m = fig.add_subplot(gs_maps[0, idx])
        im   = ax_m.imshow(res['rate'], cmap=cmap, vmin=vmin, vmax=vmax,
                           origin='lower', aspect='equal',
                           extent=[-0.5, GRID_COLS-0.5, -0.5, GRID_ROWS-0.5])
        # Block outline
        rx = BLOCK_X/TILE_W - 0.5;  ry = BLOCK_Y/TILE_H - 0.5
        ax_m.add_patch(Rectangle((rx, ry), BLOCK_W/TILE_W, BLOCK_H/TILE_H,
                                  lw=1.2, ec='black', fc='none', ls='--', zorder=5))
        ax_m.set_xlabel('Tile col', labelpad=1, fontsize=6.5)
        ax_m.set_ylabel('Tile row', labelpad=1, fontsize=6.5)
        ax_m.set_xticks(range(GRID_COLS));  ax_m.set_yticks(range(GRID_ROWS))
        ax_m.tick_params(labelsize=6)

        shadow_mean = res['rate'][shadow_mask].mean()
        open_mean   = res['rate'][~shadow_mask].mean()
        contrast    = (open_mean - shadow_mean) / open_mean * 100 if open_mean > 0 else 0

        subtitle = (f'{label}\n'
                    f'abs={atten_pct:.0f}%  '
                    f'shadow={contrast:.0f}% dip')
        ax_m.set_title(subtitle, fontsize=6.2, pad=2)
        ax_m.text(0.03, 0.97, panel_labels[idx], transform=ax_m.transAxes,
                  fontsize=7, fontweight='bold', va='top',
                  bbox=dict(boxstyle='round,pad=0.1', fc='white', alpha=0.7, ec='none'))

        cb = plt.colorbar(im, ax=ax_m, fraction=0.046, pad=0.04)
        cb.set_label('Rel. rate', fontsize=5.5);  cb.ax.tick_params(labelsize=5)

        # ── Scatter angle histogram (Row 1 left) ───────────────────────────
        angs = res['scatter_angles']
        if len(angs) > 0 and mat != 'vacuum':
            cap_ang = min(float(np.percentile(angs, 99)), 25.0)
            ax_hist.hist(angs, bins=60, range=(0, cap_ang), density=True,
                         histtype='step', linewidth=1.1, color=color, label=label)

        # ── SNR curve (Row 1 right) ─────────────────────────────────────────
        if res['snr_history']:
            ax_snr.plot(res['muon_history'], res['snr_history'],
                        color=color, linewidth=1.1, label=label)
            mu3 = res['muons_to_snr3']
            if mu3 < N_MUONS:
                ax_snr.axvline(mu3, color=color, linewidth=0.5, linestyle=':')

    # Histogram axes
    ax_hist.set_xlabel('3-D scatter angle (degrees)', labelpad=2)
    ax_hist.set_ylabel('Probability density', labelpad=2)
    ax_hist.set_title('({})  Scatter angle distribution'.format(
        chr(ord('a') + n_mat)), fontsize=8, fontweight='bold', loc='left', pad=3)
    ax_hist.legend(frameon=True, framealpha=0.9, edgecolor='#bbbbbb',
                   fontsize=6, loc='upper right')
    ax_hist.set_xlim(left=0)

    # Note: at tile resolution, scatter halos are not spatially resolved
    tile_size_cm = TILE_W
    ax_hist.text(0.98, 0.60,
                 f'Tile size = {tile_size_cm:.0f} cm\n'
                 f'Sub-tile halos not\nresolvable in map',
                 transform=ax_hist.transAxes, fontsize=6, ha='right', va='top',
                 color='#555555', style='italic',
                 bbox=dict(boxstyle='round,pad=0.3', fc='#f8f8f8', ec='#cccccc', lw=0.5))

    # SNR axes
    ax_snr.axhline(3.0, color='black', linewidth=0.65, linestyle='--', label='SNR = 3')
    ax_snr.set_xlabel('Simulated muon tracks', labelpad=2)
    ax_snr.set_ylabel('Detection SNR', labelpad=2)
    ax_snr.set_title('({})  Muons needed for SNR = 3'.format(
        chr(ord('a') + n_mat + 1)), fontsize=8, fontweight='bold', loc='left', pad=3)
    ax_snr.legend(frameon=True, framealpha=0.9, edgecolor='#bbbbbb',
                  fontsize=6, loc='upper left')
    ax_snr.set_xlim(left=0);  ax_snr.set_ylim(bottom=0)

    # Caption
    snr_summary = ', '.join(
        '{}: {:,}'.format(MATERIALS[m]['label'], results[m]['muons_to_snr3'])
        for m in mats if m != 'vacuum'
    )
    cap = (
        f'Figure 2.  Material deflection comparison at tile resolution '
        f'({GRID_ROWS}×{GRID_COLS} tiles, {TILE_W:.0f}×{TILE_H:.0f} cm each). '
        f'Row 1: shadow maps show that denser materials produce deeper shadows '
        f'(higher absorption) but scatter halos are not spatially resolved at '
        f'{TILE_W:.0f} cm tile pitch — scatter mostly displaces muons by '
        f'< 1 tile width. '
        f'Row 2: scatter angle distributions reveal material identity in angle space '
        f'even when spatial resolution is insufficient. '
        f'Row 3: SNR curves show muons needed to reach SNR = 3 '
        f'({snr_summary}). '
        f'Block: {BLOCK_W:.0f}×{BLOCK_H:.0f}×{BLOCK_THICK:.0f} cm, '
        f'pμ = {MUON_BCP_MEV/1000:.1f} GeV/c.'
    )
    ax_cap.text(0.5, 0.85, cap, ha='center', va='top', fontsize=6.5,
                style='italic', color='#222222', transform=ax_cap.transAxes,
                wrap=True)

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       'muography_material_comparison.png')
    plt.savefig(out, dpi=300, bbox_inches='tight', facecolor='white')
    print(f'Saved: {out}')
    plt.show()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    rng = np.random.default_rng(42)

    print(f"Generating {N_MUONS:,} muon tracks...")
    x0, y0, theta, phi = generate_muons(N_MUONS, rng)

    # ── Figure 1: shadow imager Config A vs B ────────────────────────────────
    print("\nFigure 1 — Config A vs B shadow imager")
    print("Running Config A...")
    res_a = run_snr_test(simulate_A, x0, y0, theta, phi, rng)

    print("Running Config B...")
    res_b = run_snr_test(simulate_B, x0, y0, theta, phi, rng)

    ev_a, mu_a = res_a[1], res_a[2]
    ev_b, mu_b = res_b[1], res_b[2]
    print(f"  Config A: {ev_a:,} valid events, needs ~{mu_a:,} muons for SNR>=3")
    print(f"  Config B: {ev_b:,} valid events, needs ~{mu_b:,} muons for SNR>=3")
    plot_results(res_a, res_b)

    # ── Figure 2: material deflection comparison ──────────────────────────────
    print("\nFigure 2 — material deflection comparison (tile resolution)")
    mat_results = run_material_comparison(x0, y0, theta, phi, rng)
    plot_material_comparison(mat_results)


if __name__ == '__main__':
    main()