"""
muography_sim.py
-------------------
Simulates two detector configurations for a binary muon shadow imager.

  Config A: 2 large single-plane trigger paddles (top) + imaging grid (bottom)
  Config B: Imaging grid on top AND bottom (tiled both layers)

Physical units throughout are cm.
Muon tracks follow a realistic cos^2(theta) zenith distribution (explained in report intro)

Usage:
  python3 muography_sim.py

Edit the constants block below to change geometry.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle

# ── Physical geometry (all units in cm) ──────────────────────────────────────

# Scintillator active area
SCINT_WIDTH  = 20.0   # cm
SCINT_HEIGHT = 20.0   # cm

# Grid resolution — set directly
GRID_COLS = 4
GRID_ROWS = 4

# Tile size derived automatically
TILE_W = SCINT_WIDTH  / GRID_COLS
TILE_H = SCINT_HEIGHT / GRID_ROWS

# Layer heights (cm above bottom imaging grid)
Z_TOP_PADDLE  = 40.0  # top trigger paddle / top imaging grid (Config B)
Z_BOT_PADDLE  = 30.0  # second trigger paddle (Config A only)
Z_IMAGING_BOT =  0.0  # bottom imaging grid — reference plane

# Dense block
BLOCK_X     =  5.0    # cm  left edge
BLOCK_Y     =  5.0    # cm  bottom edge
BLOCK_W     = 10.0    # cm  width
BLOCK_H     = 10.0    # cm  height
BLOCK_Z     = 30.0    # cm  height above bottom grid (sits between layers)
BLOCK_ATTEN =  0.70   # fraction of muons blocked (0=transparent, 1=opaque)

# Simulation
N_MUONS = 100_000

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
    """
    x0 = rng.uniform(0, SCINT_WIDTH,  n)
    y0 = rng.uniform(0, SCINT_HEIGHT, n)

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
    """
    Return (x, y) position of a muon track at z_target.
    Muon enters at (x0, y0) at height z_start travelling at (theta, phi).
    """
    dz    = z_start - z_target
    drift = dz * np.tan(theta)
    x     = x0 + drift * np.cos(phi)
    y     = y0 + drift * np.sin(phi)
    return x, y

# ── Block intersection ────────────────────────────────────────────────────────

def check_block(x0, y0, theta, phi, rng):
    """
    Compute where each muon passes through BLOCK_Z and check if it
    hits the block. Apply attenuation probabilistically.
    Returns boolean mask — True means muon survived.
    """
    x_block, y_block = track_position_at_z(
        x0, y0, theta, phi,
        z_start  = Z_TOP_PADDLE,
        z_target = BLOCK_Z
    )

    in_block = (
        (x_block >= BLOCK_X) & (x_block <= BLOCK_X + BLOCK_W) &
        (y_block >= BLOCK_Y) & (y_block <= BLOCK_Y + BLOCK_H)
    )

    absorbed = in_block & (rng.random(len(x0)) < BLOCK_ATTEN)
    return ~absorbed


def pos_to_tile(x, y):
    """
    Convert physical (x, y) in cm to (row, col) tile index.
    Returns -1 for positions outside the active area.
    """
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
    Config A: large single-plane paddles on top, imaging grid on bottom.
    Valid event = muon survived block AND lands in active area of bottom grid.
    Position determined entirely by bottom grid tile.
    """
    hit_map = np.zeros((GRID_ROWS, GRID_COLS), dtype=int)
    total_events = 0

    x_bot, y_bot = track_position_at_z(
        x0[survived], y0[survived], theta[survived], phi[survived],
        z_start  = Z_TOP_PADDLE,
        z_target = Z_IMAGING_BOT
    )

    row_bot, col_bot = pos_to_tile(x_bot, y_bot)
    in_bounds = (row_bot >= 0) & (col_bot >= 0)

    for r, c in zip(row_bot[in_bounds], col_bot[in_bounds]):
        hit_map[r, c] += 1
        total_events += 1

    return hit_map, total_events


def simulate_B(x0, y0, theta, phi, survived, rng):
    """
    Config B: imaging grid on top AND bottom.
    Valid event = muon survived block AND fires the same tile on both grids.
    Angled muons that drift across a tile boundary are rejected.
    """
    hit_map = np.zeros((GRID_ROWS, GRID_COLS), dtype=int)
    total_events = 0

    # Top grid — muon enters here so no drift from top paddle
    x_top = x0[survived].copy()
    y_top = y0[survived].copy()

    # Bottom grid — muon has drifted
    x_bot, y_bot = track_position_at_z(
        x0[survived], y0[survived], theta[survived], phi[survived],
        z_start  = Z_TOP_PADDLE,
        z_target = Z_IMAGING_BOT
    )

    row_top, col_top = pos_to_tile(x_top, y_top)
    row_bot, col_bot = pos_to_tile(x_bot, y_bot)

    in_bounds = (row_top >= 0) & (col_top >= 0) & \
                (row_bot >= 0) & (col_bot >= 0)

    same_tile = (row_top == row_bot) & (col_top == col_bot)

    valid = in_bounds & same_tile

    for r, c in zip(row_bot[valid], col_bot[valid]):
        hit_map[r, c] += 1
        total_events += 1

    return hit_map, total_events

# ── SNR test ──────────────────────────────────────────────────────────────────

def run_snr_test(config_fn, x0, y0, theta, phi, rng,
                 target_snr=3.0, step=500):
    """
    Incrementally accumulate events and track when SNR first hits target.

    SNR = (mean open count - mean shadow count) / sqrt(mean open count)

    Poisson SNR — noise floor is sqrt(N) for counting statistics.
    Stable regardless of hit map uniformity, unlike the std-based version.
    """
    hit_map = np.zeros((GRID_ROWS, GRID_COLS), dtype=int)
    total_events = 0
    snr_history = []
    muon_history = []
    muons_to_detect = None

    # Pre-classify tiles as shadow or open based on block projection
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
        bx  = x0[i:i+step]
        by  = y0[i:i+step]
        bth = theta[i:i+step]
        bph = phi[i:i+step]

        survived = check_block(bx, by, bth, bph, rng)
        batch_map, batch_events = config_fn(bx, by, bth, bph, survived, rng)

        hit_map      += batch_map
        total_events += batch_events

        if total_events < 20:
            continue

        # Work in raw counts — Poisson noise scales as sqrt(N)
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
    import os
    hit_a, ev_a, mu_a, snr_a, mh_a = res_a
    hit_b, ev_b, mu_b, snr_b, mh_b = res_b

    rate_a = hit_a / ev_a if ev_a > 0 else hit_a.astype(float)
    rate_b = hit_b / ev_b if ev_b > 0 else hit_b.astype(float)

    FLUX   = 1.0
    area   = SCINT_WIDTH * SCINT_HEIGHT
    rate   = FLUX * area
    time_a = mu_a / rate
    time_b = mu_b / rate

    # ── Academic rcParams ─────────────────────────────────────────────────────
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

    # ── Layout: two rows, explicit figure height gives caption room ───────────
    # Row 0: three hit maps  Row 1: SNR plot + table
    # Caption lives in its own axes below row 1 — no fig.text overlap
    fig = plt.figure(figsize=(10.0, 7.8), facecolor='white')

    # Three-row gridspec: maps | plots | caption
    outer = gridspec.GridSpec(
        3, 1, figure=fig,
        height_ratios=[2.6, 2.6, 0.8],
        hspace=0.38,
        left=0.06, right=0.98,
        top=0.95, bottom=0.02
    )

    # Top row: three heatmaps
    gs_top = gridspec.GridSpecFromSubplotSpec(
        1, 3, subplot_spec=outer[0],
        wspace=0.55, hspace=0
    )

    # Middle row: SNR plot + table — give table more horizontal room
    gs_mid = gridspec.GridSpecFromSubplotSpec(
        1, 2, subplot_spec=outer[1],
        wspace=0.45, width_ratios=[1.8, 1.2]
    )

    cmap = plt.cm.viridis
    vmax = max(rate_a.max(), rate_b.max())

    def draw_block(ax):
        rx = BLOCK_X / TILE_W - 0.5
        ry = BLOCK_Y / TILE_H - 0.5
        rw = BLOCK_W / TILE_W
        rh = BLOCK_H / TILE_H
        ax.add_patch(Rectangle(
            (rx, ry), rw, rh,
            linewidth=0.9, edgecolor='white',
            facecolor='none', linestyle='--', zorder=5
        ))

    def style_heatmap(ax, panel, subtitle):
        ax.set_xlabel('Tile column', labelpad=2)
        ax.set_ylabel('Tile row',    labelpad=2)
        ax.set_xticks(range(GRID_COLS))
        ax.set_yticks(range(GRID_ROWS))
        ax.text(0.03, 0.97, panel, transform=ax.transAxes,
                fontsize=8, fontweight='bold', va='top', ha='left',
                color='white',
                bbox=dict(boxstyle='round,pad=0.12', facecolor='black',
                          alpha=0.5, edgecolor='none'))
        ax.set_title(subtitle, fontsize=7.5, pad=3, loc='left', style='italic')

    # ── (a) Config A ─────────────────────────────────────────────────────────
    ax_a = fig.add_subplot(gs_top[0, 0])
    im_a = ax_a.imshow(rate_a, cmap=cmap, vmin=0, vmax=vmax,
                       origin='lower', aspect='equal',
                       extent=[-0.5, GRID_COLS-0.5, -0.5, GRID_ROWS-0.5])
    draw_block(ax_a)
    style_heatmap(ax_a, '(a)', 'Config A: paddles + bottom grid')
    cb = plt.colorbar(im_a, ax=ax_a, fraction=0.046, pad=0.04)
    cb.set_label('Rel. hit rate', fontsize=7)
    cb.ax.tick_params(labelsize=6.5)

    # ── (b) Config B ─────────────────────────────────────────────────────────
    ax_b = fig.add_subplot(gs_top[0, 1])
    im_b = ax_b.imshow(rate_b, cmap=cmap, vmin=0, vmax=vmax,
                       origin='lower', aspect='equal',
                       extent=[-0.5, GRID_COLS-0.5, -0.5, GRID_ROWS-0.5])
    draw_block(ax_b)
    style_heatmap(ax_b, '(b)', 'Config B: dual imaging grids')
    cb = plt.colorbar(im_b, ax=ax_b, fraction=0.046, pad=0.04)
    cb.set_label('Rel. hit rate', fontsize=7)
    cb.ax.tick_params(labelsize=6.5)

    # ── (c) Difference ───────────────────────────────────────────────────────
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

    # ── (d) SNR curve ─────────────────────────────────────────────────────────
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

    # ── (e) Parameter table ───────────────────────────────────────────────────
    ax_t = fig.add_subplot(gs_mid[0, 1])
    ax_t.axis('off')
    ax_t.set_title('(e)', fontsize=8, fontweight='bold', loc='left', pad=3)

    tbl_data = [
        ['Active area',      f'{SCINT_WIDTH:.0f}\u00d7{SCINT_HEIGHT:.0f} cm'],
        ['Grid',             f'{GRID_ROWS}\u00d7{GRID_COLS} tiles'],
        ['Tile size',        f'{TILE_W:.1f}\u00d7{TILE_H:.1f} cm'],
        ['Layer spacing',    f'{Z_TOP_PADDLE:.0f} cm'],
        ['Block size',       f'{BLOCK_W:.0f}\u00d7{BLOCK_H:.0f} cm'],
        ['Attenuation',      f'{BLOCK_ATTEN*100:.0f}%'],
        ['Sea-level flux',   '1 cm\u207b\u00b2 min\u207b\u00b9'],
        ['A valid events',   f'{ev_a:,}'],
        ['B valid events',   f'{ev_b:,}'],
        ['A \u2192 SNR\u22653', f'{mu_a:,} \u03bc'],
        ['B \u2192 SNR\u22653', f'{mu_b:,} \u03bc'],
        ['A image time',     f'~{time_a:.1f} min'],
        ['B image time',     f'~{time_b:.1f} min'],
    ]

    tbl = ax_t.table(
        cellText=tbl_data,
        colLabels=['Parameter', 'Value'],
        loc='upper center',
        cellLoc='left',
        bbox=[0.0, 0.0, 1.0, 1.0]
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(6.5)

    # Set explicit column widths: 58% param, 42% value
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

    # ── Caption in its own axes (no overlap) ──────────────────────────────────
    ax_cap = fig.add_subplot(outer[2])
    ax_cap.axis('off')
    line1 = (
        f'Figure 1.  Simulated muon hit-rate maps and detection statistics for two detector '
        f'configurations ({GRID_ROWS}\u00d7{GRID_COLS} grid, {TILE_W:.1f}\u00d7{TILE_H:.1f}\u2009cm tiles, '
        f'{SCINT_WIDTH:.0f}\u00d7{SCINT_HEIGHT:.0f}\u2009cm active area, layer spacing {Z_TOP_PADDLE:.0f}\u2009cm).'
    )
    line2 = (
        f'Dense block {BLOCK_W:.0f}\u00d7{BLOCK_H:.0f}\u2009cm at {BLOCK_ATTEN*100:.0f}% attenuation '
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

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    rng = np.random.default_rng(42)

    print(f"Generating {N_MUONS:,} muon tracks...")
    x0, y0, theta, phi = generate_muons(N_MUONS, rng)

    print("Running Config A...")
    res_a = run_snr_test(simulate_A, x0, y0, theta, phi, rng)

    print("Running Config B...")
    res_b = run_snr_test(simulate_B, x0, y0, theta, phi, rng)

    ev_a, mu_a = res_a[1], res_a[2]
    ev_b, mu_b = res_b[1], res_b[2]
    print(f"\nConfig A: {ev_a:,} valid events, needs ~{mu_a:,} muons for SNR≥3")
    print(f"Config B: {ev_b:,} valid events, needs ~{mu_b:,} muons for SNR≥3")

    plot_results(res_a, res_b)


if __name__ == '__main__':
    main() ---- building off of this base code, I want to see the deflection of the muons off of different materials, can you implement this