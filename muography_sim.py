"""
muography_sim.py
-------------------
Simulates two detector configurations for a binary muon shadow imager.

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
Z_TOP_PADDLE  = 30.0  # top trigger paddle / top imaging grid (Config B)
Z_BOT_PADDLE  = 20.0  # second trigger paddle (Config A only)
Z_IMAGING_BOT =  0.0  # bottom imaging grid — reference plane

# Dense block
BLOCK_X     =  5.0    # cm  left edge
BLOCK_Y     =  5.0    # cm  bottom edge
BLOCK_W     = 10.0    # cm  width
BLOCK_H     = 10.0    # cm  height
BLOCK_Z     = 15.0    # cm  height above bottom grid (sits between layers)
BLOCK_ATTEN =  0.85   # fraction of muons blocked (0=transparent, 1=opaque)

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
    SNR = (mean open rate - mean shadow rate) / std(open rate)
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

        rate_map     = hit_map / total_events
        open_rates   = rate_map[~shadow_mask]
        shadow_rates = rate_map[shadow_mask]

        if open_rates.std() == 0 or len(shadow_rates) == 0:
            continue

        snr = (open_rates.mean() - shadow_rates.mean()) / open_rates.std()
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

    # Imaging time estimate — ~1 muon/cm^2/min at sea level
    FLUX   = 1.0
    area   = SCINT_WIDTH * SCINT_HEIGHT
    rate   = FLUX * area
    time_a = mu_a / rate
    time_b = mu_b / rate

    # ── Academic style ────────────────────────────────────────────────────────
    plt.rcParams.update({
        'font.family':       'serif',
        'font.serif':        ['DejaVu Serif', 'Times New Roman', 'Times', 'serif'],
        'font.size':         9,
        'axes.labelsize':    9,
        'axes.titlesize':    9,
        'xtick.labelsize':   8,
        'ytick.labelsize':   8,
        'legend.fontsize':   8,
        'axes.linewidth':    0.6,
        'xtick.major.width': 0.6,
        'ytick.major.width': 0.6,
        'xtick.direction':   'in',
        'ytick.direction':   'in',
        'xtick.top':         True,
        'ytick.right':       True,
        'axes.grid':         False,
    })

    fig = plt.figure(figsize=(7.2, 8.5), facecolor='white', dpi=300)

    # Caption-style title at the very top
    fig.text(
        0.5, 0.985,
        'Figure 1. Simulated muon hit-rate maps and detection statistics '
        'for two scintillator detector configurations.',
        ha='center', va='top', fontsize=8, style='italic',
        color='black', wrap=True,
        transform=fig.transFigure
    )

    gs = gridspec.GridSpec(
        2, 3, figure=fig,
        hspace=0.55, wspace=0.42,
        left=0.09, right=0.97,
        top=0.92, bottom=0.09
    )

    cmap = plt.cm.viridis   # perceptually uniform, prints well in greyscale
    vmax = max(rate_a.max(), rate_b.max())

    def draw_block(ax):
        rx = BLOCK_X / TILE_W - 0.5
        ry = BLOCK_Y / TILE_H - 0.5
        rw = BLOCK_W / TILE_W
        rh = BLOCK_H / TILE_H
        ax.add_patch(Rectangle(
            (rx, ry), rw, rh,
            linewidth=0.8, edgecolor='white',
            facecolor='none', linestyle='--', zorder=5
        ))

    def style_ax(ax, label, subtitle):
        """Label format: bold panel letter + subtitle below, no box title."""
        ax.set_xlabel('Tile column', labelpad=3)
        ax.set_ylabel('Tile row',    labelpad=3)
        ax.set_xticks(range(GRID_COLS))
        ax.set_yticks(range(GRID_ROWS))
        # Panel label in top-left corner inside axes
        ax.text(0.03, 0.97, label, transform=ax.transAxes,
                fontsize=9, fontweight='bold', va='top', ha='left',
                color='white',
                bbox=dict(boxstyle='round,pad=0.15', facecolor='black',
                          alpha=0.45, edgecolor='none'))
        # Subtitle below axes
        ax.set_title(subtitle, fontsize=8, pad=4, loc='left', style='italic')

    # ── (a) Config A hit map ──────────────────────────────────────────────────
    ax_a = fig.add_subplot(gs[0, 0])
    im_a = ax_a.imshow(rate_a, cmap=cmap, vmin=0, vmax=vmax,
                       origin='lower', aspect='equal',
                       extent=[-0.5, GRID_COLS-0.5, -0.5, GRID_ROWS-0.5])
    draw_block(ax_a)
    style_ax(ax_a, '(a)', 'Config A: paddles + bottom grid')
    cb_a = plt.colorbar(im_a, ax=ax_a, fraction=0.046, pad=0.04)
    cb_a.set_label('Relative hit rate', fontsize=7.5)
    cb_a.ax.tick_params(labelsize=7)

    # ── (b) Config B hit map ──────────────────────────────────────────────────
    ax_b = fig.add_subplot(gs[0, 1])
    im_b = ax_b.imshow(rate_b, cmap=cmap, vmin=0, vmax=vmax,
                       origin='lower', aspect='equal',
                       extent=[-0.5, GRID_COLS-0.5, -0.5, GRID_ROWS-0.5])
    draw_block(ax_b)
    style_ax(ax_b, '(b)', 'Config B: dual imaging grids')
    cb_b = plt.colorbar(im_b, ax=ax_b, fraction=0.046, pad=0.04)
    cb_b.set_label('Relative hit rate', fontsize=7.5)
    cb_b.ax.tick_params(labelsize=7)

    # ── (c) Difference map ────────────────────────────────────────────────────
    ax_d = fig.add_subplot(gs[0, 2])
    diff = rate_a - rate_b
    lim  = np.abs(diff).max() if np.abs(diff).max() > 0 else 1
    im_d = ax_d.imshow(diff, cmap='RdBu_r', vmin=-lim, vmax=lim,
                       origin='lower', aspect='equal',
                       extent=[-0.5, GRID_COLS-0.5, -0.5, GRID_ROWS-0.5])
    draw_block(ax_d)
    style_ax(ax_d, '(c)', 'Difference map (A\u2212B)')
    cb_d = plt.colorbar(im_d, ax=ax_d, fraction=0.046, pad=0.04)
    cb_d.set_label('\u0394 hit rate', fontsize=7.5)
    cb_d.ax.tick_params(labelsize=7)

    # ── (d) SNR vs muon count ─────────────────────────────────────────────────
    ax_snr = fig.add_subplot(gs[1, :2])
    ax_snr.plot(mh_a, snr_a, color='#2166ac', linewidth=1.2,
                label='Config A (paddles + grid)')
    ax_snr.plot(mh_b, snr_b, color='#d6604d', linewidth=1.2,
                label='Config B (dual grid)')
    ax_snr.axhline(3.0, color='black', linewidth=0.7,
                   linestyle='--', label='SNR\u2009=\u20093 threshold')
    if mu_a < N_MUONS:
        ax_snr.axvline(mu_a, color='#2166ac', linewidth=0.7, linestyle=':')
    if mu_b < N_MUONS:
        ax_snr.axvline(mu_b, color='#d6604d', linewidth=0.7, linestyle=':')
    ax_snr.set_xlabel('Simulated muon tracks')
    ax_snr.set_ylabel('Detection SNR')
    ax_snr.set_title('(d)', fontsize=9, fontweight='bold', loc='left', pad=4)
    ax_snr.legend(frameon=True, framealpha=0.9, edgecolor='#cccccc',
                  fontsize=7.5, loc='upper left')
    ax_snr.set_xlim(left=0)
    ax_snr.set_ylim(bottom=0)

    # ── Parameter table ───────────────────────────────────────────────────────
    ax_t = fig.add_subplot(gs[1, 2])
    ax_t.axis('off')
    ax_t.set_title('(e)', fontsize=9, fontweight='bold', loc='left', pad=4)

    rows = [
        ['Parameter',                   'Value'],
        ['Active area',                 f'{SCINT_WIDTH}\u00d7{SCINT_HEIGHT}\u2009cm'],
        ['Grid',                        f'{GRID_ROWS}\u00d7{GRID_COLS} tiles'],
        ['Tile size',                   f'{TILE_W:.1f}\u00d7{TILE_H:.1f}\u2009cm'],
        ['Layer spacing',               f'{Z_TOP_PADDLE:.0f}\u2009cm'],
        ['Block dimensions',            f'{BLOCK_W:.0f}\u00d7{BLOCK_H:.0f}\u2009cm'],
        ['Block attenuation',           f'{BLOCK_ATTEN*100:.0f}%'],
        ['Muon flux (sea level)',        '1\u2009cm\u207b\u00b2\u2009min\u207b\u00b9'],
        ['Config A valid events',       f'{ev_a:,}'],
        ['Config B valid events',       f'{ev_b:,}'],
        ['Config A muons to SNR\u22653', f'{mu_a:,}'],
        ['Config B muons to SNR\u22653', f'{mu_b:,}'],
        ['Config A est. image time',    f'~{time_a:.1f}\u2009min'],
        ['Config B est. image time',    f'~{time_b:.1f}\u2009min'],
    ]

    tbl = ax_t.table(
        cellText=rows[1:],
        colLabels=rows[0],
        loc='center',
        cellLoc='left',
        bbox=[0.0, 0.0, 1.0, 1.0]
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7)

    # Style header row
    for col in range(2):
        cell = tbl[0, col]
        cell.set_facecolor('#dddddd')
        cell.set_text_props(fontweight='bold')
        cell.set_edgecolor('#aaaaaa')

    # Style data rows
    for row in range(1, len(rows)):
        for col in range(2):
            cell = tbl[row, col]
            cell.set_facecolor('white' if row % 2 == 0 else '#f7f7f7')
            cell.set_edgecolor('#cccccc')

    tbl.auto_set_column_width([0, 1])

    # ── Caption block at bottom ───────────────────────────────────────────────
    caption = (
        f'Simulation parameters: {GRID_ROWS}\u00d7{GRID_COLS} scintillator tile grid '
        f'({TILE_W:.1f}\u00d7{TILE_H:.1f}\u2009cm tiles) over a '
        f'{SCINT_WIDTH:.0f}\u00d7{SCINT_HEIGHT:.0f}\u2009cm active area. '
        f'Layer separation {Z_TOP_PADDLE:.0f}\u2009cm. Dense block '
        f'{BLOCK_W:.0f}\u00d7{BLOCK_H:.0f}\u2009cm at {BLOCK_ATTEN*100:.0f}\\% attenuation, '
        f'centred at ({BLOCK_X+BLOCK_W/2:.0f}, {BLOCK_Y+BLOCK_H/2:.0f})\u2009cm. '
        f'Muon tracks sampled from a cos\u00b2(\u03b8) zenith distribution '
        f'(\u03b8\u2009<\u200970\u00b0). Dashed white rectangles in (a)\u2013(c) '
        f'indicate the projected block boundary. Dotted vertical lines in (d) '
        f'mark the muon count at which SNR\u2009=\u20093 is first reached.'
    )
    fig.text(0.5, 0.005, caption, ha='center', va='bottom',
             fontsize=6.5, style='italic', color='#333333',
             wrap=True, transform=fig.transFigure,
             multialignment='center')

    import os
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
    main()