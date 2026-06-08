"""
muography_sim_v2.py
-------------------
Simulates two detector configurations for a binary muon shadow imager.

  Config A: 2 large single-plane trigger paddles (top) + imaging grid (bottom)
  Config B: Imaging grid on top AND bottom (tiled both layers)

Physical units throughout are cm.
Muon tracks follow a realistic cos^2(theta) zenith distribution.

Usage:
  python3 muography_sim_v2.py

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

    fig = plt.figure(figsize=(15, 10), facecolor='#0d1117')
    fig.suptitle(
        f'Muography Sim  |  {GRID_ROWS}×{GRID_COLS} grid  |  '
        f'{SCINT_WIDTH}×{SCINT_HEIGHT} cm  |  '
        f'Layer spacing {Z_TOP_PADDLE} cm  |  '
        f'Block {BLOCK_W}×{BLOCK_H} cm  atten {BLOCK_ATTEN*100:.0f}%',
        color='white', fontsize=11, fontweight='bold', y=0.99
    )

    gs = gridspec.GridSpec(2, 3, figure=fig,
                           hspace=0.45, wspace=0.35,
                           left=0.07, right=0.97,
                           top=0.88, bottom=0.07)

    cmap = plt.cm.inferno
    vmax = max(rate_a.max(), rate_b.max())

    def draw_block(ax):
        rx = BLOCK_X / TILE_W - 0.5
        ry = BLOCK_Y / TILE_H - 0.5
        rw = BLOCK_W / TILE_W
        rh = BLOCK_H / TILE_H
        ax.add_patch(Rectangle(
            (rx, ry), rw, rh,
            linewidth=2, edgecolor='#00ffcc',
            facecolor='none', linestyle='--'
        ))

    def style_ax(ax, title):
        ax.set_title(title, color='white', fontsize=10)
        ax.set_xlabel('Tile col', color='#aaa', fontsize=9)
        ax.set_ylabel('Tile row', color='#aaa', fontsize=9)
        ax.tick_params(colors='#aaa')
        ax.set_facecolor('#0d1117')
        for sp in ax.spines.values():
            sp.set_edgecolor('#333')
        ax.set_xticks(range(GRID_COLS))
        ax.set_yticks(range(GRID_ROWS))

    # Config A
    ax_a = fig.add_subplot(gs[0, 0])
    im_a = ax_a.imshow(rate_a, cmap=cmap, vmin=0, vmax=vmax,
                       origin='lower', aspect='equal',
                       extent=[-0.5, GRID_COLS-0.5, -0.5, GRID_ROWS-0.5])
    draw_block(ax_a)
    style_ax(ax_a, 'Config A — Hit Rate\n(large paddles top, grid bottom)')
    plt.colorbar(im_a, ax=ax_a, label='Hit rate').ax.yaxis.label.set_color('#aaa')

    # Config B
    ax_b = fig.add_subplot(gs[0, 1])
    im_b = ax_b.imshow(rate_b, cmap=cmap, vmin=0, vmax=vmax,
                       origin='lower', aspect='equal',
                       extent=[-0.5, GRID_COLS-0.5, -0.5, GRID_ROWS-0.5])
    draw_block(ax_b)
    style_ax(ax_b, 'Config B — Hit Rate\n(imaging grid top AND bottom)')
    plt.colorbar(im_b, ax=ax_b, label='Hit rate').ax.yaxis.label.set_color('#aaa')

    # Difference map
    ax_d = fig.add_subplot(gs[0, 2])
    diff = rate_a - rate_b
    lim  = np.abs(diff).max() if np.abs(diff).max() > 0 else 1
    im_d = ax_d.imshow(diff, cmap='RdBu_r', vmin=-lim, vmax=lim,
                       origin='lower', aspect='equal',
                       extent=[-0.5, GRID_COLS-0.5, -0.5, GRID_ROWS-0.5])
    draw_block(ax_d)
    style_ax(ax_d, 'Difference (A − B)\nper tile hit rate')
    plt.colorbar(im_d, ax=ax_d, label='Δ rate').ax.yaxis.label.set_color('#aaa')

    # SNR curves
    ax_snr = fig.add_subplot(gs[1, :2])
    ax_snr.set_facecolor('#0d1117')
    ax_snr.plot(mh_a, snr_a, color='#7F77DD', linewidth=1.5, label='Config A')
    ax_snr.plot(mh_b, snr_b, color='#E07B54', linewidth=1.5, label='Config B')
    ax_snr.axhline(3.0, color='white', linewidth=0.8,
                   linestyle='--', alpha=0.5, label='SNR = 3 threshold')
    if mu_a < N_MUONS:
        ax_snr.axvline(mu_a, color='#7F77DD', linewidth=0.8, linestyle=':')
    if mu_b < N_MUONS:
        ax_snr.axvline(mu_b, color='#E07B54', linewidth=0.8, linestyle=':')
    ax_snr.set_xlabel('Muons simulated', color='#aaa', fontsize=9)
    ax_snr.set_ylabel('SNR', color='#aaa', fontsize=9)
    ax_snr.set_title('Statistical Comparison — muons needed to reach SNR ≥ 3',
                     color='white', fontsize=10)
    ax_snr.tick_params(colors='#aaa')
    ax_snr.legend(facecolor='#1a1f2e', edgecolor='#333',
                  labelcolor='white', fontsize=9)
    for sp in ax_snr.spines.values():
        sp.set_edgecolor('#333')

    # Summary stats
    ax_s = fig.add_subplot(gs[1, 2])
    ax_s.set_facecolor('#0d1117')
    ax_s.axis('off')
    ax_s.set_title('Summary', color='white', fontsize=10)

    lines = [
        ('#7F77DD', 'Config A',         ''),
        ('#aaa',    '  Valid events',    f'{ev_a:,}'),
        ('#aaa',    '  Muons to SNR≥3', f'{mu_a:,}'),
        ('#aaa',    '  Est. image time', f'~{time_a:.1f} min'),
        ('#aaa',    '',                  ''),
        ('#E07B54', 'Config B',         ''),
        ('#aaa',    '  Valid events',    f'{ev_b:,}'),
        ('#aaa',    '  Muons to SNR≥3', f'{mu_b:,}'),
        ('#aaa',    '  Est. image time', f'~{time_b:.1f} min'),
        ('#aaa',    '',                  ''),
        ('#1D9E75', 'Geometry',         ''),
        ('#aaa',    '  Active area',     f'{SCINT_WIDTH}×{SCINT_HEIGHT} cm'),
        ('#aaa',    '  Tile size',       f'{TILE_W:.1f}×{TILE_H:.1f} cm'),
        ('#aaa',    '  Layer spacing',   f'{Z_TOP_PADDLE} cm'),
        ('#aaa',    '  Block',           f'{BLOCK_W}×{BLOCK_H} cm @ {BLOCK_ATTEN*100:.0f}% atten'),
    ]

    y = 0.97
    for color, left, right in lines:
        ax_s.text(0.02, y, left,  transform=ax_s.transAxes,
                  color=color, fontsize=8.5, va='top', fontfamily='monospace')
        ax_s.text(0.98, y, right, transform=ax_s.transAxes,
                  color='white', fontsize=8.5, va='top', ha='right',
                  fontfamily='monospace')
        y -= 0.063

    import os
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'muography_sim_v2_results.png')
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='#0d1117')
    print(f"Saved: {out}")
    print("Saved: muography_sim_v2_results.png")
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