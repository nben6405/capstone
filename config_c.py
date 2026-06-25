"""
muography_sim.py
-------------------
Simulates three detector configurations for a binary muon shadow imager.

  Config A: 2 large single-plane trigger paddles (top) + imaging grid (bottom)
  Config B: 3 imaging grids — top + mid trigger, bottom imaging grid.
            Top+bottom same-tile coincidence; mid must be in active area.
  Config C: 1 imaging grid (top) + 1 imaging grid (bottom).
            Same-tile coincidence on both planes — simplest 2-plane tracker.

Physical units throughout are cm.
Muon tracks follow a realistic cos^2(theta) zenith distribution.
Angular cutoff 30 degrees — captures ~35% of sea-level flux while keeping
geometric efficiency reasonable for an 80 cm stack.

Usage:
  python3 muography_sim.py
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
import os

# ── Physical geometry (all units in cm) ──────────────────────────────────────

SCINT_WIDTH  = 20.0
SCINT_HEIGHT = 20.0
GRID_COLS    = 4
GRID_ROWS    = 4

TILE_W = SCINT_WIDTH  / GRID_COLS   # 5.0 cm
TILE_H = SCINT_HEIGHT / GRID_ROWS   # 5.0 cm

# Layer heights (cm above bottom imaging grid)
Z_TOP_PADDLE  = 80.0   # top trigger paddle (A) / top imaging grid (B, C)
Z_BOT_PADDLE  = 75.0   # second trigger paddle (Config A)
Z_MID_GRID    = 75.0   # mid imaging grid (Config B)
Z_IMAGING_BOT =  0.0   # bottom imaging grid — reference plane (all configs)

# Dense block
BLOCK_X     =  5.0
BLOCK_Y     =  5.0
BLOCK_W     = 10.0
BLOCK_H     = 10.0
BLOCK_Z     = 65.0    # cm above bottom grid — close to top triggers
BLOCK_ATTEN =  0.35   # 35% — 10 cm tungsten

THETA_MAX_DEG = 30.0
N_MUONS = 3_000_000

# ── Track generation ──────────────────────────────────────────────────────────

def generate_muons(n, rng):
    margin = Z_TOP_PADDLE * np.tan(np.radians(THETA_MAX_DEG))
    x0 = rng.uniform(-margin, SCINT_WIDTH  + margin, n)
    y0 = rng.uniform(-margin, SCINT_HEIGHT + margin, n)

    theta = []
    while len(theta) < n:
        candidates  = rng.uniform(0, np.radians(THETA_MAX_DEG), n)
        accept_prob = np.cos(candidates) ** 2
        mask        = rng.uniform(0, 1, n) < accept_prob
        theta.extend(candidates[mask].tolist())
    theta = np.array(theta[:n])

    phi = rng.uniform(0, 2 * np.pi, n)
    return x0, y0, theta, phi


def track_position_at_z(x0, y0, theta, phi, z_start, z_target):
    dz    = z_start - z_target
    drift = dz * np.tan(theta)
    return x0 + drift * np.cos(phi), y0 + drift * np.sin(phi)

# ── Block intersection ────────────────────────────────────────────────────────

def check_block(x0, y0, theta, phi, rng):
    x_block, y_block = track_position_at_z(
        x0, y0, theta, phi,
        z_start=Z_TOP_PADDLE, z_target=BLOCK_Z
    )
    in_block = (
        (x_block >= BLOCK_X) & (x_block <= BLOCK_X + BLOCK_W) &
        (y_block >= BLOCK_Y) & (y_block <= BLOCK_Y + BLOCK_H)
    )
    absorbed = in_block & (rng.random(len(x0)) < BLOCK_ATTEN)
    return ~absorbed


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
    Config A: 2 large paddles trigger, bottom imaging grid records hit tile.
    No angular constraint — any track landing in the active area is accepted.
    """
    hit_map      = np.zeros((GRID_ROWS, GRID_COLS), dtype=int)
    total_events = 0

    x_bot, y_bot = track_position_at_z(
        x0[survived], y0[survived], theta[survived], phi[survived],
        z_start=Z_TOP_PADDLE, z_target=Z_IMAGING_BOT
    )
    row_bot, col_bot = pos_to_tile(x_bot, y_bot)
    valid = (row_bot >= 0) & (col_bot >= 0)

    for r, c in zip(row_bot[valid], col_bot[valid]):
        hit_map[r, c] += 1
        total_events  += 1

    return hit_map, total_events


def simulate_B(x0, y0, theta, phi, survived, rng):
    """
    Config B: top grid + mid grid + bottom grid.
    Top and bottom must fire in the same tile.
    Mid grid must be in active area but tile match not required —
    realistic for 80 cm stack where strict triple coincidence would
    accept only muons within ~3.6° of vertical.
    """
    hit_map      = np.zeros((GRID_ROWS, GRID_COLS), dtype=int)
    total_events = 0

    sv = survived

    x_top, y_top = track_position_at_z(
        x0[sv], y0[sv], theta[sv], phi[sv],
        z_start=Z_TOP_PADDLE, z_target=Z_TOP_PADDLE
    )
    x_mid, y_mid = track_position_at_z(
        x0[sv], y0[sv], theta[sv], phi[sv],
        z_start=Z_TOP_PADDLE, z_target=Z_MID_GRID
    )
    x_bot, y_bot = track_position_at_z(
        x0[sv], y0[sv], theta[sv], phi[sv],
        z_start=Z_TOP_PADDLE, z_target=Z_IMAGING_BOT
    )

    row_top, col_top = pos_to_tile(x_top, y_top)
    row_mid, col_mid = pos_to_tile(x_mid, y_mid)
    row_bot, col_bot = pos_to_tile(x_bot, y_bot)

    in_bounds = (
        (row_top >= 0) & (col_top >= 0) &
        (row_mid >= 0) & (col_mid >= 0) &
        (row_bot >= 0) & (col_bot >= 0)
    )
    same_tile = (row_top == row_bot) & (col_top == col_bot)
    valid     = in_bounds & same_tile

    for r, c in zip(row_bot[valid], col_bot[valid]):
        hit_map[r, c] += 1
        total_events  += 1

    return hit_map, total_events


def simulate_C(x0, y0, theta, phi, survived, rng):
    """
    Config C: 1 imaging grid on top + 1 imaging grid on bottom.
    Same-tile coincidence on both planes required.
    Simplest 2-plane tracker — no middle plane. Higher acceptance than B
    (no mid-plane veto) but better background rejection than A (angular cut
    from same-tile requirement). The top grid position here is Z_TOP_PADDLE.
    """
    hit_map      = np.zeros((GRID_ROWS, GRID_COLS), dtype=int)
    total_events = 0

    sv = survived

    x_top, y_top = track_position_at_z(
        x0[sv], y0[sv], theta[sv], phi[sv],
        z_start=Z_TOP_PADDLE, z_target=Z_TOP_PADDLE
    )
    x_bot, y_bot = track_position_at_z(
        x0[sv], y0[sv], theta[sv], phi[sv],
        z_start=Z_TOP_PADDLE, z_target=Z_IMAGING_BOT
    )

    row_top, col_top = pos_to_tile(x_top, y_top)
    row_bot, col_bot = pos_to_tile(x_bot, y_bot)

    in_bounds = (
        (row_top >= 0) & (col_top >= 0) &
        (row_bot >= 0) & (col_bot >= 0)
    )
    same_tile = (row_top == row_bot) & (col_top == col_bot)
    valid     = in_bounds & same_tile

    for r, c in zip(row_bot[valid], col_bot[valid]):
        hit_map[r, c] += 1
        total_events  += 1

    return hit_map, total_events

# ── SNR test ──────────────────────────────────────────────────────────────────

def run_snr_test(config_fn, x0, y0, theta, phi, rng,
                 target_snr=3.0, step=500):
    hit_map         = np.zeros((GRID_ROWS, GRID_COLS), dtype=int)
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

        survived  = check_block(bx, by, bth, bph, rng)
        batch_map, batch_events = config_fn(bx, by, bth, bph, survived, rng)

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

    return hit_map, total_events, muons_to_detect, snr_history, muon_history

# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_results(res_a, res_b, res_c):
    hit_a, ev_a, mu_a, snr_a, mh_a = res_a
    hit_b, ev_b, mu_b, snr_b, mh_b = res_b
    hit_c, ev_c, mu_c, snr_c, mh_c = res_c

    rate_a = hit_a / ev_a if ev_a > 0 else hit_a.astype(float)
    rate_b = hit_b / ev_b if ev_b > 0 else hit_b.astype(float)
    rate_c = hit_c / ev_c if ev_c > 0 else hit_c.astype(float)

    FLUX        = 1.0
    ANG_CAPTURE = 0.35
    area        = SCINT_WIDTH * SCINT_HEIGHT
    real_rate   = FLUX * area * ANG_CAPTURE
    time_a      = mu_a / real_rate
    time_b      = mu_b / real_rate
    time_c      = mu_c / real_rate

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
        'xtick.direction':   'in',
        'ytick.direction':   'in',
        'xtick.top':          True,
        'ytick.right':        True,
        'axes.grid':          False,
        'figure.dpi':         300,
    })

    fig = plt.figure(figsize=(13.0, 9.0), facecolor='white')

    outer = gridspec.GridSpec(3, 1, figure=fig,
                              height_ratios=[2.8, 2.8, 0.8],
                              hspace=0.40,
                              left=0.05, right=0.98,
                              top=0.95,  bottom=0.02)
    # Top row: 4 heatmaps (A, B, C, diff C-A)
    gs_top = gridspec.GridSpecFromSubplotSpec(1, 4, subplot_spec=outer[0], wspace=0.50)
    # Mid row: SNR curves + table
    gs_mid = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[1],
                                              wspace=0.40, width_ratios=[2.0, 1.0])

    def make_norm(rate_map):
        sm = np.zeros((GRID_ROWS, GRID_COLS), dtype=bool)
        for r in range(GRID_ROWS):
            for c in range(GRID_COLS):
                tx = (c + 0.5) * TILE_W
                ty = (r + 0.5) * TILE_H
                if (BLOCK_X <= tx <= BLOCK_X + BLOCK_W and
                        BLOCK_Y <= ty <= BLOCK_Y + BLOCK_H):
                    sm[r, c] = True
        return rate_map[sm].mean() * 0.85, rate_map[~sm].mean() * 1.05

    cmap = plt.cm.RdYlGn
    # Shared colour scale across A, B, C
    vmins, vmaxs = zip(make_norm(rate_a), make_norm(rate_b), make_norm(rate_c))
    vmin = min(vmins)
    vmax = max(vmaxs)

    def draw_block(ax):
        ax.add_patch(Rectangle(
            (BLOCK_X / TILE_W - 0.5, BLOCK_Y / TILE_H - 0.5),
            BLOCK_W / TILE_W, BLOCK_H / TILE_H,
            linewidth=1.2, edgecolor='black',
            facecolor='none', linestyle='--', zorder=5
        ))

    def style_heatmap(ax, panel, subtitle):
        ax.set_xlabel('Tile col', labelpad=2)
        ax.set_ylabel('Tile row', labelpad=2)
        ax.set_xticks(range(GRID_COLS))
        ax.set_yticks(range(GRID_ROWS))
        ax.text(0.03, 0.97, panel, transform=ax.transAxes,
                fontsize=8, fontweight='bold', va='top', ha='left',
                bbox=dict(boxstyle='round,pad=0.12', facecolor='white',
                          alpha=0.7, edgecolor='none'))
        ax.set_title(subtitle, fontsize=7.0, pad=3, loc='left', style='italic')

    ext = [-0.5, GRID_COLS - 0.5, -0.5, GRID_ROWS - 0.5]

    # (a) Config A
    ax_a = fig.add_subplot(gs_top[0, 0])
    im_a = ax_a.imshow(rate_a, cmap=cmap, vmin=vmin, vmax=vmax,
                       origin='lower', aspect='equal', extent=ext)
    draw_block(ax_a)
    style_heatmap(ax_a, '(a)', 'Config A: paddles + grid')
    cb = plt.colorbar(im_a, ax=ax_a, fraction=0.046, pad=0.04)
    cb.set_label('Rel. hit rate', fontsize=6.5); cb.ax.tick_params(labelsize=6)

    # (b) Config B
    ax_b = fig.add_subplot(gs_top[0, 1])
    im_b = ax_b.imshow(rate_b, cmap=cmap, vmin=vmin, vmax=vmax,
                       origin='lower', aspect='equal', extent=ext)
    draw_block(ax_b)
    style_heatmap(ax_b, '(b)', 'Config B: triple grid')
    cb = plt.colorbar(im_b, ax=ax_b, fraction=0.046, pad=0.04)
    cb.set_label('Rel. hit rate', fontsize=6.5); cb.ax.tick_params(labelsize=6)

    # (c) Config C
    ax_c = fig.add_subplot(gs_top[0, 2])
    im_c = ax_c.imshow(rate_c, cmap=cmap, vmin=vmin, vmax=vmax,
                       origin='lower', aspect='equal', extent=ext)
    draw_block(ax_c)
    style_heatmap(ax_c, '(c)', 'Config C: 2-plane grid')
    cb = plt.colorbar(im_c, ax=ax_c, fraction=0.046, pad=0.04)
    cb.set_label('Rel. hit rate', fontsize=6.5); cb.ax.tick_params(labelsize=6)

    # (d) Difference C − A
    ax_d = fig.add_subplot(gs_top[0, 3])
    diff = rate_c - rate_a
    lim  = np.abs(diff).max() or 1
    im_d = ax_d.imshow(diff, cmap='RdBu_r', vmin=-lim, vmax=lim,
                       origin='lower', aspect='equal', extent=ext)
    draw_block(ax_d)
    style_heatmap(ax_d, '(d)', 'Difference (C\u2212A)')
    cb = plt.colorbar(im_d, ax=ax_d, fraction=0.046, pad=0.04)
    cb.set_label('\u0394 hit rate', fontsize=6.5); cb.ax.tick_params(labelsize=6)

    # (e) SNR curves
    ax_snr = fig.add_subplot(gs_mid[0, 0])
    ax_snr.plot(mh_a, snr_a, color='#2166ac', linewidth=1.1,
                label='Config A (paddles + grid)')
    ax_snr.plot(mh_b, snr_b, color='#d6604d', linewidth=1.1,
                label='Config B (triple grid)')
    ax_snr.plot(mh_c, snr_c, color='#1a9641', linewidth=1.1,
                label='Config C (2-plane grid)')
    ax_snr.axhline(3.0, color='black', linewidth=0.65,
                   linestyle='--', label='SNR\u2009=\u20093')
    for mu, col in [(mu_a, '#2166ac'), (mu_b, '#d6604d'), (mu_c, '#1a9641')]:
        if mu < N_MUONS:
            ax_snr.axvline(mu, color=col, linewidth=0.65, linestyle=':')
    ax_snr.set_xlabel('Simulated muon tracks', labelpad=2)
    ax_snr.set_ylabel('Detection SNR', labelpad=2)
    ax_snr.set_title('(e)', fontsize=8, fontweight='bold', loc='left', pad=3)
    ax_snr.legend(frameon=True, framealpha=0.9, edgecolor='#bbbbbb',
                  fontsize=7, loc='upper left')
    ax_snr.set_xlim(left=0); ax_snr.set_ylim(bottom=0)

    # (f) Parameter table
    ax_t = fig.add_subplot(gs_mid[0, 1])
    ax_t.axis('off')
    ax_t.set_title('(f)', fontsize=8, fontweight='bold', loc='left', pad=3)

    tbl_data = [
        ['Active area',         f'{SCINT_WIDTH:.0f}\u00d7{SCINT_HEIGHT:.0f} cm'],
        ['Grid',                f'{GRID_ROWS}\u00d7{GRID_COLS} tiles'],
        ['Tile size',           f'{TILE_W:.1f}\u00d7{TILE_H:.1f} cm'],
        ['Stack height',        f'{Z_TOP_PADDLE:.0f} cm'],
        ['Block height',        f'{BLOCK_Z:.0f} cm above bot'],
        ['Block size',          f'{BLOCK_W:.0f}\u00d7{BLOCK_H:.0f} cm'],
        ['Attenuation',         f'{BLOCK_ATTEN*100:.0f}% (W, 10 cm)'],
        ['\u03b8 cutoff',       f'{THETA_MAX_DEG:.0f}\u00b0'],
        ['A valid events',      f'{ev_a:,}'],
        ['B valid events',      f'{ev_b:,}'],
        ['C valid events',      f'{ev_c:,}'],
        ['A \u2192 SNR\u22653', f'{mu_a:,} \u03bc'],
        ['B \u2192 SNR\u22653', f'{mu_b:,} \u03bc'],
        ['C \u2192 SNR\u22653', f'{mu_c:,} \u03bc'],
        ['A image time',        f'~{time_a:.0f} min'],
        ['B image time',        f'~{time_b:.0f} min'],
        ['C image time',        f'~{time_c:.0f} min'],
    ]

    tbl = ax_t.table(cellText=tbl_data,
                     colLabels=['Parameter', 'Value'],
                     loc='upper center', cellLoc='left',
                     bbox=[0.0, 0.0, 1.0, 1.0])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(6.0)
    for row in range(len(tbl_data) + 1):
        tbl[row, 0].set_width(0.58)
        tbl[row, 1].set_width(0.42)
    for col in range(2):
        tbl[0, col].set_facecolor('#d0d0d0')
        tbl[0, col].set_text_props(fontweight='bold')
        tbl[0, col].set_edgecolor('#999999')
    for row in range(1, len(tbl_data) + 1):
        for col in range(2):
            tbl[row, col].set_facecolor('white' if row % 2 == 1 else '#f4f4f4')
            tbl[row, col].set_edgecolor('#cccccc')

    # Caption
    ax_cap = fig.add_subplot(outer[2])
    ax_cap.axis('off')
    lines = [
        (f'Figure 1.  Simulated muon hit-rate maps for three detector configurations '
         f'({GRID_ROWS}\u00d7{GRID_COLS} grid, {TILE_W:.1f}\u00d7{TILE_H:.1f}\u2009cm tiles, '
         f'{SCINT_WIDTH:.0f}\u00d7{SCINT_HEIGHT:.0f}\u2009cm active area, {Z_TOP_PADDLE:.0f}\u2009cm stack). '
         f'Config C is a simple 2-plane tracker: top and bottom grids, same-tile coincidence only.'),
        (f'Block {BLOCK_W:.0f}\u00d7{BLOCK_H:.0f}\u2009cm tungsten at {BLOCK_ATTEN*100:.0f}% attenuation, '
         f'{BLOCK_Z:.0f}\u2009cm above bottom grid. '
         f'Angular cutoff {THETA_MAX_DEG:.0f}\u00b0 (cos\u00b2\u03b8, ~35% of sea-level flux). '
         f'Image times: flux\u2009=\u20091\u2009cm\u207b\u00b2\u2009min\u207b\u00b9 \u00d7 35% capture.'),
        ('Dashed rectangles: projected block boundary. '
         'Dotted verticals in (e): muon count at SNR\u2009=\u20093. '
         'Difference map (d) shows Config C minus Config A hit rate.'),
    ]
    for i, line in enumerate(lines):
        ax_cap.text(0.5, 0.92 - i * 0.32, line,
                    ha='center', va='top', fontsize=6.8,
                    style='italic', color='#222222',
                    transform=ax_cap.transAxes)

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       'muography_sim_results.png')
    plt.savefig(out, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"Saved: {out}")
    print(f"File size: {os.path.getsize(out):,} bytes")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    rng = np.random.default_rng(42)

    print(f"Generating {N_MUONS:,} muon tracks (theta < {THETA_MAX_DEG}°)...")
    x0, y0, theta, phi = generate_muons(N_MUONS, rng)

    print("Running Config A...")
    res_a = run_snr_test(simulate_A, x0, y0, theta, phi, rng)

    print("Running Config B...")
    res_b = run_snr_test(simulate_B, x0, y0, theta, phi, rng)

    print("Running Config C...")
    res_c = run_snr_test(simulate_C, x0, y0, theta, phi, rng)

    ev_a, mu_a = res_a[1], res_a[2]
    ev_b, mu_b = res_b[1], res_b[2]
    ev_c, mu_c = res_c[1], res_c[2]
    print(f"\nConfig A: {ev_a:,} valid events, needs ~{mu_a:,} muons for SNR≥3")
    print(f"Config B: {ev_b:,} valid events, needs ~{mu_b:,} muons for SNR≥3")
    print(f"Config C: {ev_c:,} valid events, needs ~{mu_c:,} muons for SNR≥3")

    plot_results(res_a, res_b, res_c)


if __name__ == '__main__':
    main()
