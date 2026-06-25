"""
muography_design.py
Three-layer muon detector design simulator.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
import warnings
warnings.filterwarnings('ignore')

# ── Key constants ──────────────────────────────────────────────────────────────
TRIG_BAR_W  = 10    # cm, trigger bar width
TRIG_BAR_H  = 10    # cm, trigger bar height
TRIG_COLS   = 4
TRIG_ROWS   = 4

IMG_BAR_W   = 5     # cm, imaging bar width
IMG_BAR_H   = 5     # cm, imaging bar height
IMG_COLS    = 8
IMG_ROWS    = 8

Z_TOP  = 60         # cm
Z_MID  = 30         # cm
Z_BOT  = 0          # cm

BLOCK_X     = 10    # cm, block lower-left x
BLOCK_Y     = 10    # cm, block lower-left y
BLOCK_W     = 20    # cm
BLOCK_H     = 20    # cm
BLOCK_Z     = 15    # cm, block top z (sits between mid and bot)
BLOCK_THICK = 10    # cm

BLOCK_MATERIAL = 'iron'
MUON_BCP_MEV   = 1000.0   # MeV/c baseline muon momentum
N_MUONS        = 500_000

# ── Materials (X0 in cm, lambda_I in cm) from PDG ─────────────────────────────
MATERIALS = {
    'vacuum':   {'X0': 1e12,  'lambda_I': 1e12},
    'water':    {'X0': 36.08, 'lambda_I': 83.3},
    'concrete': {'X0': 26.7,  'lambda_I': 99.9},
    'iron':     {'X0': 1.757, 'lambda_I': 16.77},
    'lead':     {'X0': 0.5612,'lambda_I': 17.59},
    'tungsten': {'X0': 0.3504,'lambda_I': 9.946},
}

RNG = np.random.default_rng(42)

# ── Physics helpers ────────────────────────────────────────────────────────────

def highland_theta0(x_cm, X0_cm, bcp_MeV=MUON_BCP_MEV):
    """Highland MCS RMS projected angle (radians)."""
    if x_cm <= 0 or X0_cm >= 1e9:
        return 0.0
    ratio = x_cm / X0_cm
    return (13.6 / bcp_MeV) * np.sqrt(ratio) * (1.0 + 0.038 * np.log(ratio))


def absorption_prob(slant_cm, lambda_I_cm):
    """Fraction of muons absorbed."""
    if lambda_I_cm >= 1e9:
        return 0.0
    return 1.0 - np.exp(-slant_cm / lambda_I_cm)


def bar_center(hit_coord, bar_width):
    """Snap coordinate to nearest bar centre."""
    return (np.floor(hit_coord / bar_width) + 0.5) * bar_width


def angular_resolution(bar_w, z_top, z_mid):
    """1-sigma angular resolution from bar discretisation."""
    sigma_pos = (bar_w / np.sqrt(12.0)) * np.sqrt(2.0)
    return sigma_pos / (z_top - z_mid)


# ── Layer / detector geometry helpers ─────────────────────────────────────────

def trig_layer_extent():
    return TRIG_COLS * TRIG_BAR_W, TRIG_ROWS * TRIG_BAR_H   # cm


def img_layer_extent():
    return IMG_COLS * IMG_BAR_W, IMG_ROWS * IMG_BAR_H


def make_hit_map(x_hits, y_hits, cols, rows, bar_w, bar_h):
    """Return 2-D hit count array."""
    hmap = np.zeros((rows, cols), dtype=int)
    for x, y in zip(x_hits, y_hits):
        col = int(x / bar_w)
        row = int(y / bar_h)
        if 0 <= col < cols and 0 <= row < rows:
            hmap[row, col] += 1
    return hmap


# ── Core simulation ────────────────────────────────────────────────────────────

def simulate(n_muons=N_MUONS,
             block_material=BLOCK_MATERIAL,
             trig_bar_w=TRIG_BAR_W,
             img_bar_w=IMG_BAR_W,
             z_top=Z_TOP, z_mid=Z_MID, z_bot=Z_BOT,
             bcp=MUON_BCP_MEV,
             block_x=BLOCK_X, block_y=BLOCK_Y,
             block_w=BLOCK_W, block_h=BLOCK_H,
             block_z=BLOCK_Z, block_thick=BLOCK_THICK):
    """
    Returns dict with hit arrays and deflection arrays.
    """
    mat    = MATERIALS[block_material]
    X0     = mat['X0']
    lam_I  = mat['lambda_I']

    trig_lx = TRIG_COLS * trig_bar_w
    trig_ly = TRIG_ROWS * TRIG_BAR_H
    img_lx  = IMG_COLS  * img_bar_w
    img_ly  = IMG_ROWS  * IMG_BAR_H

    # Use the larger of the two for isotropic generation
    det_lx = max(trig_lx, img_lx)
    det_ly = max(trig_ly, img_ly)

    # Generate muons at top layer, uniform over detector face
    x0 = RNG.uniform(0, det_lx, n_muons)
    y0 = RNG.uniform(0, det_ly, n_muons)

    # Small random angular spread (typical cosmic distribution smeared)
    max_angle = 0.3   # radians
    ax = RNG.uniform(-max_angle, max_angle, n_muons)
    ay = RNG.uniform(-max_angle, max_angle, n_muons)

    dz_top_mid = z_top - z_mid
    dz_mid_bot = z_mid - z_bot

    # Hits at middle layer (no block between top and mid)
    x_mid_true = x0 + ax * dz_top_mid
    y_mid_true = y0 + ay * dz_top_mid

    # Snap top and mid to bar centres for track reconstruction
    x_top_meas = bar_center(x0,        trig_bar_w)
    y_top_meas = bar_center(y0,        TRIG_BAR_H)
    x_mid_meas = bar_center(x_mid_true, trig_bar_w)
    y_mid_meas = bar_center(y_mid_true, TRIG_BAR_H)

    # Projected angles from measured bar centres
    ax_reco = (x_mid_meas - x_top_meas) / dz_top_mid
    ay_reco = (y_mid_meas - y_top_meas) / dz_top_mid

    # True bottom hit (continuing straight from mid)
    x_bot_true = x_mid_true + ax * dz_mid_bot
    y_bot_true = y_mid_true + ay * dz_mid_bot

    # Determine which muons pass through the block
    # Block sits between z_mid and z_bot at (block_x, block_y) to (block_x+block_w, block_y+block_h)
    # Interpolate muon position at block mid-z
    z_block_mid = block_z - block_thick / 2.0
    dz_mid_blockm = z_mid - z_block_mid
    x_at_block = x_mid_true + ax * dz_mid_blockm
    y_at_block = y_mid_true + ay * dz_mid_blockm

    in_block = (
        (x_at_block >= block_x) & (x_at_block <= block_x + block_w) &
        (y_at_block >= block_y) & (y_at_block <= block_y + block_h)
    )

    # MCS deflection for muons through block
    theta0 = highland_theta0(block_thick, X0, bcp)
    # Two independent projected plane deflections
    dtheta_x = RNG.normal(0, theta0, n_muons)
    dtheta_y = RNG.normal(0, theta0, n_muons)

    # Apply deflection only to in-block muons
    ax_after = np.where(in_block, ax + dtheta_x, ax)
    ay_after = np.where(in_block, ay + dtheta_y, ay)

    # Actual bottom hit after possible deflection
    dz_block_bot = block_z - block_thick / 2.0   # distance from block mid to bot
    x_bot_actual = x_mid_true + ax * dz_mid_bot + np.where(in_block, dtheta_x * dz_block_bot, 0.0)
    y_bot_actual = y_mid_true + ay * dz_mid_bot + np.where(in_block, dtheta_y * dz_block_bot, 0.0)

    # Absorption
    abs_prob = absorption_prob(block_thick, lam_I) if np.any(in_block) else 0.0
    absorbed = in_block & (RNG.uniform(0, 1, n_muons) < abs_prob)
    survived = ~absorbed

    # Projected bottom from track reco
    x_bot_proj = x_mid_meas + ax_reco * dz_mid_bot
    y_bot_proj = y_mid_meas + ay_reco * dz_mid_bot

    # Deflection = actual - projected
    defl_x = x_bot_actual - x_bot_proj
    defl_y = y_bot_actual - y_bot_proj

    # Only keep survived muons that hit all three layers (within detector bounds)
    in_top  = (x0 >= 0) & (x0 < trig_lx) & (y0 >= 0) & (y0 < trig_ly)
    in_mid  = (x_mid_true >= 0) & (x_mid_true < trig_lx) & (y_mid_true >= 0) & (y_mid_true < trig_ly)
    in_bot  = (x_bot_actual >= 0) & (x_bot_actual < img_lx) & (y_bot_actual >= 0) & (y_bot_actual < img_ly)
    valid   = survived & in_top & in_mid & in_bot

    return {
        'x_top': x0[valid], 'y_top': y0[valid],
        'x_mid': x_mid_true[valid], 'y_mid': y_mid_true[valid],
        'x_bot': x_bot_actual[valid], 'y_bot': y_bot_actual[valid],
        'x_bot_proj': x_bot_proj[valid], 'y_bot_proj': y_bot_proj[valid],
        'defl_x': defl_x[valid], 'defl_y': defl_y[valid],
        'in_block': in_block[valid],
        'n_valid': np.sum(valid),
        'img_bar_w': img_bar_w,
        'trig_bar_w': trig_bar_w,
    }


def compute_snr(res, img_bar_w=None):
    """
    Compute shadow SNR: mean deflection magnitude in block region vs
    std of deflection magnitude outside block.
    """
    if img_bar_w is None:
        img_bar_w = res['img_bar_w']
    dm = np.sqrt(res['defl_x']**2 + res['defl_y']**2)
    in_b = res['in_block']
    signal_hits = dm[in_b]
    noise_hits  = dm[~in_b]
    if len(signal_hits) == 0 or len(noise_hits) == 0:
        return 0.0
    signal = np.mean(signal_hits)
    noise  = np.std(noise_hits) + 1e-9
    return signal / noise


# ── Figure 1: single-config results ───────────────────────────────────────────

def figure1(res):
    trig_lx = TRIG_COLS * TRIG_BAR_W
    trig_ly = TRIG_ROWS * TRIG_BAR_H
    img_lx  = IMG_COLS  * IMG_BAR_W
    img_ly  = IMG_ROWS  * IMG_BAR_H

    fig = plt.figure(figsize=(18, 12))
    fig.suptitle(f'Three-Layer Muon Detector — Single Config ({BLOCK_MATERIAL}, '
                 f'bcp={MUON_BCP_MEV} MeV/c)', fontsize=13)

    # Hit maps (3 layers)
    ax1 = fig.add_subplot(2, 4, 1)
    hm_top = make_hit_map(res['x_top'], res['y_top'],
                          TRIG_COLS, TRIG_ROWS, TRIG_BAR_W, TRIG_BAR_H)
    im = ax1.imshow(hm_top, origin='lower', aspect='equal',
                    extent=[0, trig_lx, 0, trig_ly])
    ax1.set_title('Top layer hits'); ax1.set_xlabel('x (cm)'); ax1.set_ylabel('y (cm)')
    plt.colorbar(im, ax=ax1, label='counts')

    ax2 = fig.add_subplot(2, 4, 2)
    hm_mid = make_hit_map(res['x_mid'], res['y_mid'],
                          TRIG_COLS, TRIG_ROWS, TRIG_BAR_W, TRIG_BAR_H)
    im2 = ax2.imshow(hm_mid, origin='lower', aspect='equal',
                     extent=[0, trig_lx, 0, trig_ly])
    ax2.set_title('Mid layer hits'); ax2.set_xlabel('x (cm)')
    plt.colorbar(im2, ax=ax2, label='counts')

    ax3 = fig.add_subplot(2, 4, 3)
    hm_bot = make_hit_map(res['x_bot'], res['y_bot'],
                          IMG_COLS, IMG_ROWS, IMG_BAR_W, IMG_BAR_H)
    im3 = ax3.imshow(hm_bot, origin='lower', aspect='equal',
                     extent=[0, img_lx, 0, img_ly])
    ax3.set_title('Bot layer hits (imaging)'); ax3.set_xlabel('x (cm)')
    plt.colorbar(im3, ax=ax3, label='counts')
    # Overlay block outline
    rect = mpatches.Rectangle((BLOCK_X, BLOCK_Y), BLOCK_W, BLOCK_H,
                               linewidth=2, edgecolor='red', facecolor='none')
    ax3.add_patch(rect)

    # Deflection scatter map
    ax4 = fig.add_subplot(2, 4, 4)
    ax4.scatter(res['x_bot'][~res['in_block']],
                res['y_bot'][~res['in_block']],
                s=0.3, alpha=0.3, c='steelblue', label='no block')
    ax4.scatter(res['x_bot'][res['in_block']],
                res['y_bot'][res['in_block']],
                s=0.5, alpha=0.6, c='red', label='through block')
    rect2 = mpatches.Rectangle((BLOCK_X, BLOCK_Y), BLOCK_W, BLOCK_H,
                                linewidth=2, edgecolor='black', facecolor='none')
    ax4.add_patch(rect2)
    ax4.set_title('Bottom layer positions'); ax4.set_xlabel('x (cm)'); ax4.set_ylabel('y (cm)')
    ax4.legend(markerscale=5, fontsize=7)

    # SNR curves per material (vary N_muons sub-samples)
    ax5 = fig.add_subplot(2, 4, 5)
    n_vals = np.logspace(3, np.log10(N_MUONS), 12).astype(int)
    for mat_name in ['water', 'concrete', 'iron', 'lead', 'tungsten']:
        snrs = []
        for n in n_vals:
            r = simulate(n_muons=n, block_material=mat_name)
            snrs.append(compute_snr(r))
        ax5.semilogx(n_vals, snrs, marker='o', markersize=3, label=mat_name)
    ax5.set_xlabel('N muons'); ax5.set_ylabel('Shadow SNR')
    ax5.set_title('SNR vs N muons (per material)')
    ax5.legend(fontsize=7); ax5.grid(True, alpha=0.3)

    # Scatter angle histogram
    ax6 = fig.add_subplot(2, 4, 6)
    dm_in  = np.sqrt(res['defl_x'][res['in_block']]**2  + res['defl_y'][res['in_block']]**2)
    dm_out = np.sqrt(res['defl_x'][~res['in_block']]**2 + res['defl_y'][~res['in_block']]**2)
    bins = np.linspace(0, np.percentile(np.concatenate([dm_in, dm_out]), 99), 50)
    ax6.hist(dm_out, bins=bins, density=True, alpha=0.5, label='outside block', color='steelblue')
    ax6.hist(dm_in,  bins=bins, density=True, alpha=0.6, label='through block', color='red')
    ax6.set_xlabel('Deflection magnitude (cm)')
    ax6.set_ylabel('Density')
    ax6.set_title('Deflection distribution')
    ax6.legend(fontsize=8); ax6.grid(True, alpha=0.3)

    # 2D deflection map (gridded)
    ax7 = fig.add_subplot(2, 4, 7)
    dm_all = np.sqrt(res['defl_x']**2 + res['defl_y']**2)
    hm_defl, xedge, yedge = np.histogram2d(
        res['x_bot'], res['y_bot'], bins=20,
        range=[[0, img_lx], [0, img_ly]],
        weights=dm_all)
    counts_map, _, _ = np.histogram2d(
        res['x_bot'], res['y_bot'], bins=20,
        range=[[0, img_lx], [0, img_ly]])
    with np.errstate(invalid='ignore'):
        mean_defl = np.where(counts_map > 0, hm_defl / counts_map, 0)
    im7 = ax7.imshow(mean_defl.T, origin='lower', aspect='equal',
                     extent=[0, img_lx, 0, img_ly])
    rect3 = mpatches.Rectangle((BLOCK_X, BLOCK_Y), BLOCK_W, BLOCK_H,
                                linewidth=2, edgecolor='red', facecolor='none')
    ax7.add_patch(rect3)
    ax7.set_title('Mean deflection map (cm)'); ax7.set_xlabel('x (cm)'); ax7.set_ylabel('y (cm)')
    plt.colorbar(im7, ax=ax7, label='cm')

    # Text summary
    ax8 = fig.add_subplot(2, 4, 8)
    ax8.axis('off')
    snr_val = compute_snr(res)
    theta0  = highland_theta0(BLOCK_THICK, MATERIALS[BLOCK_MATERIAL]['X0'])
    ang_res = angular_resolution(TRIG_BAR_W, Z_TOP, Z_MID)
    txt = (f"Material:  {BLOCK_MATERIAL}\n"
           f"N valid:   {res['n_valid']:,}\n"
           f"In block:  {np.sum(res['in_block']):,}\n"
           f"Shadow SNR: {snr_val:.2f}\n"
           f"theta0 MCS: {np.degrees(theta0):.3f} deg\n"
           f"Angular res: {np.degrees(ang_res):.3f} deg\n"
           f"Block thick: {BLOCK_THICK} cm\n"
           f"Block mat:   {BLOCK_MATERIAL}\n"
           f"bcp:         {MUON_BCP_MEV} MeV/c\n")
    ax8.text(0.05, 0.95, txt, transform=ax8.transAxes, fontsize=9,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    fig.tight_layout()
    fname = 'muography_design_fig1_single_config.png'
    fig.savefig(fname, dpi=120)
    plt.close(fig)
    print(f'Saved {fname}')


# ── Figure 2: analytical design sweep ─────────────────────────────────────────

def figure2():
    trig_widths  = np.array([2, 5, 10, 20, 30])        # cm
    layer_spaces = np.array([10, 20, 30, 60, 100, 150]) # cm (z_top - z_mid)

    sweep_mats = ['water', 'iron', 'tungsten']
    fig, axes = plt.subplots(2, len(sweep_mats) + 1, figsize=(18, 10))
    fig.suptitle('Analytical Design Sweep: Deflection SNR vs (Trig Bar Width, Layer Spacing)',
                 fontsize=12)

    snr_grids = {}
    ang_grids = {}

    for mi, mat in enumerate(sweep_mats):
        mat_props = MATERIALS[mat]
        snr_grid = np.zeros((len(layer_spaces), len(trig_widths)))
        ang_grid = np.zeros((len(layer_spaces), len(trig_widths)))

        for li, ls in enumerate(layer_spaces):
            for wi, tw in enumerate(trig_widths):
                z_t = ls + 30.0    # keep z_mid = 30 fixed, vary z_top
                z_m = 30.0
                z_b = 0.0
                # Analytical SNR approximation:
                # signal ~ theta0 * dz_block
                # noise  ~ angular_res * dz_mid_bot
                theta0  = highland_theta0(BLOCK_THICK, mat_props['X0'])
                ang_res = angular_resolution(tw, z_t, z_m)
                dz_mid_bot = z_m - z_b
                signal = theta0 * (BLOCK_Z - BLOCK_THICK / 2.0)
                noise  = ang_res * dz_mid_bot + 1e-6
                snr_grid[li, wi] = signal / noise
                ang_grid[li, wi] = np.degrees(ang_res)

        snr_grids[mat] = snr_grid
        ang_grids[mat] = ang_grid

        ax = axes[0, mi]
        im = ax.imshow(snr_grid, origin='lower', aspect='auto',
                       extent=[trig_widths[0], trig_widths[-1],
                               layer_spaces[0], layer_spaces[-1]])
        ax.set_title(f'{mat}\nAnalytical Deflection SNR')
        ax.set_xlabel('Trig bar width (cm)')
        ax.set_ylabel('Layer spacing (cm)')
        plt.colorbar(im, ax=ax, label='SNR')

        ax2 = axes[1, mi]
        im2 = ax2.imshow(ang_grid, origin='lower', aspect='auto',
                         extent=[trig_widths[0], trig_widths[-1],
                                 layer_spaces[0], layer_spaces[-1]])
        ax2.set_title(f'{mat}\nAngular Resolution (deg)')
        ax2.set_xlabel('Trig bar width (cm)')
        ax2.set_ylabel('Layer spacing (cm)')
        plt.colorbar(im2, ax=ax2, label='deg')

    # Last column: angular resolution vs spacing for fixed bar widths
    ax_ar = axes[0, -1]
    for tw in trig_widths:
        ang_res_arr = [np.degrees(angular_resolution(tw, ls + 30.0, 30.0))
                       for ls in layer_spaces]
        ax_ar.plot(layer_spaces, ang_res_arr, marker='o', markersize=4,
                   label=f'w={tw} cm')
    ax_ar.set_xlabel('Layer spacing (cm)')
    ax_ar.set_ylabel('Angular resolution (deg)')
    ax_ar.set_title('Angular Resolution\nvs Layer Spacing')
    ax_ar.legend(fontsize=7); ax_ar.grid(True, alpha=0.3)

    ax_snr = axes[1, -1]
    for mat in sweep_mats:
        # Slice at middle trig_width index
        wi_mid = len(trig_widths) // 2
        ax_snr.plot(layer_spaces, snr_grids[mat][:, wi_mid], marker='s',
                    markersize=4, label=mat)
    ax_snr.set_xlabel('Layer spacing (cm)')
    ax_snr.set_ylabel('Analytical SNR')
    ax_snr.set_title(f'SNR vs Layer Spacing\n(trig bar w={trig_widths[wi_mid]} cm)')
    ax_snr.legend(fontsize=7); ax_snr.grid(True, alpha=0.3)

    fig.tight_layout()
    fname = 'muography_design_fig2_design_sweep.png'
    fig.savefig(fname, dpi=120)
    plt.close(fig)
    print(f'Saved {fname}')


# ── Figure 3: imaging resolution trade-off ────────────────────────────────────

def figure3():
    img_bar_sizes = [2, 5, 10, 20]   # cm
    snr_per_size  = []

    for ibs in img_bar_sizes:
        r = simulate(n_muons=N_MUONS, img_bar_w=ibs)
        snr_per_size.append(compute_snr(r, img_bar_w=ibs))

    fig, axes = plt.subplots(2, len(img_bar_sizes) + 1, figsize=(18, 8))
    fig.suptitle('Imaging Resolution Trade-off: Shadow SNR vs Bar Size', fontsize=12)

    # SNR vs bar size curve
    ax_snr = axes[0, 0]
    ax_snr.plot(img_bar_sizes, snr_per_size, marker='D', color='darkgreen', linewidth=2)
    ax_snr.set_xlabel('Imaging bar size (cm)')
    ax_snr.set_ylabel('Shadow SNR')
    ax_snr.set_title('Shadow SNR\nvs imaging bar size')
    ax_snr.grid(True, alpha=0.3)

    # Toy images at first 3 resolutions
    toy_sizes = img_bar_sizes[:4]
    for i, ibs in enumerate(toy_sizes):
        r = simulate(n_muons=N_MUONS, img_bar_w=ibs)
        n_cols = int(IMG_COLS * IMG_BAR_W / ibs)
        n_rows = int(IMG_ROWS * IMG_BAR_H / ibs)
        n_cols = max(n_cols, 1); n_rows = max(n_rows, 1)

        # Build deflection image
        img_lx = IMG_COLS * IMG_BAR_W
        img_ly = IMG_ROWS * IMG_BAR_H
        dm = np.sqrt(r['defl_x']**2 + r['defl_y']**2)
        hm_defl, xedge, yedge = np.histogram2d(
            r['x_bot'], r['y_bot'], bins=[n_cols, n_rows],
            range=[[0, img_lx], [0, img_ly]], weights=dm)
        counts_map, _, _ = np.histogram2d(
            r['x_bot'], r['y_bot'], bins=[n_cols, n_rows],
            range=[[0, img_lx], [0, img_ly]])
        with np.errstate(invalid='ignore'):
            mean_defl = np.where(counts_map > 0, hm_defl / counts_map, np.nan)

        ax_top = axes[0, i + 1]
        im = ax_top.imshow(mean_defl.T, origin='lower', aspect='equal',
                           extent=[0, img_lx, 0, img_ly])
        rect = mpatches.Rectangle((BLOCK_X, BLOCK_Y), BLOCK_W, BLOCK_H,
                                   linewidth=2, edgecolor='red', facecolor='none')
        ax_top.add_patch(rect)
        ax_top.set_title(f'Bar size {ibs} cm\nSNR={snr_per_size[i]:.2f}')
        ax_top.set_xlabel('x (cm)'); ax_top.set_ylabel('y (cm)')
        plt.colorbar(im, ax=ax_top, label='mean defl (cm)')

        # Hit map
        hm_hit = make_hit_map(r['x_bot'], r['y_bot'], n_cols, n_rows, ibs, ibs)
        ax_bot = axes[1, i + 1]
        im2 = ax_bot.imshow(hm_hit, origin='lower', aspect='equal',
                             extent=[0, img_lx, 0, img_ly])
        rect2 = mpatches.Rectangle((BLOCK_X, BLOCK_Y), BLOCK_W, BLOCK_H,
                                    linewidth=2, edgecolor='red', facecolor='none')
        ax_bot.add_patch(rect2)
        ax_bot.set_title(f'Hit map {ibs} cm bars')
        ax_bot.set_xlabel('x (cm)'); ax_bot.set_ylabel('y (cm)')
        plt.colorbar(im2, ax=ax_bot, label='counts')

    # Bottom-left: SNR bar chart
    ax_bar = axes[1, 0]
    bars = ax_bar.bar([str(s) for s in img_bar_sizes], snr_per_size, color='steelblue')
    ax_bar.set_xlabel('Bar size (cm)')
    ax_bar.set_ylabel('Shadow SNR')
    ax_bar.set_title('SNR bar chart')
    for b, v in zip(bars, snr_per_size):
        ax_bar.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.02,
                    f'{v:.2f}', ha='center', va='bottom', fontsize=9)
    ax_bar.grid(True, alpha=0.3, axis='y')

    fig.tight_layout()
    fname = 'muography_design_fig3_imaging_resolution.png'
    fig.savefig(fname, dpi=120)
    plt.close(fig)
    print(f'Saved {fname}')


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print('Running single-config simulation...')
    res = simulate()
    print(f'  N valid muons: {res["n_valid"]:,}')
    print(f'  In block:      {np.sum(res["in_block"]):,}')
    snr = compute_snr(res)
    print(f'  Shadow SNR:    {snr:.3f}')
    figure1(res)

    print('Running design sweep (Fig 2)...')
    figure2()

    print('Running imaging resolution trade-off (Fig 3)...')
    figure3()

    print('All done. Three PNG files saved.')
