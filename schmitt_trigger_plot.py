"""
schmitt_trigger_plot.py
Academic-style Schmitt trigger vs single-threshold comparator figure.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import rcParams

# ── Style ─────────────────────────────────────────────────────────────────────
rcParams.update({
    'font.family':       'serif',
    'font.serif':        ['DejaVu Serif', 'Times New Roman', 'Times'],
    'font.size':         8,
    'axes.labelsize':    8,
    'axes.titlesize':    9,
    'xtick.labelsize':   7,
    'ytick.labelsize':   7,
    'axes.linewidth':    0.6,
    'xtick.major.width': 0.5,
    'ytick.major.width': 0.5,
    'xtick.direction':   'in',
    'ytick.direction':   'in',
    'figure.dpi':        300,
})

# ── Signals ───────────────────────────────────────────────────────────────────
t = np.linspace(0, 10, 10000)

# Single threshold signal — smooth wave crossing threshold many times
signal_single = 0.3*np.sin(2.1*t) + 0.15*np.sin(4.3*t) + 0.1*np.sin(7*t) + 0.5
Vth = 0.55
digital_single = (signal_single > Vth).astype(float)
switch_idx_single = np.where(np.diff(digital_single) != 0)[0] + 1
switch_times_single = t[switch_idx_single]

# Dual threshold signal — noisy baseline, one dip below V-, one rise above V+
signal_dual = 0.08*np.sin(8*t) + 0.06*np.sin(13*t) + 0.55
signal_dual -= 0.45 * np.exp(-((t - 5.0)**2) / 0.4)
signal_dual += 0.45 * np.exp(-((t - 8.5)**2) / 0.2)
Vpos = 0.62
Vneg = 0.35

digital_schmitt = np.zeros(len(t))
state = 1
for i in range(len(t)):
    if state == 1 and signal_dual[i] < Vneg:
        state = 0
    elif state == 0 and signal_dual[i] > Vpos:
        state = 1
    digital_schmitt[i] = state

switch_idx_dual = np.where(np.diff(digital_schmitt) != 0)[0] + 1
switch_times_dual = t[switch_idx_dual]

# ── Colors ────────────────────────────────────────────────────────────────────
col_sig    = '#A60C0C'
col_dig    = '#1A3388'
col_thresh = '#333333'
col_vpos   = '#146614'
col_vneg   = '#B85400'
col_band   = '#E8F5E8'
col_sw     = '#888888'

# ── Figure layout ─────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(5.5, 7.0), facecolor='white')

outer = gridspec.GridSpec(
    2, 1, figure=fig,
    height_ratios=[1, 1],
    hspace=0.18,
    left=0.13, right=0.82,
    top=0.96, bottom=0.07
)

gs_top = gridspec.GridSpecFromSubplotSpec(
    2, 1, subplot_spec=outer[0],
    hspace=0.08, height_ratios=[1.3, 1]
)
gs_bot = gridspec.GridSpecFromSubplotSpec(
    2, 1, subplot_spec=outer[1],
    hspace=0.08, height_ratios=[1.3, 1]
)

ax1 = fig.add_subplot(gs_top[0])   # analog single
ax2 = fig.add_subplot(gs_top[1])   # digital single
ax3 = fig.add_subplot(gs_bot[0])   # analog dual
ax4 = fig.add_subplot(gs_bot[1])   # digital dual

def draw_switch_lines(ax, switch_times, ymin, ymax, color, lw=0.5):
    for tx in switch_times:
        ax.axvline(tx, color=color, linewidth=lw, linestyle='--',
                   zorder=1, alpha=0.85)

# ── (a) Single threshold — Analog Input ──────────────────────────────────────
draw_switch_lines(ax1, switch_times_single, 0.05, 1.00, col_sw)
ax1.plot(t, signal_single, color=col_sig, linewidth=1.1, zorder=3)
ax1.axhline(Vth, color=col_thresh, linewidth=0.8, zorder=2)
ax1.text(10.15, Vth, r'$V_\mathrm{th}$', fontsize=9, color=col_thresh,
         va='center', clip_on=False)
ax1.text(0.02, 0.93, '(a)', transform=ax1.transAxes,
         fontsize=8, fontweight='bold', va='top')
ax1.set_ylabel('Analog Input (V)')
ax1.set_title('Single-threshold comparator', fontweight='bold', pad=4)
ax1.set_xlim(0, 10)
ax1.set_ylim(0.05, 1.00)
ax1.set_yticks([])
ax1.set_xticklabels([])
for spine in ax1.spines.values():
    spine.set_linewidth(0.6)

# ── (b) Single threshold — Digital Output ─────────────────────────────────────
draw_switch_lines(ax2, switch_times_single, -0.32, 1.42, col_sw)
ax2.step(t, digital_single, where='post', color=col_dig, linewidth=1.1, zorder=3)
ax2.text(0.02, 0.90, '(b)', transform=ax2.transAxes,
         fontsize=8, fontweight='bold', va='top')
ax2.set_ylabel('Digital Output')
ax2.text(10.15, 1.0, 'Hi', fontsize=8, color='#555555', va='center',
         clip_on=False)
ax2.text(10.15, 0.0, 'Lo', fontsize=8, color='#555555', va='center',
         clip_on=False)
ax2.text(2.8, -0.20, 'chatter', fontsize=7.5, color='#777777',
         ha='center', style='italic')
ax2.set_xlim(0, 10)
ax2.set_ylim(-0.32, 1.42)
ax2.set_yticks([])
ax2.set_xticklabels([])
for spine in ax2.spines.values():
    spine.set_linewidth(0.6)

# Divider between sections
fig.add_artist(plt.Line2D(
    [0.04, 0.96], [0.505, 0.505],
    transform=fig.transFigure,
    color='#bbbbbb', linewidth=0.7, linestyle='-'
))

# ── (c) Schmitt trigger — Analog Input ────────────────────────────────────────
ax3.fill_between(t, Vneg, Vpos, color=col_band, zorder=1)
draw_switch_lines(ax3, switch_times_dual, 0.05, 1.08, col_sw)
ax3.plot(t, signal_dual, color=col_sig, linewidth=1.1, zorder=3)
ax3.axhline(Vpos, color=col_vpos, linewidth=0.8, zorder=2)
ax3.axhline(Vneg, color=col_vneg, linewidth=0.8, zorder=2)
ax3.text(10.15, Vpos, r'$V_+$', fontsize=9, color=col_vpos,
         va='center', clip_on=False)
ax3.text(10.15, Vneg, r'$V_-$', fontsize=9, color=col_vneg,
         va='center', clip_on=False)
ax3.text(0.55, (Vpos+Vneg)/2, 'hysteresis', fontsize=7,
         color='#1a6e1a', va='center', style='italic')
ax3.text(0.02, 0.93, '(c)', transform=ax3.transAxes,
         fontsize=8, fontweight='bold', va='top')
ax3.set_ylabel('Analog Input (V)')
ax3.set_title('Schmitt trigger (dual threshold)', fontweight='bold', pad=4)
ax3.set_xlim(0, 10)
ax3.set_ylim(0.05, 1.08)
ax3.set_yticks([])
ax3.set_xticklabels([])
for spine in ax3.spines.values():
    spine.set_linewidth(0.6)

# ── (d) Schmitt trigger — Digital Output ──────────────────────────────────────
draw_switch_lines(ax4, switch_times_dual, -0.32, 1.42, col_sw)
ax4.step(t, digital_schmitt, where='post', color=col_dig, linewidth=1.1, zorder=3)
ax4.text(0.02, 0.90, '(d)', transform=ax4.transAxes,
         fontsize=8, fontweight='bold', va='top')
ax4.set_ylabel('Digital Output')
ax4.set_xlabel('Time (arb. units)')
ax4.text(10.15, 1.0, 'Hi', fontsize=8, color='#555555', va='center',
         clip_on=False)
ax4.text(10.15, 0.0, 'Lo', fontsize=8, color='#555555', va='center',
         clip_on=False)
ax4.text(5.5, -0.20, 'clean pulse', fontsize=7.5, color='#777777',
         ha='center', style='italic')
ax4.set_xlim(0, 10)
ax4.set_ylim(-0.32, 1.42)
ax4.set_yticks([])
for spine in ax4.spines.values():
    spine.set_linewidth(0.6)

plt.savefig('/mnt/user-data/outputs/schmitt_trigger_comparison.png',
            dpi=300, bbox_inches='tight', facecolor='white')
print("Saved: schmitt_trigger_comparison.png")
plt.show()
