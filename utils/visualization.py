import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

NAVY, TEAL, RED = "#0B1F3A", "#00C8A6", "#E5484D"


def gait_plot(window, fs, fog=False):
    """window: (n, 6) resampled pose channels. Plots ankle heights and knee angles."""
    fig, ax = plt.subplots(2, 1, figsize=(6, 3.4), sharex=True, facecolor=NAVY)
    t = np.arange(len(window)) / fs
    for a in ax:
        a.set_facecolor(NAVY); a.tick_params(colors="white", labelsize=7)
        for s in a.spines.values(): s.set_color("#335")
    ax[0].plot(t, window[:, 0], color=TEAL, label="L ankle"); ax[0].plot(t, window[:, 1], color="white", label="R ankle")
    ax[0].set_ylabel("ankle y (torso units)", color="white", fontsize=7)
    ax[1].plot(t, window[:, 2], color=TEAL); ax[1].plot(t, window[:, 3], color="white")
    ax[1].set_ylabel("knee angle (deg)", color="white", fontsize=7); ax[1].set_xlabel("s", color="white", fontsize=7)
    if fog:
        for a in ax: a.set_facecolor("#3a1420")
    ax[0].legend(fontsize=6, facecolor=NAVY, labelcolor="white", loc="upper right")
    fig.tight_layout()
    return fig
