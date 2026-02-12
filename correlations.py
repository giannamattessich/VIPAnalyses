from utils.stats import *

def plot_synchrony(traces, t=None, *, z=True, thr=2.0, smooth=None, title = None, axs= None):
    """
    traces: (n_cells, n_timepoints) e.g. dF/F or deconv spikes (continuous)
    z: robust z-score per cell before computing synchrony
    thr: threshold for "active" (in z units if z=True)
    smooth: int window (samples) for simple moving average (optional)
    """
    X = np.asarray(traces)
    if t is None:
        t = np.arange(X.shape[1])

    if z:
        X = zscore_robust(X, axis=1)

    pop_mean = X.mean(axis=0)
    frac_active = (X > thr).mean(axis=0)

    if smooth and smooth > 1:
        k = np.ones(smooth) / smooth
        pop_mean = np.convolve(pop_mean, k, mode="same")
        frac_active = np.convolve(frac_active, k, mode="same")

    if axs is None:
        plt.figure()
        plt.plot(t, pop_mean, label="Population mean")
        plt.plot(t, frac_active, label=f"Fraction active (>{thr})")
        plt.xlabel("Time")
        plt.ylabel("a.u.")
        plt.legend()
        plt.tight_layout()
        if title is not None:
            plt.title(title)
        plt.show()
        return
    else:
        axs.plot(t, pop_mean, label="Population mean")
        axs.plot(t, frac_active, label=f"Fraction active (>{thr})")
        axs.set_xlabel("Time")
        axs.set_ylabel("a.u.")
        axs.legend()
        if title is not None:
            axs.set_title(title)
        return axs

