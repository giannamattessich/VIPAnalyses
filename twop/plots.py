import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as ticker
from scipy.ndimage import gaussian_filter
from sklearn.preprocessing import MinMaxScaler
import matplotlib.ticker as mticker
from matplotlib.colors import Normalize

def axs_provided(axs=None, **fig_kwargs):
    if axs is None:
        fig, axs = plt.subplots(**fig_kwargs)
        return axs, False
    return axs, True

def plot_dff_traces(s2p_out, cell_range=None, axs=None, **fig_kwargs):
    """
    Plot ΔF/F traces for cells across frames.
    
    Parameters
    ----------
    dff_df : DataFrame
        ΔF/F traces for each cell across frames.
    cell_range : tuple (start, stop)
        Range of cells to plot.
    axs : matplotlib axis, optional
        Axis to plot into. If None, a new figure+axis is created.
    **fig_kwargs : kwargs
        Passed to plt.subplots if axs is None.
    """

    axis, axis_provided = axs_provided(axs, **fig_kwargs)
    if cell_range is None:
        cell_range = [0, s2p_out.num_cells]
    frame_list = list(range(s2p_out.F.shape[1]))
    for i in range(cell_range[0], cell_range[1]):
        axis.plot(frame_list, s2p_out.F[i], label=f"Cell {i}")

    axis.set_xlabel("Frame")
    axis.set_ylabel("ΔF/F")
    axis.legend()

    if not axis_provided: 
        return axis
    
def spikeevent_plot(s2p_out, cell_range=None, axs=None, vertical_spacing=1, **fig_kwargs):
    """
    Raster-style spike events for cells in [cell_range[0], cell_range[1]).
    If axs is None, a new axis is created via axs_provided.
    """
    axis, _ = axs_provided(axs, **fig_kwargs)

    spikes = s2p_out.get_cell_spikes()
    # subset to requested cells
    if cell_range:
        spikes = spikes[cell_range[0]:cell_range[1], :]

    for cell_idx in range(spikes.shape[0]):
        spike_times = np.where(spikes[cell_idx] != 0)[0]
        axis.vlines(
            spike_times,
            ymin=cell_idx * vertical_spacing,
            ymax=(cell_idx + 1) * vertical_spacing,
            color="black",
            linewidth=1)

    # labels/formatting
    axis.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    axis.set_xlabel("Frame #")
    axis.set_ylabel("Cell # (relative in range)")
    return axis


def smoothed_event_plot(s2p_out, cell_range=None, axs=None, smoothing_sigma=0.5, **fig_kwargs):
    """
    Heatmap of (optionally smoothed) spike events for cells in [cell_range[0], cell_range[1]).
    If axs is None, a new axis is created via axs_provided.
    """
    axis, _ = axs_provided(axs, **fig_kwargs)

    # Extract the spikes in the specified range of cells
    spks = s2p_out.spks[s2p_out.cell_indices, :]
    if cell_range is None:
        cell_range = [0, s2p_out.num_cells]
    spikes = spks[cell_range[0]:cell_range[1], :]

    # Apply smoothing (Gaussian filter) to the spike matrix
    smoothed_matrix = gaussian_filter(spikes, sigma=smoothing_sigma)

    # # Normalize data (fit on original, transform original → then plot smoothed as-is, or
    # # normalize smoothed; here we keep your original intent and normalize the raw spikes)
    # scaler = MinMaxScaler()
    # _ = scaler.fit(spikes)
    # spikes_norm = scaler.transform(spikes)  # kept in case you want to plot normalized later

    # Create the heatmap plot
    cax = axis.imshow(
        smoothed_matrix,
        aspect="auto",
        cmap="hot",
        origin="lower",
        interpolation="none",
    )

    # Add colorbar
    plt.colorbar(cax, ax=axis, label="Spike Intensity")

    # Set labels and formatting
    axis.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    axis.set_xlabel("Frame (#)")
    axis.set_ylabel("Cell (#, relative in range)")

    return axis


def spike_raster(spikes, recording_days):
    fig, axs = plt.subplots(1, len(spikes), figsize=(50,4))

    plt.suptitle("Spiking: Front_left")
    #plt.supylabel('Spike amplitude')
    for ax_idx in range(len(spikes)):
        plot_spikes = spikes[ax_idx]
        norm = Normalize(vmin=np.percentile(plot_spikes, 1), vmax=np.percentile(plot_spikes, 99))
        axs[ax_idx].imshow(plot_spikes, aspect='auto', cmap='binary', origin='lower', norm=norm)
        axs[ax_idx].set_title(recording_days[ax_idx])

        # Remove x-axis ticks and labels
        axs[ax_idx].xaxis.set_major_locator(mticker.NullLocator())
        axs[ax_idx].xaxis.set_major_formatter(mticker.NullFormatter())

        # Remove y-axis ticks and labels
        axs[ax_idx].yaxis.set_major_locator(mticker.NullLocator())
        axs[ax_idx].yaxis.set_major_formatter(mticker.NullFormatter())
    plt.show()
 
# def single_spike_raster(spikes):
#     fig, axs = plt.subplots(1, len(spikes), figsize=(50,4))
#     plt.suptitle("Spiking: Front_left")
#     #plt.supylabel('Spike amplitude')
#     for ax_idx in range(len(spikes)):
#     plot_spikes = spikes[ax_idx]
#     norm = Normalize(vmin=np.percentile(plot_spikes, 1), vmax=np.percentile(plot_spikes, 99))
#     axs[ax_idx].imshow(plot_spikes, aspect='auto', cmap='binary', origin='lower', norm=norm)
#     axs[ax_idx].set_title(recording_days[ax_idx])

#     # Remove x-axis ticks and labels
#     axs[ax_idx].xaxis.set_major_locator(mticker.NullLocator())
#     axs[ax_idx].xaxis.set_major_formatter(mticker.NullFormatter())

#     # Remove y-axis ticks and labels
#     axs[ax_idx].yaxis.set_major_locator(mticker.NullLocator())
#     axs[ax_idx].yaxis.set_major_formatter(mticker.NullFormatter())
#     plt.show()