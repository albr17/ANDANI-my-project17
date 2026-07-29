import numpy as np
import elephant
import matplotlib.pyplot as plt

# Shared figure style, so the spike-only and the combined analysis produce
# figures that can be put side by side.
EVENT_COLOR = "crimson"
SPIKE_COLOR = "#1f77b4"
LFP_COLOR = "#d95f02"


def use_analysis_style():
    """Apply the matplotlib defaults shared by the analysis scripts."""
    plt.rcParams.update({"figure.constrained_layout.use": True,
                         "axes.grid": True,
                         "grid.alpha": 0.25,
                         "axes.titlesize": 11})


def mark_event(ax, time, label="ketamine injection"):
    """Draw a vertical marker at `time` (seconds) on a time-domain axis."""
    ax.axvline(float(time), color=EVENT_COLOR, ls="--", lw=1.5, label=label)


def zscore(x, axis=0):
    """Z-score along `axis` (used to draw signals on a common scale)."""
    return (x - np.mean(x, axis=axis)) / np.std(x, axis=axis)


def plot_scree(explained_ratio, title, ax, threshold=0.9):
    """
    Plot per-component and cumulative explained variance on one axis.

    Parameters
    ----------
    explained_ratio : ndarray
        Fraction of variance carried by each component, e.g.
        ``sklearn.decomposition.PCA.explained_variance_ratio_``.

    title : str
        Title of the axes.

    ax : matplotlib.axes.Axes
        Axes to draw on.

    threshold : float
        Cumulative variance target to mark, as a fraction.

    Returns
    -------
    cumulative : ndarray
        Cumulative explained-variance ratio.

    n_needed : int
        Number of components needed to pass `threshold`. Note this is a count,
        not a 0-based index: reaching the threshold at index 8 means 9
        components are needed.
    """
    cumulative = np.cumsum(explained_ratio)
    n_needed = int(np.argmax(cumulative > threshold)) + 1
    components = np.arange(1, cumulative.size + 1)

    ax.bar(components, explained_ratio, color="0.7", label="per component")
    ax.plot(components, cumulative, "o-", ms=4, color=SPIKE_COLOR, label="cumulative")
    ax.axhline(threshold, color=EVENT_COLOR, ls=":", lw=1.2,
               label=f"{threshold:.0%} of variance")
    ax.axvline(n_needed, color=EVENT_COLOR, ls="--", lw=1.2,
               label=f"{n_needed} components needed")
    ax.set_xlabel("Principal component")
    ax.set_ylabel("Explained variance (fraction)")
    ax.set_title(title)
    ax.legend(fontsize=8)
    return cumulative, n_needed


def plot_cwt_spectogram(signal, frequencies=150):
    """
    Plot a Continuous Wavelet Transform (CWT) spectrogram of a signal.

    The function computes the complex wavelet transform using Elephant,
    converts the coefficients to their absolute values, normalizes the
    wavelet power by the mean power of each channel across time, and displays the
    result as a log-scaled spectrogram.

    Parameters
    ----------
    signal : neo.AnalogSignal
        Input signal to analyze. The signal must contain sampling rate
        information (`sampling_rate`) and be compatible with
        `elephant.signal_processing.wavelet_transform`.

    frequencies : array-like of float
        Frequencies (Hz) at which the wavelet transform is computed.

    Notes
    -----
    - The y-axis is displayed on a logarithmic scale.
    - Wavelet amplitudes are normalized by the mean amplitude across
      frequencies for each time point.
    - A vertical reference to ketamine incjetion time.
    - The spectrogram is displayed in decibels:
      ``10 * log10(normalized_amplitude)``.
    """
    fs = float(signal.sampling_rate)
    times = np.arange(len(signal)) / fs
    wavelet = elephant.signal_processing.wavelet_transform(signal,frequency=frequencies,sampling_frequency=fs)
    
    wavelet_abs= np.abs(wavelet)
    wavelet_norm = wavelet_abs/np.mean(wavelet_abs,axis=0)
    
    # %%
    fig,axes= plt.subplots(figsize=(20,10))
    
    axes.pcolormesh(times,frequencies,10*np.log10(wavelet_norm[:,0,:].T),cmap="nipy_spectral")
    axes.axvline(x=372,color="crimson",lw=2)
    axes.set_yscale("log")
    plt.ylabel("frequency HZ")
    plt.show()

def process_lfp_pca(signals, frequency):
    """
    Compute PCA of the wavelet-transformed LFP across electrodes.

    The function computes the complex wavelet transform using Elephant 
    wavelet_transform method for a single frequency. Then takes the
    real part, and performs PCA across electrodes by eigendecomposing
    the electrode-electrode covariance matrix.

    Parameters
    ----------
    signals : neo.AnalogSignal
        Input multi-electrode signal to analyze. The signal must contain
        sampling rate information (`sampling_rate`) and be compatible with
        `elephant.signal_processing.wavelet_transform`.

    frequency : float
        Frequency (Hz) at which the wavelet transform is computed.

    Returns
    -------
    eigvals : ndarray, shape (n_electrodes,)
        Eigenvalues of the electrode covariance matrix, sorted in
        descending order and clipped to be non-negative.

    eigvecs : ndarray, shape (n_electrodes, n_electrodes)
        Corresponding eigenvectors (principal components), columns ordered
        to match `eigvals`.

    wavelet_real_signals : ndarray, shape (n_electrodes, n_time)
        Real part of the wavelet-transformed signal, mean-centered input
        to the PCA.
    """
    fs = signals.sampling_rate
    signals_wavelets = elephant.signal_processing.wavelet_transform(signals, frequency=[frequency], sampling_frequency=fs)

    # squeeze() is (n_time, n_electrodes); transpose so electrodes are the rows
    # to reduce the electrode direction -> C is (n_electrodes, n_electrodes)
    wavelet_real_signals = signals_wavelets.squeeze().real.T     # (n_electrodes, n_time)
    print(wavelet_real_signals.shape)

    mean_over_channels = wavelet_real_signals.mean(axis=1, keepdims=True)
    X = wavelet_real_signals - mean_over_channels  # (n_electrodes, n_time)

    C = (X @ X.T) / (X.shape[1] - 1)          # (n_electrodes, n_electrodes)

    eigvals, eigvecs = np.linalg.eigh(C)
    order = np.argsort(eigvals)[::-1]
    eigvals = np.clip(eigvals[order], 0, None)
    eigvecs = eigvecs[:, order]

    return eigvals, eigvecs, wavelet_real_signals, mean_over_channels

def perform_pca(X):


    mean_over_channel = X.mean(axis=1, keepdims=True)
    X_C = X - mean_over_channel  # (n_electrodes, n_time)

    C = (X_C @ X_C.T) / (X_C.shape[1] - 1)          # (n_electrodes, n_electrodes)

    eigvals, eigvecs = np.linalg.eigh(C)
    order = np.argsort(eigvals)[::-1]
    eigvals = np.clip(eigvals[order], 0, None)
    eigvecs = eigvecs[:, order]

    return eigvals, eigvecs, 

