import numpy as np
import elephant
import matplotlib.pyplot as plt

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
