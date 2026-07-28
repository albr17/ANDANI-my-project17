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
