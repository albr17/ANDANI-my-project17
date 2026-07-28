---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.19.4
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

# LFP


## Loading data

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import neo
from scipy.signal import spectrogram
from neo.io import NixIO
import viziphant
import elephant
import quantities as pq
import utils
%matplotlib inline
```

```python
with NixIO("rat_retreat_meta_1kHz.nix", mode="ro") as io:
    block = io.read_block()

seg = block.segments[0]

#print(block.description)
#print(seg.description)

#for st in seg.spiketrains:
#    print(st.name, st.annotations)
```

```python

```

## Applying bandpass filter

```python
signals = seg.analogsignals[0]
```

```python
fs = signals.sampling_rate
```

We see multiple narrowbands in the CWT, so we want to compare how the spiking activity relates to this different narrowbands. And this is why we are obtaing information about phase and amplitude.

```python
signals_filt = elephant.signal_processing.butter(signals, highpass_frequency=130*pq.Hz, lowpass_frequency=180*pq.Hz, filter_function="sosfiltfilt")
```

```python
plt.plot(signals[400:450, 0], label="original")
plt.plot(signals_filt[400:450, 0], label="filt")
plt.legend()
plt.show()
```

We see in the wavelet naroowband activity, so we proceed to use that and focus on the wavelt transformation.

```python
signals_wavelets = elephant.signal_processing.wavelet_transform(signals, frequency=[150*pq.Hz], sampling_frequency=fs)
```

```python
signals_wavelets.shape
```

```python
fig, axes = plt.subplots(3, 1, figsize=(15, 5))
axes[0].plot(signals[400:500, 0], label="original")
axes[1].plot(signals_filt[400:500, 0], label="filt")
axes[2].plot(signals_wavelets[400:500, 0], label="wv")
axes[2].plot(np.abs(signals_wavelets[400:500, 0]), label="wv")
plt.legend()
plt.show()
```

How wavelet signals looks across channel?

```python
max_val = np.median(np.abs(signals_wavelets))

fig, axes = plt.subplots(figsize=(30, 20))
for i in range(signals_wavelets.shape[1]):
    axes.plot(np.abs(signals_wavelets[400:1000, i])+max_val*i)
plt.show()
```

```python
electrode=10

signal = seg.analogsignals[0][:,electrode]
fs = float(seg.analogsignals[0].sampling_rate)
frequencies=np.logspace(0,8,num=100,base=2)*pq.Hz

def plot_cwt_spectogram(signal,frequencies):
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

plot_cwt_spectogram(signal,frequencies)
```

```python
signals_wavelets.shape
```

```python
signals.shape
```

```python
def process_lfp_pca(signals,frequency):
    fs =signals.sampling_rate
    signals_wavelets = elephant.signal_processing.wavelet_transform(signals, frequency=[frequency], sampling_frequency=fs)
    X = signals_wavelets.squeeze().real
    X = X - X.mean(axis=0, keepdims=True)
    C = (X @ X.T) / (X.shape[1] - 1)          # (n_neurons, n_neurons)

    eigvals, eigvecs = np.linalg.eigh(C)
    order = np.argsort(eigvals)[::-1]
    eigvals = np.clip(eigvals[order], 0, None)
    eigvecs = eigvecs[:, order]

    return eigvals, eigvecs


process_lfp_pca(signals, frequency=150*pq.Hz)
PR = eigvals.sum()**2 / np.sum(eigvals**2)
print(f"{eigvals.size} eigenvalues, PR = {PR:.2f}")

proj = X.T @ eigvecs[:, :3]               # (n_timebins, 3)

    
```

```python
signal.sampling_rate

```

```python

```
