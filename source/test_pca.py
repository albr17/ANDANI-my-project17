
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import neo
from scipy.signal import spectrogram
import mne
import spectral 
from viziphant.rasterplot import rasterplot
from viziphant.statistics import plot_instantaneous_rates_colormesh
from elephant import kernels
from elephant.statistics import instantaneous_rate
import elephant
import quantities as pq
from neo.io import NixIO

with NixIO("rat_retreat_meta_1kHz.nix", mode="ro") as io:
    block_check = io.read_block()

seg_check = block_check.segments[0]

print(block_check.description)
print(seg_check.description)

for st in seg_check.spiketrains:
    print(st.name, st.annotations)

seg = block_check.segments[0]
print(seg)



sts = [st for st in seg.spiketrains]


for i in range(len(sts)):
    sts[i].t_stop=705.0*pq.s

#%%
fig, ax = plt.subplots(figsize=(30, 5))
rasterplot(sts, axes=ax,s=20,marker ='|', color='k',lw=0.5,alpha=0.5)
ax.axvline(x=372, color='crimson', ls='--', lw=1.5)
plt.show()



#%% plot instantanoues
kernel = kernels.GaussianKernel(sigma=100*pq.ms)

inst_rate = instantaneous_rate(sts,sampling_period=0.001*pq.s,kernel=kernel)

spike_ids= [st.annotations["spike_id"] for st in sts]
print(len(spike_ids))
fig,axes = plt.subplots(figsize=(20,10))
plot_instantaneous_rates_colormesh(inst_rate,axes=axes)
axes.set_yticks(range(13))
axes.set_yticklabels(spike_ids)
axes.axvline(x=372,color="crimson",lw=2)
plt.show()


# %% creating the wavelet
electrode=10
signal = seg.analogsignals[0][:,electrode]
fs = float(seg.analogsignals[0].sampling_rate)
times = np.arange(len(signal)) / fs
frequencies=np.logspace(0,8,num=100,base=2)*pq.Hz
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
# %%

sig = seg.analogsignals[0]

print(sig.shape)

print(sig.sampling_rate)

signal = np.asarray(sig[:, 18]).squeeze() #example channel
fs = float(sig.sampling_rate)

times = np.arange(len(signal)) / fs

f, t, Sxx = spectrogram(signal, fs=fs, nperseg=16384)

power = 10 * np.log10(Sxx+1e-20)
plt.figure(figsize=(12, 5))
plt.pcolormesh(
    t,
    f,
    power,
    shading="auto",
    cmap="Greys",
    vmin=-115,
    vmax=-95
)
plt.ylim(0, 200)
plt.xlabel("Time")
plt.ylabel("Frequency")
plt.title("Spectrogram - Channel 19")
plt.show()
# %% 
sigma = 0.100
dt = 0.005
t0 = 0
t1 = 700

bin_edges = np.arange(t0, t1 + dt, dt)
t_eval = bin_edges[:-1] + dt / 2

spike_matrix = np.zeros((len(seg.spiketrains), len(t_eval)))
for i, st in enumerate(seg.spiketrains):
    spike_times = np.asarray(st.rescale('s').magnitude).ravel()
    spike_matrix[i], _ = np.histogram(spike_times, bins=bin_edges)

def gaussian_kernel(sigma_bins, n_sigma=4):
    half = int(np.ceil(n_sigma * sigma_bins))
    x = np.arange(-half, half + 1)
    kernel = np.exp(-x**2 / (2 * sigma_bins**2))
    return kernel / kernel.sum()

kernel = gaussian_kernel(sigma / dt)
smoothed_spike_matrix = np.zeros_like(spike_matrix)
for i, count_train in enumerate(spike_matrix):
    smoothed_spike_matrix[i] = np.convolve(count_train, kernel, mode='same') / dt

plt.figure(figsize=(20, 5))
for i, smoothed_spike_train in enumerate(smoothed_spike_matrix):
    plt.plot(t_eval, smoothed_spike_train+ i * 35, lw=0.8)
plt.yticks([])
plt.xlabel("Time (s)")
plt.ylabel("Smoothed Spike Rate (Hz)")
plt.axvline(x=372, color='crimson', ls='--', lw=1.5)
plt.title("Smoothed Spike Trains")
plt.show()

time_mask = (350,500)
plt.figure(figsize=(20, 5))
for i, smoothed_spike_train in enumerate(smoothed_spike_matrix):
    plt.plot(t_eval, smoothed_spike_train+ i * 35, lw=0.8)
plt.xlim(time_mask)
plt.yticks([])
plt.xlabel("Time (s)")
plt.ylabel("Smoothed Spike Rate (Hz)")
plt.title("Smoothed Spike Trains")
plt.show()


# %%
# PCA on the smoothed matrix: neurons are variables, time bins are samples
X = smoothed_spike_matrix - smoothed_spike_matrix.mean(axis=1, keepdims=True)
C = (X @ X.T) / (X.shape[1] - 1)          # (n_neurons, n_neurons)

eigvals, eigvecs = np.linalg.eigh(C)
order = np.argsort(eigvals)[::-1]
eigvals = np.clip(eigvals[order], 0, None)
eigvecs = eigvecs[:, order]

PR = eigvals.sum()**2 / np.sum(eigvals**2)
print(f"{eigvals.size} eigenvalues, PR = {PR:.2f}")

proj = X.T @ eigvecs[:, :3]               # (n_timebins, 3)

fig, ax = plt.subplots(2, 1, figsize=(12, 4.5))
ax[0].plot(np.arange(1, eigvals.size + 1), eigvals / eigvals.sum(), 'o-', ms=3)
ax[0].axvline(PR, color='crimson', ls='--', label=f"PR = {PR:.2f}")
ax[0].set_yscale('log')
ax[0].set_xlabel("Component"); ax[0].set_ylabel("Variance fraction"); ax[0].legend()

for k in range(3):
    ax[1].plot(t_eval, proj[:, k], lw=0.8, label=f"PC{k+1}")
ax[1].set_xlabel("Time"); ax[1].set_ylabel("Projection"); ax[1].legend()
plt.tight_layout(); plt.show()


# %%

