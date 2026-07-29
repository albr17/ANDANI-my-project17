"""
LFP spectral peaks -> wavelet scalogram -> coupling with the spike components.

Stands on its own next to `combined_analysis.py`. Nothing about the spikes is
recomputed here: the components come straight out of the cache written by
`spike_data_pca.py` (``../data/spike_pca.npz``).

Pipeline
--------
1. Welch PSD of every electrode, each one decomposed by specparam into its
   aperiodic (1/f) background and its periodic (oscillatory) peaks
2. Continuous wavelet transform of one electrode over the whole recording
3. Correlate the power of each peak band, and the total power, with the
   spike PCs - is it the whole spectrum that tracks the population, or one band?

The bands in steps 2 and 3 are the peaks of the chosen electrode's own fit. The
mean spectrum is plotted for orientation only; nothing is derived from it.

Run as VS Code interactive cells (`# %%`).
"""

# %% ---------------------------------------------------------------------
# 0. Imports and configuration
# -------------------------------------------------------------------------
import sys

import numpy as np
import matplotlib.pyplot as plt
import elephant
from neo.io import NixIO
from scipy.ndimage import gaussian_filter1d
from scipy.signal import welch
from specparam import SpectralGroupModel

sys.path.insert(0, "../source")
from utils import EVENT_COLOR, LFP_COLOR, SPIKE_COLOR, mark_event, use_analysis_style

NIX_FILE = "../data/rat_retreat_meta_1kHz.nix"
SPIKE_PCA_FILE = "../data/spike_pca.npz"    # written by spike_data_pca.py
POSITIONS_FILE = "../data/channel_positions.pkl"    # {channel 1-64: (shank, depth)}

PSD_RANGE = (1, 250)        # Hz, range fitted by specparam
WELCH_SECONDS = 4           # Welch segment length -> 0.25 Hz resolution
PEAK_WIDTH_LIMITS = (6.0, 60.0)     # Hz, narrowest and widest peak allowed
MAX_N_PEAKS = 6             # peaks specparam may place per electrode
MIN_PEAK_HEIGHT = 0.15      # log10 power a peak must stand above the 1/f fit

ELECTRODE = None            # 0-based; None -> the electrode with the strongest peaks
CWT_FREQS = np.geomspace(2, 250, 60)    # log-spaced, like a scalogram axis
N_CYCLES = 6.0              # wavelet width; more cycles = sharper in frequency
STEP = 20                   # average the wavelet power in 20 ms blocks -> 50 Hz
N_PCS = 3                   # spike components correlated against the bands

plt.switch_backend("module://matplotlib_inline.backend_inline")
use_analysis_style()


def block_average(x, step=STEP):
    """Mean of `x` over consecutive blocks of `step` samples.

    Used to bring the 1 kHz wavelet power, the time axis and the spike scores
    onto one coarser grid. Averaging (rather than picking every `step`-th
    sample) keeps it from aliasing the fast envelope fluctuations down.
    """
    n = x.size // step * step
    return x[:n].reshape(-1, step).mean(axis=1)


# %% ---------------------------------------------------------------------
# 1. Load the LFP and the cached spike components
# -------------------------------------------------------------------------
with NixIO(NIX_FILE, mode="ro") as io:
    block = io.read_block()

signals = block.segments[0].analogsignals[0]    # (n_time, n_channels)
fs = float(signals.sampling_rate)
n_channels = signals.shape[1]
times_lfp = signals.times.rescale("s").magnitude

spike_pca = np.load(SPIKE_PCA_FILE)
scores_spike = spike_pca["scores"]              # (n_time, n_pc), 1 kHz like the LFP
ratio_spike = spike_pca["ratio"]
ketamine_time = float(spike_pca["ketamine_time"])

# Probe geometry, used to lay the per-electrode grids out on the array.
electrode_mapping = np.load(POSITIONS_FILE, allow_pickle=True)
positions = np.array(list(electrode_mapping.values()))
n_rows, n_cols = positions[:, 1].max() + 1, positions[:, 0].max() + 1

print(f"LFP: {signals.shape} at {fs:g} Hz, {times_lfp[-1]:.0f} s")
print(f"cached spike scores: {scores_spike.shape}, ketamine at {ketamine_time:.0f} s")
print(f"probe: {n_rows} positions x {n_cols} shanks")


# %% ---------------------------------------------------------------------
# 2. specparam on every electrode: aperiodic background + periodic peaks
# -------------------------------------------------------------------------
# Every electrode is fitted separately (SpectralGroupModel = one independent
# SpectralModel per spectrum), so both the 1/f background and the peaks are that
# electrode's own. The mean spectrum is never fitted - it is drawn in the first
# panel purely for orientation.
freqs_psd, psd = welch(signals.magnitude, fs=fs, nperseg=int(WELCH_SECONDS * fs), axis=0)
in_range = (freqs_psd >= PSD_RANGE[0]) & (freqs_psd <= PSD_RANGE[1])
freqs_psd = freqs_psd[in_range]
psd_all = psd[in_range].T                       # (n_channels, n_freq)

group = SpectralGroupModel(peak_width_limits=PEAK_WIDTH_LIMITS,
                           max_n_peaks=MAX_N_PEAKS,
                           min_peak_height=MIN_PEAK_HEIGHT,
                           aperiodic_mode="fixed", verbose=False)
group.fit(freqs_psd, psd_all, PSD_RANGE)

periodic = group.get_params("periodic")     # (n_peaks, 4): CF, PW, BW, electrode
aperiodic = group.get_params("aperiodic")   # (n_channels, 2): offset, exponent
r_squared = group.get_metrics("gof")        # (n_channels,)


def peaks_of(electrode):
    """The (CF, PW, BW) rows specparam fitted on one electrode, low to high."""
    rows = periodic[periodic[:, 3] == electrode][:, :3]
    return rows[np.argsort(rows[:, 0])]


def flattened(electrode):
    """log10 PSD of one electrode with its own aperiodic fit subtracted.

    What is left is the periodic part: flat around zero, with the oscillations
    standing up as peaks. In "fixed" mode specparam's background is simply
    offset - exponent * log10(f), so it can be rebuilt here in one line.
    """
    offset, exponent = aperiodic[electrode]
    return np.log10(psd_all[electrode]) - (offset - exponent * np.log10(freqs_psd))


# Pick the electrode carrying the most oscillatory power, unless one was set.
if ELECTRODE is None:
    ELECTRODE = int(np.argmax([peaks_of(e)[:, 1].sum() for e in range(n_channels)]))

peak_cf, peak_pw, peak_bw = peaks_of(ELECTRODE).T

print(f"specparam: {len(periodic)} peaks over {n_channels} electrodes, "
      f"R2 = {r_squared.mean():.3f} +- {r_squared.std():.3f}")
print(f"aperiodic exponent = {aperiodic[:, 1].mean():.2f} +- {aperiodic[:, 1].std():.2f} "
      f"(range {aperiodic[:, 1].min():.2f} - {aperiodic[:, 1].max():.2f})")
print(f"\nelectrode {ELECTRODE} - its own peaks, and the bands used from here on:")
for cf, pw, bw in zip(peak_cf, peak_pw, peak_bw):
    print(f"   {cf:7.2f} Hz   power {pw:.2f}   width {bw:5.2f} Hz "
          f"-> band {cf - bw / 2:.1f} - {cf + bw / 2:.1f} Hz")

fig, axes = plt.subplots(2, 2, figsize=(15, 9))

# (a) every electrode, plus the mean - orientation only, nothing is fitted here
axes[0, 0].loglog(freqs_psd, psd_all.T, color="0.85", lw=0.3)
axes[0, 0].loglog(freqs_psd, psd_all.mean(axis=0), color="k", lw=1.6,
                  label="mean over electrodes (not fitted)")
axes[0, 0].loglog(freqs_psd, psd_all[ELECTRODE], color=LFP_COLOR, lw=1.4,
                  label=f"electrode {ELECTRODE}")
axes[0, 0].set_xlabel("Frequency (Hz)")
axes[0, 0].set_ylabel(f"PSD ({signals.units.dimensionality}$^2$/Hz)")
axes[0, 0].set_title("Welch PSD of all 64 electrodes")
axes[0, 0].legend(fontsize=8)

# (b) the decomposition of the electrode the rest of the script uses
model = group.get_model(ELECTRODE, regenerate=True)
axes[0, 1].plot(model.data.freqs, model.data.power_spectrum, color="0.4", lw=1.2,
                label="spectrum")
axes[0, 1].plot(model.data.freqs, model.results.model.modeled_spectrum, color="k",
                lw=1.6, label=f"specparam fit (R2 = {r_squared[ELECTRODE]:.3f})")
axes[0, 1].plot(model.data.freqs, model.results.model._ap_fit, color=LFP_COLOR,
                ls="--", lw=1.4,
                label=f"aperiodic, exponent {aperiodic[ELECTRODE, 1]:.2f}")
for cf, bw in zip(peak_cf, peak_bw):
    axes[0, 1].axvspan(cf - bw / 2, cf + bw / 2, color=LFP_COLOR, alpha=0.15)
axes[0, 1].set_xscale("log")
axes[0, 1].set_xlabel("Frequency (Hz)")
axes[0, 1].set_ylabel("log10 power")
axes[0, 1].set_title(f"Electrode {ELECTRODE}: periodic vs aperiodic "
                     f"(shaded = the bands used below)")
axes[0, 1].legend(fontsize=8)

# (c) periodic across the probe: where each electrode puts its peaks
scatter = axes[1, 0].scatter(periodic[:, 0], periodic[:, 3], c=periodic[:, 1],
                             s=10, cmap="viridis")
axes[1, 0].axhline(ELECTRODE, color=LFP_COLOR, ls="--", lw=1.2,
                   label=f"electrode {ELECTRODE}")
axes[1, 0].set_xscale("log")
axes[1, 0].set_xlabel("Peak centre frequency (Hz)")
axes[1, 0].set_ylabel("Electrode")
axes[1, 0].set_title(f"Periodic: {len(periodic)} peaks, fitted per electrode")
axes[1, 0].legend(fontsize=8)
fig.colorbar(scatter, ax=axes[1, 0], label="Peak power (log10 over 1/f)")

# (d) aperiodic across the probe
axes[1, 1].plot(aperiodic[:, 1], np.arange(n_channels), "o-", ms=3, lw=0.8,
                color=LFP_COLOR, label="exponent")
axes[1, 1].set_xlabel("Aperiodic exponent (1/f slope)")
axes[1, 1].set_ylabel("Electrode")
twin_ap = axes[1, 1].twiny()
twin_ap.plot(aperiodic[:, 0], np.arange(n_channels), "o-", ms=3, lw=0.8,
             color=SPIKE_COLOR, label="offset")
twin_ap.set_xlabel("Aperiodic offset")
twin_ap.grid(False)
axes[1, 1].set_title("Aperiodic: 1/f background per electrode")
axes[1, 1].legend(handles=axes[1, 1].get_lines() + twin_ap.get_lines(), fontsize=8)

fig.suptitle("specparam decomposition of every electrode")
plt.show()

# The flattened spectrum of every electrode on its own axes, laid out on the
# probe: each electrode's aperiodic fit removed, its own peaks shaded. This is
# what the bands are read off, one electrode at a time.
fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 16), sharex=True, sharey=True)
for channel_id, (shank, depth) in electrode_mapping.items():
    e = channel_id - 1
    ax = axes[n_rows - 1 - depth, shank]     # tip of the shank at the bottom
    ax.plot(freqs_psd, flattened(e), color="k", lw=0.9)
    for cf, _, bw in peaks_of(e):
        ax.axvspan(cf - bw / 2, cf + bw / 2, color=LFP_COLOR, alpha=0.25)
    ax.axhline(0, color="0.5", lw=0.6)
    ax.set_xscale("log")
    ax.tick_params(labelsize=7)
    ax.set_title(f"ch {channel_id}   exp {aperiodic[e, 1]:.2f}", fontsize=7)
    if e == ELECTRODE:
        for spine in ax.spines.values():
            spine.set(color=LFP_COLOR, lw=2.5)

fig.supxlabel("Frequency (Hz)")
fig.supylabel("log10 power - that electrode's own aperiodic fit")
fig.suptitle("Flattened spectrum of every electrode, shaded = its own peaks "
             f"(orange frame = electrode {ELECTRODE}, 'exp' = aperiodic exponent)")
plt.show()


# %% ---------------------------------------------------------------------
# 3. Wavelet transform of that electrode over the whole recording
# -------------------------------------------------------------------------
times_cwt = block_average(times_lfp)


def scalogram(electrode):
    """Wavelet power of one electrode over the whole recording.

    Returns (n_frequency, n_time) on the CWT_FREQS grid, block-averaged down to
    fs / STEP. One frequency at a time: the full complex transform of 705 s x 60
    frequencies would be several GB, while the averaged power is a few MB.
    """
    sig = np.asarray(signals[:, electrode].magnitude).ravel()
    out = np.empty((CWT_FREQS.size, times_cwt.size), dtype=np.float32)
    for i, frequency in enumerate(CWT_FREQS):
        coeffs = elephant.signal_processing.wavelet_transform(
            sig, frequency=frequency, sampling_frequency=fs, n_cycles=N_CYCLES)
        out[i] = block_average(np.abs(coeffs) ** 2)
    return out


sig = np.asarray(signals[:, ELECTRODE].magnitude).ravel()
power = scalogram(ELECTRODE)
print(f"scalogram: {power.shape} (n_freq, n_time) at {fs / STEP:g} Hz")

# Each frequency is shown relative to its own mean over time, otherwise the 1/f
# slope alone would leave everything above ~20 Hz black. The light smoothing is
# for the display only - 705 s squeezed into ~1500 pixels otherwise aliases into
# vertical stripes. The correlations below use the unsmoothed `power`.
power_db = 10 * np.log10(power / power.mean(axis=1, keepdims=True))
power_db = gaussian_filter1d(power_db, 10, axis=1)      # 10 blocks = 200 ms

fig, axes = plt.subplots(3, 1, figsize=(20, 10), sharex=True,
                         gridspec_kw={"height_ratios": [1, 3, 1.2]})

axes[0].plot(times_lfp, sig, color="k", lw=0.2)
axes[0].set_ylabel(f"LFP ({signals.units.dimensionality})")
axes[0].set_title(f"Raw LFP - electrode {ELECTRODE}")

mesh = axes[1].pcolormesh(times_cwt, CWT_FREQS, power_db, cmap="viridis",
                          shading="nearest", rasterized=True,
                          vmin=np.percentile(power_db, 1),
                          vmax=np.percentile(power_db, 99.5))
for cf in peak_cf:
    axes[1].axhline(cf, color="w", ls=":", lw=1.0, alpha=0.7)
axes[1].set_yscale("log")
axes[1].set_ylabel("Frequency (Hz)")
axes[1].set_title(f"Wavelet power ({N_CYCLES:.0f} cycles), dB relative to each "
                  f"frequency's mean - dotted lines are this electrode's own peaks")
fig.colorbar(mesh, ax=axes[1], label="Power (dB re. mean)", pad=0.01)

for pc in range(2):
    axes[2].plot(times_cwt, block_average(scores_spike[:, pc]), lw=0.5, alpha=0.8,
                 label=f"PC{pc + 1} ({ratio_spike[pc]:.0%} of variance)")
axes[2].set_ylabel("Score (a.u.)")
axes[2].set_xlabel("Time (s)")
axes[2].set_title("Spike PCA (cached)")
axes[2].legend(fontsize=9, loc="upper left")

for ax in axes:
    mark_event(ax, ketamine_time)
axes[0].set_xlim(times_cwt[0], times_cwt[-1])
fig.suptitle(f"Electrode {ELECTRODE}: LFP, wavelet scalogram and the spike components")
plt.show()


# %% ---------------------------------------------------------------------
# 4. Does the wavelet power track the spike components?
# -------------------------------------------------------------------------
# One time series per peak band of THIS electrode (the specparam peaks from
# section 2): the power summed over the frequencies inside the band. Plus the
# total over the whole scalogram, as the reference for "does anything specific
# about the peaks matter, or is it just overall power?".
EPOCH_GUARD = 5.0           # s dropped either side of the injection

band_power = {"total": power.sum(axis=0)}
for cf, bw in zip(peak_cf, peak_bw):
    inside = (CWT_FREQS >= cf - bw / 2) & (CWT_FREQS <= cf + bw / 2)
    if not inside.any():        # peak narrower than the spacing of the wavelet grid
        inside[np.abs(CWT_FREQS - cf).argmin()] = True
    band_power[f"{cf:.0f} Hz"] = power[inside].sum(axis=0)

names = list(band_power)
pc_scores = np.stack([block_average(scores_spike[:, k]) for k in range(N_PCS)], axis=1)
n = min(times_cwt.size, pc_scores.shape[0])     # the two grids differ by a sample or two

# Over the whole recording both the band power and the spike scores step up at
# the injection, and a correlation across that step mostly measures the step.
# So each epoch is correlated on its own. The guard band drops the transition
# itself: the 2 Hz wavelet is ~3 s long, so power right at the injection is a
# mixture of both sides of it.
epochs = {"before ketamine": times_cwt[:n] < ketamine_time - EPOCH_GUARD,
          "after ketamine": times_cwt[:n] > ketamine_time + EPOCH_GUARD}


def corr_table(mask):
    """Pearson r of every band against every spike PC, over the samples in `mask`.

    No p-values: neighbouring samples of a 100 ms-smoothed rate are nowhere near
    independent, so a test against n ~ 18,000 would return p ~ 0 regardless.
    """
    return np.array([[np.corrcoef(band_power[name][:n][mask], pc_scores[:n, k][mask])[0, 1]
                      for k in range(N_PCS)] for name in names])


r = {epoch: corr_table(mask) for epoch, mask in epochs.items()}

for epoch, table in r.items():
    print(f"\n{epoch} ({epochs[epoch].sum()} samples)")
    print(f"{'band':>18} | " + " | ".join(f"{f'PC{k + 1}':>7}" for k in range(N_PCS)))
    for name, row in zip(names, table):
        print(f"{name:>18} | " + " | ".join(f"{v:+7.3f}" for v in row))

# Band shown in the time course: the one tracking PC1 best on average.
best = int(np.mean([np.abs(t[:, 0]) for t in r.values()], axis=0).argmax())
print(f"\nbest PC1 coupling: {names[best]}, "
      + ", ".join(f"{epoch} r = {t[best, 0]:+.3f}" for epoch, t in r.items()))

fig = plt.figure(figsize=(18, 9))
grid = fig.add_gridspec(2, 2, height_ratios=[1, 1.3])

ax_time = fig.add_subplot(grid[0, :])
ax_time.plot(times_cwt[:n], band_power[names[best]][:n], color=LFP_COLOR, lw=0.7,
             label=f"{names[best]} wavelet power")
ax_time.axvspan(ketamine_time - EPOCH_GUARD, ketamine_time + EPOCH_GUARD,
                color="0.5", alpha=0.4, label="excluded transition")
mark_event(ax_time, ketamine_time)
ax_time.set_xlabel("Time (s)")
ax_time.set_ylabel("Power (a.u.)")
twin = ax_time.twinx()
twin.plot(times_cwt[:n], pc_scores[:n, 0], color=SPIKE_COLOR, lw=0.7, alpha=0.8,
          label="spike PC1")
twin.set_ylabel("PC1 score (a.u.)")
twin.grid(False)
ax_time.set_title(f"{names[best]} vs spike PC1 - "
                  + ", ".join(f"{epoch} r = {t[best, 0]:+.3f}"
                              for epoch, t in r.items()))
ax_time.legend(handles=ax_time.get_lines() + twin.get_lines(), fontsize=9,
               loc="upper left")

# Same row of bars twice, once per epoch, on a shared y so they can be compared.
x = np.arange(len(names))
width = 0.8 / N_PCS
y_max = 1.15 * max(np.abs(t).max() for t in r.values())
for col, (epoch, table) in enumerate(r.items()):
    ax = fig.add_subplot(grid[1, col])
    for k in range(N_PCS):
        ax.bar(x + (k - (N_PCS - 1) / 2) * width, table[:, k], width=width,
               label=f"PC{k + 1} ({ratio_spike[k]:.0%} of variance)")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(x, labels=names, fontsize=9)
    ax.set_ylim(-y_max, y_max)
    ax.set_xlabel("Wavelet band")
    ax.set_ylabel("Pearson r")
    ax.set_title(epoch)
    if col == 0:
        ax.legend(fontsize=9)
fig.suptitle(f"Electrode {ELECTRODE}: band power vs spike PCA, "
             f"each epoch correlated on its own")
plt.show()


# %% ---------------------------------------------------------------------
# 5. The same question for all 64 electrodes, laid out on the probe
# -------------------------------------------------------------------------
# Total wavelet power of every electrode against every spike PC, in each epoch.
# One cell per electrode, placed at its own (shank, depth) - so a hot spot on
# the array shows up as a cluster of cells, not as a run of channel numbers.
# The wavelet transform is redone per electrode: about 2 s each, ~2 min total.
r_total = np.zeros((n_channels, len(epochs), N_PCS))
for e in range(n_channels):
    total_power = scalogram(e).sum(axis=0)[:n]
    for j, mask in enumerate(epochs.values()):
        for k in range(N_PCS):
            r_total[e, j, k] = np.corrcoef(total_power[mask], pc_scores[:n, k][mask])[0, 1]
    print(f"\r  electrode {e + 1}/{n_channels}", end="")

top_e, top_j, top_k = np.unravel_index(np.abs(r_total).argmax(), r_total.shape)
print(f"\nstrongest anywhere on the probe: electrode {top_e + 1}, "
      f"{list(epochs)[top_j]}, PC{top_k + 1}, r = {r_total[top_e, top_j, top_k]:+.3f}")
for j, epoch in enumerate(epochs):
    print(f"  {epoch}: |r| max {np.abs(r_total[:, j]).max():.3f}, "
          f"median {np.median(np.abs(r_total[:, j])):.3f}")

fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 20), sharex=True, sharey=True)
x = np.arange(N_PCS)
width = 0.4
y_max = 1.1 * np.abs(r_total).max()
epoch_colors = ("0.55", EVENT_COLOR)        # before, after

for channel_id, (shank, depth) in electrode_mapping.items():
    e = channel_id - 1
    ax = axes[n_rows - 1 - depth, shank]    # tip of the shank at the bottom
    for j, epoch in enumerate(epochs):
        ax.bar(x + (j - 0.5) * width, r_total[e, j], width=width,
               color=epoch_colors[j], label=epoch)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_ylim(-y_max, y_max)
    ax.set_xticks(x, labels=[f"PC{k + 1}" for k in range(N_PCS)], fontsize=7)
    ax.tick_params(labelsize=7)
    # the electrode's own specparam peaks, so each cell says what it is made of
    ax.set_title(f"ch {channel_id}   "
                 + "/".join(f"{cf:.0f}" for cf in peaks_of(e)[:, 0]) + " Hz",
                 fontsize=7)
    if e == ELECTRODE:
        for spine in ax.spines.values():
            spine.set(color=LFP_COLOR, lw=2.5)

axes[0, 0].legend(fontsize=9, loc="upper left")
fig.supxlabel("Shank (left -> right)")
fig.supylabel("Position along shank (tip at the bottom)")
fig.suptitle("Total wavelet power vs the spike PCs, every electrode on the probe "
             f"(orange frame = electrode {ELECTRODE}, titles list each "
             f"electrode's own specparam peaks)")
plt.show()
