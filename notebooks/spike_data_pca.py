"""
Stage 1 of the spike / LFP analysis: the spike data on its own.

Loads the sorted units, smooths them with a Gaussian kernel into instantaneous
firing rates, runs PCA over the neurons, and saves both results to
``../data/spike_pca.npz`` so that `combined_analysis.py` can reuse them without
recomputing the (slow) rate estimate.

Contents of the cache
---------------------
times        (n_time,)            time axis of the rates, in seconds
rates        (n_time, n_neurons)  Gaussian-smoothed instantaneous rates, in Hz
scores       (n_time, n_pc)       projection of the rates onto each component
loadings     (n_pc, n_neurons)    neuron weights of each component
ratio        (n_pc,)              explained-variance fraction per component
spike_ids    (n_neurons,)         cluster id of each unit
unit_labels  (n_neurons,)         cluster id + SU/MU tag, for figure labels
ketamine_time, recording_duration, kernel_sigma_ms, sampling_period_ms  scalars

`rates` and `scores` are stored as float32 to keep the file around 70 MB; that
is well below the precision the downstream correlations need.

Run as VS Code interactive cells (`# %%`).
"""


# %% ---------------------------------------------------------------------
# 0. Imports and configuration
# -------------------------------------------------------------------------
import sys

import numpy as np
import matplotlib.pyplot as plt
import quantities as pq
from elephant import kernels
from elephant.statistics import instantaneous_rate
from neo.io import NixIO
from sklearn.decomposition import PCA
from viziphant.rasterplot import rasterplot
from viziphant.statistics import plot_instantaneous_rates_colormesh
%matplotlib inline
sys.path.insert(0, "../source")
from utils import SPIKE_COLOR, mark_event, plot_scree, use_analysis_style

NIX_FILE = "../data/rat_retreat_meta_1kHz.nix"
SPIKE_PCA_FILE = "../data/spike_pca.npz"    # written at the end of this script

# Draw the figures inline in the interactive window. Same effect as the
# `%matplotlib inline` magic, but plain Python, so the editor does not flag it.
plt.switch_backend("module://matplotlib_inline.backend_inline")
use_analysis_style()


# %% ---------------------------------------------------------------------
# 1. Load the units and read the experiment metadata from the file
# -------------------------------------------------------------------------
with NixIO(NIX_FILE, mode="ro") as io:
    block = io.read_block()

seg = block.segments[0]

# seg.description is a "key=value; key=value" string holding the protocol, e.g.
# "ketamine_injection_time_s=372; baseline_interval_s=0-372; ..."
metadata = dict(field.split("=", 1) for field in seg.description.split("; "))
ketamine_time = float(metadata["ketamine_injection_time_s"]) * pq.s
recording_duration = seg.analogsignals[0].t_stop    # ground truth for the session

# Each SpikeTrain stores its own t_stop (the time of its last spike), so align
# them to the recording length before rasterising / estimating rates.
sts = list(seg.spiketrains)
for st in sts:
    st.t_stop = recording_duration

n_neurons = len(sts)
spike_ids = [st.annotations["spike_id"] for st in sts]
unit_labels = [f"{st.annotations['spike_id']} "
               f"({'SU' if st.annotations['unit_type'] == 'single_unit' else 'MU'})"
               for st in sts]

print(f"{block.description} | {seg.description}")
print(f"recording: {recording_duration}, ketamine at {ketamine_time}")
print(f"{n_neurons} units (SU = single unit, MU = multi unit): {unit_labels}")


# %% 
for st in sts:
    st.t_stop = recording_duration


# %% ---------------------------------------------------------------------
# 2. Spike raster
# -------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(30, 5))
rasterplot(sts, axes=ax, s=20, marker="|", color="k", lw=0.5, alpha=0.5)
mark_event(ax, ketamine_time)
ax.set_xlabel("Time (s)")
ax.set_ylabel("Unit")
ax.set_yticks(range(n_neurons))
ax.set_yticklabels(unit_labels)
ax.set_title(f"Spike raster of {n_neurons} sorted units - olfactory bulb, "
             f"ketamine at {float(ketamine_time):.0f} s")
ax.legend(loc="upper right", fontsize=9)
plt.show()


# %% ---------------------------------------------------------------------
# 3. Instantaneous firing rate (Gaussian kernel)
# -------------------------------------------------------------------------
KERNEL_SIGMA = 100 * pq.ms          # smoothing width of the rate estimate
RATE_SAMPLING_PERIOD = 1 * pq.ms    # sample the rate at 1 kHz, like the LFP

kernel = kernels.GaussianKernel(sigma=KERNEL_SIGMA)
inst_rate = instantaneous_rate(sts, sampling_period=RATE_SAMPLING_PERIOD, kernel=kernel)
times_rate = inst_rate.times.rescale("s").magnitude
print(f"instantaneous rate: {inst_rate.shape} (n_time, n_neurons)")

fig, ax = plt.subplots(figsize=(20, 10))
# units="s": without it viziphant labels the time axis in the units of
# sampling_period (ms here) and the event marker would land at 372 ms.
plot_instantaneous_rates_colormesh(inst_rate, axes=ax, units="s")
mark_event(ax, ketamine_time)
# One mesh cell per unit, so the row centres sit at i + 0.5.
ax.set_yticks(np.arange(n_neurons) + 0.5)
ax.set_yticklabels(unit_labels)
ax.set_xlabel("Time (s)")
ax.set_ylabel("Unit")
ax.set_title(f"Instantaneous firing rate (Gaussian kernel, "
             f"sigma = {KERNEL_SIGMA.item():.0f} ms)")
ax.legend(loc="upper right", fontsize=9)
plt.show()


# %% ---------------------------------------------------------------------
# 4. PCA on the firing rates (samples = time bins, features = neurons)
# -------------------------------------------------------------------------
VARIANCE_THRESHOLD = 0.9    # cumulative variance target reported on the scree plot

pca_spike = PCA()
scores_spike = pca_spike.fit_transform(inst_rate.magnitude)   # (n_time, n_neurons)
loadings_spike = pca_spike.components_                        # (n_pc, n_neurons)
ratio_spike = pca_spike.explained_variance_ratio_

fig, ax = plt.subplots(figsize=(7, 4))
cum_spike, n_pc_spike = plot_scree(
    ratio_spike, "Scree plot - PCA of the smoothed spike trains", ax,
    threshold=VARIANCE_THRESHOLD)
plt.show()

print(f"spikes: {n_pc_spike} components explain >{VARIANCE_THRESHOLD:.0%} "
      f"of the variance")
print(np.round(cum_spike, 3))


# %% ---------------------------------------------------------------------
# 5. Loadings - which neurons build each mode
# -------------------------------------------------------------------------
N_PCS_TO_SHOW = min(4, n_neurons)   # components shown in the detailed panels

# Full loading matrix: signed values -> diverging colormap, symmetric limits.
vmax = np.abs(loadings_spike).max()
fig, ax = plt.subplots(figsize=(10, 6))
mesh = ax.pcolormesh(loadings_spike, cmap="coolwarm", vmin=-vmax, vmax=vmax)
ax.set_xticks(np.arange(n_neurons) + 0.5, labels=unit_labels, rotation=90)
ax.set_yticks(np.arange(n_neurons) + 0.5,
              labels=[f"PC{i + 1}" for i in range(n_neurons)])
ax.grid(False)
ax.set_xlabel("Unit")
ax.set_ylabel("Principal component")
ax.set_title("Spike PCA loadings - all components")
fig.colorbar(mesh, ax=ax, label="Loading (a.u.)")
plt.show()

# Bar view of the leading components.
fig, axes = plt.subplots(N_PCS_TO_SHOW, 1, figsize=(12, 2.2 * N_PCS_TO_SHOW),
                         sharex=True)
for i, ax in enumerate(axes):
    ax.bar(np.arange(n_neurons), loadings_spike[i], color=SPIKE_COLOR)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_ylabel("Loading")
    ax.set_title(f"PC{i + 1} ({ratio_spike[i]:.1%} of variance)")
axes[-1].set_xticks(np.arange(n_neurons), labels=unit_labels, rotation=90)
axes[-1].set_xlabel("Unit")
fig.suptitle("Spike PCA - neuron contributions to the leading components")
plt.show()


# %% ---------------------------------------------------------------------
# 6. Score time courses
# -------------------------------------------------------------------------
fig, axes = plt.subplots(N_PCS_TO_SHOW, 1, figsize=(20, 2.5 * N_PCS_TO_SHOW),
                         sharex=True)
for i, ax in enumerate(axes):
    ax.plot(times_rate, scores_spike[:, i], lw=0.8, color=SPIKE_COLOR)
    mark_event(ax, ketamine_time)
    ax.set_ylabel("Score (a.u.)")
    ax.set_title(f"PC{i + 1} ({ratio_spike[i]:.1%} of variance)")
axes[0].legend(loc="upper right", fontsize=9)
axes[-1].set_xlabel("Time (s)")
fig.suptitle("Spike PCA - projection of the population rate onto each component")
plt.show()


# %% ---------------------------------------------------------------------
# 7. Save the smoothed rates and the PCA for the combined analysis
# -------------------------------------------------------------------------
np.savez(SPIKE_PCA_FILE,
         times=times_rate,
         rates=inst_rate.magnitude.astype(np.float32),
         scores=scores_spike.astype(np.float32),
         loadings=loadings_spike,
         ratio=ratio_spike,
         spike_ids=np.array(spike_ids),
         unit_labels=np.array(unit_labels),
         ketamine_time=float(ketamine_time),
         recording_duration=float(recording_duration),
         kernel_sigma_ms=float(KERNEL_SIGMA.rescale("ms")),
         sampling_period_ms=float(RATE_SAMPLING_PERIOD.rescale("ms")))

print(f"saved -> {SPIKE_PCA_FILE}")
print("next: run combined_analysis.py")

# %%
