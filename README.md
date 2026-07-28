# Rat Olfactory Bulb Electrophysiology Dataset
## Overview
This dataset contains extracellular electrophysiological recordings acquired from the olfactory bulb of an adult rat using 8×8 microelectrode array (Neuronexus).
The recording captures approximately 10 minutes of neural activity sampled at 30 kHz from 64 channel.

The data are organized as a continuous recording with shape:
```text
(21,150,000 samples, 64 channels)
```

- Sampling frequency: 30,000 Hz
- Number of channels: 64
- Electrode layout: 8 × 8 microelectrode array
- Recording duration: approximately 705 seconds (~11 minutes)

The dataset includes:

- Continuous recording
- Spike times 


## Data Structure

1. Block
2. Segment
3. Analog Signal (64 channels)
3. SpikeTrain (unit_32)
3. SpikeTrain (unit_36)
3. SpikeTrain (unit_38)
3. SpikeTrain (unit_60)

## Electrode Positions Description

A Python pickle file called *channel_positions.pkl* containing a dictionary that maps each NeuroNexus A8×8 probe channel ID to its two-dimensional position on the probe.
`` dict[int, tuple[int, int]] ``

**Dictionary format**
Key: Channel ID (1–64)
Value: (x, y) coordinate
where:
x (0–7) is the shank index, counted from left to right.
y (0–7) is the electrode position along the shank, counted from the probe tip (bottom) toward the probe base (top).


## Experimental Design

The recording consists of three phases:

1. **Baseline activity**
   
   Spontaneous neural activity before ketamine administration.

2. **Ketamine administration**
   
   Systemic ketamine injection.

3. **Post-ketamine activity**
   
   Neural activity recorded following ketamine administration, capturing ketamine-induced changes in olfactory bulb network dynamics.



My goal: understand single unit properties

I would like to use this dataset to learn how to explore and characterize neuronal spike trains. 
- examine the relationship between spiking activity and network oscillations observed in the local field potential (LFP).
The recording contains a baseline period dominated by gamma-band activity and a post-ketamine period characterized by HFO ~150 Hz, providing an opportunity to study how neuronal firing patterns relate to different oscillatory states


## ANDA-NI License

The data and source code in this repository is made available solely for use by students and instructors for educational purposes during the ANDA-NI 2026 school held from June 15-July 31, 2026 at the Forschungszentrum Julich, Germany. Unathorized copying and use of the files in this repository, via any medium, is prohibited unless specified otherwise.