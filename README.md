# gnuradio--telecomms--labs

This repository contains four laboratory exercises in wireless communications and signal processing. The work covers LDPC decoding, MIMO detection, spectrum sensing, and interference mitigation.

Each laboratory includes a simulation implemented in GNU Radio and Python. The experiments examine receiver performance under different channel and interference conditions. The results are presented in technical reports with supporting measurements.

## Laboratory Exercises

**[Lab 1 — Adaptive LDPC-Coded OFDM](Lab1_LDPC_OFDM/)**

Comparison of fixed and adaptive LDPC decoding under multipath fading and narrowband interference.

**[Lab 2 — 2×2 MIMO-OFDM Detection](Lab2_MIMO_OFDM/)**

Comparison of ZF, MMSE, and MMSE-SIC detection under channel correlation and estimation error.

**[Lab 3 — Cognitive Spectrum Access](Lab3_Cognitive_Spectrum_Access/)**

Wideband spectrum sensing with automatic channel selection and adaptive sensing intervals.

**[Lab 4 — Chirp Acquisition and Packet Recovery](Lab4_Chirp_Receiver/)**

Chirp synchronization and packet recovery using interference suppression under different jamming conditions.

## Implementation

The signal-processing algorithms are written in Python and integrated into GNU Radio through Embedded Python Blocks. GNU Radio provides the flowgraphs, controls, and signal displays.

Each laboratory includes source code, recorded results, validation tests, and a technical report. All results are from software simulations.

##
This work was conducted by Balraj Minhas at the Wireless Technology Center in Purdue University Fort Wayne (PFW), Indiana, USA.
