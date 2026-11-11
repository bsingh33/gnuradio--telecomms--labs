# Lab 2 — 2×2 MIMO-OFDM Detection

This laboratory implements a 2×2 MIMO-OFDM receiver with two independent QPSK data streams. The receiver uses ZF, MMSE, and ordered MMSE-SIC detection.

The experiments examine how spatial correlation and channel-estimation error affect detection. Channel estimates are obtained using pilot symbols. BER and channel-matrix condition numbers are measured for each detector.

## Results

At 12 dB SNR, increasing spatial correlation from 0 to 0.95 raises the median channel condition number from 2.96 to 29.09.

Under these conditions, ZF BER increases from 0.02840 to 0.34464. MMSE-SIC BER increases from 0.00757 to 0.12120.

The results show that poorly conditioned channels make it more difficult to separate the transmitted signals. MMSE and MMSE-SIC improve detection but require additional receiver processing.

The study includes 22 operating points with 800 independent fading realizations per point.

## Files

[Report](report/) | [GNU Radio and Python code](code/) | [Results](results/) | [Tests](tests/)
