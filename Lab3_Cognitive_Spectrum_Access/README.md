# Lab 3 — Closed-Loop Cognitive Spectrum Access

This laboratory implements a spectrum-monitoring system with eight frequency channels. An FFT-based detector estimates channel occupancy and tracks changes in receiver noise.

A secondary transmitter uses the sensing results to select available channels. It stops transmitting or changes channels when primary-user activity is detected.

The experiments compare fixed and adaptive sensing methods. The measurements include detection probability, false alarms, channel-vacation delay, collision time, and secondary goodput.

## Results

Under nominal traffic, adaptive sensing reduces primary-user collision time from 321.3 ms to 174.7 ms per run. This is a **45.6% reduction** compared with fixed-period sensing using the same detector.

Secondary goodput decreases from 56.01 to 52.55 kb/s because more frequent sensing leaves less time for transmission.

The study includes 27 simulation runs. Goodput is calculated using a collision-erasure model rather than physical-layer packet decoding.

## Files

[Report](report/) | [GNU Radio and Python code](code/) | [Results](results/) | [Tests](tests/)
