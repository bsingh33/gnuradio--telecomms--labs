# Lab 1 — Complexity-Adaptive LDPC-Coded OFDM

This laboratory implements an LDPC-coded QPSK-OFDM system. The transmitted signal passes through a six-tap Rayleigh fading channel with added noise and narrowband interference.

Two LDPC decoding methods are compared. The fixed decoder uses the full iteration budget. The adaptive decoder stops when the parity checks are satisfied. The experiments vary SNR, interference level, code rate, and maximum iterations.

## Results

At SNR = 8 dB and JSR = −10 dB, the first valid decoding decision occurs after 8.35 iterations on average. The fixed decoder uses 20 iterations. This corresponds to a **58.25% reduction in required iterations** with comparable BER and FER.

The recorded experiment includes 26,000 frames. The original run records the stopping iteration but continues processing. It therefore does not measure execution-time savings.

## Files

[Report](report/) | [GNU Radio and Python code](code/) | [Results](results/) | [Tests](tests/)
