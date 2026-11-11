# Lab 4 — Robust Chirp Acquisition and Packet Recovery

This laboratory implements a chirp communication receiver. The transmitter generates a preamble and data symbols. The receiver estimates packet timing, demodulates the symbols, and verifies the recovered data using a CRC.

Four receiver methods are compared: unmodified reception, amplitude blanking, spectral excision, and combined mitigation.

The experiments examine broadband barrage, swept-frequency, pulsed, and event-triggered interference. All four receivers process the same received signals.

## Results

At SNR = −7 dB and JSR = 10 dB, spectral excision improves packet recovery under swept interference from **38.3% to 98.0%**.

At JSR = 15 dB, amplitude blanking improves packet recovery under event-triggered interference from **5.3% to 43.3%**.

The results also show that interference suppression does not always improve reception. Blanking can remove useful signal samples and reduce packet recovery under pulsed interference.

The study includes 5,100 transmitted packets. It uses a simplified chirp packet format rather than the complete LoRa physical layer.

## Files

[Report](report/) | [GNU Radio and Python code](code/) | [Results](results/) | [Tests](tests/)
