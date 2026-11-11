import csv
import time
from collections import deque
from pathlib import Path
from datetime import datetime, timezone
import os

import numpy as np
try:
    from gnuradio import gr
except ImportError:
    # Allow the identical engine to run and be tested without GNU Radio.
    class _SyncBlock:
        def __init__(self, *args, **kwargs):
            pass
    class _GR:
        sync_block = _SyncBlock
    gr = _GR()


class blk(gr.sync_block):
    """LDPC-coded OFDM link with fixed and adaptive min-sum decoding."""

    def __init__(
        self,
        snr_db=10.0,
        jsr_db=-20.0,
        max_iters=20,
        rate_sel=0,
        seed=2026,
        window_frames=1000,
        report_mode=True,
        frames_per_point=1000,
        extended_report=False,
        csi_mode="perfect",
        code_family="random_eira",
        quant_bits=0,
        run_label="",
    ):
        gr.sync_block.__init__(
            self,
            name="LDPC-OFDM Engine",
            in_sig=None,
            out_sig=[
                np.complex64, np.complex64, np.complex64,
                np.float32, np.float32, np.float32, np.float32,
                np.float32, np.float32, np.float32, np.float32,
                np.float32, np.float32,
            ],
        )

        self.snr_db = float(snr_db)
        self.jsr_db = float(jsr_db)
        self.max_iters = int(max_iters)
        self.rate_sel = int(rate_sel)

        self.seed = int(seed)
        self.window_frames = int(window_frames)
        self.rng = np.random.default_rng(self.seed)

        self.nfft = 64
        self.cp = 16
        self.k_info = 96
        self.active = np.r_[np.arange(1, 25), np.arange(40, 64)]
        self.interferer_bin = 12

        self.report_mode = bool(report_mode)
        self.frames_per_point = int(frames_per_point)
        self.extended_report = bool(extended_report)
        if csi_mode not in ("perfect", "pilot_ls"):
            raise ValueError("csi_mode must be 'perfect' or 'pilot_ls'")
        if code_family not in ("random_eira", "structured_eira"):
            raise ValueError("code_family must be 'random_eira' or 'structured_eira'")
        if int(quant_bits) not in (0, 4, 6, 8):
            raise ValueError("quant_bits must be 0, 4, 6, or 8")
        self.csi_mode = csi_mode
        self.code_family = code_family
        self.quant_bits = int(quant_bits)
        self.run_label = str(run_label)
        self.report_plan = self._make_report_plan()
        self.report_index = 0
        self.report_frame = 0
        self.report_done = False

        self._control_key = None
        self._frame_count = 0

        # Never overwrite previous data when a new GUI or headless run starts.
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
        run_id = f"{stamp}_{os.getpid()}"
        if self.run_label:
            safe_label = "".join(c if c.isalnum() or c in "-_" else "_" for c in self.run_label)
            run_id += "_" + safe_label[:48]
        self.results_dir = Path.cwd() / "results" / "runs" / run_id
        self.results_dir.mkdir(parents=True, exist_ok=False)

        self.csv_path = self.results_dir / "Lab1_LDPC_OFDM_results.csv"
        self.capture_path = self.results_dir / "representative_capture.npz"

        self.rep_tx = []
        self.rep_rx = []
        self.rep_waveform = []
        self.last_outputs = None
        self.last_metrics = None

        self._build_code()
        self._reset_stats()
        self._init_csv()

        print("[Lab1] Results:", self.results_dir.resolve())

    # ------------------------------------------------------------------
    # Experiment plan
    # ------------------------------------------------------------------

    def _make_report_plan(self):
        plan = [
            ("representative", 8.0, -10.0, 0, 20),
        ]

        for snr in [0, 2, 4, 6, 8, 10, 12, 14, 16]:
            plan.append(("snr_sweep", float(snr), -10.0, 0, 20))

        for jsr in [-30, -25, -20, -15, -10, -5, 0]:
            plan.append(("jsr_sweep", 8.0, float(jsr), 0, 20))

        if self.extended_report:
            for rate in [0, 1, 2]:
                plan.append(("rate_sweep", 6.0, -15.0, rate, 20))

            for iters in [5, 10, 15, 20, 25, 30]:
                plan.append(("iteration_sweep", 6.0, -15.0, 0, iters))

        return plan

    def _current_settings(self):
        if self.report_mode and not self.report_done:
            name, snr, jsr, rate, iters = self.report_plan[self.report_index]
            return name, snr, jsr, rate, iters

        return (
            "interactive",
            float(self.snr_db),
            float(self.jsr_db),
            int(np.clip(self.rate_sel, 0, 2)),
            max(1, int(self.max_iters)),
        )

    def _apply_settings(self, snr, jsr, rate, iters):
        key = (snr, jsr, rate, iters)

        if key == self._control_key:
            return

        old_rate = None if self._control_key is None else self._control_key[2]

        self.snr_db = snr
        self.jsr_db = jsr
        self.rate_sel = rate
        self.max_iters = iters

        if old_rate is None or rate != old_rate:
            self._build_code()

        self._reset_stats()
        self._control_key = key

        print(
            f"[Lab1] SNR={snr:.1f} dB, JSR={jsr:.1f} dB, "
            f"rate={self.code_rate:.3f}, Imax={iters}"
        )

    def _advance_report(self):
        if not self.report_mode:
            return

        self.report_frame += 1
        if self.report_frame < self.frames_per_point:
            return

        name = self.report_plan[self.report_index][0]

        if name == "representative":
            np.savez_compressed(
                self.capture_path,
                ideal_qpsk=np.concatenate(self.rep_tx),
                equalized_qpsk=np.concatenate(self.rep_rx),
                rx_waveform=np.concatenate(self.rep_waveform),
                samp_rate=64000.0,
            )
            print("[Lab1] Saved representative IQ capture.")

        self.report_index += 1
        self.report_frame = 0

        if self.report_index >= len(self.report_plan):
            self.report_done = True
            print("[Lab1] REPORT RUN COMPLETE")
        else:
            next_name = self.report_plan[self.report_index][0]
            print(f"[Lab1] Starting {next_name}")

    # ------------------------------------------------------------------
    # LDPC code
    # ------------------------------------------------------------------

    def _build_code(self):
        rates = [0.5, 2.0 / 3.0, 0.75]
        idx = int(np.clip(self.rate_sel, 0, 2))

        self.code_rate = rates[idx]
        self.m_checks = int(
            round(self.k_info * (1.0 - self.code_rate) / self.code_rate)
        )

        rng = np.random.default_rng(self.seed + 1000 + idx)

        A = np.zeros((self.m_checks, self.k_info), dtype=np.uint8)
        if self.code_family == "structured_eira":
            # Circulant-permutation structure in A, accumulator B as below.
            # Illustrative structured code; NOT a standards-defined QC-LDPC.
            z = 8
            row_blocks = self.m_checks // z
            col_blocks = self.k_info // z
            for cb in range(col_blocks):
                used = set()
                for branch in range(3):
                    rb = (cb + branch * max(1, row_blocks // 3)) % row_blocks
                    while rb in used:
                        rb = (rb + 1) % row_blocks
                    used.add(rb)
                    shift = (3 * cb + 5 * branch + cb * branch) % z
                    for pos in range(z):
                        A[rb * z + pos, cb * z + (pos + shift) % z] = 1
        else:
            for col in range(self.k_info):
                rows = rng.choice(self.m_checks, size=3, replace=False)
                A[rows, col] = 1

        B = np.eye(self.m_checks, dtype=np.uint8)
        for row in range(1, self.m_checks):
            B[row, row - 1] = 1

        self.A = A
        self.H = np.concatenate((A, B), axis=1)
        self.edge_count = int(np.count_nonzero(self.H))

        self.check_vars = [
            np.flatnonzero(self.H[row])
            for row in range(self.m_checks)
        ]
        self.var_checks = [
            np.flatnonzero(self.H[:, col])
            for col in range(self.H.shape[1])
        ]

    def _encode(self, bits):
        s = (self.A @ bits) & 1

        parity = np.empty(self.m_checks, dtype=np.uint8)
        parity[0] = s[0]
        for i in range(1, self.m_checks):
            parity[i] = s[i] ^ parity[i - 1]

        codeword = np.concatenate((bits, parity))

        if np.any((self.H @ codeword) & 1):
            raise RuntimeError("LDPC encoding failed parity check.")

        return codeword

    # ------------------------------------------------------------------
    # Normalized min-sum decoder
    # ------------------------------------------------------------------

    def _quantize(self, x):
        """Uniform signed-message approximation; off when quant_bits=0."""
        if not self.quant_bits:
            return x
        clip = 20.0
        max_code = (1 << (self.quant_bits - 1)) - 1
        step = clip / max_code
        return np.round(np.clip(x, -clip, clip) / step) * step

    def _decode(self, llr, max_iters, early_stop):
        """Independent normalized-min-sum execution with optional REAL exit."""
        alpha = 0.80
        m, n = self.H.shape
        llr = self._quantize(np.asarray(llr, dtype=float))
        v2c = np.zeros((m, n), dtype=float)
        c2v = np.zeros((m, n), dtype=float)
        for row, variables in enumerate(self.check_vars):
            v2c[row, variables] = llr[variables]

        hard = np.zeros(n, dtype=np.uint8)
        elapsed_start = time.perf_counter_ns()
        for iteration in range(1, max_iters + 1):
            for row, variables in enumerate(self.check_vars):
                values = v2c[row, variables]
                signs = np.where(values >= 0, 1.0, -1.0)
                magnitudes = np.abs(values)
                first = int(np.argmin(magnitudes))
                min1 = magnitudes[first]
                temp = magnitudes.copy()
                temp[first] = np.inf
                min2 = np.min(temp)
                sign_product = np.prod(signs)
                for pos, variable in enumerate(variables):
                    magnitude = min2 if pos == first else min1
                    c2v[row, variable] = (
                        alpha * sign_product * signs[pos] * magnitude
                    )
            if self.quant_bits:
                c2v = self._quantize(c2v)

            posterior = llr.copy()
            for variable, checks in enumerate(self.var_checks):
                if len(checks) == 0:
                    continue
                posterior[variable] += np.sum(c2v[checks, variable])
                for check in checks:
                    v2c[check, variable] = posterior[variable] - c2v[check, variable]
            if self.quant_bits:
                v2c = self._quantize(v2c)
                posterior = self._quantize(posterior)

            hard = (posterior < 0).astype(np.uint8)
            if early_stop and not np.any((self.H @ hard) & 1):
                break

        microseconds = (time.perf_counter_ns() - elapsed_start) / 1000.0
        return hard, iteration, microseconds

    def _decode_pair(self, llr, max_iters):
        """Two independent decoders get an identical initial LLR vector."""
        fixed_word, fixed_iter, fixed_us = self._decode(llr, max_iters, False)
        adaptive_word, adaptive_iter, adaptive_us = self._decode(llr, max_iters, True)
        return fixed_word, adaptive_word, fixed_iter, adaptive_iter, fixed_us, adaptive_us

    # ------------------------------------------------------------------
    # Modulation and channel
    # ------------------------------------------------------------------

    @staticmethod
    def _qpsk_mod(bits):
        if len(bits) % 2:
            bits = np.r_[bits, 0]

        i = 1.0 - 2.0 * bits[0::2].astype(float)
        q = 1.0 - 2.0 * bits[1::2].astype(float)
        return (i + 1j * q) / np.sqrt(2.0)

    def _ofdm_tx(self, symbols):
        carriers = len(self.active)
        blocks = int(np.ceil(len(symbols) / carriers))

        padded = np.zeros(blocks * carriers, dtype=complex)
        padded[:len(symbols)] = symbols

        waveform = []

        if self.csi_mode == "pilot_ls":
            # One OFDM training symbol with known QPSK phase on active carriers.
            pilot_X = np.zeros(self.nfft, dtype=complex)
            pilot_X[self.active] = (1.0 + 1.0j) / np.sqrt(2.0)
            pilot = np.fft.ifft(pilot_X) * np.sqrt(self.nfft)
            waveform.append(np.r_[pilot[-self.cp:], pilot])

        for block in range(blocks):
            X = np.zeros(self.nfft, dtype=complex)
            start = block * carriers
            stop = (block + 1) * carriers
            X[self.active] = padded[start:stop]

            x = np.fft.ifft(X) * np.sqrt(self.nfft)
            waveform.append(np.r_[x[-self.cp:], x])

        return np.concatenate(waveform), blocks, len(symbols)

    def _channel_taps(self):
        power = 0.65 ** np.arange(6)
        power = power / np.sum(power)

        taps = (
            self.rng.normal(size=6)
            + 1j * self.rng.normal(size=6)
        ) * np.sqrt(power / 2.0)

        return taps

    def _impair(self, tx, channel, snr_db, jsr_db):
        faded = np.convolve(tx, channel)[:len(tx)]
        signal_power = np.mean(np.abs(faded) ** 2)

        noise_power = signal_power / (10.0 ** (snr_db / 10.0))
        noise = np.sqrt(noise_power / 2.0) * (
            self.rng.normal(size=len(faded))
            + 1j * self.rng.normal(size=len(faded))
        )

        jammer_power = signal_power * (10.0 ** (jsr_db / 10.0))
        n = np.arange(len(faded))
        jammer = np.sqrt(jammer_power) * np.exp(
            1j * 2.0 * np.pi * self.interferer_bin * n / self.nfft
        )

        return faded + noise + jammer, noise_power

    # ------------------------------------------------------------------
    # OFDM receiver
    # ------------------------------------------------------------------

    def _ofdm_rx(
        self,
        received,
        channel,
        blocks,
        number_of_symbols,
        noise_power,
    ):
        H = np.fft.fft(
            np.r_[channel, np.zeros(self.nfft - len(channel))]
        )
        pilot_offset = 0
        if self.csi_mode == "pilot_ls":
            # One full-band, known training OFDM symbol per frame.
            pilot_raw = received[self.cp:self.cp + self.nfft]
            pilot_fft = np.fft.fft(pilot_raw) / np.sqrt(self.nfft)
            pilot_value = (1.0 + 1.0j) / np.sqrt(2.0)
            H = np.zeros(self.nfft, dtype=complex)
            H[self.active] = pilot_fft[self.active] / pilot_value
            pilot_offset = self.nfft + self.cp

        symbols = []
        variances = []
        block_len = self.nfft + self.cp

        for block in range(blocks):
            start = pilot_offset + block * block_len
            stop = start + block_len

            y = received[start:stop]
            Y = np.fft.fft(y[self.cp:]) / np.sqrt(self.nfft)

            H_active = H[self.active]
            z = Y[self.active] / (H_active + 1e-12)

            symbols.extend(z)
            variances.extend(
                noise_power / (np.abs(H_active) ** 2 + 1e-12)
            )

        z = np.asarray(symbols[:number_of_symbols])
        variance = np.asarray(variances[:number_of_symbols])

        llr = np.empty(2 * len(z), dtype=float)
        llr[0::2] = 2.0 * np.sqrt(2.0) * z.real / variance
        llr[1::2] = 2.0 * np.sqrt(2.0) * z.imag / variance

        return z, llr

    # ------------------------------------------------------------------
    # Statistics and logging
    # ------------------------------------------------------------------

    def _reset_stats(self):
        n = max(10, self.window_frames)

        self.be_fixed = deque(maxlen=n)
        self.be_adaptive = deque(maxlen=n)
        self.fe_fixed = deque(maxlen=n)
        self.fe_adaptive = deque(maxlen=n)
        self.iter_fixed = deque(maxlen=n)
        self.iter_adaptive = deque(maxlen=n)

        self._frame_count = 0

    def _metrics(self):
        frames = max(1, len(self.be_fixed))
        bits = frames * self.k_info

        ber_fixed = sum(self.be_fixed) / bits
        ber_adaptive = sum(self.be_adaptive) / bits
        fer_fixed = sum(self.fe_fixed) / frames
        fer_adaptive = sum(self.fe_adaptive) / frames

        iter_fixed = sum(self.iter_fixed) / frames
        iter_adaptive = sum(self.iter_adaptive) / frames

        work_fixed = iter_fixed * self.m_checks / self.k_info
        work_adaptive = iter_adaptive * self.m_checks / self.k_info
        saving = 100.0 * (1.0 - work_adaptive / max(work_fixed, 1e-12))

        return (
            ber_fixed,
            ber_adaptive,
            fer_fixed,
            fer_adaptive,
            iter_fixed,
            iter_adaptive,
            work_fixed,
            work_adaptive,
            saving,
            float(self._frame_count),
        )

    def _init_csv(self):
        with self.csv_path.open("w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([
                "experiment",
                "time_unix",
                "snr_db",
                "jsr_db",
                "code_rate",
                "max_iters",
                "frame",
                "bit_errors_fixed",
                "bit_errors_adaptive",
                "frame_error_fixed",
                "frame_error_adaptive",
                "iterations_fixed",
                "iterations_adaptive",
                "decoder_work_fixed",
                "decoder_work_adaptive",
                "decoder_time_fixed_us",
                "decoder_time_adaptive_us",
                "edge_updates_fixed",
                "edge_updates_adaptive",
                "syndrome_checks_adaptive",
                "csi_mode",
                "code_family",
                "quant_bits",
                "seed",
            ])

    def _log_frame(
        self,
        experiment,
        snr,
        jsr,
        max_iters,
        errors_fixed,
        errors_adaptive,
        fixed_iterations,
        adaptive_iterations,
        fixed_us,
        adaptive_us,
    ):
        work_fixed = fixed_iterations * self.m_checks / self.k_info
        work_adaptive = adaptive_iterations * self.m_checks / self.k_info

        with self.csv_path.open("a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([
                experiment,
                f"{time.time():.6f}",
                f"{snr:.3f}",
                f"{jsr:.3f}",
                f"{self.code_rate:.6f}",
                max_iters,
                self._frame_count,
                errors_fixed,
                errors_adaptive,
                int(errors_fixed > 0),
                int(errors_adaptive > 0),
                fixed_iterations,
                adaptive_iterations,
                f"{work_fixed:.6f}",
                f"{work_adaptive:.6f}",
                f"{fixed_us:.3f}",
                f"{adaptive_us:.3f}",
                2 * self.edge_count * fixed_iterations,
                2 * self.edge_count * adaptive_iterations,
                adaptive_iterations,
                self.csi_mode,
                self.code_family,
                self.quant_bits,
                self.seed,
            ])

    @staticmethod
    def _repeat_into(output, data):
        data = np.asarray(data)

        if len(data) == 0:
            output[:] = 0
        else:
            output[:] = np.resize(data, len(output)).astype(
                output.dtype,
                copy=False,
            )

    def simulate_frame(self, experiment, snr, jsr, rate, max_iters):
        """Run one independent simulated frame; GUI and headless share this."""
        self._apply_settings(snr, jsr, rate, max_iters)
        information_bits = self.rng.integers(0, 2, self.k_info, dtype=np.uint8)
        coded_bits = self._encode(information_bits)
        tx_symbols = self._qpsk_mod(coded_bits)
        tx_waveform, blocks, symbol_count = self._ofdm_tx(tx_symbols)
        channel = self._channel_taps()
        received_waveform, noise_power = self._impair(tx_waveform, channel, snr, jsr)
        received_symbols, llr = self._ofdm_rx(
            received_waveform, channel, blocks, symbol_count, noise_power
        )
        llr = llr[:len(coded_bits)]
        (fixed_codeword, adaptive_codeword, fixed_iterations,
         adaptive_iterations, fixed_us, adaptive_us) = self._decode_pair(llr, max_iters)

        fixed_errors = int(np.count_nonzero(
            fixed_codeword[:self.k_info] != information_bits
        ))
        adaptive_errors = int(np.count_nonzero(
            adaptive_codeword[:self.k_info] != information_bits
        ))
        self.be_fixed.append(fixed_errors)
        self.be_adaptive.append(adaptive_errors)
        self.fe_fixed.append(int(fixed_errors > 0))
        self.fe_adaptive.append(int(adaptive_errors > 0))
        self.iter_fixed.append(fixed_iterations)
        self.iter_adaptive.append(adaptive_iterations)
        self._frame_count += 1
        self._log_frame(
            experiment, snr, jsr, max_iters, fixed_errors, adaptive_errors,
            fixed_iterations, adaptive_iterations, fixed_us, adaptive_us
        )
        if experiment == "representative":
            self.rep_tx.append(tx_symbols.copy())
            self.rep_rx.append(received_symbols.copy())
            self.rep_waveform.append(received_waveform.copy())

        return tx_symbols, received_symbols, received_waveform, self._metrics()

    # ------------------------------------------------------------------
    # GNU Radio work function
    # ------------------------------------------------------------------

    def work(self, input_items, output_items):
        if self.report_done:
            # Keep last GUI results visible rather than clearing plots to zero.
            if self.last_outputs is not None:
                for i in range(3):
                    self._repeat_into(output_items[i], self.last_outputs[i])
                for port, value in enumerate(self.last_metrics, start=3):
                    output_items[port][:] = np.float32(value)
            else:
                for output in output_items:
                    output[:] = 0
            time.sleep(0.05)
            return len(output_items[0])

        experiment, snr, jsr, rate, max_iters = self._current_settings()
        tx, rx, waveform, metrics = self.simulate_frame(
            experiment, snr, jsr, rate, max_iters
        )
        self.last_outputs = (tx, rx, waveform)
        self.last_metrics = metrics
        self._repeat_into(output_items[0], tx)
        self._repeat_into(output_items[1], rx)
        self._repeat_into(output_items[2], waveform)
        for port, value in enumerate(metrics, start=3):
            output_items[port][:] = np.float32(value)
        self._advance_report()
        if not self.report_mode:
            time.sleep(0.05)
        return len(output_items[0])
