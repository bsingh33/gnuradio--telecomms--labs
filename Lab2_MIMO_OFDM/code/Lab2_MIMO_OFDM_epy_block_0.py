#!/usr/bin/env python3
"""2x2 spatial-multiplexing MIMO-OFDM: ZF, MMSE, and ordered MMSE-SIC.

This module is both the GNU Radio Embedded Python Block source and a standalone
reproducibility entry point. The study uses no external radio hardware.
"""
from __future__ import annotations
import argparse
import csv
from pathlib import Path
import numpy as np

try:
    from gnuradio import gr
except ImportError:  # Headless reproducibility does not require GNU Radio.
    class _HeadlessGR:
        class sync_block:
            def __init__(self, *args, **kwargs):
                pass
    gr = _HeadlessGR()

NFFT, CP, NT, NR, L = 64, 16, 2, 2, 6
ACTIVE = np.r_[np.arange(1, 27), np.arange(38, 64)]
NC = len(ACTIVE)
BITS_PER_FRAME = NC * NT * 2
PDP = np.exp(-np.arange(L, dtype=float) / 1.5)
PDP /= PDP.sum()
DET = ('ZF', 'MMSE', 'MMSE-SIC')


def cn(rng, shape):
    return (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)) / np.sqrt(2.0)


def qpsk(bits):
    return ((1.0 - 2.0 * bits[..., 0]) + 1j * (1.0 - 2.0 * bits[..., 1])) / np.sqrt(2.0)


def hard_bits(z):
    return np.stack((np.real(z) < 0.0, np.imag(z) < 0.0), axis=-1).astype(np.uint8)


def slicer(z):
    return qpsk(hard_bits(z))


def correlation_root(rho):
    # Symmetric positive-definite matrix sqrt of [[1,rho],[rho,1]].
    a = (np.sqrt(1.0 + rho) + np.sqrt(1.0 - rho)) / 2.0
    b = (np.sqrt(1.0 + rho) - np.sqrt(1.0 - rho)) / 2.0
    return np.array([[a, b], [b, a]])


def channel_taps(rng, nframes, rho):
    if not (0.0 <= rho < 1.0):
        raise ValueError('rho must lie in [0, 1)')
    w = cn(rng, (nframes, NR, NT, L))
    r = correlation_root(rho)
    taps = np.einsum('ri,bijl,jt->brtl', r, w, r, optimize=True)
    taps *= np.sqrt(PDP)[None, None, None, :]
    return taps


def channel_freq(taps):
    return np.transpose(np.fft.fft(taps, n=NFFT, axis=-1)[..., ACTIVE], (0, 3, 1, 2))


def time_channel(x_freq, taps, noise_var, rng):
    """Unitary IFFT, CP, linear convolution, time-domain noise, CP removal, FFT.

    x_freq: (frame, tx, tone); taps: (frame, rx, tx, delay)
    output: (frame, active_tone, rx), received CP-free time samples (frame,rx,64)
    """
    x = np.fft.ifft(x_freq, axis=-1, norm='ortho')
    xcp = np.concatenate((x[..., -CP:], x), axis=-1)
    y = np.zeros((len(x), NR, NFFT + CP), dtype=np.complex128)
    for ell in range(L):
        y[..., ell:] += np.einsum('brt,bts->brs', taps[..., ell], xcp[..., :NFFT+CP-ell], optimize=True)
    y += np.sqrt(noise_var) * cn(rng, y.shape)
    useful = y[..., CP:]
    yf = np.fft.fft(useful, axis=-1, norm='ortho')
    return np.transpose(yf[..., ACTIVE], (0, 2, 1)), useful


def solve_detectors(h_est, y, noise_var):
    """Vectorized matched linear detectors and ordered MMSE-SIC.

    All three detectors receive the *same* tone-level y and estimated H.
    MMSE-SIC first detects the stream with the smaller estimated MMSE error
    covariance diagonal; the remaining stream is detected after cancellation.
    """
    h = h_est.reshape(-1, NR, NT)
    y = y.reshape(-1, NR)
    n = len(h)
    hh = h.conj().swapaxes(1, 2)
    gram = hh @ h
    rhs = (hh @ y[..., None])[..., 0]
    eye = np.eye(NT, dtype=np.complex128)[None, :, :]
    zf = np.linalg.solve(gram, rhs[..., None])[..., 0]
    reg = gram + noise_var * eye
    # First right-hand side gives MMSE estimate, other two give covariance
    # geometry for stream ordering (single 2x2 factorization per tone).
    rhs3 = np.concatenate((rhs[..., None], np.broadcast_to(eye, (n, NT, NT))), axis=-1)
    sol = np.linalg.solve(reg, rhs3)
    mmse = sol[..., 0]
    inv_reg = sol[..., 1:]
    order = np.argmin(np.real(np.diagonal(inv_reg, axis1=1, axis2=2)), axis=1)
    other = 1 - order
    row = np.arange(n)
    first_hard = slicer(mmse[row, order])
    residual = y - h[row, :, order] * first_hard[:, None]
    h_other = h[row, :, other]
    second_soft = np.sum(h_other.conj() * residual, axis=1) / (
        np.sum(np.abs(h_other)**2, axis=1) + noise_var
    )
    sic = np.empty((n, NT), dtype=np.complex128)
    sic[row, order] = first_hard
    sic[row, other] = slicer(second_soft)
    shape = h_est.shape[:-2] + (NT,)
    return {'ZF': zf.reshape(shape), 'MMSE': mmse.reshape(shape), 'MMSE-SIC': sic.reshape(shape)}


def generate_batch(rng, count, snr_db, rho, pilot_snr_db=None, pilot_rng=None):
    esn0 = 10.0**(snr_db/10.0)
    n0 = 1.0/esn0
    taps = channel_taps(rng, count, rho)
    h_true = channel_freq(taps)
    bits = rng.integers(0, 2, size=(count, NT, NC, 2), dtype=np.uint8)
    symbols = qpsk(bits)
    x_freq = np.zeros((count, NT, NFFT), dtype=np.complex128)
    x_freq[:, :, ACTIVE] = symbols
    y, rx_useful = time_channel(x_freq, taps, n0, rng)
    if pilot_snr_db is None:
        h_est = h_true.copy()
    else:
        if pilot_rng is None:
            pilot_rng = rng
        p0 = np.zeros_like(x_freq)
        p1 = np.zeros_like(x_freq)
        p0[:, 0, ACTIVE] = 1.0
        p1[:, 1, ACTIVE] = 1.0
        n0_pilot = 10.0**(-pilot_snr_db/10.0)
        y0, _ = time_channel(p0, taps, n0_pilot, pilot_rng)
        y1, _ = time_channel(p1, taps, n0_pilot, pilot_rng)
        h_est = np.stack((y0, y1), axis=-1)
    # Output data arranged (frame, tone, stream, bit).
    truth = np.transpose(bits, (0,2,1,3))
    tx_syms = np.transpose(symbols, (0,2,1))
    return truth, tx_syms, y, h_true, h_est, rx_useful, n0


def experiment_plan():
    plan = []
    for snr in [0, 4, 8, 12, 16, 20]:
        plan.append(('snr', float(snr), 0.3, None, 1001))
    for rho in [0.0, 0.3, 0.6, 0.8, 0.95]:
        plan.append(('correlation', 12.0, rho, None, 2002))
    for p in [None, 8.0, 14.0, 20.0, 26.0]:
        plan.append(('estimation', 12.0, 0.6, p, 3003))
    for rho in [0.0, 0.6, 0.9]:
        for p in [12.0, 24.0]:
            plan.append(('joint', 12.0, rho, p, 4004))
    return plan


CONDITION_EDGES = np.array([1., 2., 4., 8., 16., np.inf])
CONDITION_LABELS = ('[1,2)', '[2,4)', '[4,8)', '[8,16)', '[16,inf)')


def run_point(experiment, snr_db, rho, pilot_snr_db, seed, frames=800,
              batch=100, capture=False):
    bit_errors = {d:0 for d in DET}
    bit_errors_by_bin = {d:np.zeros(5,dtype=np.int64) for d in DET}
    n_by_bin = np.zeros(5, dtype=np.int64)
    frame_bit_errors = {d:[] for d in DET}
    frame_bin_bit_errors = {d:[[] for _ in range(5)] for d in DET}
    frame_bin_bits = [[] for _ in range(5)]
    nmse_numerator = 0.0
    nmse_denominator = 0.0
    conditions = []
    enhancements = []
    first_capture = None
    for start in range(0, frames, batch):
        count = min(batch, frames-start)
        # Keep data, channel and data noise identical across conditions within
        # each sweep; pilots have an independent random stream.
        data_rng = np.random.default_rng(np.random.SeedSequence([seed, start, 0]))
        pilot_rng = np.random.default_rng(np.random.SeedSequence([seed, start, 1]))
        truth, tx, y, ht, he, waveform, n0 = generate_batch(
            data_rng, count, snr_db, rho, pilot_snr_db, pilot_rng
        )
        nmse_numerator += float(np.sum(np.abs(ht-he)**2))
        nmse_denominator += float(np.sum(np.abs(ht)**2))
        singular = np.linalg.svd(ht.reshape(-1, 2, 2), compute_uv=False)
        cond = singular[:,0] / singular[:,1]
        conditions.append(cond)
        enhancements.append((1/singular[:,0]**2+1/singular[:,1]**2)/2.0)
        got = solve_detectors(he,y,n0)
        bi = np.searchsorted(CONDITION_EDGES[1:], cond, side='right')
        counts = np.bincount(bi, minlength=5)
        n_by_bin += counts * NT * 2
        label_matrix=bi.reshape(count, NC)
        for k in range(5):
            frame_bin_bits[k].extend((NT*2*np.sum(label_matrix==k,axis=1)).tolist())
        for d in DET:
            errors = np.count_nonzero(hard_bits(got[d]) != truth, axis=(2,3))
            bit_errors[d] += int(errors.sum())
            frame_bit_errors[d].extend(errors.sum(axis=1).astype(int).tolist())
            for k in range(5):
                frame_bin_bit_errors[d][k].extend(
                    np.sum(errors*(label_matrix==k),axis=1).astype(int).tolist())
            bit_errors_by_bin[d] += np.bincount(bi, weights=errors.reshape(-1),minlength=5).astype(np.int64)
        if capture and first_capture is None:
            keep = min(count, 80)
            first_capture = dict(tx=tx[:keep].copy(), y=y[:keep].copy(),
                                 zf=got['ZF'][:keep].copy(), mmse=got['MMSE'][:keep].copy(),
                                 sic=got['MMSE-SIC'][:keep].copy(),
                                 h_true=ht[:keep].copy(), h_est=he[:keep].copy(),
                                 waveform=waveform[:keep].copy())
    condall = np.concatenate(conditions)
    enhall = np.concatenate(enhancements)
    nmse = nmse_numerator / nmse_denominator
    total_bits = frames * BITS_PER_FRAME
    # Bootstrap whole OFDM frames, not individual bits; tone errors within a
    # frame share a frequency-selective channel and are not independent.
    rng_ci = np.random.default_rng(np.random.SeedSequence([seed, 991]))
    sample_indices = rng_ci.integers(0, frames, size=(800, frames))
    rows = []
    for d in DET:
        errors = bit_errors[d]
        frame_err=np.asarray(frame_bit_errors[d],dtype=int)
        distribution=frame_err[sample_indices].sum(axis=1)/total_bits
        lo,hi = np.quantile(distribution,[.025,.975])
        rows.append(dict(experiment=experiment, snr_db=snr_db, rho=rho,
                 pilot_snr_db='perfect' if pilot_snr_db is None else pilot_snr_db,
                 detector=d, frames=frames, bits=total_bits, bit_errors=errors,
                 ber=errors/total_bits, ci95_low=lo, ci95_high=hi, nmse=nmse,
                 median_cond=float(np.median(condall)),
                 p90_cond=float(np.quantile(condall,.9)),
                 median_zf_noise_enhancement=float(np.median(enhall)), seed=seed))
    bins=[]
    for i,label in enumerate(CONDITION_LABELS):
        for d in DET:
            cnt=int(n_by_bin[i]); err=int(bit_errors_by_bin[d][i])
            frame_num=np.asarray(frame_bin_bit_errors[d][i],dtype=float)
            frame_den=np.asarray(frame_bin_bits[i],dtype=float)
            denom=frame_den[sample_indices].sum(axis=1)
            ratio=np.divide(frame_num[sample_indices].sum(axis=1),denom,
                    out=np.full(len(denom),np.nan),where=denom>0)
            lo,hi=np.nanquantile(ratio,[.025,.975]) if np.any(np.isfinite(ratio)) else (np.nan,np.nan)
            bins.append(dict(experiment=experiment,snr_db=snr_db,rho=rho,
                  pilot_snr_db='perfect' if pilot_snr_db is None else pilot_snr_db,
                  condition_bin=label,detector=d,bits=cnt,bit_errors=err,
                  ber=err/cnt if cnt else np.nan,ci95_low=lo,ci95_high=hi))
    return rows, bins, first_capture


def _write_csv(path, rows):
    with Path(path).open('w', newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0].keys()))
        writer.writeheader(); writer.writerows(rows)


def run_study(output_dir, frames=800, seed=2026, batch=100, quiet=False):
    output_dir=Path(output_dir);output_dir.mkdir(parents=True,exist_ok=True)
    rows=[]; bins=[]; capture=None
    plan=experiment_plan()
    for i,(experiment,snr,rho,pilot,group) in enumerate(plan,1):
        r,b,c=run_point(experiment,snr,rho,pilot,seed+group,
                        frames=frames,batch=batch,
                        capture=(experiment=='estimation' and pilot==20.0))
        rows.extend(r);bins.extend(b)
        if c is not None: capture=c
        if not quiet:
            print(f'[{i:02d}/{len(plan)}] {experiment:11s} Es/N0={snr:4.0f} dB '
                  f'rho={rho:0.2f} pilot={pilot}  '
                  f'BER(ZF/MMSE/SIC)='+ '/'.join(f'{x["ber"]:.4g}' for x in r),flush=True)
    _write_csv(output_dir/'Lab2_MIMO_OFDM_results.csv',rows)
    _write_csv(output_dir/'Lab2_condition_bins.csv',bins)
    if capture is not None:
        np.savez_compressed(output_dir/'representative_capture.npz',**capture)
    return rows,bins


# GNU Radio GUI outputs: QPSK TX, ZF, MMSE, and SIC;
# dashboard streams: BER ZF/MMSE/SIC, median condition, channel NMSE.
class blk(gr.sync_block):
    def __init__(self, snr_db=12.0, rho=0.6, pilot_snr_db=20.0, seed=2026):
        gr.sync_block.__init__(self, name='2x2 MIMO-OFDM experiment',in_sig=None,
              out_sig=[np.complex64, np.complex64, np.complex64,
                  np.complex64, np.float32, np.float32, np.float32,
                  np.float32, np.float32])
        self.snr_db=float(snr_db)
        self.rho=float(rho)
        self.pilot_snr_db=float(pilot_snr_db)
        self.seed=int(seed)
        self.rng=np.random.default_rng(self.seed)
        self.buffer=[]
    def work(self,input_items,output_items):
        n=len(output_items[0]); offset=0
        while offset<n:
            if not self.buffer:
                truth,tx,y,ht,he,wave,n0=generate_batch(
                    self.rng,1,self.snr_db,self.rho,self.pilot_snr_db)
                d=solve_detectors(he,y,n0)
                sv=np.linalg.svd(ht.reshape(-1,2,2),compute_uv=False)
                cond=float(np.median(sv[:,0]/sv[:,1]))
                nmse=float(np.sum(np.abs(he-ht)**2)/np.sum(np.abs(ht)**2))
                base=[tx[0,:,0],d['ZF'][0,:,0],d['MMSE'][0,:,0],
                      d['MMSE-SIC'][0,:,0]]
                bvals=[float(np.mean(hard_bits(d[name])!=truth)) for name in DET]
                bvals += [cond,nmse]
                # Display each constellation stream repeatedly; never confuse
                # these monitor samples with a physical streaming MIMO interface.
                width=max(len(q) for q in base)
                self.buffer=[np.resize(z,width).astype(np.complex64) for z in base]+[
                    np.full(width,v,dtype=np.float32) for v in bvals]
            take=min(n-offset,len(self.buffer[0]))
            for j,arr in enumerate(output_items):arr[offset:offset+take]=self.buffer[j][:take]
            self.buffer=[v[take:] for v in self.buffer]
            if not len(self.buffer[0]):self.buffer=[]
            offset+=take
        return n

if __name__=='__main__':
    ap=argparse.ArgumentParser(description='Run the complete 2x2 MIMO-OFDM study')
    ap.add_argument('--output',default='results')
    ap.add_argument('--frames',type=int,default=800)
    ap.add_argument('--seed',type=int,default=2026)
    ap.add_argument('--batch',type=int,default=100)
    args=ap.parse_args()
    run_study(args.output, frames=args.frames, seed=args.seed,batch=args.batch)
