#!/usr/bin/env python3
"""Lab 3: causal closed-loop cognitive spectrum access in a simulated 8-band environment.

The standalone study runs using NumPy. GNU Radio is optional and only required
for the Embedded Python Block live visualization. No physical RF is transmitted.
"""
from __future__ import annotations
import argparse
import csv
from pathlib import Path
import numpy as np

try:
    from gnuradio import gr
except ImportError:
    class _DummyGR:
        class sync_block:
            def __init__(self,*args,**kwargs): pass
    gr = _DummyGR()

NB = 8
NFFT = 128
BINS = NFFT // NB
SNAPS = 8
DT = 0.002  # simulated seconds per discrete observation/decision tick
FS = NFFT * SNAPS / DT  # 512 ksample/s: 8 x 128 samples in each 2-ms tick
PAYLOAD_BITS = 128  # one logical payload packet per eligible 2-ms tick
CAL_TICKS = 40  # all channels are vacant during documented receiver calibration
THRESHOLD_FACTOR = 1.20  # fixed multiplier; empirical Pfa, not asserted as ideal CFAR
POLICIES = ('fixed_threshold', 'fixed_period', 'adaptive')
SCENARIOS = {
    'nominal': dict(pu_snr_db=5.0, noise_rise_db=4.0, p01=0.008, p10=0.015),
    'weak_primary': dict(pu_snr_db=-4.0, noise_rise_db=4.0, p01=0.008, p10=0.015),
    'busy_band': dict(pu_snr_db=5.0, noise_rise_db=4.0, p01=0.016, p10=0.015),
}


def complex_gaussian(rng, shape):
    return (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)) / np.sqrt(2)


def generate_environment(seed=2026, scenario='nominal', ticks=3000, capture=False, store_iq=False):
    """Generate one shared time series, independent of the secondary controller.

    Eight frequency slots occupy disjoint blocks of sixteen 128-point DFT bins.
    Eight independent 128-sample FFT snapshots form each 2-ms sensing statistic.
    The primary follows a per-channel Markov on/off process. Noise changes over time.
    """
    if scenario not in SCENARIOS: raise ValueError('unknown scenario')
    if ticks <= CAL_TICKS + 10: raise ValueError('ticks must exceed calibration period')
    c = SCENARIOS[scenario]
    rng = np.random.default_rng(seed)
    occupancy = np.zeros((ticks, NB), dtype=bool)
    for t in range(CAL_TICKS + 1, ticks):
        old = occupancy[t-1]
        u = rng.random(NB)
        occupancy[t] = np.where(old, u >= c['p10'], u < c['p01'])
    # Slowly varying received noise power; known to the simulator, not to controller.
    time = np.arange(ticks, dtype=float)
    noise_db = c['noise_rise_db'] * time / (ticks - 1) + 0.25 * np.sin(2 * np.pi * time / (ticks / 3))
    noise_power = 10**(noise_db / 10)
    pu_power = 10**(c['pu_snr_db'] / 10)  # constant absolute PU power relative to initial noise
    energies = np.empty((ticks, NB), dtype=np.float64)
    example = None
    full_iq = np.empty((ticks,SNAPS*NFFT),dtype=np.complex64) if store_iq else None
    # Batch IFFT -> add noise in time -> FFT -> square-and-average over each subband.
    for start in range(0, ticks, 80):
        end = min(start+80, ticks)
        count = end - start
        symbols = (2*rng.integers(0, 2, size=(count, SNAPS, NFFT))-1 +
                   1j*(2*rng.integers(0, 2, size=(count, SNAPS, NFFT))-1)) / np.sqrt(2)
        mask = np.repeat(occupancy[start:end], BINS, axis=1)
        X = symbols * mask[:,None,:] * np.sqrt(pu_power)
        x = np.fft.ifft(X, axis=-1, norm='ortho')
        x += complex_gaussian(rng, x.shape) * np.sqrt(noise_power[start:end,None,None])
        Y = np.fft.fft(x, axis=-1, norm='ortho')
        if store_iq: full_iq[start:end] = x.reshape(count,-1).astype(np.complex64)
        energies[start:end] = np.mean(np.abs(Y.reshape(count,SNAPS,NB,BINS))**2, axis=(1,3))
        if capture and start <= min(CAL_TICKS+150, ticks-1) < end:
            ix = min(CAL_TICKS+150, ticks-1)-start
            example = dict(wideband_iq=x[ix].reshape(-1), fft_power=np.mean(np.abs(Y[ix])**2,axis=0))
    baseline = float(np.mean(energies[:CAL_TICKS]))
    # Account for downward sampling bias when using the low three of eight bands.
    low3_baseline = float(np.mean(np.sort(energies[:CAL_TICKS],axis=1)[:,:3]))
    bias_correction = baseline / low3_baseline
    return dict(energies=energies, occupancy=occupancy, noise_power=noise_power,
                baseline=baseline, bias_correction=bias_correction, example=example,
                seed=seed, scenario=scenario, ticks=ticks, pu_power=pu_power, iq=full_iq)


def choose_channel(energies, detected, current):
    available = np.flatnonzero(~detected)
    if current >= 0 and not detected[current]: return current
    if len(available) == 0: return -1
    # A low measured energy is preferred, without using ground-truth occupancy.
    return int(available[np.argmin(energies[available])])


def simulate_controller(env, policy='adaptive', trace=False):
    """Causal half-duplex listen-before-transmit policy.

    At a sensing tick the secondary is muted; a channel change requires the
    subsequent tick as a retuning guard, with payload resuming two ticks
    after sensing starts. The controller sees only energy samples on sensing ticks.
    """
    if policy not in POLICIES: raise ValueError(policy)
    energies, occ = env['energies'], env['occupancy']
    ticks = len(energies)
    floor_est = env['baseline']
    fixed_threshold = THRESHOLD_FACTOR * floor_est
    next_sense = CAL_TICKS
    mute_until = 0
    current = -1
    n_sense=retunes=transmitted=delivered=collisions=0
    positives=negatives=hits=false_alarms=0
    episode_start = None
    episode_ch = -1
    n_arrivals_while_transmitting=0
    vacate_delays=[]
    sense_to_retune=[]
    traced=[]
    for t in range(CAL_TICKS,ticks):
        sensing = (t >= next_sense)
        est_noise = floor_est
        detected = np.full(NB, False, dtype=bool)
        target = current
        interval = 16 if policy != 'adaptive' else 8
        switched = False
        if sensing:
            n_sense += 1
            if policy != 'fixed_threshold':
                low3 = np.mean(np.sort(energies[t])[:3]) * env['bias_correction']
                floor_est = .60 * floor_est + .40 * low3
                est_noise = floor_est
            threshold = fixed_threshold if policy=='fixed_threshold' else THRESHOLD_FACTOR * floor_est
            detected = energies[t] > threshold
            positives += int(np.count_nonzero(occ[t]))
            negatives += int(np.count_nonzero(~occ[t]))
            hits += int(np.count_nonzero(detected & occ[t]))
            false_alarms += int(np.count_nonzero(detected & ~occ[t]))
            target = choose_channel(energies[t], detected, current)
            switched = (target != current)
            current = target
            if switched:
                if target >= 0:
                    retunes += 1
                    mute_until = t+2  # sensing tick, then one guard tick; TX resumes t+2
                    sense_to_retune.append(2*DT*1000) # sensing start to first TX eligibility
                else:
                    mute_until = t+1 # becoming idle is not a retune
            if policy == 'adaptive':
                if target < 0 or switched: interval=2
                else:
                    ratio = energies[t,target] / max(threshold, 1e-12)
                    interval = 2 if ratio >= 1.0 else (4 if ratio >= .9 else 8)
            next_sense=t+interval
        is_tx = (not sensing) and (t >= mute_until) and (current >= 0)
        is_collision=bool(is_tx and occ[t,current])
        if is_tx:
            transmitted+=1
            if is_collision: collisions+=1
            else: delivered+=1
        # A PU return is protected only once the SU actually abandons the
        # affected channel (or enters idle). A sensing gap alone is not vacating.
        if episode_start is not None:
            if not occ[t,episode_ch]:
                episode_start=None  # PU disappeared before the SU vacated
            elif current != episode_ch:
                vacate_delays.append((t-episode_start)*DT*1000)
                episode_start=None
        if episode_start is None and is_collision and t>CAL_TICKS and not occ[t-1,current]:
            episode_start=t
            episode_ch=current
            n_arrivals_while_transmitting+=1
        if trace:
            traced.append(dict(tick=t,time_ms=round(1000*t*DT,3),
                primary_mask=''.join(str(int(v)) for v in occ[t]),
                band0_power=float(energies[t,0]), noise_true=float(env['noise_power'][t]),
                noise_est=float(est_noise), channel=int(current), sensing=int(sensing),
                tx=int(is_tx), collision=int(is_collision), payload_delivered_bits=PAYLOAD_BITS if is_tx and not is_collision else 0,
                interval_ticks=int(interval) if sensing else 0, change=int(switched)))
    duration=(ticks-CAL_TICKS)*DT
    metrics=dict(scenario=env['scenario'], seed=env['seed'], policy=policy,
        ticks=ticks-CAL_TICKS, duration_s=duration, sensing_ticks=n_sense,
        sensing_fraction=n_sense/(ticks-CAL_TICKS), detected_primary=hits, sensed_primary=positives,
        false_alarms=false_alarms, sensed_vacant=negatives,
        pd=hits/positives if positives else float('nan'),
        pfa=false_alarms/negatives if negatives else float('nan'),
        retunes=retunes, sensing_to_retune_ms=float(np.mean(sense_to_retune)) if sense_to_retune else float('nan'),
        transmitted_packets=transmitted, successful_packets=delivered,
        collision_packets=collisions, primary_collision_ms=collisions*DT*1000,
        collision_fraction_tx=collisions/transmitted if transmitted else 0.0,
        goodput_kbps=delivered*PAYLOAD_BITS/duration/1000,
        pu_return_exposures=n_arrivals_while_transmitting, exposures_vacated=len(vacate_delays),
        median_onset_to_vacate_ms=float(np.median(vacate_delays)) if vacate_delays else float('nan'))
    return metrics,traced


def write_csv(path, rows):
    path = Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    if not rows: return
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f, fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def run_study(directory='results',ticks=3000,seeds=(2026,2027,2028)):
    out=Path(directory);out.mkdir(parents=True,exist_ok=True)
    metrics=[]; representative=None
    for scenario in SCENARIOS:
        for seed in seeds:
            env=generate_environment(seed, scenario, ticks, capture=(scenario=='nominal' and seed==seeds[0]))
            if scenario=='nominal' and seed==seeds[0]:
                representative=env
                np.savez_compressed(out/'representative_spectrum.npz',
                    fft_power=env['example']['fft_power'],iq=env['example']['wideband_iq'],
                    energy=env['energies'],occupancy=env['occupancy'].astype(np.uint8),
                    noise_power=env['noise_power'],fs=FS,dt=DT)
            for policy in POLICIES:
                m,trace=simulate_controller(env,policy,trace=(scenario=='nominal' and seed==seeds[0] and policy in ('fixed_period','adaptive')))
                metrics.append(m)
                print(f"{scenario:14s} seed={seed} {policy:16s} Pd={m['pd']:.3f} Pfa={m['pfa']:.3f} "
                      f"collision={m['primary_collision_ms']:.0f}ms goodput={m['goodput_kbps']:.2f}kb/s")
                if trace: write_csv(out/f'{policy}_trace.csv',trace)
    write_csv(out/'Lab3_spectrum_access_results.csv',metrics)
    return metrics


class blk(gr.sync_block):
    """GNU Radio display replay of the recorded causal, half-duplex controller.

    No hardware transmission; the controller is evaluated by simulate_controller.
    The spectrum display uses the very same complex samples used for detection.
    """
    def __init__(self, seed=2026):
        gr.sync_block.__init__(self,name='Closed-loop cognitive access',in_sig=None,
                            out_sig=[np.complex64,np.complex64,np.float32,np.float32,np.float32,np.float32])
        self.env=generate_environment(int(seed),'nominal',ticks=3000,store_iq=True)
        self.metrics,self.trace=simulate_controller(self.env,'adaptive',trace=True)
        self.rng=np.random.default_rng(seed+9000)
        self.tick_index=0
        self.offset=SNAPS*NFFT
        self.env_samples=np.zeros(SNAPS*NFFT,dtype=np.complex64)
        self.su_samples=np.zeros(SNAPS*NFFT,dtype=np.complex64)
        self.status=np.zeros(4,dtype=np.float32)

    def _next_tick(self):
        k=self.tick_index % len(self.trace)
        record=self.trace[k]
        tick=int(record['tick'])
        self.env_samples=self.env['iq'][tick]
        self.su_samples.fill(0)
        if record['tx']:
            ch=record['channel']
            bit_i=self.rng.integers(0,2,size=(SNAPS,BINS))
            bit_q=self.rng.integers(0,2,size=(SNAPS,BINS))
            symbols=((1-2*bit_i)+1j*(1-2*bit_q))/np.sqrt(2)
            frequency=np.zeros((SNAPS,NFFT),dtype=np.complex128)
            frequency[:,ch*BINS:(ch+1)*BINS]=symbols
            self.su_samples=np.fft.ifft(frequency,axis=-1,norm='ortho').astype(np.complex64).reshape(-1)
        self.status[:]=[record['channel'],record['sensing'],record['collision'],record['noise_est']]
        self.tick_index+=1
        self.offset=0

    def work(self,input_items,output_items):
        count=len(output_items[0]);done=0
        while done<count:
            if self.offset >= SNAPS*NFFT:self._next_tick()
            width=min(count-done,SNAPS*NFFT-self.offset)
            x=slice(self.offset,self.offset+width)
            o=slice(done,done+width)
            output_items[0][o]=self.env_samples[x]
            output_items[1][o]=self.su_samples[x]
            for j in range(4): output_items[j+2][o]=self.status[j]
            done+=width;self.offset+=width
        return count


if __name__=='__main__':
    ap=argparse.ArgumentParser(description='Reproduce Lab 3 closed-loop access study')
    ap.add_argument('--output',default='results')
    ap.add_argument('--ticks',type=int,default=3000)
    ap.add_argument('--seeds',type=int,nargs='+',default=[2026,2027,2028])
    a=ap.parse_args()
    run_study(a.output,a.ticks,tuple(a.seeds))
