#!/usr/bin/env python3
"""Reproducible complex-baseband chirp receiver and interference experiments.

Signal model: 128-sample quadratic chirps, four-chirp known preamble,
12 chirp-encoded payload symbols, 9 data bytes and a CRC-8. This is a
laboratory framing convention, not a full LoRa PHY. No real RF is emitted.
"""
from __future__ import annotations
import argparse
import csv
from pathlib import Path
import numpy as np
from scipy.signal import correlate

try:
    from gnuradio import gr
except ImportError:
    class _GR:
        class sync_block:
            def __init__(self,*a,**kw): pass
    gr = _GR()

N = 128
FS = 125_000.0
PRE = 4
PAY_SYMBOLS = 12
DATA_BYTES = 9
PAY_BITS = 8*(DATA_BYTES+1)
FRAME = (PRE+PAY_SYMBOLS)*N
OBS = FRAME+192
SEARCH = OBS-FRAME+1
SNR_DB = -7.0
SEEDS=(2026,2027,2028)
JSR_LEVELS=(0.,5.,10.,15.)
KINDS=('barrage','sweep','pulsed','event_triggered')
METHODS=('raw','blank','spectral','hybrid')
TOL=2
W=64
# Thresholds are defined before the simulation and held fixed for all cases.
BLANK_MULT=2.7
SPECTRAL_RATIO=14.0
SPECTRAL_FRACTION=.18

n=np.arange(N)
UP=np.exp(1j*np.pi*(n*n/N-n))
PREAMBLE=np.tile(UP, PRE)
TONE_BANK=np.exp(2j*np.pi*np.outer(np.arange(N),n)/N)


def cn(rng, count):
    return (rng.standard_normal(count)+1j*rng.standard_normal(count))/np.sqrt(2)


def crc8(data):
    """CRC-8, polynomial 0x07, init=0, MSB-first, no reflection or xor-out."""
    crc=0
    for byte in data:
        crc ^= int(byte)
        for _ in range(8):
            crc=((crc<<1)^ (7 if (crc&128) else 0))&255
    return crc


def frame_from_bytes(data):
    b=np.asarray(data,dtype=np.uint8)
    if b.size != DATA_BYTES:raise ValueError('expected 9 payload bytes')
    packet=np.r_[b, np.uint8(crc8(b))]
    bit_stream=np.unpackbits(packet)
    padded=np.pad(bit_stream,(0, PAY_SYMBOLS*7-PAY_BITS))
    values=padded.reshape(PAY_SYMBOLS,7)@np.array([64,32,16,8,4,2,1])
    payload=np.concatenate([UP*TONE_BANK[int(k)] for k in values])
    return np.r_[PREAMBLE,payload],packet,values


def make_record(rng, snr_db=SNR_DB, present=True):
    tau=int(rng.integers(32,161))
    data=rng.integers(0,256,DATA_BYTES,dtype=np.uint8)
    tx,packet,values=frame_from_bytes(data)
    x=np.sqrt(10**(-snr_db/10))*cn(rng,OBS)
    if present:
        x[tau:tau+FRAME]+=tx*np.exp(1j*rng.uniform(-np.pi,np.pi))
    return x,tau,packet,values


def make_interference(rng,kind,tau=None):
    """One record-normalized waveform. Triggered case has simulator-known onset."""
    if kind=='none':return np.zeros(OBS,complex)
    ix=np.arange(OBS,dtype=float)
    if kind=='barrage': j=cn(rng,OBS)
    elif kind=='sweep':
        f0=rng.uniform(-.34,-.26)
        f1=rng.uniform(.26,.34)
        slope=(f1-f0)/(OBS-1)
        j=np.exp(2j*np.pi*(f0*ix+0.5*slope*ix*ix)+1j*rng.uniform(0,2*np.pi))
    elif kind=='pulsed':
        j=np.zeros(OBS,complex)
        duration=int(.20*OBS)
        onset=int(rng.integers(0,OBS-duration+1))
        j[onset:onset+duration]=cn(rng,duration)
    elif kind=='event_triggered':
        j=np.zeros(OBS,complex)
        if tau is None:return j
        onset=int(tau+N//2)
        duration=3*N
        j[onset:onset+duration]=cn(rng,duration)
    else:raise ValueError(kind)
    p=np.mean(np.abs(j)**2)
    return j/np.sqrt(p) if p>0 else j


def mitigate(x, method):
    x=np.asarray(x,dtype=np.complex128)
    if method not in METHODS:raise ValueError(method)
    if method=='raw':return x.copy(),dict(blanked=0,excised=0)
    y=x.copy()
    blanked=0; excised=0
    if method in ('blank','hybrid'):
        amp=np.abs(y)
        limit=BLANK_MULT*np.median(amp)
        mask=amp>limit
        blanked=int(np.count_nonzero(mask))
        y[mask]=0
    if method in ('spectral','hybrid'):
        assert y.size%W==0
        Y=np.fft.fft(y.reshape(-1,W),axis=1)
        power=np.abs(Y)**2
        med=np.median(power,axis=1)
        maxbin=np.argmax(power,axis=1)
        peak=power[np.arange(power.shape[0]),maxbin]
        flag=(peak>SPECTRAL_RATIO*np.maximum(med,1e-12)) & (peak>SPECTRAL_FRACTION*np.sum(power,axis=1))
        for row, k in enumerate(maxbin):
            if flag[row]:
                Y[row, np.array([k-1,k,k+1])%W]=0
                excised+=3
        y=np.fft.ifft(Y,axis=1).ravel()
    return y, dict(blanked=blanked,excised=excised)


def match_statistic(x):
    """Normalized known-preamble matched filter, all admissible packet onsets."""
    c=correlate(x,PREAMBLE,mode='valid',method='fft')[:SEARCH]
    en=np.r_[0,np.cumsum(np.abs(x)**2)]
    power=(en[PRE*N:PRE*N+SEARCH]-en[:SEARCH])
    return (np.abs(c)**2/(PRE*N*np.maximum(power,1e-12)))


def acquire(x,threshold):
    stat=match_statistic(x)
    t=int(np.argmax(stat))
    v=float(stat[t])
    return (t if v>threshold else -1),v


def decode(x,start):
    if start<0 or start+FRAME>len(x):return None,None,False
    begin=start+PRE*N
    chunks=x[begin:begin+PAY_SYMBOLS*N].reshape(PAY_SYMBOLS,N)
    spec=np.fft.fft(chunks*np.conj(UP),axis=1)
    syms=np.argmax(np.abs(spec)**2,axis=1)
    bits=((syms[:,None]>>np.arange(6,-1,-1))&1).astype(np.uint8).ravel()[:PAY_BITS]
    decoded=np.packbits(bits)
    crc_ok=bool(crc8(decoded[:-1])==int(decoded[-1]))
    return decoded,syms,crc_ok


def calibrate(ntrials=1200,seed=44123,pfa=.01):
    rng=np.random.default_rng(seed)
    values={m:np.empty(ntrials) for m in METHODS}
    for i in range(ntrials):
        x,_,_,_=make_record(rng,present=False)
        for m in METHODS:
            y,_=mitigate(x,m)
            values[m][i]=np.max(match_statistic(y))
    return {m:float(np.quantile(values[m],1-pfa)) for m in METHODS}


def process(x,method,threshold):
    y,stats=mitigate(x,method)
    t,peak=acquire(y,threshold)
    decoded,syms,crc_ok=decode(y,t)
    return dict(start=t,peak=peak,decoded=decoded,crc_ok=crc_ok,**stats)


def simulate(seed,kind,jsr_db,thresholds,nframes=160,nh0=35,snr_db=SNR_DB):
    rng=np.random.default_rng(int(seed))
    entries=[]
    records={m:dict(correct=0,miss=0,false_lock=0,packet_ok=0,crc_ok=0,crc_false_accept=0,
        bit_errors=0,bit_trials=0,timing=[],peaks=[],blanked=[],excised=[],false_alarm=0) for m in METHODS}
    for k in range(nframes):
        clean,tau,packet,_=make_record(rng,snr_db=snr_db)
        jammer=make_interference(rng,kind,tau)
        x=clean+np.sqrt(10**(jsr_db/10))*jammer if kind!='none' else clean
        for m in METHODS:
            r=process(x,m,thresholds[m]);s=records[m]
            correct=r['start']>=0 and abs(r['start']-tau)<=TOL
            if r['start']<0:s['miss']+=1
            elif correct:s['correct']+=1;s['timing'].append(abs(r['start']-tau))
            else:s['false_lock']+=1
            errors=''
            is_good=False
            if correct and r['decoded'] is not None:
                errors=int(np.count_nonzero(np.unpackbits(r['decoded'])[:DATA_BYTES*8] != np.unpackbits(packet)[:DATA_BYTES*8]))
                s['bit_errors']+=errors;s['bit_trials']+=DATA_BYTES*8
                is_good=bool(errors==0 and r['crc_ok'])
            if r['crc_ok']:
                s['crc_ok']+=1
                if not is_good:s['crc_false_accept']+=1
            if is_good:s['packet_ok']+=1
            s['peaks'].append(r['peak']);s['blanked'].append(r['blanked']);s['excised'].append(r['excised'])
            entries.append(dict(seed=seed,kind=kind,jsr_db=('' if kind=='none' else jsr_db),
               frame=k,method=m,true_start=tau,estimated_start=r['start'],
               outcome='correct' if correct else ('miss' if r['start']<0 else 'false_lock'),
               payload_bit_errors=errors,crc_ok=int(r['crc_ok']),packet_ok=int(is_good),
               peak=r['peak'],blanked_samples=r['blanked'],excised_bins=r['excised']))
    # Separate H0 realizations: no payload and no preamble. Triggered interference is silent.
    for _ in range(nh0):
        null,_,_,_=make_record(rng,snr_db=snr_db,present=False)
        j=make_interference(rng,kind,None)
        if kind!='none':null=null+np.sqrt(10**(jsr_db/10))*j
        for m in METHODS:
            r=process(null,m,thresholds[m])
            records[m]['false_alarm']+=int(r['start']>=0)
    rows=[]
    for m,r in records.items():
        rows.append(dict(seed=seed,kind=kind,jsr_db=('' if kind=='none' else jsr_db),snr_db=snr_db,
             method=m,frames=nframes,absent_trials=nh0,correct=r['correct'],miss=r['miss'],false_lock=r['false_lock'],
             packet_ok=r['packet_ok'],crc_ok=r['crc_ok'],crc_false_accept=r['crc_false_accept'],
             bit_errors_on_correct_locks=r['bit_errors'],bit_trials_on_correct_locks=r['bit_trials'],
             false_alarms=r['false_alarm'],p_correct=r['correct']/nframes,
             p_false_lock=r['false_lock']/nframes,p_packet=r['packet_ok']/nframes,
             ber_given_correct_lock=r['bit_errors']/r['bit_trials'] if r['bit_trials'] else '',
             pfa_h0=r['false_alarm']/nh0,mean_abs_timing_given_correct=float(np.mean(r['timing'])) if r['timing'] else '',
             mean_peak=float(np.mean(r['peaks'])),mean_blanked=float(np.mean(r['blanked'])),
             mean_excised_bins=float(np.mean(r['excised']))))
    return entries, rows


def write_csv(path,rows):
    if not rows:return
    with Path(path).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def experiment(out,nframes=160,nh0=35,seeds=SEEDS,calibration=1200):
    dest=Path(out);dest.mkdir(exist_ok=True,parents=True)
    thresholds=calibrate(ntrials=calibration)
    plan=[('none',0.)]+[(k,jsr) for k in KINDS for jsr in JSR_LEVELS]
    all_details=[];summary=[]
    for seed in seeds:
        for kind,jsr in plan:
            detail,aggregate=simulate(seed,kind,jsr,thresholds,nframes,nh0)
            all_details+=detail;summary+=aggregate
            print('seed',seed,'condition',kind,jsr,'packet ok',[(r['method'],r['packet_ok']) for r in aggregate],flush=True)
    write_csv(dest/'Lab4_receiver_results.csv',summary)
    write_csv(dest/'Lab4_packet_trials.csv',all_details)
    write_csv(dest/'Lab4_thresholds.csv',[dict(method=m,threshold=thresholds[m],noise_only_trials=calibration,
       target_pfa=.01,seed=44123) for m in METHODS])
    return summary


class blk(gr.sync_block):
    """Display one paired synthetic frame and the four end-to-end CRC outcomes."""
    def __init__(self,snr_db=-7.,jsr_db=10.,kind=2,seed=2026):
        gr.sync_block.__init__(self,name='Chirp Packet Receiver',in_sig=None,
            out_sig=[np.complex64,np.float32,np.float32,np.float32,np.float32,np.float32])
        self.rng=np.random.default_rng(int(seed));self.snr_db=float(snr_db);self.jsr_db=float(jsr_db)
        self.kind=int(kind)%4
        self.thresholds=calibrate(ntrials=500)
        self.samples=np.zeros(OBS,np.complex64);self.val=np.zeros(5,np.float32);self.offset=OBS
    def new_frame(self):
        clean,tau,packet,_=make_record(self.rng,self.snr_db)
        j=make_interference(self.rng,KINDS[self.kind],tau)
        x=clean+np.sqrt(10**(self.jsr_db/10))*j
        flags=[]
        for m in METHODS:
            r=process(x,m,self.thresholds[m])
            good=r['start']>=0 and abs(r['start']-tau)<=TOL and r['decoded'] is not None and np.array_equal(r['decoded'],packet)
            flags.append(float(good))
        self.samples=x.astype(np.complex64);self.val=np.array([float(tau),*flags],dtype=np.float32)
        self.offset=0
    def work(self,input_items,output_items):
        todo=len(output_items[0]);done=0
        while done<todo:
            if self.offset>=OBS:self.new_frame()
            take=min(todo-done,OBS-self.offset)
            output_items[0][done:done+take]=self.samples[self.offset:self.offset+take]
            for m in range(5):output_items[m+1][done:done+take]=self.val[m]
            self.offset+=take;done+=take
        return todo

if __name__=='__main__':
    parser=argparse.ArgumentParser(description='End-to-end simulated chirp receiver under interference')
    parser.add_argument('--out',default='results')
    parser.add_argument('--frames',type=int,default=160)
    parser.add_argument('--h0',type=int,default=35)
    parser.add_argument('--calibration',type=int,default=1200)
    parser.add_argument('--seeds',nargs='+',type=int,default=list(SEEDS))
    args=parser.parse_args()
    experiment(args.out,args.frames,args.h0,args.seeds,args.calibration)
