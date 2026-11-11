"""Independent numerical checks for Lab 2."""
import importlib.util
import numpy as np
from pathlib import Path
src=Path(__file__).resolve().parents[1]/'Lab2_MIMO_OFDM_epy_block_0.py'
spec=importlib.util.spec_from_file_location('lab2',src)
m=importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

def test_qpsk():
    b=np.array([[[0,0],[0,1],[1,0],[1,1]]],dtype=np.uint8)
    assert np.array_equal(m.hard_bits(m.qpsk(b)),b)
    assert np.allclose(np.abs(m.qpsk(b)),1)

def test_ofdm_convolution():
    rng=np.random.default_rng(99)
    t=m.channel_taps(rng,30,.6)
    xf=np.zeros((30,2,m.NFFT),complex)
    xf[:,:,m.ACTIVE]=m.cn(rng,(30,2,m.NC))
    y,_=m.time_channel(xf,t,0,rng)
    h=m.channel_freq(t)
    expected=np.einsum('bkrt,bkt->bkr',h,np.transpose(xf[:,:,m.ACTIVE],(0,2,1)))
    assert np.max(np.abs(y-expected))<1e-12

def test_zf_noise_free():
    rng=np.random.default_rng(7)
    h=m.channel_freq(m.channel_taps(rng,30,.8))
    x=m.qpsk(rng.integers(0,2,(30,m.NC,2,2)))
    y=np.einsum('bkrt,bkt->bkr',h,x)
    z=m.solve_detectors(h,y,1e-12)
    for a in z.values():
        assert np.array_equal(m.hard_bits(a),m.hard_bits(x))

def test_estimation_error_and_correlation():
    rng=np.random.default_rng(26)
    b,tx,y,h,he,w,n0=m.generate_batch(rng,800,12,.6,20)
    nmse=np.sum(np.abs(he-h)**2)/np.sum(np.abs(h)**2)
    assert .008 < nmse < .012, nmse
    z=h[:,0,0,0]; w=h[:,0,0,1]; v=h[:,0,1,0]
    txcorr=np.real(np.vdot(z,w))/np.sqrt(np.vdot(z,z).real*np.vdot(w,w).real)
    rxcorr=np.real(np.vdot(z,v))/np.sqrt(np.vdot(z,z).real*np.vdot(v,v).real)
    assert abs(txcorr-.6)<.09,(txcorr,rxcorr)
    assert abs(rxcorr-.6)<.09,(txcorr,rxcorr)

def test_shared_conditions():
    a=m.run_point('test',12,.6,None,123,frames=45,batch=15)[0]
    b=m.run_point('test',12,.6,20,123,frames=45,batch=15)[0]
    assert a[0]['median_cond']==b[0]['median_cond']
    assert a[0]['nmse']==0 and b[0]['nmse']>0

def test_reproducibility():
    x=m.run_point('test',12,.6,20,123,frames=45,batch=15)[0]
    y=m.run_point('test',12,.6,20,123,frames=45,batch=15)[0]
    assert x==y

if __name__=='__main__':
    for n,fn in sorted(globals().items()):
        if n.startswith('test_'):
            fn();print('PASS',n)
