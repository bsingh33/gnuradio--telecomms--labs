"""Signal-model and archived-results checks for Lab 4.
Run with: python -m unittest discover -s tests -v
"""
import sys,csv,unittest
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import Lab4_Chirp_Receiver_epy_block_0 as s

class TestReceiver(unittest.TestCase):
    def test_crc_reference(self):
        self.assertEqual(s.crc8(b'123456789'),0xF4)
    def test_chirp_unit_magnitude(self):
        self.assertTrue(np.allclose(np.abs(s.UP),1))
    def test_symbol_mapping_all_128_bins(self):
        c=s.UP[None,:]*s.TONE_BANK
        peaks=np.argmax(np.abs(np.fft.fft(c*np.conj(s.UP),axis=1))**2,axis=1)
        self.assertTrue(np.array_equal(peaks,np.arange(128)))
    def test_exact_frame_recovery(self):
        r=np.random.default_rng(69)
        x,tau,pkt,_=s.make_record(r,snr_db=50)
        rcv=s.process(x,'raw',.012)
        self.assertEqual(rcv['start'],tau)
        self.assertTrue(rcv['crc_ok'])
        self.assertTrue(np.array_equal(rcv['decoded'],pkt))
    def test_unknown_phase(self):
        r=np.random.default_rng(93)
        data=np.frombuffer(b'HELLO SDR',dtype=np.uint8)
        frame,pkt,_=s.frame_from_bytes(data)
        x=np.zeros(s.OBS,dtype=complex)
        tau=57
        x[tau:tau+s.FRAME]=frame*np.exp(1j*2.4)
        t,_=s.acquire(x,.01)
        decoded,_,crc=s.decode(x,t)
        self.assertEqual(t,tau);self.assertTrue(crc)
        self.assertTrue(np.array_equal(decoded,pkt))
    def test_interference_normalization(self):
        for kind in s.KINDS:
            j=s.make_interference(np.random.default_rng(6),kind,tau=45)
            self.assertAlmostEqual(float(np.mean(np.abs(j)**2)),1.,places=11)
    def test_triggered_null_silent(self):
        j=s.make_interference(np.random.default_rng(4),'event_triggered',None)
        self.assertEqual(float(np.sum(np.abs(j)**2)),0.)
    def test_blanking_clips_large_impulse(self):
        x=np.ones(s.OBS,dtype=complex);x[100]=500
        y,stats=s.mitigate(x,'blank')
        self.assertEqual(y[100],0);self.assertEqual(stats['blanked'],1)
    def test_no_excision_without_outlier(self):
        x=np.zeros(s.OBS,dtype=complex)
        for m in s.METHODS:
            y,stats=s.mitigate(x,m)
            self.assertTrue(np.allclose(y,0))
            self.assertEqual(stats['excised'],0)
    def test_reproducible_frame(self):
        one=s.make_record(np.random.default_rng(77))
        two=s.make_record(np.random.default_rng(77))
        for a,b in zip(one,two):self.assertTrue(np.array_equal(a,b))
    def test_no_leaked_true_start(self):
        # Process only the supplied IQ vector and a fixed threshold.
        x,tau,_,_=s.make_record(np.random.default_rng(100),snr_db=45)
        for m in s.METHODS:
            self.assertEqual(s.process(x,m,.01)['start'],tau)
    def test_archived_counts(self):
        with open(ROOT/'results/Lab4_receiver_results.csv') as f:rows=list(csv.DictReader(f))
        self.assertEqual(len(rows),3*(1+4*4)*len(s.METHODS))
        self.assertEqual({int(r['frames']) for r in rows},{100})
        self.assertEqual({int(r['absent_trials']) for r in rows},{40})
        for r in rows:
            self.assertEqual(int(r['correct'])+int(r['miss'])+int(r['false_lock']),int(r['frames']))
            self.assertLessEqual(int(r['packet_ok']),int(r['correct']))
    def test_trial_csv_reconstructs_report(self):
        with open(ROOT/'results/Lab4_receiver_results.csv') as f:summary=list(csv.DictReader(f))
        with open(ROOT/'results/Lab4_packet_trials.csv') as f:trial=list(csv.DictReader(f))
        lookup={}
        for r in trial:
            key=(r['seed'],r['kind'],r['jsr_db'],r['method'])
            if key not in lookup:lookup[key]=[0,0,0,0,0]
            z=lookup[key];z[0]+=1
            z[1]+=int(r['packet_ok']);z[2]+=int(r['outcome']=='correct')
            z[3]+=int(r['outcome']=='miss');z[4]+=int(r['outcome']=='false_lock')
        for r in summary:
            key=(r['seed'],r['kind'],r['jsr_db'],r['method'])
            self.assertEqual(lookup[key],list(map(int,[r['frames'],r['packet_ok'],r['correct'],r['miss'],r['false_lock']])))

if __name__=='__main__':unittest.main()
