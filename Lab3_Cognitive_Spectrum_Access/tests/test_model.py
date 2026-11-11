"""Deterministic tests of Lab 3 sensing, timing, and access causality."""
import unittest
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import Lab3_Cognitive_Access_epy_block_0 as m

class SpectrumAccessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env=m.generate_environment(seed=42,scenario='nominal',ticks=300,store_iq=True)

    def test_unitary_fft_energy_statistic(self):
        e=self.env
        t=m.CAL_TICKS+150
        Y=np.fft.fft(e['iq'][t].reshape(m.SNAPS,m.NFFT),axis=-1,norm='ortho')
        observed=np.mean(np.abs(Y.reshape(m.SNAPS,m.NB,m.BINS))**2,axis=(0,2))
        np.testing.assert_allclose(observed,e['energies'][t],rtol=2e-7,atol=2e-7)

    def test_calibration_precedes_control(self):
        _,trace=m.simulate_controller(self.env,'adaptive',trace=True)
        self.assertEqual(trace[0]['tick'],m.CAL_TICKS)
        self.assertTrue(np.all(~self.env['occupancy'][:m.CAL_TICKS+1]))

    def test_all_policies_mute_while_sensing(self):
        for p in m.POLICIES:
            stats,trace=m.simulate_controller(self.env,p,trace=True)
            self.assertTrue(all(not x['tx'] for x in trace if x['sensing']))
            self.assertTrue(all(not x['tx'] or x['channel']>=0 for x in trace))
            self.assertEqual(stats['collision_packets'],sum(x['collision'] for x in trace))
            self.assertAlmostEqual(stats['primary_collision_ms'],2*sum(x['collision'] for x in trace))

    def test_retune_guard(self):
        _,trace=m.simulate_controller(self.env,'adaptive',trace=True)
        for j,r in enumerate(trace[:-1]):
            if r['change'] and r['channel']>=0:
                self.assertEqual(r['tx'],0)
                self.assertEqual(trace[j+1]['tx'],0)

    def test_observed_detection_counts(self):
        for p in m.POLICIES:
            mtr,tr=m.simulate_controller(self.env,p,trace=True)
            self.assertEqual(mtr['sensed_primary']+mtr['sensed_vacant'],mtr['sensing_ticks']*m.NB)
            self.assertLessEqual(mtr['detected_primary'],mtr['sensed_primary'])
            self.assertLessEqual(mtr['false_alarms'],mtr['sensed_vacant'])

    def test_goodput_is_delivered_bits_per_time(self):
        mtr,tr=m.simulate_controller(self.env,'adaptive',trace=True)
        delivered=sum(r['payload_delivered_bits'] for r in tr)
        self.assertEqual(delivered,mtr['successful_packets']*m.PAYLOAD_BITS)
        self.assertAlmostEqual(delivered/(mtr['duration_s']*1000),mtr['goodput_kbps'])
        self.assertLessEqual(mtr['goodput_kbps'],m.PAYLOAD_BITS/m.DT/1000)

    def test_perfect_primary_detection_at_strong_snr_not_assumed(self):
        e=m.generate_environment(seed=77,scenario='weak_primary',ticks=300)
        v,_=m.simulate_controller(e,'adaptive')
        self.assertTrue(0<=v['pd']<=1)
        self.assertTrue(0<=v['pfa']<=1)

    def test_paired_replay_and_repeatability(self):
        a=m.generate_environment(seed=123,scenario='nominal',ticks=180)
        b=m.generate_environment(seed=123,scenario='nominal',ticks=180)
        np.testing.assert_array_equal(a['occupancy'],b['occupancy'])
        np.testing.assert_array_equal(a['energies'],b['energies'])
        for p in m.POLICIES:
            z,_=m.simulate_controller(a,p)
            zz,_=m.simulate_controller(b,p)
            for key in z:
                self.assertTrue((isinstance(z[key],float) and np.isnan(z[key]) and np.isnan(zz[key])) or z[key]==zz[key])

    def test_gnuradio_replay_samples_match_detector_waveform(self):
        # Headless stand-in checks source-block output alignment and silence.
        # Creating the block is intentionally independent of the real GR runtime.
        live=m.blk(seed=2026)
        count=m.SNAPS*m.NFFT*3
        o=[np.zeros(count,dtype=np.complex64),np.zeros(count,dtype=np.complex64)] + [
            np.zeros(count,dtype=np.float32) for _ in range(4)]
        self.assertEqual(live.work([],o),count)
        source_idx=m.CAL_TICKS
        np.testing.assert_array_equal(o[0][:m.SNAPS*m.NFFT],live.env['iq'][source_idx])
        self.assertEqual(o[3][0],1.0)  # first measured tick is sensing
        self.assertTrue(np.all(o[1][:m.SNAPS*m.NFFT]==0))

    def test_policy_never_uses_future_observations(self):
        a=m.generate_environment(seed=123,scenario='nominal',ticks=190)
        b={**a,'energies':a['energies'].copy(),'occupancy':a['occupancy'].copy()}
        b['energies'][140:]*=1000
        b['occupancy'][140:]=~b['occupancy'][140:]
        for p in m.POLICIES:
            _,tr_a=m.simulate_controller(a,p,trace=True)
            _,tr_b=m.simulate_controller(b,p,trace=True)
            for x,y in zip(tr_a,tr_b):
                if x['tick']<140:
                    self.assertEqual(x,y)

if __name__=='__main__': unittest.main()
