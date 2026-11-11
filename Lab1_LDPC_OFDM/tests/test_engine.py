"""Standard-library regression tests; no SDR or GNU Radio installation required."""
import csv
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from Lab1_Adaptive_LDPC_OFDM_epy_block_0 import blk


class EngineTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.cwd=patch('os.getcwd',return_value=self.tmp.name)
        # Engine uses Path.cwd(), change process CWD instead.
        import os
        self.original_cwd=Path.cwd()
        os.chdir(self.tmp.name)

    def tearDown(self):
        import os
        os.chdir(self.original_cwd)
        self.tmp.cleanup()

    def make(self,**kwargs):
        return blk(report_mode=False,run_label='unit_test',**kwargs)

    def test_encode_syndrome_three_rates_and_two_code_families(self):
        for family in ('random_eira','structured_eira'):
            engine=self.make(code_family=family)
            for rate in (0,1,2):
                engine._apply_settings(8.,-10.,rate,20)
                rng=np.random.default_rng(100+rate)
                info=rng.integers(0,2,engine.k_info,dtype=np.uint8)
                word=engine._encode(info)
                self.assertEqual(len(word),96+engine.m_checks)
                self.assertTrue(np.all((engine.H@word)%2==0))

    def test_adaptive_stops_really_and_matches_fixed_on_clean_codeword(self):
        engine=self.make()
        info=np.zeros(96,dtype=np.uint8)
        cw=engine._encode(info)
        llr=(1-2*cw.astype(int)).astype(float)*80
        f,a,ni,na,tf,ta=engine._decode_pair(llr,20)
        self.assertTrue(np.array_equal(f,cw))
        self.assertTrue(np.array_equal(a,cw))
        self.assertEqual(ni,20)
        self.assertLess(na,ni)
        self.assertGreater(tf,0)
        self.assertGreater(ta,0)

    def test_single_frame_perfect_and_pilot_modes(self):
        for mode in ('perfect','pilot_ls'):
            engine=self.make(csi_mode=mode)
            tx,rx,signal,metrics=engine.simulate_frame('unit',8.,-10.,0,8)
            self.assertEqual(len(tx),96)
            self.assertEqual(len(rx),96)
            self.assertEqual(len(signal),160+(80 if mode=='pilot_ls' else 0))
            self.assertTrue(np.isfinite(rx).all())
            self.assertLessEqual(metrics[5],metrics[4])
            with engine.csv_path.open() as f:
                row=list(csv.DictReader(f))[0]
            self.assertEqual(row['csi_mode'],mode)
            self.assertIn('decoder_time_adaptive_us',row)

    def test_quantized_decoder_runs_and_is_finite(self):
        for bits in (4,6,8):
            e=self.make(quant_bits=bits)
            result=e.simulate_frame('quant',8.,-10.,0,5)
            self.assertTrue(np.isfinite(np.array(result[-1])).all())
            self.assertTrue(e.report_plan)

    def test_run_paths_never_overwrite(self):
        a=self.make();b=self.make()
        self.assertNotEqual(a.csv_path,b.csv_path)
        self.assertTrue(a.csv_path.exists())
        self.assertTrue(b.csv_path.exists())

    def test_grc_embeds_same_engine(self):
        import yaml
        doc=yaml.safe_load((ROOT/'Lab1_Adaptive_LDPC_OFDM.grc').read_text())
        block=next(b for b in doc['blocks'] if b.get('id')=='epy_block')
        self.assertEqual(block['parameters']['_source_code'],(ROOT/'Lab1_Adaptive_LDPC_OFDM_epy_block_0.py').read_text())
        self.assertEqual(block['parameters']['extended_report'],'True')


if __name__=='__main__':
    unittest.main()
