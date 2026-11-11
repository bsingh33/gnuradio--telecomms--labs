"""Guard against mismatches between GNU Radio output ports and generated streams."""
import importlib.util
from pathlib import Path
import numpy as np

p=Path(__file__).resolve().parents[1]/"Lab2_MIMO_OFDM_epy_block_0.py"
spec=importlib.util.spec_from_file_location("lab2_gui_engine",p)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def test_all_nine_outputs_have_expected_types():
    e=m.blk(seed=2026)
    lengths=[8,20,52,71]
    for n in lengths:
        out=[np.empty(n,dtype=np.complex64) for _ in range(4)] + [np.empty(n,dtype=np.float32) for _ in range(5)]
        assert e.work([],out)==n
        assert len(out)==9
        for x in out:
            assert np.isfinite(x).all()
        for x in out[4:7]:
            assert ((x>=0)&(x<=1)).all()
        assert np.all(out[7]>=1)
        assert np.all(out[8]>=0)
