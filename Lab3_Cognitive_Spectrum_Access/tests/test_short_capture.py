"""Capture must also exist in a valid, shortened exploratory study."""
import importlib.util
from pathlib import Path
import tempfile
import numpy as np
p=Path(__file__).resolve().parents[1]/"Lab3_Cognitive_Access_epy_block_0.py"
spec=importlib.util.spec_from_file_location("lab3_capture_engine",p)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def test_short_run_capture():
    env=m.generate_environment(seed=2026,scenario="nominal",ticks=140,capture=True)
    assert env["example"] is not None
    assert np.isfinite(env["example"]["fft_power"]).all()
    with tempfile.TemporaryDirectory() as temp:
        results=m.run_study(temp,ticks=140,seeds=(2026,))
        assert len(results)==9
        assert (Path(temp)/"representative_spectrum.npz").exists()
