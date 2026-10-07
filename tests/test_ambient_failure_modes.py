"""Regression checks for section-location reproducibility."""
import os
import subprocess
import sys


def test_stem_cutoff_is_independent_of_hash_seed():
    code = '''
from llm_distract.ambient.failure_modes import _stems
import json
row = {"distractor_topic": "nasal polyp surgery", "assigned_content":
       "softball team polyps removed finally breathing through nose"}
print(json.dumps(_stems(row)))
'''
    outputs = [subprocess.check_output([sys.executable, "-c", code],
               env={**os.environ, "PYTHONHASHSEED": str(seed)}, text=True) for seed in [1, 7, 99]]
    assert len(set(outputs)) == 1
