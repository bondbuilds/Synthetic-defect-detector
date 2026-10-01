'''Automated check that generator.py works 
to be run with pytest - automatically finds files named test_*.py and runs every function which has a name tsrting with test_
if all 'Asserts' are true then test passes'''

import csv
from generate import generate_dataset


def test_dataset_is_balanced_and_complete(tmp_path):
    generate_dataset(tmp_path, n=20, seed=0)
    with open(tmp_path / "labels.csv") as f:
    #generate small dataset in temp folder and read every row.
    rows = list(csv.Dictreader(f))
    #rows to dict keyed by column header
    assert len(rows) == 20
    assert sum(int(r["label"]) for r in rows)
    assert all((tmp_path / "images" / r["filename"]).exists() for r in rows)