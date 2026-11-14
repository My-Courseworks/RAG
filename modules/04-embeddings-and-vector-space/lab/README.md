# M4 lab

Runnable code for [M4 · Embeddings and Vector Space](../README.md).

## `run_matryoshka_recipe.py`: train a Matryoshka embedder and its control

Runs the recipe in [M4 · Worked: training a Matryoshka embedder yourself](../README.md#worked-training-a-matryoshka-embedder-yourself).

- **It runs the README's own code.** The script reads the Python blocks of Steps 1–7 out of `../README.md`
  and executes them as written, so the module's code and this lab can't drift apart. It adds only the data
  and the Step 8 comparison.
- **What it prints:** recall@1 at every width (32 → 384) for the untrained base model, the
  Matryoshka-trained model, and a control trained on the full width only.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python run_matryoshka_recipe.py                    # Natural Questions: 3,000 train / 500 test, ~4 min on a laptop CPU
python run_matryoshka_recipe.py --data synthetic   # smoke test only: proves the code runs
python run_matryoshka_recipe.py --seed 1           # another seed; run a few before trusting a gap
```

The first run downloads `all-MiniLM-L6-v2` (~90 MB) and, for `--data nq`, the Natural Questions pairs.

### Recorded result (2026-11-14, seed 0, the versions in `requirements.txt`)

```
base model, untrained        {32: 0.8, 64: 0.916, 128: 0.958, 256: 0.964, 384: 0.974}
Matryoshka (DIMS)            {32: 0.892, 64: 0.936, 128: 0.952, 256: 0.966, 384: 0.976}
control (dims=[384])         {32: 0.808, 64: 0.926, 128: 0.958, 256: 0.97, 384: 0.98}
```

What it shows and its caveats are in the module, under the Step 8 results table.

### `--data synthetic` result

After one epoch, both trained models score ~1.0 at every width, because every query names a unique
contract number. That run only proves the code executes end to end; it can't show the Matryoshka effect.
