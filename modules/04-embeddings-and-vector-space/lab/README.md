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


## `run_threshold_distributions.py`: score duplicates, hard negatives and random pairs on real models

Measures the table in [M4 · Worked: setting a similarity threshold from two distributions](../README.md#worked-setting-a-similarity-threshold-from-two-distributions).

- **Chunks:** 200 clause-shaped LogiTrack chunks from 16 templates inside the script (surcharges, free time,
  claims, payment, reefer set points, …), each with a source header line (`title | filename | p. N`).
  It doesn't use [`labs/generate_corpus.py`](../../../labs/generate_corpus.py): that generator runs, but
  about 94% of its words are filler drawn from a 30-word vocabulary, and its clause text comes from four
  sentence templates.
- **Pairs, built from the chunks (seeded):**
  - **120 duplicates**, never byte copies, in three kinds:
    - `reupload`: a new filename and page number in the header.
    - `other_carrier`: the same clause under another carrier's agreement header.
    - `reformatted`: line breaks moved, `’`/`–` normalized to `'`/`-`, the clause title sometimes upper-cased.
  - **120 hard negatives:** exactly one figure changed (a percentage, a USD amount, a day count or the
    day of a date). Every other byte is identical.
  - **1,000 random pairs:** two different chunks.
- **Models:** local CPU only, cosine on L2-normalized embeddings, prefix and pooling as each card says.
  - `intfloat/e5-base-v2`, mean pooling, prefix `query: `. Its [card](https://huggingface.co/intfloat/e5-base-v2)
    says *"Use "query: " prefix for symmetric tasks such as semantic similarity, paraphrase retrieval"*, and
    under *"Why does the cosine similarity scores distribute around 0.7 to 1.0?"*: *"This is a known and
    expected behavior as we use a low temperature 0.01 for InfoNCE contrastive loss."*
  - `BAAI/bge-base-en-v1.5`, CLS pooling, no instruction. Its [card](https://huggingface.co/BAAI/bge-base-en-v1.5)
    says the query instruction is only for short-query retrieval: *"in other cases, no instruction is
    needed, just use the original query directly. In all cases, **no instruction** needs to be added to
    passages."* It also says v1.5 was released *"to alleviate the issue of the similarity distribution"*.
    Its FAQ, which recommends v1.5 over the earlier models, says BGE scores are *"about in the interval [0.6, 1]"*.
  - `bert-base-uncased`, mean pooling: never trained for similarity, so it shows the raw cone.
  - **Opt-in, paid:** `ada002`, `te3small` and `te3large` (`text-embedding-ada-002`, `-3-small`, `-3-large`).
    They call the OpenAI embeddings API, need a key and cost money. A seed has 440 unique texts, about 45,000
    tokens per model, so about $0.011 per seed for all three at $0.10 / $0.02 / $0.13 per 1M tokens (an
    estimate, counted with BERT's tokenizer). The default `--models` never includes them.
    - The key comes from `OPENAI_API_KEY`, or else from `~/.config/openai_key` (a bare key on one line).
      The script never prints it, and errors show only the HTTP status and error code.
    - Embeddings are cached in `~/.cache/rag-course/m4-openai-embeddings.sqlite` (`--cache`), keyed by model
      and text hash, so reruns don't re-bill. The run prints the tokens billed and the cost.
    - If a model is retired (HTTP 404 or `model_not_found`), the run prints `UNAVAILABLE` and carries on.
- **What it prints, per model:**
  - the percentiles from the module's table;
  - the gap (dup p5 − hard-negative p99), and the share of hard negatives above the duplicates' median;
  - three cutoffs, each with the dups caught and hard negatives dropped:
    - the module's rule: the lowest value with ≤ 1% of hard negatives above it;
    - the value that maximizes caught − dropped;
    - a naive cutoff at the duplicates' 5th percentile, which ignores the hard negatives.
  - the medians by pair kind, and the carry-over: each model's cutoff applied to the other models' scores.
  A pair counts as caught or dropped when its score is strictly above the cutoff.

```bash
python run_threshold_distributions.py                       # seed 0, all three models, ~15 s after downloads
python run_threshold_distributions.py --seed 1              # another corpus and pair draw
python run_threshold_distributions.py --models e5 bge --examples   # also print one pair of each kind
python run_threshold_distributions.py --models e5 bge ada002 te3small te3large   # paid: needs an OpenAI key
```

The first run downloads the three open models from the Hugging Face Hub (about 440 MB each). Without the
opt-in models, no embedding API is called.

### Recorded result (2026-11-14, seed 0, e5 / BGE / BERT, the versions in `requirements.txt`)

Open models only. The run with the OpenAI models is recorded in the next section.

```
seed 0 | 200 chunks | duplicates 120 | hard_negatives 120 | random 1000 | torch 2.15.0 | transformers 5.23.0
scored e5       intfloat/e5-base-v2 @ f52bf8ec8c
scored bge      BAAI/bge-base-en-v1.5 @ a5beb1e3e6
scored bert     bert-base-uncased @ 86b5e09344

                                                e5       bge      bert
random pairs, p5                          0.814443  0.582078  0.843227
random pairs, median                      0.864531  0.658290  0.913449
hard negatives, median                    0.997324  0.999806  0.999875
hard negatives, p99                       0.999984  0.999986  0.999967
duplicates, p5                            0.971962  0.930884  0.988810
duplicates, median                        0.999192  0.997516  0.998143
gap (dup p5 - hardneg p99)               -0.028022 -0.069102 -0.011157
clean gap (dup min > hardneg max)?              no        no        no
hard negs above dup median                   24.2%     98.3%    100.0%
CUTOFF (<=1% hard negs above it)          0.999985  0.999986  0.999968
  dups caught                                 0.0%      0.0%      0.0%
  hard negs dropped                           0.8%      0.8%      0.8%
balance cutoff (max caught - dropped)     0.999082  0.999987  0.999974
  dups caught                                54.2%      0.0%      0.0%
  hard negs dropped                          25.8%      0.0%      0.0%
naive cutoff (dup p5, no hard negs)       0.971962  0.930884  0.988810
  dups caught                                95.0%     95.0%     95.0%
  hard negs dropped                         100.0%    100.0%    100.0%

Duplicates, median score by kind:
  other_carrier  n=43         e5 0.978285      bge 0.964368     bert 0.993441
  reformatted    n=37         e5 0.999368      bge 0.999829     bert 0.999528
  reupload       n=40         e5 0.999266      bge 0.997898     bert 0.998501

Hard negatives, median score by kind:
  date           n=17         e5 0.999961      bge 0.999908     bert 0.999892
  days           n=36         e5 0.996360      bge 0.999530     bert 0.999896
  pct            n=43         e5 0.997280      bge 0.999891     bert 0.999853
  usd            n=24         e5 0.994044      bge 0.999597     bert 0.999868

Carry-over of the CUTOFF: cutoff from one model applied to another's scores
  (from cutoff -> to: dups caught / hard negs dropped)
  e5       0.999985 -> bge     :   0.0% /   1.7%
  e5       0.999985 -> bert    :   0.0% /   0.0%
  bge      0.999986 -> e5      :   0.0% /   0.8%
  bge      0.999986 -> bert    :   0.0% /   0.0%
  bert     0.999968 -> e5      :   0.0% /   5.0%
  bert     0.999968 -> bge     :   0.8% /  15.0%

Carry-over of the naive cutoff: cutoff from one model applied to another's scores
  (from cutoff -> to: dups caught / hard negs dropped)
  e5       0.971962 -> bge     :  77.5% / 100.0%
  e5       0.971962 -> bert    : 100.0% / 100.0%
  bge      0.930884 -> e5      : 100.0% / 100.0%
  bge      0.930884 -> bert    : 100.0% / 100.0%
  bert     0.988810 -> e5      :  65.8% /  99.2%
  bert     0.988810 -> bge     :  65.0% / 100.0%
```

**Seeds 1 and 2** (other chunks, other pairs) move each score statistic by less than 0.02:

| | e5 | BGE | BERT |
|---|---|---|---|
| random pairs, median | 0.865 / 0.864 / 0.864 | 0.658 / 0.657 / 0.662 | 0.913 / 0.910 / 0.915 |
| hard negatives, p99 | 0.99998 / 0.99998 / 0.99998 | 0.99999 / 0.99997 / 0.99998 | 0.99997 / 0.99997 / 0.99996 |
| duplicates, p5 | 0.972 / 0.974 / 0.974 | 0.931 / 0.941 / 0.950 | 0.989 / 0.989 / 0.990 |
| duplicates, median | 0.9992 / 0.9992 / 0.9993 | 0.9975 / 0.9981 / 0.9978 | 0.9981 / 0.9984 / 0.9982 |
| hard negs above dup median | 24% / 22.5% / 19% | 98% / 88% / 97% | 100% / 100% / 100% |
| dups caught at the module's cutoff | 0% / 0% / 0% | 0% / 0.8% / 0% | 0% / 0% / 0% |
| e5's naive cutoff → BGE: dups caught | | 77.5% / 78.3% / 77.5% | |

(seed 0 / 1 / 2)

**What it shows:**
- **No cosine cutoff separates the two sets, on any of the three models.** One changed figure in a
  ~70-word clause lowers the score by a median of under 0.003 on e5 and under 0.0002 on BGE and BERT.
  Swapping the carrier in the header (`other_carrier`) lowers it more: by about 0.007 (BERT), 0.02 (e5)
  and 0.035–0.04 (BGE). On BGE and BERT, 88–100% of hard negatives score above the duplicates' median.
  On every model, a cutoff that keeps 99% of the hard negatives catches no duplicates. This is the
  module's step-4 case: a cosine score alone can't do this dedup, so add the exact check on numbers
  and identifiers.
- **e5's scores are compressed, as its card says.** Its random pairs run 0.81–0.86 (5th percentile to
  median), against 0.58–0.66 for BGE. Raw BERT is higher still (0.84–0.92): the cone.
- **Carrying a cutoff over still breaks.** The naive cutoff (the duplicates' 5th percentile) is 0.972 on e5.
  Applied to BGE it catches 78% of duplicates, not 95%.
- **e5 reacts most to a changed figure.** Its hard negatives sit at a median of 0.997,
  below its `reupload` and `reformatted` duplicates (0.999). That's why e5 is the only model whose best
  caught − dropped cutoff catches anything (54%), and that cutoff still drops 26% of the hard negatives.

### Recorded result with OpenAI models (2026-11-14, seed 0, e5 / BGE / ada-002 / 3-small / 3-large)

`text-embedding-ada-002` still answered on 2026-11-14. Seeds 0, 1 and 2 cost $0.0117 each (about 140k
tokens), $0.035 in all.

```
seed 0 | 200 chunks | duplicates 120 | hard_negatives 120 | random 1000 | torch 2.15.0 | transformers 5.23.0
scored e5       intfloat/e5-base-v2 @ f52bf8ec8c
scored bge      BAAI/bge-base-en-v1.5 @ a5beb1e3e6
scored ada002   text-embedding-ada-002 @ api (0 cached, 440 sent)
scored te3small text-embedding-3-small @ api (0 cached, 440 sent)
scored te3large text-embedding-3-large @ api (0 cached, 440 sent)

                                                e5       bge    ada002  te3small  te3large
random pairs, p5                          0.814443  0.582078  0.786833  0.374706  0.393499
random pairs, median                      0.864531  0.658290  0.870610  0.619010  0.601370
hard negatives, median                    0.997324  0.999806  0.999680  0.999423  0.997918
hard negatives, p99                       0.999984  0.999986  0.999985  0.999979  0.999912
duplicates, p5                            0.971962  0.930884  0.953521  0.841745  0.880172
duplicates, median                        0.999192  0.997516  0.997484  0.988393  0.988940
gap (dup p5 - hardneg p99)               -0.028022 -0.069102 -0.046464 -0.158234 -0.119739
clean gap (dup min > hardneg max)?              no        no        no        no        no
hard negs above dup median                   24.2%     98.3%     99.2%    100.0%     94.2%
CUTOFF (<=1% hard negs above it)          0.999985  0.999986  0.999985  0.999980  0.999917
  dups caught                                 0.0%      0.0%      0.0%      0.0%      0.0%
  hard negs dropped                           0.8%      0.8%      0.8%      0.8%      0.8%
balance cutoff (max caught - dropped)     0.999082  0.999987  0.999986  0.999987  0.999944
  dups caught                                54.2%      0.0%      0.0%      0.0%      0.0%
  hard negs dropped                          25.8%      0.0%      0.0%      0.0%      0.0%
naive cutoff (dup p5, no hard negs)       0.971962  0.930884  0.953521  0.841745  0.880172
  dups caught                                95.0%     95.0%     95.0%     95.0%     95.0%
  hard negs dropped                         100.0%    100.0%    100.0%    100.0%    100.0%

Duplicates, median score by kind:
  other_carrier  n=43         e5 0.978285      bge 0.964368   ada002 0.965267 te3small 0.891912 te3large 0.914806
  reformatted    n=37         e5 0.999368      bge 0.999829   ada002 0.998370 te3small 0.993612 te3large 0.993185
  reupload       n=40         e5 0.999266      bge 0.997898   ada002 0.998813 te3small 0.996506 te3large 0.993170

Hard negatives, median score by kind:
  date           n=17         e5 0.999961      bge 0.999908   ada002 0.999836 te3small 0.999746 te3large 0.999287
  days           n=36         e5 0.996360      bge 0.999530   ada002 0.999327 te3small 0.999139 te3large 0.997247
  pct            n=43         e5 0.997280      bge 0.999891   ada002 0.999811 te3small 0.999752 te3large 0.998170
  usd            n=24         e5 0.994044      bge 0.999597   ada002 0.999669 te3small 0.999367 te3large 0.996402

Carry-over of the CUTOFF: cutoff from one model applied to another's scores
  (from cutoff -> to: dups caught / hard negs dropped)
  e5       0.999985 -> bge     :   0.0% /   1.7%
  e5       0.999985 -> ada002  :   0.0% /   2.5%
  e5       0.999985 -> te3small:   0.0% /   0.8%
  e5       0.999985 -> te3large:   0.0% /   0.0%
  bge      0.999986 -> e5      :   0.0% /   0.8%
  bge      0.999986 -> ada002  :   0.0% /   0.0%
  bge      0.999986 -> te3small:   0.0% /   0.8%
  bge      0.999986 -> te3large:   0.0% /   0.0%
  ada002   0.999985 -> e5      :   0.0% /   0.8%
  ada002   0.999985 -> bge     :   0.0% /   1.7%
  ada002   0.999985 -> te3small:   0.0% /   0.8%
  ada002   0.999985 -> te3large:   0.0% /   0.0%
  te3small 0.999980 -> e5      :   0.0% /   4.2%
  te3small 0.999980 -> bge     :   0.0% /   5.8%
  te3small 0.999980 -> ada002  :   0.0% /   4.2%
  te3small 0.999980 -> te3large:   0.0% /   0.0%
  te3large 0.999917 -> e5      :   1.7% /  14.2%
  te3large 0.999917 -> bge     :   2.5% /  29.2%
  te3large 0.999917 -> ada002  :   0.0% /  15.0%
  te3large 0.999917 -> te3small:   0.0% /  10.8%

Carry-over of the naive cutoff: cutoff from one model applied to another's scores
  (from cutoff -> to: dups caught / hard negs dropped)
  e5       0.971962 -> bge     :  77.5% / 100.0%
  e5       0.971962 -> ada002  :  69.2% / 100.0%
  e5       0.971962 -> te3small:  62.5% / 100.0%
  e5       0.971962 -> te3large:  61.7% / 100.0%
  bge      0.930884 -> e5      : 100.0% / 100.0%
  bge      0.930884 -> ada002  : 100.0% / 100.0%
  bge      0.930884 -> te3small:  65.0% / 100.0%
  bge      0.930884 -> te3large:  73.3% / 100.0%
  ada002   0.953521 -> e5      : 100.0% / 100.0%
  ada002   0.953521 -> bge     :  85.8% / 100.0%
  ada002   0.953521 -> te3small:  64.2% / 100.0%
  ada002   0.953521 -> te3large:  65.0% / 100.0%
  te3small 0.841745 -> e5      : 100.0% / 100.0%
  te3small 0.841745 -> bge     : 100.0% / 100.0%
  te3small 0.841745 -> ada002  : 100.0% / 100.0%
  te3small 0.841745 -> te3large:  99.2% / 100.0%
  te3large 0.880172 -> e5      : 100.0% / 100.0%
  te3large 0.880172 -> bge     : 100.0% / 100.0%
  te3large 0.880172 -> ada002  : 100.0% / 100.0%
  te3large 0.880172 -> te3small:  85.8% / 100.0%

OpenAI tokens billed this run (cached texts cost nothing):
  ada002     46,744 tokens  $0.00467
  te3small   46,744 tokens  $0.00093
  te3large   46,744 tokens  $0.00608
  total     140,232 tokens  $0.01169
```

**Seeds 1 and 2** don't change the conclusion:
- There's no clean gap on any model or seed.
- The share of hard negatives above the duplicates' median is ada-002 99 / 98 / 99%, 3-large 94 / 92 / 92%,
  and 3-small 100% on every seed.
- ada-002's by-eye cutoff (its duplicates' 5th percentile) carried over to 3-large catches 65.0 / 70.0 / 71.7%
  of duplicates. Carried over to BGE it catches 85.8 / 89.2 / 93.3%. It drops 100% of hard negatives in both
  cases.
