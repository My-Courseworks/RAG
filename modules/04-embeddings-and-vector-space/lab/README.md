# M4 lab

Runnable code for [M4 · Embeddings and Vector Space](../README.md).

## `run_matryoshka_recipe.py`: train a Matryoshka embedder and its control

Runs the recipe in [M4 · Worked: training a Matryoshka embedder yourself](../README.md#worked-training-a-matryoshka-embedder-yourself).

- **It runs the README's own code.** The script reads the Python blocks of Steps 1–7 out of `../README.md`
  and executes them as written, so the module's code and this lab can't drift apart. It adds only the data
  and the Step 8 comparison.
- **What it prints:** recall@1 at every width (32 → 384) for the untrained base model, the
  Matryoshka-trained model, and a control trained on the full width only. Then a second table that adds a
  PCA baseline for each of the three models (below).
- **Seeded:** `--seed` picks the data sample and also fixes the training batch order and dropout. Both
  trained models start from the same order, so a seed always gives the same output.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python run_matryoshka_recipe.py                    # Natural Questions: 3,000 train / 500 test, ~4 min on a laptop CPU
python run_matryoshka_recipe.py --data synthetic   # smoke test only: proves the code runs
python run_matryoshka_recipe.py --seed 1           # another seed; run a few before trusting a gap
```

The first run downloads `all-MiniLM-L6-v2` (~90 MB) and, for `--data nq`, the Natural Questions pairs.

### Recorded result (2026-11-14, seed 0, the versions in `requirements.txt`)

The first recorded run, which the module's Step 8 table shows. It came from the script before seeding was
added, so the training batch order was random:

```
base model, untrained        {32: 0.8, 64: 0.916, 128: 0.958, 256: 0.964, 384: 0.974}
Matryoshka (DIMS)            {32: 0.892, 64: 0.936, 128: 0.952, 256: 0.966, 384: 0.976}
control (dims=[384])         {32: 0.808, 64: 0.926, 128: 0.958, 256: 0.97, 384: 0.98}
```

What it shows and its caveats are in the module, under the Step 8 results table.

The current, seeded script on the same seed 0 (the same output on a second run). The base model has no
training, so its line is identical. The two trained models move by at most 0.008 at any width, because the
batch order is now fixed rather than random:

```
base model, untrained        {32: 0.8, 64: 0.916, 128: 0.958, 256: 0.964, 384: 0.974}
Matryoshka (DIMS)            {32: 0.888, 64: 0.934, 128: 0.956, 256: 0.966, 384: 0.97}
control (dims=[384])         {32: 0.816, 64: 0.93, 128: 0.966, 256: 0.972, 384: 0.98}
```

### PCA baseline: is Matryoshka better than the best post-hoc compression?

Comparing Matryoshka prefixes with the first $k$ coordinates of an ordinary model is a weak
possible baseline: an ordinary model puts no order on its coordinates. The strongest cheap baseline is PCA,
which picks the $k$ directions that carry the most variance. So the script also scores **PCA-$k$** for
each of the three models:

- **Fit the way production would.** Embed the indexed corpus (the 3,000 training passages, not the test
  set and not the queries) with that model at full width, as unit vectors. $\mu$ is their mean. $U_k$ is
  the top $k$ principal directions, from the SVD of the centred matrix. Each model gets its own fit, on its
  own embeddings.
- **Apply it frozen.** Every test query and test passage becomes
  $\text{normalize}\big((x - \mu)\,U_k\big)$. Recall@1 is then scored exactly as the recipe scores it.
- **Centred** is the full-width score after subtracting $\mu$ only. At $k = 384$, PCA is centring plus a
  rotation, and a rotation changes no cosine. So PCA-384 must equal centred, and it does, on every model and
  seed: that is the sanity check.
- PCA directions are ordered too, so a PCA-rotated index can be cut at any $k$, like a Matryoshka
  vector. What PCA costs: you fit and store $\mu$ and $U$ (here 384 × 384), you multiply every query by
  them, and if the corpus drifts you refit and re-project the whole index.

**Seed 0** (recall@1 on the 500 held-out pairs; *prefix* = first $k$ coordinates, re-normalized):

```
recall@1 at width k: prefix = first k coordinates (embed_at); PCA = fitted on the training passages, applied frozen
model       method        32      64     128     256     384   centred
base        prefix     0.800   0.916   0.958   0.964   0.974
base        PCA        0.870   0.950   0.976   0.976   0.976     0.976
Matryoshka  prefix     0.888   0.934   0.956   0.966   0.970
Matryoshka  PCA        0.904   0.946   0.956   0.972   0.970     0.970
control     prefix     0.816   0.930   0.966   0.972   0.980
control     PCA        0.880   0.946   0.980   0.980   0.982     0.982
(centred = full width, corpus mean subtracted, no projection; it must equal PCA at the last width)
```

**All three seeds** (`--seed 1` and `--seed 2` draw other pairs and another batch order). The small widths, where the
question is decided:

| Seed | k | Matryoshka prefix | Base PCA | Control PCA | Matryoshka PCA |
|---|---|---|---|---|---|
| 0 | 32 | **0.888** | 0.870 | 0.880 | 0.904 |
| 1 | 32 | 0.848 | **0.876** | 0.834 | 0.870 |
| 2 | 32 | 0.904 | **0.906** | 0.870 | 0.910 |
| 0 | 64 | 0.934 | **0.950** | 0.946 | 0.946 |
| 1 | 64 | 0.924 | **0.940** | 0.934 | 0.934 |
| 2 | 64 | 0.952 | **0.966** | 0.962 | 0.964 |

Bold marks the best of Matryoshka prefix, base PCA and control PCA. Matryoshka PCA is shown for reference: it needs both the
fine-tuning and a fit.

- **On this data, Matryoshka fine-tuning does not beat PCA on the untrained model.** At $k = 32$ the
  two tie: the Matryoshka prefix wins on seed 0 (+0.018), loses on seed 1 (−0.028), and is level on seed 2
  (−0.002), with means of 0.880 and 0.884. At $k = 64$ base PCA wins on all three seeds, by 0.014–0.016.
  The base model was already trained on Natural Questions, so the fine-tuning has little to add here and
  the comparison favours PCA. On data the base model hasn't seen, fine-tuning has more to teach, and the
  answer may differ.
- **Against the control, Matryoshka training wins only at the smallest width.** At $k = 32$ the
  Matryoshka prefix beats control PCA on every seed (+0.008, +0.014, +0.034). At $k = 64$ control PCA wins
  on every seed (+0.012, +0.010, +0.010). So if you fine-tune anyway, the ordinary loss plus PCA is as good
  as Matryoshka from 64 dimensions up.
- **The naive prefix is a weak baseline.** On seed 0, PCA lifts every model at $k = 32$: base +0.070, control
  +0.064, Matryoshka +0.016. Matryoshka gains least because its first coordinates already carry most of
  the information. With 500 test queries, one query is 0.002 and the standard error of a single score
  near 0.88 is about 0.015. Read gaps under ~0.02 on a single seed as noise.

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
    - **Nothing is sent without `--allow-api`.** Without that flag the OpenAI models are cache-only: a model
      with uncached texts prints `SKIPPED … pass --allow-api`, and the key is never read.
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
python run_threshold_distributions.py --models e5 bge ada002 te3small te3large --allow-api   # paid: needs an OpenAI key
python run_threshold_distributions.py --models e5 bge --transform none center whiten   # see "Centering and whitening"
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

### Centering and whitening: do they separate duplicates from hard negatives?

`--transform none center whiten` applies a post-embedding transform and reports each one separately
(`whiten` once per `--whiten-k`, default 32, 64 and 128).

- **Fit on the ingestion set only:** each model's embeddings of the 200 base chunks, before any pair
  variant exists. μ is their mean, and Σ = X_cᵀX_c/(n−1) is their covariance. U_k and λ_k are Σ's top-k
  eigenvectors and eigenvalues, taken from the SVD of X_c (same eigenpairs, no D×D matrix).
- **Apply the same frozen transform to every text in every pair,** then L2-renormalize:
  - `center`: normalize(x − μ);
  - `whiten`: normalize((x − μ) U_k / √λ_k). k must be < the number of fit vectors.
- **Sanity checks it prints:**
  - after centering, the mean pairwise cosine of the fit chunks (≈ 0; with n unit vectors that sum to zero it
    would be exactly −1/(n−1) ≈ −0.005);
  - after whitening, before renormalizing, the max |cov − I| of the fit chunks.
- Every row of the main table is printed per model × transform, followed by a summary table. The by-kind
  medians and the carry-over use the first transform listed. With the default `--transform none`, the
  output gains a `[transform: none]` header and the summary; the numbers in the results above are unchanged.

### Recorded result (2026-11-14, seed 0, transforms, e5 / BGE / ada-002 / 3-small / 3-large)

`python run_threshold_distributions.py --models e5 bge ada002 te3small te3large --transform none center whiten`.
The OpenAI vectors came from the cache filled by the run above, so this run sent 0 texts and billed $0.
Excerpts of the raw output (the per-transform tables are left out):

```
scored e5       intfloat/e5-base-v2 @ f52bf8ec8c
  center    mean pairwise cosine of fit chunks after centering -4.44e-03
  whiten32  max |cov - I| of whitened fit chunks (before renormalizing) 3.66e-15
  whiten64  max |cov - I| of whitened fit chunks (before renormalizing) 6.22e-15
  whiten128 max |cov - I| of whitened fit chunks (before renormalizing) 9.15e-15
scored bge      BAAI/bge-base-en-v1.5 @ a5beb1e3e6
  center    mean pairwise cosine of fit chunks after centering -4.47e-03
  whiten32  max |cov - I| of whitened fit chunks (before renormalizing) 7.49e-15
  whiten64  max |cov - I| of whitened fit chunks (before renormalizing) 7.49e-15
  whiten128 max |cov - I| of whitened fit chunks (before renormalizing) 7.49e-15
scored ada002   text-embedding-ada-002 @ api (440 cached, 0 sent)
  center    mean pairwise cosine of fit chunks after centering -3.06e-03
  whiten32  max |cov - I| of whitened fit chunks (before renormalizing) 4.70e-15
  whiten64  max |cov - I| of whitened fit chunks (before renormalizing) 4.70e-15
  whiten128 max |cov - I| of whitened fit chunks (before renormalizing) 4.70e-15
scored te3small text-embedding-3-small @ api (440 cached, 0 sent)
  center    mean pairwise cosine of fit chunks after centering -3.28e-03
  whiten32  max |cov - I| of whitened fit chunks (before renormalizing) 5.11e-15
  whiten64  max |cov - I| of whitened fit chunks (before renormalizing) 5.11e-15
  whiten128 max |cov - I| of whitened fit chunks (before renormalizing) 5.11e-15
scored te3large text-embedding-3-large @ api (440 cached, 0 sent)
  center    mean pairwise cosine of fit chunks after centering -3.77e-03
  whiten32  max |cov - I| of whitened fit chunks (before renormalizing) 9.21e-15
  whiten64  max |cov - I| of whitened fit chunks (before renormalizing) 9.21e-15
  whiten128 max |cov - I| of whitened fit chunks (before renormalizing) 9.21e-15

Summary: gap = dup p5 - hardneg p99; cutoff = lowest value with <= 1% of hard negatives above it
  model    transform         gap  clean gap  hn > dup med  dups caught
  e5       none          -0.0280         no         24.2%         0.0%
  e5       center        -0.2246         no         25.0%         0.0%
  e5       whiten32      -0.4817         no         18.3%         0.8%
  e5       whiten64      -0.5060         no         17.5%         0.8%
  e5       whiten128     -0.5780         no         41.7%         0.0%
  bge      none          -0.0691         no         98.3%         0.0%
  bge      center        -0.2396         no         97.5%         0.0%
  bge      whiten32      -0.5849         no        100.0%         0.0%
  bge      whiten64      -0.7000         no         89.2%         0.0%
  bge      whiten128     -0.6933         no         90.8%         0.0%
  ada002   none          -0.0465         no         99.2%         0.0%
  ada002   center        -0.4237         no         99.2%         0.0%
  ada002   whiten32      -0.6255         no         98.3%         0.0%
  ada002   whiten64      -0.6656         no         67.5%         0.0%
  ada002   whiten128     -0.6912         no         86.7%         0.0%
  te3small none          -0.1582         no        100.0%         0.0%
  te3small center        -0.4393         no        100.0%         0.0%
  te3small whiten32      -0.6715         no        100.0%         0.0%
  te3small whiten64      -0.6598         no         90.0%         0.0%
  te3small whiten128     -0.6926         no         95.8%         0.0%
  te3large none          -0.1197         no         94.2%         0.0%
  te3large center        -0.2954         no         94.2%         0.0%
  te3large whiten32      -0.6116         no         90.8%         0.0%
  te3large whiten64      -0.6559         no         80.0%         0.0%
  te3large whiten128     -0.6326         no         85.0%         0.0%

OpenAI tokens billed this run (cached texts cost nothing):
  ada002          0 tokens  $0.00000
  te3small        0 tokens  $0.00000
  te3large        0 tokens  $0.00000
  total           0 tokens  $0.00000
```

**Seeds 1 and 2** (cache-only, $0) give the same picture:
- No model × transform has a clean gap on any seed, and every gap is negative (−0.026 to −0.745).
- At the ≤ 1%-false-drop cutoff, no combination catches more than 2.5% of duplicates on any seed (the
  2.5% cases are e5 `whiten32` and BGE `whiten128` on seed 1). The OpenAI models catch 0% under every
  transform on every seed.

**What it shows:**
- **Neither transform separates the sets.** Centering and whitening move random pairs to about 0 and
  spread the duplicates out: the duplicates' 5th percentile falls to 0.26–0.82 across seeds. The hard
  negatives' 99th percentile stays at 0.9989 or above, so every gap gets *wider in the wrong direction*.
- **k changes the shape, not the verdict.** k = 64 lowers the share of hard negatives above the duplicates'
  median the most: ada-002 99% → 67.5–77%, 3-large 92–94% → 55–80% across seeds. But the top 1% of hard
  negatives still sits above nearly every duplicate, so the cutoff catches 0–2.5%.
- **Why:** both transforms are fit on the chunks, so they re-weight the directions in which *chunks* differ
  (topic, template, carrier). A one-figure edit barely moves the vector in any direction, so no linear
  re-weighting makes it less similar than a duplicate with a new header. The fix is still the exact check on
  numbers and identifiers from the module's step 4.

