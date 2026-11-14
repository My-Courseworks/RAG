"""Run M4's "Worked: training a Matryoshka embedder yourself" recipe, straight from the module README.

The script does not copy the recipe. It reads the Python blocks of Steps 1-7 out of
../README.md and executes them as written, so if the README's code changes, this lab tests the new code.
It then supplies data, trains the Matryoshka model and the full-width control (Step 8), and prints
recall@1 at every width.

After those lines it prints a PCA baseline for each of the three models: the best post-hoc way to
shrink an ordinary model's vectors, to compare against Matryoshka's free prefix. PCA-k is fitted the
production way, on the indexed corpus only (the training passages, embedded at full width): mu is their
mean, U_k their top-k principal directions. It is then applied frozen to the test queries and passages,
x -> normalize((x - mu) @ U_k). "centred" is the full-width score after subtracting mu only; PCA at
k = D is that plus a rotation, so the two must match.

Usage (from this folder):
    python3 -m venv .venv && source .venv/bin/activate
    pip install -r requirements.txt
    python run_matryoshka_recipe.py                    # Natural Questions: 3,000 train / 500 test
    python run_matryoshka_recipe.py --data synthetic   # 60-second smoke test, no dataset download

--data synthetic only proves the code runs: every query names a unique contract number, so all
widths score ~1.0 and the Matryoshka effect cannot show. Use the default (nq) for a real comparison.
"""
import argparse
import random
import re
from pathlib import Path

import torch
import torch.nn.functional as F

README = Path(__file__).resolve().parent.parent / "README.md"
SECTION = "### Worked: training a Matryoshka embedder yourself"


def load_recipe(readme_path):
    """Execute the recipe's definitions (Steps 1-6, plus recall_at_1 from Step 7) and return their namespace."""
    text = readme_path.read_text(encoding="utf-8")
    section = text[text.index(SECTION):]
    section = section[:section.index("\n---\n")]
    blocks = re.findall(r"```python\n(.*?)```", section, re.S)
    namespace = {}
    for block in blocks[:6]:                       # Steps 1-6: model, embed, losses, train, embed_at
        exec(block, namespace)
    exec(blocks[6].split("pairs = [")[0], namespace)  # Step 7: the recall_at_1 definition only
    return namespace


def embed_in_batches(recipe, texts, m, batch_size=250):
    """Unit vectors at width m for many texts, a batch at a time (3,000 passages at once would not fit)."""
    return torch.cat([recipe["embed_at"](texts[i:i + batch_size], m) for i in range(0, len(texts), batch_size)])


def fit_pca(corpus):
    """Mean and principal directions (columns, most variance first) of the corpus vectors."""
    mu = corpus.mean(dim=0)
    _, _, vh = torch.linalg.svd(corpus - mu, full_matrices=False)
    return mu, vh.T


def recall_from_vectors(q, p):
    """recall@1 exactly as the recipe's recall_at_1 scores it, on vectors already computed."""
    top = (q @ p.T).argmax(dim=1)
    return (top == torch.arange(len(q))).float().mean().item()


def pca_scores(recipe, corpus_passages, test_pairs, dims):
    """Fit centering + PCA on the corpus with the current model, apply it frozen to the test pairs.

    Returns ({k: recall@1 of PCA-k}, recall@1 of centring alone at full width)."""
    full = dims[-1]
    mu, directions = fit_pca(embed_in_batches(recipe, corpus_passages, full))
    queries, passages = zip(*test_pairs)
    q = recipe["embed_at"](list(queries), full)       # embedded exactly as recall_at_1 embeds them
    p = recipe["embed_at"](list(passages), full)

    def project(x, k):
        return F.normalize((x - mu) @ directions[:, :k], dim=-1)

    pca = {k: round(recall_from_vectors(project(q, k), project(p, k)), 3) for k in dims}
    centred = round(recall_from_vectors(F.normalize(q - mu, dim=-1), F.normalize(p - mu, dim=-1)), 3)
    return pca, centred


def print_comparison(rows, centred, dims):
    """rows: [(model label, method, {k: recall})]; centred: {model label: full-width centred recall}."""
    print("\nrecall@1 at width k: prefix = first k coordinates (embed_at); "
          "PCA = fitted on the training passages, applied frozen")
    print(f"{'model':<12}{'method':<8}" + "".join(f"{k:>8}" for k in dims) + f"{'centred':>10}")
    for label, method, scores in rows:
        tail = f"{centred[label]:>10.3f}" if method == "PCA" else ""
        print(f"{label:<12}{method:<8}" + "".join(f"{scores[k]:>8.3f}" for k in dims) + tail)
    print("(centred = full width, corpus mean subtracted, no projection; it must equal PCA at the last width)")


def seed_everything(seed):
    """Fix the batch order (random.shuffle in train) and dropout, so a seed gives the same run every time."""
    random.seed(seed)
    torch.manual_seed(seed)


def natural_questions_pairs(n, seed):
    from datasets import load_dataset
    ds = load_dataset("sentence-transformers/natural-questions", split="train")  # columns: query, answer
    rng = random.Random(seed)
    return [(ds[i]["query"], ds[i]["answer"]) for i in rng.sample(range(len(ds)), n)]


def synthetic_pairs(n, seed):
    rng = random.Random(seed)
    lanes = ["Rotterdam-Singapore", "Hamburg-Shanghai", "Antwerp-Mumbai", "Felixstowe-Busan", "Genoa-Dubai"]
    goods = ["lithium batteries", "frozen fish", "steel coils", "cotton bales", "wine", "solar panels"]
    pairs = []
    for i in range(n):
        lane, item = rng.choice(lanes), rng.choice(goods)
        pairs.append((f"surcharge for {item} on {lane} contract {i}?",
                      f"Contract {i}: shipping {item} on the {lane} lane carries a surcharge of "
                      f"{rng.randint(1, 9)}.{rng.randint(0, 9)}%."))
    return pairs


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", choices=["nq", "synthetic"], default="nq")
    parser.add_argument("--n-train", type=int, default=3000)
    parser.add_argument("--n-test", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    recipe = load_recipe(README)
    make_pairs = natural_questions_pairs if args.data == "nq" else synthetic_pairs
    pairs = make_pairs(args.n_train + args.n_test, args.seed)
    train_pairs, test_pairs = pairs[:args.n_train], pairs[args.n_train:]
    corpus_passages = [passage for _, passage in train_pairs]   # the indexed corpus PCA is fitted on
    dims = recipe["DIMS"]
    rows, centred = [], {}

    def report(label, short):
        scores = {m: round(recipe["recall_at_1"](test_pairs, m), 3) for m in dims}
        print(f"{label:<28}", scores, flush=True)
        pca, centred[short] = pca_scores(recipe, corpus_passages, test_pairs, dims)
        rows.extend([(short, "prefix", scores), (short, "PCA", pca)])

    report("base model, untrained", "base")
    seed_everything(args.seed)
    recipe["train"](list(train_pairs), dims)              # Steps 5-7 (a copy: train shuffles in place)
    report("Matryoshka (DIMS)", "Matryoshka")
    recipe["model"] = recipe["AutoModel"].from_pretrained(recipe["MODEL"])  # Step 8: fresh copy
    seed_everything(args.seed)                                           # same batch order as above
    recipe["train"](list(train_pairs), dims=[dims[-1]])
    report(f"control (dims=[{dims[-1]}])", "control")
    print_comparison(rows, centred, dims)


if __name__ == "__main__":
    main()
