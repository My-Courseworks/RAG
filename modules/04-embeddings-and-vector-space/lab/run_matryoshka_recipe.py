"""Run M4's "Worked: training a Matryoshka embedder yourself" recipe, straight from the module README.

The script does not copy the recipe. It reads the Python blocks of Steps 1-7 out of
../README.md and executes them as written, so if the README's code changes, this lab tests the new code.
It then supplies data, trains the Matryoshka model and the full-width control (Step 8), and prints
recall@1 at every width.

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
    dims = recipe["DIMS"]

    def report(label):
        scores = {m: round(recipe["recall_at_1"](test_pairs, m), 3) for m in dims}
        print(f"{label:<28}", scores, flush=True)

    report("base model, untrained")
    recipe["train"](train_pairs, dims)                                   # Steps 5-7
    report("Matryoshka (DIMS)")
    recipe["model"] = recipe["AutoModel"].from_pretrained(recipe["MODEL"])  # Step 8: fresh copy
    recipe["train"](train_pairs, dims=[dims[-1]])
    report(f"control (dims=[{dims[-1]}])")


if __name__ == "__main__":
    main()
