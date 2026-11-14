# Sentence-BERT, explained simply

> A no-math explanation of Sentence-BERT, the 2019 idea behind every embedding model in
> [M4](README.md). The first part is a story a five-year-old can follow. The second part says what each
> piece of the story really is. Back to [M4 · What an embedding model is optimizing](README.md#what-an-embedding-model-is-optimizing).

## Contents

- [The story](#the-story)
  - [1 · The problem: a very slow judge](#1--the-problem-a-very-slow-judge)
  - [2 · The idea: give every sentence a spot on a map](#2--the-idea-give-every-sentence-a-spot-on-a-map)
  - [3 · Why the old reader made bad maps](#3--why-the-old-reader-made-bad-maps)
  - [4 · The twins game: teaching the reader to make good maps](#4--the-twins-game-teaching-the-reader-to-make-good-maps)
  - [5 · Why it's so fast now](#5--why-its-so-fast-now)
- [For grown-ups: what each part of the story really is](#for-grown-ups-what-each-part-of-the-story-really-is)
- [Sources](#sources)

## The story

### 1 · The problem: a very slow judge

![Animation of BERT used as a cross-encoder, in 9 steps. Sentence A "the parcel is late" and sentence B "my package hasn't arrived" merge into one input, [CLS] A [SEP] B [SEP]. Each token's input is the sum of token, segment and position embeddings. Self-attention lets "late" attend to B's tokens across the [SEP] boundary, "arrived" attend back to A, and [CLS] gather from every token, repeated over 12 layers. Of the outputs only C, the [CLS] vector, is kept; a classification layer turns it into one score for the pair.](images/cross-encoder-bert.gif)

*The judge at work: BERT as a cross-encoder, built from the BERT paper [[BERT18]](#src-bert18): the
`[CLS]`/`[SEP]` input format and segment embeddings (§3), the input embedding sum (Fig. 2), C and the
classification layer (§3, §4.1). Attention weights and the score are illustrative. The paper never
says "cross-encoder"; it calls this "bidirectional cross attention between two sentences" (§3.2).*

- Imagine a classroom full of drawings, and you want to find the two drawings that look most alike.
  - **Grown-up:** the task is **semantic similarity search** over a collection: find the sentences, or
    passages, whose meaning is closest.
- You have a clever judge named BERT. BERT is very good at comparing, but only one way: you hold up
  **two drawings at once**, and BERT looks at both together and says how alike they are.
  - **Grown-up:** this is a **cross-encoder**. Both sentences go into BERT as one input, separated by a
    `[SEP]` token, so every token of one sentence attends to every token of the other. It outputs **one
    score for the whole sentence pair**, read off the `[CLS]` output by a small head on top. It's
    accurate, but it produces nothing reusable for either sentence on its own. How that `[CLS]` output,
    called **C**, comes about:
    - **Every layer, every token gathers from every token.** Each position takes a weighted sum of all
      positions' vectors (attention) and updates itself. All 13 positions do this in parallel, 12 times
      in BERT-base [[BERT18]](#src-bert18).
    - **It compounds.** In layer *k*, each token reads vectors that already gathered from everyone in
      layer *k*−1. So after 12 layers, every position has absorbed 12 rounds of the whole pair.
    - **`[CLS]` is special by role, not by computation.** It uses the same weights as every other
      position. It's special only because it sits at position 0 and is the vector the classifier reads.
      C is simply its final vector.
    - **The other tokens do real work, then their outputs go unused.** In every layer they're what C
      reads from. For a pair score, their final outputs T₁…T₁₂ are discarded. Other tasks use them:
      question answering finds the answer span from them, and Sentence-BERT averages them (§4).
  - <details>
    <summary><b>Grown-up: how <code>[SEP]</code> was trained to mean a boundary</b></summary>

    - **Placed by rule in every pretraining input.** BERT's pretraining built each example as
      `[CLS] span A [SEP] span B [SEP]`, from BooksCorpus and Wikipedia [[BERT18]](#src-bert18). Across
      millions of examples, `[SEP]` appeared exactly at the A/B boundary and at the end, and nowhere else.
    - **Started blank, like any word.** Its embedding began as a random vector, with no built-in meaning
      and no special code path. Whatever it came to mean had to be learned through ordinary updates.
    - **Next-sentence prediction made the boundary worth knowing.** To decide whether B really follows
      A, the model has to compare the end of A with the start of B, and `[SEP]` was a perfectly reliable
      cue for where that seam is. Using it as a boundary marker lowered the loss, so training pushed its
      embedding, and the attention heads that read it, that way.
    - **Masked-word prediction added pressure from the other side.** When guessing a hidden word near
      the seam, it helps to know that the text past `[SEP]` belongs to a different span, maybe a random
      one, rather than continuing the sentence.
    - **It was always visible, never a target.** Google's pretraining script never masks `[SEP]`, so the
      model never had to predict it. It was always there as a dependable signal.
    - **Segment embeddings reinforced it.** Every token up to the first `[SEP]` carries the "A"
      embedding and every token after it carries "B" [[BERT18]](#src-bert18). `[SEP]` always sits where
      that label switches, so both signals teach the same boundary.

    </details>
- With 10 drawings, that's quick. With a whole school's worth, you'd have to hold up every possible
  pair. That's so many pairs that the judge would be busy for days.
  - **Grown-up:** *n* sentences have *n*(*n*−1)/2 pairs, so the cost grows with the square of the
    collection. For 10,000 sentences that's about 50 million BERT runs, roughly **65 hours**
    [[SB19]](#src-sb19).

### 2 · The idea: give every sentence a spot on a map

- What if, instead, each drawing got a **sticker with a spot on a big map**?
  - **Grown-up:** the sticker is a **sentence embedding**: a fixed-length vector of D numbers (768 for
    BERT-base), computed from one sentence alone.
- Drawings that look alike get spots **close together**. Drawings that look different get spots
  **far apart**.
  - **Grown-up:** "close" is measured with **cosine similarity**, or dot product on normalized vectors
    ([M4 · Distance](README.md#distance-geometry-and-the-normalization-decision)). The geometry is the
    meaning: nothing else is stored.
- Now you never need the judge to look at pairs. To find the two most alike drawings, you just look at
  the map and find the two closest stickers. Measuring how far apart two stickers are takes a second.
  - **Grown-up:** this is **nearest-neighbour search**. One comparison is about D multiply-adds, and an
    ANN index avoids comparing against every vector ([→ M5](../05-vector-databases-and-ann/README.md)).
- Sentences work the same way. Each sentence gets one spot on the map, and sentences that **mean** the
  same thing sit close together, even if they use different words.
  - **Grown-up:** that's the advantage over lexical matching such as BM25, which needs shared words
    ([→ M7](../07-sparse-dense-hybrid/README.md)). It's also the weakness: an exact code or identifier
    can get blurred into its neighbours.

### 3 · Why the old reader made bad maps

- People first tried asking BERT to place the stickers. It did a bad job.
  - **Grown-up:** the first attempts used pretrained BERT as it was, taking either the `[CLS]` vector or
    the mean of the token vectors as the sentence embedding, then comparing with cosine. This was the
    common practice. In the paper's words, *"The most commonly used approach is to average the BERT
    output layer … or by using the output of the first token (the [CLS] token)"*, and the popular
    bert-as-service tool offered both. Nobody had checked it: *"there is so far no evaluation if these
    methods lead to useful sentence embeddings"* [[SB19]](#src-sb19). The pipeline is the same one
    Sentence-BERT uses later (embed, pool, compare). Only the training behind the vectors differs.
- That's because BERT had only ever practised a different game: **guessing hidden words** in a sentence,
  like a fill-in-the-blank puzzle. It got really good at words, but nobody ever taught it to put whole
  sentences on a map.
  - **Grown-up:** BERT's pretraining was **masked-language modelling** (predict hidden tokens) and
    **next-sentence prediction**, which reads the two sentences *together*. Neither objective requires
    that two separately encoded sentences with similar meaning get nearby vectors.
- So its map was messy: sentences about totally different things ended up close together. It was even
  worse than a very old, simple trick of just averaging the words' own stickers.
  - **Grown-up:** raw BERT vectors sit in a narrow cone, so most pairs score high
    ([M4 · The cone effect](README.md#the-cone-effect-raw-embeddings-are-not-isotropic)). The paper
    measured the common practice for the first time. Its verdict: *"this common practice yields
    rather bad sentence embeddings, often worse than averaging GloVe embeddings"* [[SB19]](#src-sb19).
    Spearman correlation × 100 with human similarity scores (higher is better), no task-specific
    training:

    | Sentence vector | Average over 7 STS tasks | STS benchmark alone |
    |---|---|---|
    | BERT `[CLS]` vector | 29.19 | 16.50 |
    | BERT, average of token vectors | 54.81 | 46.35 |
    | Average of GloVe word vectors (2014) | 61.32 | 58.02 |
    | InferSent (2017, siamese LSTM trained on NLI) | 65.01 | 68.03 |
    | Universal Sentence Encoder (2018) | 71.22 | 74.92 |

    The two encoders actually trained to produce sentence vectors, InferSent and USE, beat raw BERT
    easily, even though BERT was the stronger model. The vectors were the problem, not the pipeline
    [[SB19]](#src-sb19) (Table 1).

### 4 · The twins game: teaching the reader to make good maps

![Sentence-BERT architecture, Figures 1 and 2 of the paper. Left, training: Sentence A and Sentence B each pass through BERT and pooling to give vectors u and v; (u, v, |u−v|) feeds a softmax classifier. Right, inference: the same two towers give u and v, and cosine-sim(u, v) gives a score from −1 to 1.](images/sbert-architecture-figures-1-2.png)

*Figures 1 and 2 from the paper [[SB19]](#src-sb19), page 3. **Left:** how it's trained on NLI (natural language
inference), which is the twins game below. **Right:** how it's used afterwards, comparing two stickers by cosine. The two "BERT"
boxes are the twins: one network with tied weights, drawn twice.*

- Sentence-BERT's idea was to **teach BERT the map game** directly.
  - **Grown-up:** keep BERT's architecture and pretrained weights, and **fine-tune** it with an
    objective defined on the sentence vectors themselves.
- Picture twins who share **one brain**. Each twin reads one sentence, and each makes a sticker for it.
  Because they share a brain, they turn a sentence into a sticker spot in exactly the same way.
  - **Grown-up:** a **siamese network**, or **bi-encoder**: the same BERT, with the same weights, encodes
    each sentence separately. Sharing weights guarantees both vectors live in the same space, so
    comparing them is meaningful.
- A teacher shows the twins two sentences and says whether they **mean the same thing**, **mean
  opposite things**, or **have nothing to do with each other**.
  - **Grown-up:** these are **NLI** labels (entailment, contradiction, neutral) from the SNLI and
    MultiNLI datasets [[SB19]](#src-sb19).

    <details>
    <summary><b>NLI: the task</b></summary>

    NLI is a task, not a dataset (also called *recognizing textual entailment*). You get a **premise**
    and a **hypothesis**, and decide: if the premise is true, what follows for the hypothesis?

    | Label | Meaning | Premise → hypothesis (real SNLI examples [[SNLI15]](#src-snli15)) |
    |---|---|---|
    | **Entailment** | the hypothesis must be true | "A soccer game with multiple males playing." → "Some men are playing a sport." |
    | **Contradiction** | the hypothesis can't be true | "A black race car starts up in front of a crowd of people." → "A man is driving down a lonely road." |
    | **Neutral** | it might or might not be true | "A smiling costumed woman is holding an umbrella." → "A happy woman in a fairy costume holds an umbrella." |

    It was built to test whether a model understands sentences, not for search. Sentence-BERT borrowed
    it as training data.

    </details>

    <details>
    <summary><b>SNLI: the Stanford NLI corpus (2015)</b></summary>

    - **Size:** about **570,000** human-written sentence pairs [[SNLI15]](#src-snli15).
    - **Premises:** image captions from the Flickr30k corpus, short literal descriptions of everyday
      scenes.
    - **Hypotheses:** written by crowdworkers (about 2,500 of them), one new sentence per label for each
      caption.
    - **Character:** simple, concrete, visual language. Everything is about a scene you could photograph.

    </details>

    <details>
    <summary><b>MultiNLI: the Multi-Genre NLI corpus (2018)</b></summary>

    - **Size:** about **433,000** pairs, with the same format and the same three labels as SNLI
      [[MNLI18]](#src-mnli18).
    - **Premises:** **ten genres** of written and spoken English, among them fiction, Slate pop-culture
      articles, transcribed telephone conversations, Berlitz travel guides and Oxford University Press
      non-fiction.
    - **Why it exists:** SNLI's caption-only language was too narrow. Only five genres appear in
      training, so a model can be tested on *matched* text (genres it saw) and *mismatched* text (genres
      it didn't).

    </details>
- After every pair, the shared brain adjusts a little, so that next time it places sentences with the
  same meaning closer together and different ones further apart.
  - **Grown-up:** each step is a gradient update. On NLI the paper trained a small classifier that reads
    the two sentence vectors *u* and *v*, together with |*u* − *v*| (element-wise absolute difference), and predicts the label. Making that
    prediction possible is what reshapes the space. The paper also used a regression objective on
    similarity scores and a triplet objective [[SB19]](#src-sb19).

    <details>
    <summary><b>What Sentence-BERT did with SNLI and MultiNLI</b></summary>

    - **The setup:** the two datasets combined, about a million pairs. One epoch of the
      (u, v, |u − v|) classifier, batch size 16, Adam at learning rate 2e-5, mean pooling
      [[SB19]](#src-sb19) (§3.1).
    - **All ~110 million BERT weights get updated.** Fine-tuning here doesn't mean freezing BERT and
      training a small network on top. Training starts from pretrained BERT, and every weight is
      nudged at a small learning rate, the same way BERT's own paper fine-tunes [[BERT18]](#src-bert18).
      The only new weights are the classifier's, about 6.9k (3 × 768 inputs × 3 labels). The pooling is
      an average and has no weights. Where BERT-base's weights live:

    | Inside BERT-base | Weights | Updated by Sentence-BERT? |
    |---|---|---|
    | Token embeddings (30,522 vocabulary entries × 768) | 23.4M | ✅ |
    | Position (512 × 768) and segment (2 × 768) embeddings | 0.4M | ✅ |
    | **12 layers**, each about 7.1M: | 85.1M | ✅ |
    | – attention: query, key, value and output matrices (4 × 768 × 768) | 2.4M per layer | ✅ |
    | – feed-forward network (768 → 3,072 → 768) | 4.7M per layer | ✅ |
    | – layer norms | ~3k per layer | ✅ |
    | Pooler layer on top of `[CLS]` (in the Hugging Face model) | 0.6M | ❌ unused, since Sentence-BERT averages tokens instead |
    | **Total** | **≈ 110M** | |

    - **The function chain.** Call BERT's weights θ and the classifier's weights W<sub>t</sub>:
      - u = f<sub>θ</sub>(A): sentence A through the 12 layers, then mean pooling.
      - v = f<sub>θ</sub>(B): the **same θ**, since there's one copy of BERT.
      - p = softmax(W<sub>t</sub> · [u ; v ; |u − v|]): three probabilities, one per label.
      - L(θ, W<sub>t</sub>) = −log p<sub>y</sub>: cross-entropy against the true label *y*, averaged
        over a batch of 16 pairs. The softmax gives probabilities; the cross-entropy is what makes it a
        loss.
    - **Backpropagation is the chain rule on that function**, worked from the loss backwards:
      - L → probabilities → W<sub>t</sub>, giving ∂L/∂W<sub>t</sub>.
      - → the features [u ; v ; |u − v|], giving ∂L/∂u and ∂L/∂v. Through |u − v| the two are tied:
        moving either one changes their distance.
      - → through mean pooling, where the gradient is split evenly across the tokens, into the last
        layer's token vectors.
      - → down all 12 layers, **once through A's pass and once through B's**. Because θ appears in
        both, its gradient is the sum of two paths: ∂L/∂θ = (∂L/∂u)(∂u/∂θ) + (∂L/∂v)(∂v/∂θ).
      - → finally into the embedding rows of the tokens that appeared. The tokens themselves are
        discrete and get no gradient. Then Adam nudges every weight against its gradient, and the step
        is done.
    - **The classifier is thrown away at inference.** W<sub>t</sub> only exists to turn NLI labels into
      a loss. Afterwards, only u and v are used, compared by cosine. In the paper's words, *"the
      concatenation mode is only relevant for training the softmax classifier"* [[SB19]](#src-sb19).
      What you keep is BERT with all its weights adjusted, plus mean pooling.
    - **Why NLI teaches a good map, with no search queries in it:**
      - **Entailment pairs** are near-paraphrases in different words. They teach "different words,
        same meaning → close together".
      - **Contradiction pairs** often share almost every word ("A man is sleeping" vs "A man is
        running"). They teach "same words, different meaning → far apart", which bag-of-words methods
        can't do.
    - **Later models used the labels more directly.** Supervised SimCSE treats an entailment pair as a
      positive and the contradiction pair for the same premise as a **hard negative**
      [[SIMCSE21]](#src-simcse21).

    </details>
- After practising on about a million pairs, the brain makes good maps, even for sentences it has never
  seen.
  - **Grown-up:** on the same measures as §3's table, the NLI-trained model scores **74.89** (base) and
    **76.55** (large) averaged over the 7 STS tasks, and **77.03** and **79.23** on the STS benchmark,
    against 54.81 and 46.35 for raw BERT mean pooling [[SB19]](#src-sb19) (Table 1). Same pipeline,
    re-shaped space. It generalizes beyond its training pairs, but unevenly: in the paper's own
    cross-topic test, performance dropped by about 7 points [[SB19]](#src-sb19). Domain shift is the
    reason M4 says to measure on your own queries.
- **How a sentence gets one sticker:** the brain makes a little sticker for **every word**, then mixes
  them all together, like mixing paints into one colour. That mixed colour is the sentence's sticker.
  - **Grown-up:** **mean pooling**, Sentence-BERT's default, averages the token vectors into one vector
    of the same size D ([M4 §3 · Pooling](README.md#3--pooling-the-grid-becomes-one-vector)).

### 5 · Why it's so fast now

- With the slow judge, every new comparison meant reading two sentences again.
  - **Grown-up:** a cross-encoder must run the full model once per pair, at query time.
- With the map, each sentence is read **once**, and its sticker is saved. Comparing is just measuring
  distances on the map.
  - **Grown-up:** with a bi-encoder, every document's vector is computed **ahead of time**, at ingest,
    and stored in a vector database. At query time only the query is encoded, once, and compared against
    the stored vectors.
- Finding the most alike pair among 10,000 sentences went from about **65 hours** with the judge to
  about **5 seconds** with the map.
  - **Grown-up:** those are the paper's figures [[SB19]](#src-sb19). The price is accuracy: a
    bi-encoder never lets the two texts attend to each other. The cross-encoder wasn't abandoned for
    being bad. It's still the more accurate of the two, just too slow to search with. The paper's own
    comparison, STS benchmark test set, both trained on NLI and then on the STS benchmark:

    | Model size | BERT cross-encoder | Sentence-BERT bi-encoder |
    |---|---|---|
    | base | **88.33** | 85.35 |
    | large | **88.77** | 86.10 |

    That's why pipelines often use both: a bi-encoder to fetch candidates from stored vectors, then a
    cross-encoder to re-score the top few ([→ M9](../09-ranking-fusion-reranking/README.md))
    [[SB19]](#src-sb19) (Table 2).

## For grown-ups: what each part of the story really is

| In the story | What it really is |
|---|---|
| The judge looking at two drawings at once | A **cross-encoder**: BERT reads both sentences in one input. Still the more accurate approach (88.77 vs 86.10 on the STS benchmark, §5), but you must run it again for every pair. |
| A sticker with a spot on the map | A **sentence embedding**: one vector of D numbers per sentence (768 for BERT-base). |
| Close spots mean similar meaning | **Cosine similarity** between the two vectors ([M4 · Distance](README.md#distance-geometry-and-the-normalization-decision)). |
| The fill-in-the-blank game | BERT's **pretraining**: masked-language modelling and next-sentence prediction. Neither asks for comparable sentence vectors. |
| The messy map | Raw BERT vectors compared by cosine. On sentence-similarity benchmarks, averaged GloVe word vectors score 61.32, BERT mean pooling 54.81, and BERT's `[CLS]` vector 29.19 [[SB19]](#src-sb19). |
| Twins sharing one brain | A **siamese network** (also called a **bi-encoder**): the same BERT, with the same weights, encodes each sentence separately. |
| The teacher's three answers | **NLI** pair labels (entailment, contradiction, neutral) from the SNLI and MultiNLI datasets. |
| Mixing the word stickers into one colour | **Mean pooling** over the token vectors, Sentence-BERT's default ([M4 §3 · Pooling](README.md#3--pooling-the-grid-becomes-one-vector)). |
| Saving each sticker once | Embeddings are computed **ahead of time** and stored, which is what makes a vector database possible. |
| 65 hours → 5 seconds | The paper's own figures for 10,000 sentences [[SB19]](#src-sb19). |

Two places where the story simplifies:

- **"Closer or further apart" is the effect, not the exact training rule.** On NLI, the original paper
  trained a small classifier that looks at the two sentence vectors and predicts the label. Pulling
  same-meaning sentences together is what that training does to the map. Later models train the
  "pull together, push apart" rule directly, with a contrastive loss ([M4 · Matryoshka recipe, step 3](README.md#the-recipe-step-by-step)).
- **The paper trained in other ways too:** a regression objective on sentence-similarity scores, and a
  triplet objective (an anchor sentence, a similar one and a different one).

Where Sentence-BERT sits in the bigger story, and what came after it:
[embedding-history.md · Sentence-BERT](embedding-history.md#sentence-bert-2019).

## Sources

- <a id="src-bert18"></a>*BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding* — Jacob Devlin, Ming-Wei Chang, Kenton Lee, Kristina Toutanova · arXiv:1810.04805 (NAACL 2019) · [arxiv.org/abs/1810.04805](https://arxiv.org/abs/1810.04805) · local copy: [papers/bert-1810.04805.pdf](papers/bert-1810.04805.pdf) — `[BERT18]` · owns the §1 animation's grounding: `[CLS]` and `[SEP]`, segment embeddings, the token + segment + position sum (Fig. 2), C as the `[CLS]` final hidden vector fed to a classification layer (§3, §4.1, Fig. 4a), bidirectional self-attention, and BERT-base's 12 layers / 768 dims / 12 heads / 110M parameters. Tokens come from Hugging Face's `bert-base-uncased` vocabulary, not the paper.
- <a id="src-sb19"></a>*Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks* — Nils Reimers, Iryna Gurevych · arXiv:1908.10084 (EMNLP 2019) · [arxiv.org/abs/1908.10084](https://arxiv.org/abs/1908.10084) · local copy: [papers/sentence-bert-1908.10084.pdf](papers/sentence-bert-1908.10084.pdf) — `[SB19]` · owns the architecture figure in §4 (Figures 1 and 2, page 3, rendered from the PDF), the ~50 million BERT runs and the 65-hours-to-5-seconds comparison, the ~7-point cross-topic drop, the common-practice quotes and bert-as-service (§3), Table 1's scores for `[CLS]`, BERT mean pooling, GloVe, InferSent, USE and SBERT-NLI (§3, §4), Table 2's cross-encoder vs bi-encoder scores (§5), the siamese architecture, mean pooling as default, and the NLI, regression and triplet objectives.
- <a id="src-snli15"></a>*A large annotated corpus for learning natural language inference* — Samuel R. Bowman, Gabor Angeli, Christopher Potts, Christopher D. Manning · arXiv:1508.05326 (EMNLP 2015) · [arxiv.org/abs/1508.05326](https://arxiv.org/abs/1508.05326) · local copy: [papers/snli-1508.05326.pdf](papers/snli-1508.05326.pdf) — `[SNLI15]` · owns SNLI's 570K pairs, Flickr30k captions as premises, the ~2,500 crowdworkers, and the three example pairs (its Table 1).
- <a id="src-mnli18"></a>*A Broad-Coverage Challenge Corpus for Sentence Understanding through Inference* — Adina Williams, Nikita Nangia, Samuel R. Bowman · arXiv:1704.05426 (NAACL 2018) · [arxiv.org/abs/1704.05426](https://arxiv.org/abs/1704.05426) · local copy: [papers/multinli-1704.05426.pdf](papers/multinli-1704.05426.pdf) — `[MNLI18]` · owns MultiNLI's 433k pairs, the ten genres, five genres in training, and matched vs mismatched evaluation.
- <a id="src-simcse21"></a>*SimCSE: Simple Contrastive Learning of Sentence Embeddings* — Tianyu Gao, Xingcheng Yao, Danqi Chen · arXiv:2104.08821 (EMNLP 2021) · [arxiv.org/abs/2104.08821](https://arxiv.org/abs/2104.08821) — `[SIMCSE21]` · owns entailment pairs as positives and contradiction pairs as hard negatives.

**Unsupported or simplified:** the weight counts in the "What Sentence-BERT did" table are computed from the Hugging Face `bert-base-uncased` configuration (30,522-entry vocabulary, 512 positions, feed-forward size 3,072), not quoted from a paper. in the `[SEP]` collapsible (§1), "never masked" comes from Google's released pretraining script (`create_pretraining_data.py`), not the paper, and wasn't re-checked for this file; why next-sentence and masked-word training push `[SEP]` toward a boundary meaning is this file's explanation, not a claim in the paper. the "A man is sleeping" / "A man is running" pair in §4 is illustrative, not from a dataset. The drawings, map, judge and twins are analogies written for this file, not
the paper's. The Table 1 and Table 2 scores were checked against the local PDF; the other figures match M4's
existing citation of [SB19].
