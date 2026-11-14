# The embedding journey: from GloVe to LLM-backboned embedders

> **What this covers:** how text embeddings got from one fixed vector per word (2013–14) to instruction-aware LLM embedders with Matryoshka dimensions (2024–25). For each technique: what it introduced, which earlier flaw it fixed (and whether the evidence says it really did), and where it fell short.
> **Who it's for:** anyone choosing an embedder who wants to know why today's models look the way they do, and which old problems are still open.
> **Evidence:** researched 2026-11-14 from each technique's own paper, technical report or model card. Each "Fell short" is quoted from a later paper that measured it where one exists. Otherwise it's the paper's own stated limitation.
> Back to [M4 · One vector per token](README.md#2--one-vector-per-token) · [M4 · What an embedding model is optimizing](README.md#what-an-embedding-model-is-optimizing).

## Contents

- [Summary table](#summary-table)
- [Five shifts](#five-shifts)
- [2013–2016: static word vectors](#20132016-static-word-vectors)
  - [word2vec (2013), brief context](#word2vec-2013-brief-context)
  - [GloVe (2014)](#glove-2014)
  - [fastText (2016)](#fasttext-2016)
- [2018: contextual vectors and the first sentence encoders](#2018-contextual-vectors-and-the-first-sentence-encoders)
  - [ELMo (Feb 2018)](#elmo-feb-2018)
  - [Universal Sentence Encoder (Mar 2018)](#universal-sentence-encoder-mar-2018)
  - [GPT-1 (Jun 2018)](#gpt-1-jun-2018)
  - [BERT (Oct 2018)](#bert-oct-2018)
- [2019–2021: training for the vector itself](#20192021-training-for-the-vector-itself)
  - [Sentence-BERT (2019)](#sentence-bert-2019)
  - [DPR (2020)](#dpr-2020)
  - [SimCSE (2021)](#simcse-2021)
  - [Contriever (Dec 2021)](#contriever-dec-2021)
- [2022–2023: contrastive pretraining at scale, and an early LLM embedder](#20222023-contrastive-pretraining-at-scale-and-an-early-llm-embedder)
  - [cpt-text (Jan 2022)](#cpt-text-jan-2022)
  - [E5 (Dec 2022)](#e5-dec-2022)
  - [INSTRUCTOR (Dec 2022)](#instructor-dec-2022)
  - [GTE (Aug 2023)](#gte-aug-2023)
  - [BGE / C-Pack (Sep 2023)](#bge--c-pack-sep-2023)
- [Late 2023–2024: LLM backbones](#late-20232024-llm-backbones)
  - [e5-mistral (Dec 2023)](#e5-mistral-dec-2023)
  - [LLM2Vec (Apr 2024)](#llm2vec-apr-2024)
  - [NV-Embed (May 2024)](#nv-embed-may-2024)
- [2025: lab-scale LLM embedder families](#2025-lab-scale-llm-embedder-families)
  - [Gemini Embedding (Mar 2025)](#gemini-embedding-mar-2025)
  - [Qwen3-Embedding (Jun 2025)](#qwen3-embedding-jun-2025)
  - [EmbeddingGemma (Sep 2025)](#embeddinggemma-sep-2025)
  - [OpenAI `text-embedding-ada-002` and `text-embedding-3-*`, architecture undisclosed](#openai-text-embedding-ada-002-and-text-embedding-3--architecture-undisclosed)
- [What has not been fixed](#what-has-not-been-fixed)
- [Sources](#sources)

---

## Summary table

"—" means the source doesn't state it. **Pooling** is how the per-token vectors become one vector ([M4 · Pooling](README.md#3--pooling-the-grid-becomes-one-vector)). **Max input** is in tokens unless marked otherwise.

| Year | Model | Org | Backbone | Pooling | Training signal | Max input | Dims |
|---|---|---|---|---|---|---|---|
| 2013 | word2vec | Google | Static | n/a (one word) | Predict nearby words (CBOW / skip-gram) | one word | 300–1,000 in its tables |
| 2014 | GloVe | Stanford | Static | n/a (one word) | Global co-occurrence counts | one word | 300 in its comparisons |
| 2016 | fastText | Facebook AI Research | Static + char n-grams | Sum of n-gram vectors | Skip-gram over subwords | one word | 300 |
| 2018 | ELMo | AI2, UW | BiLSTM | Learned linear mix of layers, per token | Bidirectional language modelling | — | 512-d projections per layer |
| 2018 | Universal Sentence Encoder | Google | Transformer encoder / DAN | Sum ÷ √length (transformer); average of word + bigram embeddings (DAN) | Multi-task: Skip-Thought-like, conversation, SNLI | — | 512 |
| 2018 | GPT-1 | OpenAI | Decoder (causal) | Not an embedder | Next-token prediction | 512 (training sequences) | 768 |
| 2018 | BERT | Google | Encoder (bidirectional) | `[CLS]` for classification | Masked LM + next-sentence prediction | 512 | 768 / 1,024 |
| 2019 | Sentence-BERT | UKP, TU Darmstadt | Encoder | Mean (default) | Siamese fine-tuning on NLI / STS labels | 512 (BERT) | 768 / 1,024 |
| 2020 | DPR | Facebook AI et al. | Two encoders | `[CLS]` | Question–passage pairs, in-batch + BM25 negatives | 100-word passages | 768 |
| 2021 | SimCSE | Princeton, Tsinghua | Encoder | `[CLS]` (MLP kept in training only, for unsup.) | Contrastive; dropout as augmentation, or NLI pairs | — | 768 / 1,024 |
| 2021 | Contriever | Meta AI et al. | Encoder | Mean | Unsupervised contrastive (random crops) | — | 768 (BERT-base) |
| 2022 | cpt-text | OpenAI | Decoder LLM (GPT) | `[EOS]` (last token) | Contrastive on neighbouring web text | — | 1,024–12,288 |
| 2022 | E5 | Microsoft | Encoder | Mean | Weakly supervised contrastive (CCPairs) → fine-tune | 512 (eval) | 768 / 1,024 (BERT-base / -large); small: — |
| 2022 | INSTRUCTOR | HKU, UW, Meta, AI2 | Encoder (GTR / T5) | Mean over text tokens | Multi-task contrastive with instructions (MEDI, 330 datasets) | — | — |
| 2023 | GTE | Alibaba | Encoder | Mean | ~800M weak pairs → labelled triples | 512 (fine-tune) | 384 / 768 / 1,024 (backbone hidden sizes) |
| 2023 | BGE (C-Pack) | BAAI | Encoder | `[CLS]` (bge-base card config) | RetroMAE pre-train → weak pairs → labelled + instructions | — | 768 (base) |
| 2023 | e5-mistral | Microsoft | Decoder LLM (Mistral-7B) | Last token (`[EOS]`) | LLM-generated synthetic data + labelled, <1k steps | 4,096 recommended | 4,096 |
| 2024 | LLM2Vec | McGill/Mila, ServiceNow | Decoder → bidirectional | Mean | MNTP + unsupervised SimCSE | 512 (training) | Base LLM's hidden size |
| 2024 | NV-Embed | NVIDIA | Decoder → bidirectional | Latent-attention layer | Two-stage contrastive instruction tuning | 512 (training); card 32,768 | 4,096 |
| 2025 | Gemini Embedding | Google | Gemini LLM, bidirectional | Mean + linear projection | Pre-finetune on noisy pairs → fine-tune → model soup | 2,048 (`gemini-embedding-001`) | 3,072 (MRL 768 / 1,536) |
| 2025 | Qwen3-Embedding | Alibaba (Tongyi) | Decoder LLM (causal) | Last token (`[EOS]`) | ~150M synthetic weak pairs → ~12M supervised → slerp merge | 32K | 1,024 / 2,560 / 4,096 (MRL) |
| 2025 | EmbeddingGemma | Google | Gemma 3 → encoder-decoder → encoder | Mean + projections | Distil Gemini Embedding + spread-out regulariser → souping | 2,048 | 768 (MRL 512/256/128) |
| — | `text-embedding-ada-002`, `-3-small`, `-3-large` | OpenAI | Not disclosed | Not disclosed | Not disclosed | 8,192 | 1,536 / 1,536 / 3,072 |

---

## Five shifts

1. **Static to contextual (2018).** A word used to get one vector wherever it appeared. ELMo, then BERT, made each token's vector depend on its sentence.
2. **Token vectors to sentence vectors (2018–19).** BERT's token vectors were strong, but averaging them or taking `[CLS]` made poor sentence vectors. USE and then Sentence-BERT trained directly for one comparable vector per text.
3. **Supervised pairs to contrastive pretraining at scale (2020–23).** **Contrastive learning** pulls matching pairs together and pushes the other items in the batch apart. DPR and SBERT needed labelled pairs. SimCSE, Contriever, E5, GTE and BGE got their pairs from dropout, crops, or hundreds of millions of mined web pairs.
4. **Encoder to LLM backbone (2022–24).** cpt-text started from GPT, e5-mistral from Mistral-7B. LLM2Vec and NV-Embed removed the causal mask (which lets each token see only the tokens before it) so every token sees the whole input.
5. **Single model to a family of sizes, Matryoshka dimensions and instructions (2022–25).** INSTRUCTOR made the task part of the input. Gemini Embedding, Qwen3-Embedding and EmbeddingGemma ship several sizes or several **Matryoshka (MRL)** cut points, where the first *k* numbers of the vector work on their own ([M4 · Matryoshka](README.md#dimensionality-matryoshka-truncation-and-the-storage-bill)).

---

## 2013–2016: static word vectors

### word2vec (2013), brief context
- **Introduced:** shallow models for "computing continuous vector representations of words from very large data sets". CBOW predicts a word from its context, and skip-gram predicts the context from a word [[W2V13]](#src-w2v13).
- **Fixed:** systems that "treat words as atomic units - there is no notion of similarity between words" [[W2V13]](#src-w2v13).
- **Fell short:** GloVe: skip-gram methods "poorly utilize the statistics of the corpus since they train on separate local context windows instead of on global co-occurrence counts" [[GLOVE14]](#src-glove14).
- **Facts:** Jan 2013 · Google · static · local-window prediction · one word · up to 1,000-d in its tables.

### GloVe (2014)
- **Introduced:** a "global log-bilinear regression model" trained "only on the nonzero elements in a word-word cooccurrence matrix" [[GLOVE14]](#src-glove14).
- **Fixed:** it joined the two earlier families. Matrix factorisation (LSA) used global statistics but did "relatively poorly on the word analogy task", and skip-gram did the reverse. It reports "75% accuracy on the word analogy dataset" [[GLOVE14]](#src-glove14). **Fixed, on its own benchmarks.**
- **Fell short:** one vector per word, in any context. Popular models "ignore the morphology of words, by assigning a distinct vector to each word" [[FT16]](#src-ft16). ELMo sets itself against GloVe: its representations "are functions of the entire input sentence" [[ELMO18]](#src-elmo18).
- **Facts:** EMNLP 2014 · Stanford · static · co-occurrence counts, up to 42B Common Crawl tokens · one word · "All vectors are 300-dimensional" in its comparison.

### fastText (2016)
- **Introduced:** each word is a bag of character n-grams (3 to 6 characters), and its vector is the sum of their vectors [[FT16]](#src-ft16).
- **Fixed:** rare and unseen words. It can represent "words that did not appear in the training data" [[FT16]](#src-ft16). **Fixed for morphology, not for context.**
- **Fell short:** still one vector per word. Later measurement: "less than 5% of the variance in a word's contextualized representations can be explained by a static embedding" [[ETH19]](#src-eth19).
- **Facts:** Jul 2016 (TACL 2017) · Facebook AI Research · static + subwords · sum of n-grams · skip-gram · one word · 300-d.

---

## 2018: contextual vectors and the first sentence encoders

### ELMo (Feb 2018)
- **Introduced:** word vectors that "are learned functions of the internal states of a deep bidirectional language model (biLM)" [[ELMO18]](#src-elmo18).
- **Fixed:** one vector per word. It models "how these uses vary across linguistic contexts (i.e., to model polysemy)" [[ELMO18]](#src-elmo18). **Fixed:** the 5% result above is the evidence [[ETH19]](#src-eth19).
- **Fell short:** BERT calls it "a shallow concatenation of independently trained left-to-right and right-to-left LMs" [[BERT18]](#src-bert18). It gives token vectors, not a sentence vector.
- **Facts:** Feb 2018 · AI2, UW · two-layer biLSTM · "a linear function of the internal network states" · language modelling · — · "4096 units and 512 dimension projections".

### Universal Sentence Encoder (Mar 2018)
- **Introduced:** two sentence encoders for transfer: "One makes use of the transformer … architecture, while the other is formulated as a deep averaging network (DAN)" [[USE18]](#src-use18). A **DAN** averages word and bigram embeddings and passes the average through a feed-forward network.
- **Fixed:** word-level transfer: "transfer learning using sentence embeddings tends to outperform word level transfer" [[USE18]](#src-use18). **Partly:** Sentence-BERT later beat it by 5.5 STS points [[SB19]](#src-sb19).
- **Fell short:** the transformer variant's "compute time and memory usage scaling dramatically with sentence length" [[USE18]](#src-use18). The Gemini report cites USE among embedders with "limitations in their ability to generalize effectively across varied tasks and domains" [[GEM25]](#src-gem25).
- **Facts:** 29 Mar 2018 · Google Research · transformer or DAN · sum ÷ √length (transformer) · multi-task: Skip-Thought-like, conversational response, SNLI · — · 512-d. Not shown to be the first transformer sentence encoder. At v2 the DAN "is already available" and the transformer "will be made available at a later point" [[USE18]](#src-use18).

### GPT-1 (Jun 2018)
- **Introduced:** "a 12-layer decoder-only transformer with masked self-attention heads", pre-trained on BooksCorpus, then fine-tuned per task [[GPT18]](#src-gpt18). It wasn't trained to produce embeddings.
- **Fixed:** nothing earlier in embeddings. It was a new direction: fine-tune a whole pre-trained transformer.
- **Fell short:** a similarity score needs both sentences in one input, "both possible sentence orderings", so vectors can't be precomputed [[GPT18]](#src-gpt18). In GPT, "every token can only attend to previous tokens" [[BERT18]](#src-bert18).
- **Facts:** 2018 (PDF created 2018-06-08) · OpenAI · decoder · no embedding pooling · next-token LM · "contiguous sequences of 512 tokens" · 768-d states.

### BERT (Oct 2018)
- **Introduced:** a "masked language model" objective, so an encoder can be "deep bidirectional": every token attends to every token [[BERT18]](#src-bert18).
- **Fixed:** GPT's left-to-right limit and ELMo's shallow concatenation [[BERT18]](#src-bert18). **Fixed for token and pair tasks.**
- **Fell short:** sentence vectors. Pooling BERT "yields rather bad sentence embeddings, often worse than averaging GloVe embeddings". On STS: GloVe average 61.32, BERT average 54.81, BERT `[CLS]` 29.19 [[SB19]](#src-sb19). Its token vectors are "not isotropic in any layer" [[ETH19]](#src-eth19).
- **Facts:** Oct 2018 · Google AI Language · encoder · `[CLS]` for classification · MLM + next-sentence prediction · 512 · 768 / 1,024.

---

## 2019–2021: training for the vector itself

### Sentence-BERT (2019)
- **Introduced:** siamese fine-tuning of BERT, so sentence vectors "can be compared using cosine-similarity" [[SB19]](#src-sb19). This is the **bi-encoder**: each text is encoded separately.
- **Fixed:** BERT's pairwise cost and its weak pooled vectors. Search over 10,000 sentences takes "about 5 seconds" instead of 65 hours. It beats InferSent by 11.7 STS points and USE by 5.5 [[SB19]](#src-sb19). **Fixed.**
- **Fell short:** it needs labelled pairs, and it transfers unevenly. In its own cross-topic test it shows "a performance drop of SBERT by about 7 points Spearman correlation" [[SB19]](#src-sb19).
- **Facts:** Aug 2019 · UKP, TU Darmstadt · encoder · mean ("The default configuration is MEAN") · NLI/STS labels · BERT's 512 · 768 / 1,024.

### DPR (2020)
- **Introduced:** a dual encoder trained on question–passage pairs, with **in-batch negatives** plus "one BM25 negative passage" [[DPR20]](#src-dpr20). In-batch negatives use the other questions' passages in the batch as wrong answers.
- **Fixed:** sparse retrieval for open-domain QA: "9%-19% absolute" top-20 gain over Lucene-BM25 [[DPR20]](#src-dpr20). **Fixed, in domain.**
- **Fell short:** out of domain. "DPR … overall performs the worst in generalization on the benchmark" [[BEIR21]](#src-beir21).
- **Facts:** Apr 2020 · Facebook AI, UW, Princeton · two BERT-base encoders · `[CLS]`, "so d = 768" · QA pairs · 100-word passages · 768.

### SimCSE (2021)
- **Introduced:** a sentence "predicts itself … with only standard dropout used as noise". A supervised variant uses NLI entailment pairs as positives and contradictions as **hard negatives** (wrong answers that look plausible) [[SIMCSE21]](#src-simcse21).
- **Fixed:** BERT's anisotropy. Contrastive learning "regularizes pre-trained embeddings' anisotropic space to be more uniform". STS: 76.3 (unsupervised) and 81.6 (supervised) [[SIMCSE21]](#src-simcse21). **Fixed for STS.**
- **Fell short:** retrieval. SimCSE "demonstrates limitations in text retrieval tasks" [[GTE23]](#src-gte23). DPR "is stronger for retrieval than text similarity tasks, and vice versa for SimCSE" [[INSTR22]](#src-instr22).
- **Facts:** Apr 2021 · Princeton, Tsinghua · BERT / RoBERTa · `[CLS]` (the unsupervised model keeps the MLP "only during training") · contrastive · — · 768 / 1,024.

### Contriever (Dec 2021)
- **Introduced:** unsupervised contrastive pretraining for retrieval. Two random crops of a document form a positive pair, with in-batch or MoCo negatives [[CONTR21]](#src-contr21). **MoCo** keeps a queue of negatives from a slowly updated copy of the model.
- **Fixed:** dense retrievers that "do not transfer well to new applications with no training data". It "outperforms BM25 on 11 out of 15 datasets for the Recall@100" [[CONTR21]](#src-contr21). **Partly.**
- **Fell short:** on averaged BEIR nDCG@10, E5 later reported "the first reported result that an unsupervised model can beat BM25" [[E522]](#src-e522). So by that metric, Contriever had not.
- **Facts:** Dec 2021 (TMLR 2022) · Meta AI et al. · BERT-base · mean · unsupervised crops · — · 768.

---

## 2022–2023: contrastive pretraining at scale, and an early LLM embedder

### cpt-text (Jan 2022)
- **Introduced:** "cpt-text is initialized with GPT models". It is trained contrastively on "neighboring pieces of text on the Internet as positive pairs", and the last-layer `[EOS]` state is the embedding [[CPT22]](#src-cpt22).
- **Fixed:** separate models for sentence embedding and for search. One model reports +23.4%, +14.7% and +10.6% relative on MSMARCO, NQ and TriviaQA over the best unsupervised methods [[CPT22]](#src-cpt22). **Fixed for search.**
- **Fell short:** "our models underperformed on sentence similarity tasks" [[CPT22]](#src-cpt22). On MTEB, the "OpenAI Ada Similarity" API model scored 18.36 on retrieval, below GloVe's 21.62 [[MTEB22]](#src-mteb22). That's the similarity model, not the search one.
- **Facts:** Jan 2022 · OpenAI · decoder LLM (300M to 175B) · `[EOS]` · unsupervised contrastive · — · 1,024 to 12,288. Gemini's report credits it with initialising "from decoder-only GPT-3". The same report cites earlier T5-based work (Ni et al., 2021), so "first" is unverified [[GEM25]](#src-gem25).

### E5 (Dec 2022)
- **Introduced:** contrastive pretraining on **CCPairs**: 1.3B web pairs, filtered to 270M. Inputs carry `query:` / `passage:` prefixes. Fine-tuning follows [[E522]](#src-e522).
- **Fixed:** Contriever's BM25 gap: "the first model that outperforms the strong BM25 baseline on the BEIR retrieval benchmark without using any labeled data" [[E522]](#src-e522). **Fixed, by 1.2 nDCG@10.**
- **Fell short:** the same group later wrote that multi-stage approaches "entail a complex multistage training pipeline that demands substantial engineering efforts to curate large amounts of relevance pairs" [[E5M24]](#src-e5m24).
- **Facts:** Dec 2022 · Microsoft · MiniLM / BERT-base / BERT-large · mean · weakly supervised → fine-tune · 512 at evaluation · backbone hidden size.

### INSTRUCTOR (Dec 2022)
- **Introduced:** "every text input is embedded together with instructions explaining the use case", e.g. "Represent the Wikipedia question for retrieving supporting documents:". Training uses 330 datasets (MEDI) [[INSTR22]](#src-instr22).
- **Fixed:** embeddings that "perform poorly when applied to the same type of task but in different domains" [[INSTR22]](#src-instr22). It reports "an average improvement of 3.4%" over 70 datasets. **Partly.**
- **Fell short:** sensitivity to wording. "The default prompt can both systematically understate or overstate performance" [[ONEP26]](#src-onep26).
- **Facts:** Dec 2022 · HKU, UW, Meta AI, AI2 · GTR (T5) encoder · mean over the text tokens, not the instruction · multi-task contrastive · — · —.

### GTE (Aug 2023)
- **Introduced:** "multi-stage contrastive learning": ~800M weak web pairs, then labelled triples [[GTE23]](#src-gte23).
- **Fixed:** task-specific models such as SimCSE. "GTEbase outperforms the black-box embedding API provided by OpenAI" on MTEB [[GTE23]](#src-gte23). **Fixed on MTEB.**
- **Fell short:** the multi-stage cost [[E5M24]](#src-e5m24), and a short window: 128 tokens in pretraining, "512" in fine-tuning [[GTE23]](#src-gte23).
- **Facts:** Aug 2023 · Alibaba · MiniLM-L12-H384 / BERT-base / BERT-large · mean (gte-base config, 768-d [[HF-GTEB]](#src-hf-gteb)) · weak → supervised · 512 · 384 / 768 / 1,024.

### BGE / C-Pack (Sep 2023)
- **Introduced:** an open recipe with open data. RetroMAE-style pretraining comes first. Then weakly supervised contrastive training with "a big batch size (as large as 19,200)". Then labelled fine-tuning with a query-side instruction [[CPACK23]](#src-cpack23).
- **Fixed:** closed training data. Its Chinese and English sets are "the largest public release of training data for text embeddings". Retrieval improves from 57.25 to 63.90 as the batch grows from 256 to 19,200 [[CPACK23]](#src-cpack23). **Fixed at release.**
- **Fell short:** the multi-stage cost [[E5M24]](#src-e5m24). No later source here measures a BGE-specific failure.
- **Facts:** Sep 2023 · BAAI · encoder · `[CLS]` ([[HF-BGEB]](#src-hf-bgeb)) · pretrain → weak → labelled · — · 768 (base).

---

## Late 2023–2024: LLM backbones

### e5-mistral (Dec 2023)
- **Introduced:** fine-tuning Mistral-7B with "only synthetic data and less than 1k training steps". The synthetic data comes from proprietary LLMs and covers "93 languages". Pooling is "the last layer [EOS] vector" [[E5M24]](#src-e5m24).
- **Fixed:** E5/BGE-style pipelines, and the 512-token limit: passkey retrieval worked "for inputs up to 32k tokens". Dropping the instruction costs 4.2 MTEB points [[E5M24]](#src-e5m24). **Fixed the pipeline. Long context only partly.**
- **Fell short:** "inputs longer than 4096 tokens is not recommended" [[HF-E5M]](#src-hf-e5m). Low-resource languages have "still room for improvement" [[E5M24]](#src-e5m24). It is among the models that "struggle" on LIMIT [[LIMIT25]](#src-limit25).
- **Facts:** 31 Dec 2023 · Microsoft · decoder, causal · last token · synthetic + labelled, LoRA · 4,096 recommended · 4,096.

### LLM2Vec (Apr 2024)
- **Introduced:** "1) enabling bidirectional attention, 2) masked next token prediction, and 3) unsupervised contrastive learning", applied to any decoder LLM [[LLM2VEC24]](#src-llm2vec24).
- **Fixed:** causal attention, which "prevents the representations from capturing information across the entire input sequence". Its Llama-3-8B model reaches "state-of-the-art performance among models trained only on publicly available data" [[LLM2VEC24]](#src-llm2vec24). **Fixed.**
- **Fell short:** just switching on bidirectional attention "without any training harms performance" for two of its LLMs, so the extra training step is needed [[LLM2VEC24]](#src-llm2vec24). NV-Embed calls itself "simpler yet more effective" than adding "an additional training phase with masked token prediction" [[NVE24]](#src-nve24).
- **Facts:** Apr 2024 · McGill/Mila, ServiceNow · decoder → bidirectional, 1.3B–8B · mean · MNTP + SimCSE · 512 in training · base LLM's hidden size.

### NV-Embed (May 2024)
- **Introduced:** a "latent attention layer to obtain pooled embeddings", in which token states attend to 512 trainable latents. It also removes "the causal attention mask of LLMs during contrastive training" [[NVE24]](#src-nve24).
- **Fixed:** mean and last-token pooling. Latent attention "consistently improves retrieval and downstream task accuracy" over both. It reached MTEB No. 1: 69.32 in May 2024, and 72.31 for v2 in Aug 2024 [[NVE24]](#src-nve24). **Fixed on MTEB.**
- **Fell short:** EmbeddingGemma found "simple pooling types tend to outperform attention pooling" [[EGEMMA25]](#src-egemma25). That test used a single-query pooler, not NV-Embed's layer. And the v2 card says: "should not be used for any commercial purpose" [[HF-NVE2]](#src-hf-nve2).
- **Facts:** May 2024 · NVIDIA · Mistral-7B → bidirectional · latent attention · two-stage contrastive instruction tuning · 512 in training (card example 32,768) · 4,096.

---

## 2025: lab-scale LLM embedder families

### Gemini Embedding (Mar 2025)
- **Introduced:** "a novel embedding model initialized from the powerful Gemini large language model". It is "a transformer with bidirectional attention", with mean pooling and a projection to 3,072-d with MRL. Gemini filters the data, picks the positives and negatives, and writes synthetic data. Training is pre-finetuning on noisy pairs, then fine-tuning, then a **Model Soup**, which averages several fine-tuned checkpoints [[GEM25]](#src-gem25).
- **Fixed:** the weak generalisation of USE-era general embedders. It reports first place on MTEB(Multilingual), "significantly surpassing the previous best models" [[GEM25]](#src-gem25). **Fixed at release.**
- **Fell short:** "excessive hard negatives often led to overfitting" [[GEM25]](#src-gem25). "Gemini Embed" is among the models that struggle on LIMIT [[LIMIT25]](#src-limit25). It's API-only, and its size isn't stated.
- **Facts:** 10 Mar 2025 · Google · Gemini LLM, bidirectional · mean + projection · two-stage contrastive + soup · 2,048 for `gemini-embedding-001` · 3,072, MRL 768 / 1,536. The docs now also list a multimodal `gemini-embedding-2` with an 8,192-token input [[GAPI-EMB]](#src-gapi-emb); see [image and video embeddings](image-and-video-embeddings.md).

### Qwen3-Embedding (Jun 2025)
- **Introduced:** a 0.6B / 4B / 8B family on Qwen3, "appending an [EOS] token" with causal attention, instructions on the query side, MRL, and 32K input. Qwen3 also *writes* ~150M weak-supervision pairs. ~12M filtered pairs are used for supervised training, and checkpoints are merged with slerp (spherical interpolation) [[QWEN3E25]](#src-qwen3e25).
- **Fixed:** weak pairs mined "from open-source communities such as Q&A forums" (GTE, E5, BGE) are replaced with LLM-synthesised ones. The 8B reports "70.58 on the MTEB Multilingual benchmark" [[QWEN3E25]](#src-qwen3e25). **Fixed, on its own results.**
- **Fell short:** "even models with strong performance in the MTEB multilingual benchmark may struggle in XTREME-UP, such as Qwen3 Embedding 0.6B" [[EGEMMA25]](#src-egemma25). "Qwen3 Embed" also struggles on LIMIT [[LIMIT25]](#src-limit25).
- **Facts:** Jun 2025 · Alibaba Tongyi Lab · decoder, causal · last token · synthetic → supervised → merge · 32K · 1,024 / 2,560 / 4,096.

### EmbeddingGemma (Sep 2025)
- **Introduced:** a 308M model that doesn't use the Gemma 3 decoder directly. Gemma 3 is first adapted "into an encoder-decoder model", and EmbeddingGemma starts from that encoder. It distils "the state-of-the-art Gemini Embedding", and uses "bidirectional attention", "mean pooling" and a "spread-out" regulariser [[EGEMMA25]](#src-egemma25).
- **Fixed:** small models. It "outperforms prior top models … with fewer than 500M parameters", and the lead holds at 128 dimensions or 4-bit weights [[EGEMMA25]](#src-egemma25). **Fixed for its size class.**
- **Fell short:** 2,048-token input, and activations "do not support `float16`" [[HF-EGEMMA]](#src-hf-egemma). It's "comparable to models nearly double its size", not to the 4B–8B tier [[EGEMMA25]](#src-egemma25).
- **Facts:** Sep 2025 · Google · encoder from Gemma 3 · mean + projections · distillation + contrastive + souping · 2,048 · 768, MRL 512 / 256 / 128.

### OpenAI `text-embedding-ada-002` and `text-embedding-3-*`, architecture undisclosed
- **Introduced:** hosted models. Max input 8,192. MTEB 61.0 / 62.3 / 64.6 (ada-002 / 3-small / 3-large). `dimensions` shortens `-3-*` vectors [[OAI-EMB]](#src-oai-emb). Outputs are 1,536 / 1,536 / 3,072 [[MSAOAI]](#src-msaoai).
- **Fixed / Fell short:** can't be assessed from primary sources. Neither page discloses architecture, pooling or training data [[OAI-EMB]](#src-oai-emb)[[MSAOAI]](#src-msaoai), so this file doesn't guess a lineage. Community measurements of ada-002's cone: [M4 · The cone effect](README.md#the-cone-effect-raw-embeddings-are-not-isotropic).

---

## What has not been fixed

- **Anisotropy and thresholds.** Contextual vectors are "not isotropic in any layer" [[ETH19]](#src-eth19). Contrastive training makes the space "more uniform" [[SIMCSE21]](#src-simcse21), but no model in this history makes a cosine score mean the same thing across models. Thresholds still have to be measured per model: [M4 · The cone effect](README.md#the-cone-effect-raw-embeddings-are-not-isotropic) and [M4 · Worked threshold](README.md#worked-setting-a-similarity-threshold-from-two-distributions).
- **The dimension-bounded expressivity limit.** "the number of top-k subsets of documents capable of being returned as the result of some query is limited by the dimension of the embedding" [[LIMIT25]](#src-limit25). The 2024–25 LLM embedders (E5-Mistral, Qwen3, Gemini Embedding) are the models that struggle on LIMIT, so a bigger backbone hasn't removed it. See [M4 · Choosing an instrument](README.md#choosing-an-instrument-beir-mteb-and-your-own-60-queries).
- **Domain and internal jargon.** INSTRUCTOR named the problem in 2022: embeddings "perform poorly when applied to the same type of task but in different domains" [[INSTR22]](#src-instr22). BEIR: "In-domain performance is not a good indicator for out-of-domain generalization" [[BEIR21]](#src-beir21). Instructions and synthetic data widen coverage, but they don't teach your company's words. See [M4 · What an embedding model is optimizing](README.md#what-an-embedding-model-is-optimizing) and [M15 · the model does not know this word](../15-finetuning-retrieval-models/README.md#opening-scene--the-model-does-not-know-this-word).
- **Leaderboard rank vs. your own evaluation.** MTEB found "no particular text embedding method dominates across all tasks" [[MTEB22]](#src-mteb22). Instruction-aware models can be re-ranked by prompt choice [[ONEP26]](#src-onep26). Every "Fixed" above is on the authors' benchmark, not your corpus. See [M4 · Choosing an instrument](README.md#choosing-an-instrument-beir-mteb-and-your-own-60-queries) and [M4 · Where the literature disagrees](README.md#where-the-literature-disagrees).

---

## Sources

- <a id="src-w2v13"></a>*Efficient Estimation of Word Representations in Vector Space* — Mikolov, Chen, Corrado, Dean, Google · arXiv 2013-01-16 · [arxiv.org/abs/1301.3781](https://arxiv.org/abs/1301.3781) — `[W2V13]` · paper.
- <a id="src-glove14"></a>*GloVe: Global Vectors for Word Representation* — Pennington, Socher, Manning, Stanford · EMNLP 2014 · [aclanthology.org/D14-1162](https://aclanthology.org/D14-1162/) (PDF read) — `[GLOVE14]` · paper.
- <a id="src-ft16"></a>*Enriching Word Vectors with Subword Information* — Bojanowski, Grave, Joulin, Mikolov, Facebook AI Research · arXiv 2016-07-15, TACL 2017 · [arxiv.org/abs/1607.04606](https://arxiv.org/abs/1607.04606) — `[FT16]` · paper.
- <a id="src-elmo18"></a>*Deep contextualized word representations* — Peters et al., AI2 / UW · arXiv 2018-02-15, NAACL 2018 · [arxiv.org/abs/1802.05365](https://arxiv.org/abs/1802.05365) — `[ELMO18]` · paper.
- <a id="src-use18"></a>*Universal Sentence Encoder* — Cer et al., Google Research · arXiv 2018-03-29 (v2 2018-04-12) · [arxiv.org/abs/1803.11175](https://arxiv.org/abs/1803.11175) — `[USE18]` · paper.
- <a id="src-gpt18"></a>*Improving Language Understanding by Generative Pre-Training* — Radford, Narasimhan, Salimans, Sutskever, OpenAI · 2018 (PDF metadata 2018-06-08) · [cdn.openai.com PDF](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf) — `[GPT18]` · paper (preprint).
- <a id="src-bert18"></a>*BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding* — Devlin, Chang, Lee, Toutanova, Google AI Language · arXiv:1810.04805 (NAACL 2019) · [arxiv.org/abs/1810.04805](https://arxiv.org/abs/1810.04805) — `[BERT18]` · paper.
- <a id="src-sb19"></a>*Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks* — Reimers, Gurevych, UKP TU Darmstadt · arXiv:1908.10084 (EMNLP 2019) · [arxiv.org/abs/1908.10084](https://arxiv.org/abs/1908.10084) — `[SB19]` · paper.
- <a id="src-eth19"></a>*How Contextual are Contextualized Word Representations?* — Ethayarajh, Stanford · arXiv:1909.00512 (EMNLP 2019) · [arxiv.org/abs/1909.00512](https://arxiv.org/abs/1909.00512) — `[ETH19]` · paper.
- <a id="src-dpr20"></a>*Dense Passage Retrieval for Open-Domain Question Answering* — Karpukhin et al., Facebook AI / UW / Princeton · arXiv:2004.04906 (EMNLP 2020) · [arxiv.org/abs/2004.04906](https://arxiv.org/abs/2004.04906) — `[DPR20]` · paper.
- <a id="src-beir21"></a>*BEIR: A Heterogenous Benchmark for Zero-shot Evaluation of Information Retrieval Models* — Thakur et al., UKP · arXiv:2104.08663 (NeurIPS 2021 D&B) · [arxiv.org/abs/2104.08663](https://arxiv.org/abs/2104.08663) — `[BEIR21]` · paper / benchmark.
- <a id="src-simcse21"></a>*SimCSE: Simple Contrastive Learning of Sentence Embeddings* — Gao, Yao, Chen, Princeton / Tsinghua · arXiv:2104.08821 (EMNLP 2021) · [arxiv.org/abs/2104.08821](https://arxiv.org/abs/2104.08821) — `[SIMCSE21]` · paper.
- <a id="src-contr21"></a>*Unsupervised Dense Information Retrieval with Contrastive Learning* (Contriever) — Izacard et al., Meta AI Research et al. · arXiv 2021-12-16, TMLR 08/2022 · [arxiv.org/abs/2112.09118](https://arxiv.org/abs/2112.09118) — `[CONTR21]` · paper.
- <a id="src-cpt22"></a>*Text and Code Embeddings by Contrastive Pre-Training* — Neelakantan, Xu et al., OpenAI · arXiv 2022-01-24 · [arxiv.org/abs/2201.10005](https://arxiv.org/abs/2201.10005) — `[CPT22]` · paper.
- <a id="src-mteb22"></a>*MTEB: Massive Text Embedding Benchmark* — Muennighoff, Tazi, Magne, Reimers · arXiv:2210.07316 (2022) · [arxiv.org/abs/2210.07316](https://arxiv.org/abs/2210.07316) — `[MTEB22]` · paper / benchmark.
- <a id="src-e522"></a>*Text Embeddings by Weakly-Supervised Contrastive Pre-training* — Wang et al., Microsoft · arXiv:2212.03533 (2022; v2 2024-02-22) · [arxiv.org/abs/2212.03533](https://arxiv.org/abs/2212.03533) — `[E522]` · paper.
- <a id="src-instr22"></a>*One Embedder, Any Task: Instruction-Finetuned Text Embeddings* — Su, Shi et al., HKU / UW / Meta AI / AI2 · arXiv:2212.09741 (ACL 2023 Findings) · [arxiv.org/abs/2212.09741](https://arxiv.org/abs/2212.09741) — `[INSTR22]` · paper.
- <a id="src-gte23"></a>*Towards General Text Embeddings with Multi-stage Contrastive Learning* — Li et al., Alibaba · arXiv 2023-08-07 · [arxiv.org/abs/2308.03281](https://arxiv.org/abs/2308.03281) — `[GTE23]` · paper.
- <a id="src-hf-gteb"></a>[thenlper/gte-base `1_Pooling/config.json`](https://huggingface.co/thenlper/gte-base/raw/main/1_Pooling/config.json) — Hugging Face · read 2026-11-14 — `[HF-GTEB]` · model config.
- <a id="src-cpack23"></a>*C-Pack: Packed Resources For General Chinese Embeddings* (BGE) — Xiao, Liu et al., BAAI et al. · arXiv 2023-09-14 (SIGIR 2024) · [arxiv.org/abs/2309.07597](https://arxiv.org/abs/2309.07597) — `[CPACK23]` · paper.
- <a id="src-hf-bgeb"></a>[BAAI/bge-base-en-v1.5 `1_Pooling/config.json`](https://huggingface.co/BAAI/bge-base-en-v1.5/raw/main/1_Pooling/config.json) — Hugging Face · read 2026-11-14 — `[HF-BGEB]` · model config.
- <a id="src-e5m24"></a>*Improving Text Embeddings with Large Language Models* — Wang et al., Microsoft · arXiv:2401.00368 (2023-12-31; ACL 2024) · [arxiv.org/abs/2401.00368](https://arxiv.org/abs/2401.00368) — `[E5M24]` · paper.
- <a id="src-hf-e5m"></a>[intfloat/e5-mistral-7b-instruct](https://huggingface.co/intfloat/e5-mistral-7b-instruct) — Microsoft (intfloat) · undated — `[HF-E5M]` · model card.
- <a id="src-llm2vec24"></a>*LLM2Vec: Large Language Models Are Secretly Powerful Text Encoders* — BehnamGhader, Adlakha et al., McGill/Mila / ServiceNow · arXiv 2024-04-09 (COLM 2024) · [arxiv.org/abs/2404.05961](https://arxiv.org/abs/2404.05961) — `[LLM2VEC24]` · paper.
- <a id="src-nve24"></a>*NV-Embed: Improved Techniques for Training LLMs as Generalist Embedding Models* — Lee et al., NVIDIA · arXiv 2024-05-27 (ICLR 2025) · [arxiv.org/abs/2405.17428](https://arxiv.org/abs/2405.17428) — `[NVE24]` · paper.
- <a id="src-hf-nve2"></a>[nvidia/NV-Embed-v2](https://huggingface.co/nvidia/NV-Embed-v2) — NVIDIA · undated — `[HF-NVE2]` · model card.
- <a id="src-gem25"></a>*Gemini Embedding: Generalizable Embeddings from Gemini* — Lee, Chen, Dua, Cer et al., Google · arXiv 2025-03-10 · [arxiv.org/abs/2503.07891](https://arxiv.org/abs/2503.07891) — `[GEM25]` · technical report.
- <a id="src-gapi-emb"></a>[Embeddings](https://aistudio.google.com/docs/embeddings) (redirected from ai.google.dev/gemini-api/docs/embeddings) — Google · model table "Latest update" June 2025 / April 2026 — `[GAPI-EMB]` · official doc.
- <a id="src-qwen3e25"></a>*Qwen3 Embedding: Advancing Text Embedding and Reranking Through Foundation Models* — Zhang et al., Alibaba Tongyi Lab · arXiv 2025-06-05 (v3 2025-06-11) · [arxiv.org/abs/2506.05176](https://arxiv.org/abs/2506.05176) — `[QWEN3E25]` · technical report.
- <a id="src-egemma25"></a>*EmbeddingGemma: Powerful and Lightweight Text Representations* — Schechter Vera, Dua et al., Google · arXiv 2025-09-24 (v3 2025-11-01) · [arxiv.org/abs/2509.20354](https://arxiv.org/abs/2509.20354) — `[EGEMMA25]` · technical report.
- <a id="src-hf-egemma"></a>[google/embeddinggemma-300m](https://huggingface.co/google/embeddinggemma-300m) — Google · undated — `[HF-EGEMMA]` · model card.
- <a id="src-limit25"></a>*On the Theoretical Limitations of Embedding-Based Retrieval* — Weller, Boratko, Naim, Lee, Google DeepMind / JHU · arXiv:2508.21038 (ICLR 2026) · [arxiv.org/abs/2508.21038](https://arxiv.org/abs/2508.21038) — `[LIMIT25]` · paper.
- <a id="src-onep26"></a>*One prompt is not enough: Instruction Sensitivity Undermines Embedding Model Evaluation* — Kostiuk, Enevoldsen · arXiv:2605.22544 (2026) · [arxiv.org/abs/2605.22544](https://arxiv.org/abs/2605.22544) — `[ONEP26]` · paper (abstract read).
- <a id="src-oai-emb"></a>[OpenAI · Vector embeddings guide](https://developers.openai.com/api/docs/guides/embeddings) — OpenAI · undated — `[OAI-EMB]` · official doc.
- <a id="src-msaoai"></a>[Azure OpenAI models — embeddings](https://learn.microsoft.com/en-us/azure/ai-foundry/openai/concepts/models) — Microsoft Learn · ms.date 2026-09-21 — `[MSAOAI]` · official doc.

**Unsupported or not verified:**
- **Pages that failed:** OpenAI's posts on GPT-1, `text-embedding-ada-002` and `text-embedding-3` all returned 403, so the OpenAI release dates and the GPT-1 month are unverified. The GPT-1 date comes from the PDF's creation metadata. The GloVe project page (nlp.stanford.edu) refused the connection, so the released GloVe vector sizes come only from the paper's tables.
- **Argued, not sourced:** the five-shift grouping and its year ranges; the "Fixed / Partly" verdicts, which weigh the cited numbers; "a bigger backbone hasn't removed it" (LIMIT tests several sizes but doesn't make that claim); "they don't teach your company's words".
- **Not verified:** that USE was the earliest transformer sentence embedder; that cpt-text was the first major lab's LLM embedder (the Gemini report points to earlier T5-based work); the mapping from MTEB's "OpenAI Ada Similarity" to a specific cpt-text checkpoint; Gemini Embedding's parameter count (not stated); LLM2Vec's and INSTRUCTOR's output dimensions; max inputs marked "—", which the papers don't state; whether OpenAI's hosted models descend from cpt-text.
- **Self-reported only:** every MTEB, BEIR, STS and leaderboard number above comes from the model's own authors, except the MTEB22, BEIR21 and SB19 comparisons of *other* models.
