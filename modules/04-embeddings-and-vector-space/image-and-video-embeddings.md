# Image and video embeddings: retrieval over pictures, scanned pages and video

> **What this covers:** images, scanned pages and video as corpora in their own right: shared image-text spaces and the models that make them, ColPali and its storage bill, video RAG, and what LlamaIndex supports today.
> **Who it's for:** anyone building retrieval over photos, scanned documents or recorded video.
> **Status in the course:** an extension point. Multimodal retrieval is *deferred by decision* in the frozen scope ([syllabus §1](../../syllabus.md#1-scope-knobs-frozen)), so no module teaches it and the labs stay text-only. This file is here for when that scope opens.
> **Evidence:** researched 2026-11-13. Most limits and dimensions come from vendor docs and model cards; most benchmark numbers come from the people who built the model or the benchmark. Each claim is labelled with its source type, and *derived* marks this file's own arithmetic.
> Back to [M4 · Input modalities](README.md#1--tokens) · the [storage bill](README.md#dimensionality-matryoshka-truncation-and-the-storage-bill) this file re-derives.

## Contents

- [Why text-only embedders are not enough](#why-text-only-embedders-are-not-enough)
- [How a shared image-text space works](#how-a-shared-image-text-space-works)
- [Image-text embedders](#image-text-embedders)
  - [Choosing one](#choosing-one)
- [Scanned documents as images: ColPali and late interaction](#scanned-documents-as-images-colpali-and-late-interaction)
  - [Worked: the storage bill for many vectors per page](#worked-the-storage-bill-for-many-vectors-per-page)
- [Video RAG](#video-rag)
  - [Clip embedders](#clip-embedders)
  - [Hosted video APIs and their limits](#hosted-video-apis-and-their-limits)
  - [The pipeline, step by step](#the-pipeline-step-by-step)
- [Benchmarks: what each one measures](#benchmarks-what-each-one-measures)
- [LlamaIndex today](#llamaindex-today)
- [Design decisions](#design-decisions)
  - [One shared space, or caption into text](#one-shared-space-or-caption-into-text)
  - [Cost and storage](#cost-and-storage)
  - [Thresholds are per space](#thresholds-are-per-space)
  - [Evaluation](#evaluation)
- [Where the evidence disagrees](#where-the-evidence-disagrees)
- [Sources](#sources)

---

## Why text-only embedders are not enough

- **M4's models take text only.** Tables, figures and scans reach them as the text the parser made, and what the parser dropped, no embedder recovers ([M4 · Multimodal inputs](README.md#what-an-embedding-model-is-optimizing)).
- **Text pipelines over page images are fragile.** The ColPali authors call them "lengthy and brittle processes" [[COLPALI24]](#src-colpali24). The Hugging Face write-up adds that they "tend to propagate errors" and struggle "to take into account the more visual elements of a page" [[COLPALI-BLOG]](#src-colpali-blog) (vendor blog).
- **The measured gap is large on documents.** On ViDoRe, OCR plus a text embedder scores 66.1 nDCG@5, and ColPali, which embeds the page image, scores 81.3 [[COLPALI-HTML]](#src-colpali-html).
  - **nDCG@5** is a ranking score from 0 to 100 that rewards putting the relevant page near the top of the first five results.
- **Three ways to index an image:** caption or OCR it into text (the cheap baseline); one vector per image in a space shared with text; or many vectors per page with late interaction ([compared below](#one-shared-space-or-caption-into-text)).
- **Video adds time:** clips, a transcript channel and timestamps ([Video RAG](#video-rag)).
- **LogiTrack cases that fit** (*illustrative*):
  - A **scanned, signed and stamped carrier agreement** has no text layer, so OCR must run before any parser ([M3](../03-parsing-and-chunking/README.md#what-recoverable-means)).
  - A **DG label figure** in a handling guide: what matters is the label's look, which a caption may not name in ops staff's words.

---

## How a shared image-text space works

- **Two encoders, one space.** An image encoder and a text encoder each output a vector, and training makes them comparable: the "same semantic space with the same dimensionality" in Google's words [[VERTEX-MM]](#src-vertex-mm), a "shared semantic space" in Amazon's [[TITAN-MM]](#src-titan-mm).
- **Contrastive training, as in CLIP.** CLIP learns by "predicting which caption goes with which image" [[CLIP21]](#src-clip21). A matching image and caption are pushed together relative to mismatched pairs. That is the same contrastive family as M4's text objectives ([M4](README.md#what-an-embedding-model-is-optimizing)).
  - SigLIP 2 replaces CLIP's objective with a **sigmoid loss** [[SIGLIP2-BLOG]](#src-siglip2-blog) (vendor blog).
- **Why a text query can retrieve an image.** The image encoder embeds images at ingest; the text encoder embeds the query. Because training aligned the two, the query's nearest neighbours include the images it describes. LlamaIndex calls this `text_to_image_retrieve` [[LI-MM-GUIDE]](#src-li-mm-guide).
- **Scores are still relative ranks,** as in M4, and a text-image cosine is not on a text-text scale (*argued*), so thresholds are set [per space](#thresholds-are-per-space).
- **Two separate spaces cannot be compared.** Model A's vectors and model B's vectors were never trained against each other, so a cosine between them means nothing. LlamaIndex's docs say it plainly: "Embeddings from different models are not interchangeable" ([M4 opening scene](README.md#opening-scene--the-upgrade-that-changed-nothing)).
  - **A shared space belongs to one model pair.** nomic-embed-vision-v1.5 "shares the same embedding space as" nomic-embed-text-v1.5 [[NOMIC-VIS]](#src-nomic-vis). Swap either side and you re-embed both.
  - **Two spaces means two ranked lists.** Embed the query once per space, then merge the lists by rank, e.g. RRF ([M7 · Fusion](../07-sparse-dense-hybrid/README.md#fusion-why-two-ranked-lists-cannot-simply-be-added)), never by raw score.
- **More than two modalities.** ImageBind puts six modalities in one joint space [[IMAGEBIND]](#src-imagebind). LanguageBind binds video, infrared, depth and audio to a frozen language encoder, "taking the language as the bind across different modalities" [[LANGUAGEBIND]](#src-languagebind).
- **The newest embedders are built on vision-language models (VLMs) and take interleaved input:** text and images in one input, giving one fused vector (Cohere's term) [[COHERE-MODELS]](#src-cohere-models). The table below marks which ones.

---

## Image-text embedders

Every row maps text and images into one shared space. ColPali and ColQwen2 are in [the next section](#scanned-documents-as-images-colpali-and-late-interaction); Twelve Labs Marengo is under [video](#hosted-video-apis-and-their-limits).

| Model | Access · license | Inputs | Output D | Input limits (as stated) | Source |
|---|---|---|---|---|---|
| CLIP family: CLIP (OpenAI), OpenCLIP, EVA-CLIP | Open · MIT (CLIP repo, OpenCLIP code); EVA-CLIP license not shown on page | Text, image; OpenCLIP also CLAP audio | Varies by checkpoint, not stated on pages read | CLIP text `context_length=77`; OpenCLIP checkpoints at 224/256/448 px (by name); EVA-CLIP 149M–5.0B params, one 336 px variant | [[CLIP-REPO]](#src-clip-repo)[[OPENCLIP]](#src-openclip)[[EVACLIP23]](#src-evaclip23)[[EVACLIP-REPO]](#src-evaclip-repo) |
| SigLIP 2 (Google) | Open · Apache-2.0 (so400m card) | Text, image; multilingual | Not stated | Fixed 224/256/384/512; `-naflex` = dynamic resolution | [[SIGLIP2-BLOG]](#src-siglip2-blog)[[SIGLIP2-CARD]](#src-siglip2-card) |
| jina-clip-v2 | Open weights · CC BY-NC 4.0 (commercial use via API/clouds) | Text, image; 89 languages | 1024, Matryoshka down to 64 | 512×512 image; 8,192 text tokens | [[JINA-CLIP2]](#src-jina-clip2) |
| nomic-embed-vision-v1.5 | Open · Apache-2.0 | Image; text side is nomic-embed-text-v1.5 | Not stated on card | Not stated | [[NOMIC-VIS]](#src-nomic-vis) |
| ImageBind (Meta) | Open · CC-BY-NC 4.0 | Image, text, audio, depth, thermal, IMU | Not stated | — | [[IMAGEBIND]](#src-imagebind) |
| GME-Qwen2-VL-2B | Open · Apache-2.0 | Text, image, image+text pair | 1536 | ≤1,024 visual tokens; max sequence 32,768 | [[GME]](#src-gme) |
| Qwen3-VL-Embedding-8B | Open · Apache-2.0 | "Text, images, screenshots, videos, and arbitrary multimodal combinations" | Up to 4096; MRL 64–4096 | 32k context; frame and pixel limits not stated | [[QWEN3-VL-EMB]](#src-qwen3-vl-emb) |
| Cohere embed-v4.0 / embed-v5.0-pro | Hosted · proprietary | "Text, Images, Mixed texts/images (i.e. PDFs)" | v4.0: 256/512/1024/1536 (default); v5.0-pro: 256–2048 (default 2048) | 128k context; v5.0-pro downsamples images >2,458,624 px, 5 MB/image, up to 96 images | [[COHERE-MODELS]](#src-cohere-models)[[COHERE-EMB]](#src-cohere-emb) |
| voyage-multimodal-3.5 (and -3) | Hosted · proprietary | Interleaved text + image; video on 3.5 only | 3.5: 1024 default, 256/512/2048; 3: 1024 | 32k tokens/input; 560 px = 1 token (image), 1120 px = 1 token (video); ≤16 MP, ≤20 MB | [[VOYAGE-MM]](#src-voyage-mm) |
| Amazon Titan Multimodal Embeddings G1 | Hosted (Bedrock) · proprietary | Text, image, or both; English | 1024 (default), 384, 256 | 256 text tokens; 2048×2048 px; 25 MB | [[TITAN-MM]](#src-titan-mm) |
| Google `multimodalembedding@001` | Hosted (Vertex) · proprietary | Text, image, video; no audio | 1408 default (128–1408) | Images resized to 512×512; text 32 tokens, English | [[VERTEX-MM]](#src-vertex-mm) |
| Google `gemini-embedding-2` | Hosted (Gemini API / Vertex) · proprietary | Interleaved text, image, video, audio, PDF | 3072 default (128–3072) | 8,192 tokens total; 6 images/request (258 tokens each, per Vertex); video 120 s; audio 180 s; 1 PDF of ≤6 pages | [[VERTEX-MM]](#src-vertex-mm)[[GEMINI-EMB]](#src-gemini-emb) |

Two vendor pages are inconsistent (Cohere, Voyage); see [Where the evidence disagrees](#where-the-evidence-disagrees).

### Choosing one

- **Check the license before the leaderboard.** jina-clip-v2 and ImageBind are non-commercial (CC BY-NC) [[JINA-CLIP2]](#src-jina-clip2)[[IMAGEBIND]](#src-imagebind).
- **Check the text side's input limit.** CLIP's text side takes 77 tokens, Titan's 256 and Google `@001`'s 32 (English) [[CLIP-REPO]](#src-clip-repo)[[TITAN-MM]](#src-titan-mm)[[VERTEX-MM]](#src-vertex-mm). That fits a query, not a text chunk.
  - So the text chunks need a second, text-only embedder, which means two spaces and a fusion step ([above](#how-a-shared-image-text-space-works)).
  - jina-clip-v2 (8,192 tokens), GME (32,768), Voyage (32k) and Cohere (128k) can embed chunks and images in one space.
- **Match the model to the content.**
  - **Natural photos:** a shared space is the direct route.
  - **Document pages:** a single-vector image embedder loses to the text pipeline (51.4 against 67.0 on ViDoRe) [[COLPALI-HTML]](#src-colpali-html). See [the next section](#scanned-documents-as-images-colpali-and-late-interaction).
  - **Pages mixing a figure and its text:** an interleaved model (Cohere, Voyage, `gemini-embedding-2`, GME, Qwen3-VL-Embedding).
  - **Video:** a model whose Inputs column lists video, or Marengo, or frames through an image model ([video](#video-rag)).
- **Sweep the width.** jina-clip-v2 (to 64), Qwen3-VL-Embedding (64–4096), `gemini-embedding-2` (128–3072) and `@001` (128–1408) all let you choose D, so M4's truncation sweep applies ([M4-2](README.md#dimensionality-matryoshka-truncation-and-the-storage-bill)).
- **Use universal leaderboards as a filter only** ([Benchmarks](#benchmarks-what-each-one-measures)).

---

## Scanned documents as images: ColPali and late interaction

The LogiTrack case (*illustrative*): a scanned carrier agreement with signatures, stamps and a rate table. The text route OCRs and parses it ([M3](../03-parsing-and-chunking/README.md#what-recoverable-means)); the image route embeds a screenshot of each page.

- **The idea.** ColPali is "a Vision Language Model trained to produce high-quality multi-vector embeddings from images of document pages", paired with "a late interaction matching mechanism" [[COLPALI24]](#src-colpali24). The page is indexed as an image, with no OCR and no layout parsing.
- **How late interaction scores a page.** For each query token, take the highest dot product against any of the page's patch vectors, then sum over query tokens [[COLPALI-HTML]](#src-colpali-html):

  $$\text{LI}(q,d) = \sum_i \max_j \langle E_q^{(i)}, E_d^{(j)} \rangle$$

  - A **patch** is a small square of the page image. Each patch gets its own 128-d vector.
  - It is the same MaxSim scoring as text late interaction ([M9](../09-ranking-fusion-reranking/README.md#late-interaction-paying-for-tokens-instead-of-pairs)).

| Model | License | Vectors per page | Base | Source |
|---|---|---|---|---|
| ColPali v1.3 | Gemma (PaliGemma base) + MIT adapters | ~1,030 at 128-d. A page is `(1030, 128)`, a query `(28, 128)` | `paligemma-3b-pt-448` | [[COLPALI-CARD]](#src-colpali-card)[[COLPALI-HTML]](#src-colpali-html) |
| ColQwen2 v1.0 | Apache-2.0 backbone + MIT adapters | ≤768 patches at 128-d | Dynamic resolution; "does not resize" | [[COLQWEN2]](#src-colqwen2) |

**ViDoRe results from the ColPali paper** (nDCG@5, averaged) [[COLPALI-HTML]](#src-colpali-html):

| Approach | nDCG@5 |
|---|---|
| Unstructured + OCR → BGE-M3 | 66.1 |
| Unstructured + captioning → BGE-M3 | 67.0 |
| Vanilla SigLIP (single vector per page) | 51.4 |
| ColPali | 81.3 |

The single-vector image embedder loses to the text pipeline; late interaction is what wins. The paper's costs, side by side [[COLPALI-HTML]](#src-colpali-html):

| | OCR + text embedder | ColPali (late interaction) |
|---|---|---|
| What gets embedded | Extracted text, one vector per chunk | Page image, ~1,030 vectors per page |
| Indexing time per page (paper's setup) | 7.22 s (layout detection, OCR, captioning) | 0.39 s |
| Storage per page (paper) | 8.60 KB (BGE-M3) | ~257.5 KB |
| Loses on | OCR errors, visual elements | Storage, and an ANN index that must support MaxSim |
| Keeps M4's math and thresholds | Yes | No |

### Worked: the storage bill for many vectors per page

**1 · Generalize M4's formula.** M4 prices raw vectors at $4 \times D \times N$ bytes: float32, one vector per chunk ([M4 · storage bill](README.md#dimensionality-matryoshka-truncation-and-the-storage-bill)). With many vectors per item it becomes

$$\text{bytes} = b \times D \times P \times N$$

- $b$ is bytes per dimension: 4 at float32, 2 at fp16, 1/8 when binarized.
- $P$ is vectors per item; M4's case is $P = 1$.
- For ColPali at fp16, with D = 128 [[COLPALI-HTML]](#src-colpali-html), this is $2 \times 128 \times P \times N$ (*derived*).

**2 · Price one page** (*derived*, except where the paper is cited).

| Representation | $b$ | $D$ | $P$ | Bytes per page | vs one 1,536-d vector |
|---|---|---|---|---|---|
| One 1,536-d float32 vector (M4's `text-embedding-3-small` row) | 4 | 1,536 | 1 | 6,144 | 1× |
| BM25 / BGE-M3 (paper-reported) | — | — | — | 3.00 KB / 8.60 KB | — |
| ColPali, fp16 | 2 | 128 | 1,030 | 263,680 (paper: ~257.5 KB) | 43× |
| ColQwen2 at its 768-patch cap, fp16 | 2 | 128 | 768 | 196,608 (~192 KB) | 32× |
| ColPali, token pooling factor 3 | 2 | 128 | ~343 | ~87,800 | 14× |
| ColPali, binarized | 1/8 | 128 | 1,030 | 16,480 | 2.7× |

- **The paper's verdict:** "an order of magnitude larger than BM25 and two orders of magnitude larger than BGE-M3", about 30× BGE-M3's 8.60 KB [[COLPALI-HTML]](#src-colpali-html). Its 257.5 KB matches 1,030 × 128 × 2 B if a KB is 1,024 bytes (*derived*).

**3 · Scale it to a corpus** (*derived*). Take 100,000 pages:

| Representation | Raw bytes @100,000 pages |
|---|---|
| One 1,536-d float32 vector per page | 0.61 GB |
| ColPali, fp16 | 26.4 GB |
| ColPali, pooled (factor 3) | 8.8 GB |
| ColPali, binarized | 1.65 GB |

- For scale: M4's whole hierarchical LogiTrack text index (187,043 vectors at 1,536-d) is 1.15 GB.
- Reading the paper's KB as 1,000 bytes gives 25.75 GB instead; either way, about 26 GB.
- These are vector payloads only; index structure comes on top ([M5](../05-vector-databases-and-ann/README.md)).

**4 · Shrink it.**
- **Token pooling** with factor 3 cuts vectors by 66.7% and keeps about 97.8% of performance. The most text-dense set (Shift) degrades more [[COLPALI-HTML]](#src-colpali-html).
- **Centroid pooling and binary quantization** can cut storage by roughly two orders of magnitude [[COLPALI-HTML]](#src-colpali-html). The pages read give no quality figure for binarization alone. M5 covers what quantization costs in recall ([M5 · Quantization](../05-vector-databases-and-ann/README.md#quantization-is-a-spectrum-not-a-checkbox)).
- **Or don't store page vectors at all.** Retrieve with single vectors, then rerank the image candidates with ColPali, the `ColPaliRerank` pattern [[LI-COLPALI]](#src-li-colpali). Storage then stays at the single-vector level (*argued*).

**5 · Check the index.** The ANN index must support multi-vector or MaxSim scoring, so the M5 engine choice is directly affected ([M5](../05-vector-databases-and-ann/README.md#the-engine-landscape)).

---

## Video RAG

Video is a segmentation problem first. You decide what a retrievable unit is (a clip with a start and end time), then embed each unit through one or more channels.

### Clip embedders

Open research models that embed video directly:
- **X-CLIP:** "a novel multi-grained contrastive model … for video-text retrieval", evaluated by R@1 (the share of queries whose correct clip ranks first) on MSR-VTT and four other sets [[XCLIP22]](#src-xclip22).
- **InternVideo2:** a video encoder scaled to 6B parameters, trained with masked modelling, cross-modal contrastive learning and next-token prediction on "video-audio-speech captions" [[INTERNVIDEO2]](#src-internvideo2).
- **LanguageBind:** binds video, infrared, depth and audio to a frozen language encoder (dataset VIDAL-10M) [[LANGUAGEBIND]](#src-languagebind).
- **VideoPrism:** "a general-purpose video encoder … with a single frozen model", reported state of the art on 31 of 33 benchmarks [[VIDEOPRISM]](#src-videoprism) (authors' claim).
- **VLM-based:** Qwen3-VL-Embedding takes video directly and scores 66.1 on MMEB-V2 Video Overall [[QWEN3-VL-EMB]](#src-qwen3-vl-emb) (model card).

### Hosted video APIs and their limits

| API | What it embeds | Segments and caps | Audio | Source |
|---|---|---|---|---|
| Google `multimodalembedding@001` | One embedding per interval | Essential 4/min (interval ≥15 s), Standard 8/min, Plus 15/min (4–8 s); minimum interval 4 s. Analyses "only 2 minutes of content … at a time"; default segment 0–120 s at a 16 s interval | "The model doesn't consider audio content" | [[VERTEX-MM]](#src-vertex-mm) |
| Google `gemini-embedding-2` | 1 video per request, about 120 s | Longer videos "must be split into overlapping chunks". Frames: at most 32 (AI Studio) or up to 120 at 1 FPS (Vertex); the pages disagree | Audio is a separate input, up to 180 s | [[GEMINI-EMB]](#src-gemini-emb)[[VERTEX-MM]](#src-vertex-mm) |
| Twelve Labs Marengo 3.5 | Clips or the whole asset | `dynamic` segments "adapt[s] to scene changes", 2 s minimum by default; `fixed` is 6 s by default. `asset` scope is advised only for videos of "10-30 seconds". Video length "No limit" (3.0: 4 s–4 h) | `visual` and `audio` embeddings by default; `transcription` is legacy and needs 3.0 | [[MARENGO]](#src-marengo)[[MARENGO-3X]](#src-marengo-3x)[[MARENGO-VIDEO]](#src-marengo-video) |
| voyage-multimodal-3.5 | Video inside an interleaved input | 32k tokens per input; 1120 px = 1 token for video | Not stated | [[VOYAGE-MM]](#src-voyage-mm) |

- **Marengo** takes video, audio, image and text (documents too on 3.5), at 512-d on 3.0 and 128/256/512 on 3.5 [[MARENGO-3X]](#src-marengo-3x).
- **Storage per hour** (*derived*): `@001` at 1408-d float32 is 5,632 B per vector. At 4–15 vectors a minute, that is 1.35–5.07 MB per hour of video, for the visual channel alone.

### The pipeline, step by step

1. **Ingest.** Store the file with its duration, fps and an ID you can link to a timestamp.
2. **Segment.** Cut fixed windows (the APIs above use 4–16 s) or scene-adaptive clips, overlapping at the boundaries.
3. **Visual channel.** Either sample frames and embed them with an image-text model, or embed whole clips with a video embedder. LlamaIndex's example uses CLIP at `fps=0.2`, one frame every 5 s [[LI-VIDEO]](#src-li-video).
4. **ASR channel.** **ASR** (automatic speech recognition) turns speech into a transcript. Transcribe (the example uses `recognize_whisper` [[LI-VIDEO]](#src-li-video)), chunk on the same time boundaries, and embed with the text model. Google's video embeddings ignore audio, so this channel is not optional there [[VERTEX-MM]](#src-vertex-mm).
5. **Metadata.** Put `video_id`, `t_start`, `t_end`, channel and frame paths on every vector.
6. **Index.** One collection per channel, or one collection with a channel tag.
7. **Query.** Embed the query once per space, take top-k from each channel, fuse (RRF or weighted), and merge hits that are adjacent in time.
8. **Generate.** Pass the transcript span and keyframes to a multimodal LLM, and cite `video_id@t_start–t_end`.
9. **Evaluate.** Label gold time spans and measure recall@k by temporal overlap, per channel.

---

## Benchmarks: what each one measures

| Benchmark | Measures | Size and metric | Caveat |
|---|---|---|---|
| **MMEB** | Universal multimodal embeddings over 4 meta-tasks: classification, VQA, multimodal retrieval, visual grounding | 36 datasets (20 train / 16 eval) [[MMEB24]](#src-mmeb24) | Retrieval is one task of four, so the average is not a retrieval score (*argued*) |
| **MMEB-V2** | Adds "visual document retrieval, video retrieval, temporal grounding, video classification and video question answering" | 78 datasets [[MMEB-V2]](#src-mmeb-v2); Qwen3-VL-Embedding-8B 77.9 overall [[QWEN3-VL-EMB]](#src-qwen3-vl-emb) | The headline score comes from the model's own card |
| **ViDoRe (v1)** | Page-level retrieval over visually rich documents, across domains and languages | nDCG@5 [[COLPALI24]](#src-colpali24)[[COLPALI-HTML]](#src-colpali-html) | The scores here come from the ColPali paper, and the ColPali cards are published under the ViDoRe / ILLUIN name |
| **ViDoRe V3** | Enterprise RAG retrieval, human-annotated | 10 datasets, ~26k pages, 3,099 queries, 6 query languages; nDCG@10 [[VIDORE-V3]](#src-vidore-v3) | Best models reach "65% NDCG@10 on English"; open-ended and multi-hop queries are hardest. Vendor blog (ILLUIN, with NVIDIA) |
| **MSR-VTT** | Text↔video retrieval and captioning | "10K web video clips … 200K clip-sentence pairs"; R@1 [[MSRVTT16]](#src-msrvtt16)[[XCLIP22]](#src-xclip22) | Short web clips, not long recordings (*argued*) |
| **Flickr30k** | Sentence-based image retrieval and description | 158k captions; "a standard benchmark for sentence-based image description" [[FLICKR30K]](#src-flickr30k) | Natural photos, not documents |
| **COCO Captions** | Text↔image retrieval and captioning | "over one and a half million captions describing over 330,000 images" [[COCO-CAP]](#src-coco-cap); EVA-CLIP reports COCO T2I R@5 [[EVACLIP-REPO]](#src-evaclip-repo) | Natural photos, not documents |
| **ImageNet-1K zero-shot** | Classification, used as a proxy for representation quality | Reported by CLIP, OpenCLIP and EVA-CLIP [[CLIP21]](#src-clip21)[[OPENCLIP]](#src-openclip)[[EVACLIP-REPO]](#src-evaclip-repo) | Not a retrieval metric |
| **UMRB** | GME's universal multimodal retrieval benchmark | GME-2B 64.45 [[GME]](#src-gme) | Built by the model's own authors |

- **ViDoRe v1 and V3 don't compare.** They use different metrics (nDCG@5 against nDCG@10) and different corpora.
- **None of these is your corpus.** Use ViDoRe V3 and MMEB-V2 as sanity checks, and measure on your own queries ([Evaluation](#evaluation)).

---

## LlamaIndex today

- **The index.** "`MultiModalVectorStoreIndex` supports building separate vector stores for image and text embedding vector stores." The image store "Can be arbitrary Image embedding (Default is CLIP)", and the text store's "Default is GPT3.5", in the docs' own wording [[LI-MM-GUIDE]](#src-li-mm-guide).
  - So the default is two spaces: a text query is embedded by both models, once per store.
- **What's supported.** The guide's table marks Image ✅ for retrieval and the query engine. **Audio and Video are 🛑 in every column** [[LI-MM-GUIDE]](#src-li-mm-guide). The use-case page: "All the core RAG concepts: indexing, retrieval, and synthesis, can be extended into the image setting" [[LI-MM-USE]](#src-li-mm-use).
- **Retrieval and image nodes.** One retriever returns both kinds of node, each with its own top-k. The caller tells them apart by type [[LI-MM-RETR]](#src-li-mm-retr):

  ```python
  retriever = index.as_retriever(similarity_top_k=3, image_similarity_top_k=5)
  for res_node in retriever.retrieve(query):
      if isinstance(res_node.node, ImageNode):
          ...  # an image hit
  ```

  `text_to_image_retrieve` returns image-only results [[LI-MM-GUIDE]](#src-li-mm-guide).

| Embedding class | What the docs show | Source |
|---|---|---|
| `ClipEmbedding` | A subclass of `MultiModalEmbedding`: "CLIP embedding models for encoding text and image for Multi-Modal purpose" | [[LI-CLIP-API]](#src-li-clip-api) |
| `NomicEmbedding` | `NomicEmbedding(model_name="nomic-embed-text-v1.5", vision_model_name="nomic-embed-vision-v1.5")`, passed as both `embed_model` and `image_embed_model`, so one shared space | [[LI-NOMIC]](#src-li-nomic) |
| `VoyageEmbedding` | `VoyageEmbedding(model_name="voyage-multimodal-3")` | [[LI-VOYAGE]](#src-li-voyage) |
| `CohereEmbedding` | `CohereEmbedding(model_name="embed-english-v3.0")`: "current v3 models support multimodal embeddings" | [[LI-COHERE]](#src-li-cohere) |

- **The examples found use v3 models,** not voyage-multimodal-3.5 or Cohere v4/v5.
- **ColPali is only a reranker.** `from llama_index.postprocessor.colpali_rerank import ColPaliRerank`, model `vidore/colpali-v1.2`, applied to `ImageNode`s after a Cohere-embedding retrieval [[LI-COLPALI]](#src-li-colpali). No multi-vector index turned up in the pages read.
- **Video is assembled from parts.** The only route is an example notebook: frames at `fps=0.2`, a Whisper transcript, CLIP, LanceDB `text_collection` and `image_collection`, and `OpenAIMultiModal(model="gpt-4o")` [[LI-VIDEO]](#src-li-video). That fits the 🛑 above.
- **For a build** (*argued*): images fit the framework; page multi-vector and video indexes are your own code.

---

## Design decisions

### One shared space, or caption into text

| Route | What is embedded | Keeps M4's thresholds and storage math | Wins on | Loses on |
|---|---|---|---|---|
| **Caption / OCR into text** | Text from the parser or a vision model | Yes | Documents: 66.1–67.0 against single-vector SigLIP's 51.4 [[COLPALI-HTML]](#src-colpali-html) | The describer's vocabulary, the "battery paperwork" gap ([M4](README.md#what-an-embedding-model-is-optimizing)); OCR errors; visual-only content |
| **Shared image-text space** | One vector per image | Storage yes ($P = 1$); thresholds no, it is a new space | Natural photos, where it is the direct route | Document pages (51.4 on ViDoRe) |
| **Page multi-vector (ColPali)** | ~1,030 vectors per page | No | Document pages: 81.3 | About 30× BGE-M3's storage per page; an ANN index with MaxSim |
| **Single vectors + ColPali rerank** | One vector per item; ColPali on candidates | Yes, for storage | Page detail without page storage (*argued*) | Rerank latency per query (*argued*) |

The ledger entry [M4-5](#evaluation) below makes the first row the default.

### Cost and storage

- **Three drivers:** vectors per item (1 per image, ~1,030 per page with ColPali, 4–15 per minute of video with `@001`), bytes per vector ([worked bill](#worked-the-storage-bill-for-many-vectors-per-page)), and tokens per image on hosted APIs (258 in `gemini-embedding-2`, per Vertex) [[COLPALI-HTML]](#src-colpali-html)[[VERTEX-MM]](#src-vertex-mm).
- **Re-embedding:** a shared space ties two models together, so changing either side re-embeds images and text [[NOMIC-VIS]](#src-nomic-vis).

### Thresholds are per space

- **Run M4's threshold procedure once per space** ([Worked: setting a similarity threshold from two distributions](README.md#worked-setting-a-similarity-threshold-from-two-distributions)): text-text, text-image, and each video channel. A cutoff from one space means nothing in another (*argued*).
- **Late-interaction scores grow with the query.** The score sums one maximum per query token, so a 28-token query adds 28 terms. A fixed cutoff then behaves differently on short and long queries (*argued* from the formula).
- **Threshold before fusion.** RRF combines ranks, not scores, so apply each space's cutoff to its own list first (*argued*).
- **Version each cutoff with its model id, D and normalization,** as M4-1 and M4-3 require for text.

### Evaluation

- **Tag every chunk's origin:** native text, OCR, caption, image vector, page multi-vector, video-visual or video-ASR. Report recall@10 per origin ([M4 · Multimodal inputs](README.md#what-an-embedding-model-is-optimizing)).
- **Put figure and scan queries in the golden set.** Without them, a captioning loss is invisible ([M13 · Golden set](../13-evaluation/README.md#building-and-maintaining-the-golden-set-stratification-sizing-decay)).
- **For video, label gold time spans** and score recall by temporal overlap, per channel.

> **Tradeoff (the ledger entry M4-5, image and scanned-page retrieval).** \
> *Decision:* caption figures and OCR scans into text, and embed them with the text embedder M4
> already chose. Tag every chunk with its origin and report recall@10 per origin. \
> *Costs:* the describer's vocabulary and OCR errors become the index's; visual-only content (a
> stamp, a label's layout) is lost; the OCR-and-captioning pipeline is the slow step (7.22 s per
> page in the ColPali paper's setup). \
> *Accepted because:* one embedder, one space, M4's thresholds and the $4 \times D \times N$ storage
> bill all still hold. On documents, captioning plus BGE-M3 (67.0) beat a single-vector image
> embedder (51.4). And multimodal retrieval is deferred by decision ([syllabus §1](../../syllabus.md#1-scope-knobs-frozen)). \
> *Reverses if:* (a) recall@10 on OCR or caption chunks falls clearly below native-text chunks
> on comparable queries — then rerank image candidates with ColPali first (storage unchanged), and
> move to pooled late interaction only if that is not enough, after re-deriving storage; (b) queries
> ask what images *look like*, not what they say (photos, label condition) — then a shared
> image-text space with its own thresholds; (c) the corpus gains video — then ASR transcripts as
> text plus clip embeddings, outside LlamaIndex's supported path.

---

## Where the evidence disagrees

- **"Embedding the image beats parsing the page."** Only with late interaction. Single-vector SigLIP (51.4) loses to OCR or captioning plus BGE-M3 (66.1, 67.0); ColPali (81.3) wins [[COLPALI-HTML]](#src-colpali-html). All three come from the ColPali paper ([Benchmarks](#benchmarks-what-each-one-measures)).
- **"Document-image retrieval is solved."** On ViDoRe V3 the best models reach "65% NDCG@10 on English", and open-ended and multi-hop queries are hardest [[VIDORE-V3]](#src-vidore-v3). The metric and corpus differ from v1, so the numbers can't be lined up.
- **"Pooling is nearly free."** Factor-3 pooling keeps about 97.8% on average, but the most text-dense set (Shift) degrades more [[COLPALI-HTML]](#src-colpali-html). Text-dense pages, such as contracts, are the case to test (*argued*).
- **Vendor docs disagree with themselves.** Google's two pages give different `gemini-embedding-2` video frame limits (32 frames against 120 at 1 FPS) [[GEMINI-EMB]](#src-gemini-emb)[[VERTEX-MM]](#src-vertex-mm). Voyage's page recommends both `voyage-multimodal-3` and 3.5 [[VOYAGE-MM]](#src-voyage-mm). Cohere documents v5.0-pro on one page and v4.0 on another [[COHERE-EMB]](#src-cohere-emb)[[COHERE-MODELS]](#src-cohere-models).
- **Benchmarks reported by their builders.** UMRB is GME's own benchmark [[GME]](#src-gme), and the MMEB-V2 headline for Qwen3-VL-Embedding is from its own card [[QWEN3-VL-EMB]](#src-qwen3-vl-emb).
- **Vendors embed video; the framework doesn't.** voyage-multimodal-3.5, `gemini-embedding-2` and Marengo take video [[VOYAGE-MM]](#src-voyage-mm)[[GEMINI-EMB]](#src-gemini-emb)[[MARENGO]](#src-marengo), while LlamaIndex marks video 🛑 and its examples use v3 models [[LI-MM-GUIDE]](#src-li-mm-guide).

---

## Sources

- <a id="src-clip-repo"></a>[openai/CLIP](https://github.com/openai/CLIP) — OpenAI · accessed 2026-11-13 — `[CLIP-REPO]` · official repo.
- <a id="src-clip21"></a>*Learning Transferable Visual Models From Natural Language Supervision* — OpenAI · 2021 · [arxiv.org/abs/2103.00020](https://arxiv.org/abs/2103.00020) — `[CLIP21]` · paper.
- <a id="src-openclip"></a>[open_clip](https://github.com/mlfoundations/open_clip) and its [LICENSE](https://raw.githubusercontent.com/mlfoundations/open_clip/main/LICENSE) — mlfoundations · accessed 2026-11-13 — `[OPENCLIP]` · official repo.
- <a id="src-siglip2-blog"></a>[SigLIP 2: A better multilingual vision language encoder](https://huggingface.co/blog/siglip2) — Hugging Face (Google models) · 2025-02-21 — `[SIGLIP2-BLOG]` · vendor blog.
- <a id="src-siglip2-card"></a>[google/siglip2-so400m-patch14-384](https://huggingface.co/google/siglip2-so400m-patch14-384) — Google · accessed 2026-11-13 — `[SIGLIP2-CARD]` · model card.
- <a id="src-evaclip23"></a>*EVA-CLIP: Improved Training Techniques for CLIP at Scale* — BAAI · 2023 · [arxiv.org/abs/2303.15389](https://arxiv.org/abs/2303.15389) — `[EVACLIP23]` · paper.
- <a id="src-evaclip-repo"></a>[EVA-CLIP](https://github.com/baaivision/EVA/tree/master/EVA-CLIP) — baaivision · accessed 2026-11-13 — `[EVACLIP-REPO]` · official repo.
- <a id="src-jina-clip2"></a>[jinaai/jina-clip-v2](https://huggingface.co/jinaai/jina-clip-v2) — Jina AI · paper 2024-12-11 — `[JINA-CLIP2]` · model card.
- <a id="src-nomic-vis"></a>[nomic-ai/nomic-embed-vision-v1.5](https://huggingface.co/nomic-ai/nomic-embed-vision-v1.5) — Nomic · accessed 2026-11-13 — `[NOMIC-VIS]` · model card.
- <a id="src-cohere-models"></a>[Cohere's Embed Models](https://docs.cohere.com/docs/cohere-embed) — Cohere · accessed 2026-11-13 — `[COHERE-MODELS]` · official doc.
- <a id="src-cohere-emb"></a>[Introduction to Embeddings at Cohere](https://docs.cohere.com/docs/embeddings) — Cohere · accessed 2026-11-13 — `[COHERE-EMB]` · official doc.
- <a id="src-voyage-mm"></a>[Multimodal Embeddings](https://docs.voyageai.com/docs/multimodal-embeddings) — Voyage AI · accessed 2026-11-13 — `[VOYAGE-MM]` · official doc.
- <a id="src-titan-mm"></a>[Amazon Titan Multimodal Embeddings G1 model](https://docs.aws.amazon.com/bedrock/latest/userguide/titan-multiemb-models.html) — AWS · accessed 2026-11-13 — `[TITAN-MM]` · official doc.
- <a id="src-vertex-mm"></a>[Get multimodal embeddings](https://docs.cloud.google.com/vertex-ai/generative-ai/docs/embeddings/get-multimodal-embeddings) — Google Cloud (Vertex AI) · accessed 2026-11-13 — `[VERTEX-MM]` · official doc.
- <a id="src-gemini-emb"></a>[Embeddings (Gemini API)](https://aistudio.google.com/docs/embeddings) — Google · accessed 2026-11-13 — `[GEMINI-EMB]` · official doc.
- <a id="src-imagebind"></a>[facebookresearch/ImageBind](https://github.com/facebookresearch/ImageBind) — Meta · accessed 2026-11-13 — `[IMAGEBIND]` · official repo.
- <a id="src-gme"></a>[Alibaba-NLP/gme-Qwen2-VL-2B-Instruct](https://huggingface.co/Alibaba-NLP/gme-Qwen2-VL-2B-Instruct) — Alibaba (Tongyi) · accessed 2026-11-13 — `[GME]` · model card.
- <a id="src-qwen3-vl-emb"></a>[Qwen/Qwen3-VL-Embedding-8B](https://huggingface.co/Qwen/Qwen3-VL-Embedding-8B) — Qwen (Alibaba) · paper 2026-01-08 — `[QWEN3-VL-EMB]` · model card.
- <a id="src-marengo"></a>[Marengo (overview)](https://docs.twelvelabs.io/docs/concepts/models/marengo) — Twelve Labs · accessed 2026-11-13 — `[MARENGO]` · official doc.
- <a id="src-marengo-3x"></a>[Marengo 3.0](https://docs.twelvelabs.io/v1.3/docs/concepts/models/marengo/marengo-3-0.md) and [Marengo 3.5](https://docs.twelvelabs.io/v1.3/docs/concepts/models/marengo/marengo-3-5.md) — Twelve Labs · accessed 2026-11-13 — `[MARENGO-3X]` · official doc.
- <a id="src-marengo-video"></a>[Create video embeddings (at scale)](https://docs.twelvelabs.io/v1.3/docs/guides/create-embeddings/at-scale/video.md) — Twelve Labs · accessed 2026-11-13 — `[MARENGO-VIDEO]` · official doc.
- <a id="src-colpali24"></a>*ColPali: Efficient Document Retrieval with Vision Language Models* (abstract) — ICLR 2025 · v1 2024-06-27, v6 2025-02-28 · [arxiv.org/abs/2407.01449](https://arxiv.org/abs/2407.01449) — `[COLPALI24]` · paper.
- <a id="src-colpali-html"></a>*ColPali* (full HTML, v6) — arXiv · 2025-02-28 · [arxiv.org/html/2407.01449v6](https://arxiv.org/html/2407.01449v6) — `[COLPALI-HTML]` · paper.
- <a id="src-colpali-card"></a>[vidore/colpali-v1.3](https://huggingface.co/vidore/colpali-v1.3) — ViDoRe / ILLUIN · accessed 2026-11-13 — `[COLPALI-CARD]` · model card.
- <a id="src-colqwen2"></a>[vidore/colqwen2-v1.0](https://huggingface.co/vidore/colqwen2-v1.0) — ViDoRe / ILLUIN · accessed 2026-11-13 — `[COLQWEN2]` · model card.
- <a id="src-colpali-blog"></a>[ColPali: Efficient Document Retrieval with Vision Language Models](https://huggingface.co/blog/manu/colpali) — M. Faysse, Hugging Face · 2024-07-05 — `[COLPALI-BLOG]` · vendor blog.
- <a id="src-xclip22"></a>*X-CLIP: End-to-End Multi-grained Contrastive Learning for Video-Text Retrieval* — 2022 · [arxiv.org/abs/2207.07285](https://arxiv.org/abs/2207.07285) — `[XCLIP22]` · paper.
- <a id="src-internvideo2"></a>*InternVideo2* — 2024 · [arxiv.org/abs/2403.15377](https://arxiv.org/abs/2403.15377) — `[INTERNVIDEO2]` · paper.
- <a id="src-languagebind"></a>*LanguageBind* — 2023 · [arxiv.org/abs/2310.01852](https://arxiv.org/abs/2310.01852) — `[LANGUAGEBIND]` · paper.
- <a id="src-videoprism"></a>*VideoPrism* — Google DeepMind code · 2024 · [arxiv.org/abs/2402.13217](https://arxiv.org/abs/2402.13217) — `[VIDEOPRISM]` · paper.
- <a id="src-li-video"></a>[Multimodal RAG for processing videos using OpenAI GPT4V and LanceDB vectorstore](https://developers.llamaindex.ai/python/examples/multi_modal/multi_modal_video_rag/) — LlamaIndex · accessed 2026-11-13 — `[LI-VIDEO]` · official doc (example).
- <a id="src-mmeb24"></a>*VLM2Vec / MMEB* — 2024 · [arxiv.org/abs/2410.05160](https://arxiv.org/abs/2410.05160) — `[MMEB24]` · paper.
- <a id="src-mmeb-v2"></a>*VLM2Vec-V2 / MMEB-V2* — 2025-07 · [arxiv.org/abs/2507.04590](https://arxiv.org/abs/2507.04590) — `[MMEB-V2]` · paper.
- <a id="src-vidore-v3"></a>[ViDoRe V3: a comprehensive evaluation of retrieval for enterprise use-cases](https://huggingface.co/blog/QuentinJG/introducing-vidore-v3) — Hugging Face blog (ILLUIN, with NVIDIA) · 2025-11-05 — `[VIDORE-V3]` · benchmark / vendor blog.
- <a id="src-msrvtt16"></a>[MSR-VTT: A Large Video Description Dataset](https://www.microsoft.com/en-us/research/publication/msr-vtt-a-large-video-description-dataset-for-bridging-video-and-language/) — Microsoft Research · 2016 — `[MSRVTT16]` · benchmark.
- <a id="src-flickr30k"></a>*Flickr30k Entities* — 2015 · [arxiv.org/abs/1505.04870](https://arxiv.org/abs/1505.04870) — `[FLICKR30K]` · paper / benchmark.
- <a id="src-coco-cap"></a>*Microsoft COCO Captions* — 2015 · [arxiv.org/abs/1504.00325](https://arxiv.org/abs/1504.00325) — `[COCO-CAP]` · paper / benchmark.
- <a id="src-li-mm-guide"></a>[Multi-modal models (module guide)](https://developers.llamaindex.ai/python/framework/module_guides/models/multi_modal/) — LlamaIndex · accessed 2026-11-13 — `[LI-MM-GUIDE]` · official doc.
- <a id="src-li-mm-use"></a>[Multi-modal (use case)](https://developers.llamaindex.ai/python/framework/use_cases/multimodal/) — LlamaIndex · accessed 2026-11-13 — `[LI-MM-USE]` · official doc.
- <a id="src-li-mm-retr"></a>[Multi-Modal Retrieval using GPT text embedding and CLIP image embedding](https://developers.llamaindex.ai/python/examples/multi_modal/multi_modal_retrieval/) — LlamaIndex · accessed 2026-11-13 — `[LI-MM-RETR]` · official doc (example).
- <a id="src-li-clip-api"></a>[ClipEmbedding API reference](https://developers.llamaindex.ai/python/framework-api-reference/embeddings/clip/) — LlamaIndex · accessed 2026-11-13 — `[LI-CLIP-API]` · official doc.
- <a id="src-li-nomic"></a>[Multi-Modal RAG using Nomic Embed and Anthropic](https://developers.llamaindex.ai/python/examples/multi_modal/multi_modal_rag_nomic/) — LlamaIndex · accessed 2026-11-13 — `[LI-NOMIC]` · official doc (example).
- <a id="src-li-voyage"></a>[Multi-Modal Retrieval using VoyageAI Multi-Modal Embeddings](https://developers.llamaindex.ai/python/examples/multi_modal/llamaindex_mongodb_voyageai_multimodal/) — LlamaIndex · accessed 2026-11-13 — `[LI-VOYAGE]` · official doc (example).
- <a id="src-li-cohere"></a>[Multi-Modal Retrieval using Cohere Multi-Modal Embeddings](https://developers.llamaindex.ai/python/examples/multi_modal/cohere_multi_modal/) — LlamaIndex · accessed 2026-11-13 — `[LI-COHERE]` · official doc (example).
- <a id="src-li-colpali"></a>[Reranking using ColPali, Cohere Reranker and Multi-Modal Embeddings](https://developers.llamaindex.ai/python/examples/node_postprocessor/colpalirerank/) — LlamaIndex · accessed 2026-11-13 — `[LI-COLPALI]` · official doc (example).

**Unsupported or vendor-reported:**
- **Illustrative, built for this file:** the LogiTrack cases (a scanned signed agreement, a DG label figure), the route comparison, the threshold and evaluation advice, and the ledger entry M4-5. These are this file's judgement, not findings.
- **Derived (this file's arithmetic):** the $b \times D \times P \times N$ generalization, bytes per page, the ratios against a 1,536-d vector, the 100,000-page totals, the pooled (~343 vectors) and binarized figures, the 1,024-byte reading of the paper's KB, and the video bytes per hour.
- **Argued, not sourced:** text-image cosines being on their own scale; late-interaction scores growing with query length; thresholding before RRF; rerank-only keeping storage at the single-vector level, and its latency cost; MMEB's average not being a retrieval score; MSR-VTT clips being short; contracts being the text-dense case to test for pooling; what LlamaIndex's support means for a build.
- **Vendor-reported or self-reported only:** every hosted model's limits, dimensions and segmenting rules (vendor docs); SigLIP 2's details (vendor blog); Qwen3-VL-Embedding's 77.9 and 66.1 (model card); GME's 64.45 on its own UMRB; ViDoRe V3's "65%" (vendor blog, with NVIDIA); VideoPrism's 31 of 33; ColPali's ViDoRe scores, indexing times and storage figures (the authors' own benchmark and setup), and the HF blog's framing of text pipelines.
- **Not verified in the research:** output D for CLIP and OpenCLIP checkpoints, SigLIP 2, EVA-CLIP, nomic-embed-vision and ImageBind (not stated on the pages read); EVA-CLIP's license (not shown); OpenCLIP's resolutions (read from checkpoint names); Qwen3-VL-Embedding's frame and pixel limits; Voyage's audio handling; quality after binarization alone; any LlamaIndex multi-vector index (none found in the pages read); LlamaIndex examples for voyage-multimodal-3.5 or Cohere v4/v5 (none found). Nomic's vision blog post returned 404, so only the model card was read. COCO's statistics come from the paper, because cocodataset.org showed none.
