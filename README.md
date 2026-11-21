# Geotag

Where and when was this news photo taken? Geotag retrieves news articles for a photo, lets a fine-tuned vision-language model weigh that evidence, and answers only as precisely as the evidence allows, with a conformal bound on the error rate.

> Photo of a collapsed building, no caption.
> Without the trust layer: **Port-au-Prince, Haiti, 2010-01-19**, which may be wrong in the city or the day.
> With the trust layer at ε = 10%: **Haiti, 2010**, an answer the model can stand behind, with expected error at most 10%.

## How it works

```mermaid
flowchart LR
  A["News photo"] --> B["CLIP ViT-L/14<br/>top-50 of 176k NYT articles"]
  B --> C["Location re-ranker<br/>s_bi · σ(s_loc)^0.1"]
  C --> D["Top-3 articles +<br/>4 candidate events"]
  D --> E["Qwen2.5-VL-3B + LoRA<br/>trained on wrong-place evidence"]
  E --> F["Trust layer<br/>conformal back-off"]
  F --> G["Haiti, 2010<br/>(error ≤ ε)"]
```

1. **Retriever.** Frozen CLIP ViT-L/14 scores 176,314 NYT articles (2010–2021, headline + abstract) against the photo and keeps the top 50. A small MLP re-ranks them by how well each article's location fits the photo; its weight α = 0.1 is chosen on the dev set.
2. **Generator.** Qwen2.5-VL-3B-Instruct reads the photo, the top-3 articles and four candidate (location, date) events built from the top-20 articles. LoRA (r = 64, vision tower frozen) trains it on 6,000 TARA examples in three formats (direct answer, multiple choice, verdict per article). In 15% of the examples every article is from the wrong place, so the model learns to reject evidence instead of copying it.
3. **Trust layer** (`nv/hierarchy.py`). Turns the model's multiple-choice probabilities into an answer at the right level of detail, described below.

## Trust layer

Each candidate event k gets a probability $p_k$ from the model's first answer token. Its location resolves through the GeoNames gazetteer to the path country > region > city, its date to year > month > day. Starting from the most likely candidate $k^\ast$, the confidence of the node at depth $l$ of its path is the probability of all candidates that share that node:

$$c_l = \sum_k p_k \, \mathbf{1}\left[\mathrm{path}_k[{:}l] = \mathrm{path}_{k^\ast}[{:}l]\right], \qquad c_1 \ge c_2 \ge c_3 .$$

Geotag answers with $k^\ast$'s node at the finest depth where $c_l \ge \lambda$, and gives no answer when even the country (or year) is below $\lambda$. Along one path, a correct city implies a correct region and country, so the error of the committed answer can only fall as $\lambda$ grows. That makes conformal risk control applicable: on the $n$ dev images,

$$\hat\lambda = \min\left\{\lambda : \frac{n \, \hat R_{\text{dev}}(\lambda) + 1}{n + 1} \le \varepsilon\right\}
\quad\Longrightarrow\quad \mathbb{E}\left[\text{error on a new image}\right] \le \varepsilon ,$$

assuming dev and test images are exchangeable. The guarantee holds on average over calibration sets, not for every single split.

The layer is compared with two baselines on the same images: **finest** (always answer with full detail) and **selective** (the same calibration on $p_{k^\ast}$ alone, answering in full or not at all). It reports, for ε ∈ {5%, 10%, 20%}, the test error, the share of images answered, the share answered at each depth, and specificity (mean depth / 3).

On synthetic probabilities over the real TARA labels, the average test error over 200 random dev/test splits was 4.91%, 9.90% and 19.75% for ε = 5%, 10% and 20% (location) and 4.99%, 9.93% and 19.84% (date). Synthetic probabilities cannot show whether back-off beats the selective baseline; that needs the real model's outputs.

## Results

**Status:** the code is complete; the full GPU run is pending. The tables below will report every number on the 1,571 TARA test images that have a gold location or date, with 95% bootstrap intervals, plus `GREAT@500` on the 500-image subset used by the reference run.

Reference: the NewsVQA-lite run of the same pipeline (500 test images, 1 seed). These are the numbers this repository sets out to reproduce; they are not yet results of this repository.

| Run | GREAT | Location EM@1 | Date EM@1 | Correct → wrong location (vs photo only) |
|---|---|---|---|---|
| Retrieval only (re-ranked top-1) | 54.7 | 25.5 | 4.2 | 57 |
| Qwen2.5-VL-3B, photo only | 51.1 | 24.6 | 1.2 | – |
| Qwen2.5-VL-3B + top-3 articles | 57.2 | 31.4 | 3.5 | 29 |
| Qwen2.5-VL-7B + top-3 articles | 57.8 | 27.6 | 3.9 | 42 |
| 3B + LoRA, direct answer | 62.5 | 44.2 | 4.6 | 7 |
| 3B + LoRA, multiple choice | 61.9 | 44.8 | 5.6 | 10 |

The multiple-choice confidence's expected calibration error fell from 23.6 (zero-shot) to 18.4 (LoRA). Retrieval: location R@1 on test is 52.9 with CLIP alone and 56.3 after re-ranking.

## Run it

**Data** (not in this repository):

| What | Source | Where it goes |
|---|---|---|
| TARA labels and images | [github.com/zeyofu/TARA](https://github.com/zeyofu/tara) | `data/TARA_dataset/{input,img/{train,dev,test}}` |
| NYT articles 2010–2021 | NYT Archive API, one JSON file per year | `data/processed_articles/nyt_articles_<year>.json` |
| GeoNames | `cities15000.zip`, `countryInfo.txt`, `admin1CodesASCII.txt` from [download.geonames.org](https://download.geonames.org/export/dump/), plus NewsRECON's `geonames_results.json` | `runs_3b/geonames/` |

**1. Build the corpus on a laptop** (CPU, about 15 s; needs only the Python standard library):

```bash
NV_OUT=runs_3b python -m nv.prepare
```

**2. Run everything else on Colab** with [`notebooks/geotag_colab.ipynb`](notebooks/geotag_colab.ipynb): put `data/tara_img.zip` (a zip of `data/TARA_dataset/img`), `runs_3b/prepared/` and `runs_3b/geonames/` in `My Drive/geotag/`, add a read-only GitHub token as the Colab secret `GH_TOKEN`, choose an L4 GPU and run all cells. Each step writes to Drive and is skipped once done, so a disconnected session resumes where it stopped. Expect about 4–5 L4 hours.

On a machine with its own GPU, `bash scripts/run_all.sh` runs the same steps.

| Step | Command |
|---|---|
| CLIP embeddings, retrieval, re-ranker | `python -m nv.embed`, `python -m nv.retrieve`, `python -m nv.cross_encoder` |
| Training and evaluation data | `python -m nv.build_sft --n_train 6000 --n_eval 1644 --noise 0.15` |
| LoRA training | `python -m nv.train_lora --out ckpt/lora_mixed` |
| Inference | `python -m nv.infer --split test --fmt mcqa --name lora_mixed_mcqa --adapter ckpt/lora_mixed/final` |
| Scores, intervals, flips | `python -m nv.evaluate --split test --baseline zs_noev` |
| Trust layer | `python -m nv.hierarchy --name lora_mixed_mcqa` |
| Summary for the paper | `python scripts/extract_metrics.py --src runs_3b/results --out runs_3b/metrics_summary.json` |
| Demo | `python demo.py --adapter ckpt/lora_mixed/final --share` |

## Repository layout

```
nv/                    one module per stage (python -m nv.<stage>)
  prepare.py           corpus filter and TARA normalisation (removes every dev/test source article)
  embed.py, retrieve.py, cross_encoder.py    retrieval
  build_sft.py         training and evaluation prompts
  train_lora.py, infer.py                    fine-tuning and batched generation
  evaluate.py, metrics.py, geo.py            scores, bootstrap intervals, gazetteer
  hierarchy.py         trust layer
notebooks/             geotag_colab.ipynb
scripts/               run_all.sh, extract_metrics.py, plot_loss.py
results/               reference numbers (NewsVQA-lite run)
demo.py                Gradio app
```

`ARCHITECTURE_GUIDE.md` maps every equation to the line of code that implements it.

## Metrics

- **GREAT**: mean of a date score and a location score, each 0–100 with partial credit (introduced by Siingh et al., 2025; this code follows NewsRECON's implementation, which scores the date and location parts only).
- **EM@1**: exact match; a location counts if the most specific gold place name appears in the answer.
- **Correct → wrong flips**: test images whose location was right from the photo alone and turned wrong once articles were added, i.e. blind trust in evidence.
- **ECE**: expected calibration error of the multiple-choice confidence, 10 bins.

## Credits and related work

The retrieval and fine-tuning pipeline follows NewsVQA-lite (2026). This repository rebuilds it from scratch, evaluates on the full TARA test set with bootstrap intervals, and adds the conformal trust layer.

- NewsRECON: Tonglet et al., [arXiv:2601.14121](https://arxiv.org/abs/2601.14121), retrieval design and evaluation protocol.
- LiveVQA: Fu et al., [arXiv:2504.05288](https://arxiv.org/abs/2504.05288), the three training formats.
- TARA: Fu et al., ACL 2022, [dataset](https://github.com/zeyofu/tara).
- GETReason (GREAT metric): Siingh et al., [arXiv:2505.21863](https://arxiv.org/abs/2505.21863).
- Conformal risk control: Angelopoulos et al., ICLR 2024, [arXiv:2208.02814](https://arxiv.org/abs/2208.02814).
- Closest ideas to the trust layer: conformal factuality (Mohri and Hashimoto, [ICML 2024](https://proceedings.mlr.press/v235/mohri24a.html)), hierarchical selective classification (Goren et al., NeurIPS 2024, [arXiv:2405.11533](https://arxiv.org/abs/2405.11533)), conformal linguistic calibration (Jiang et al., [arXiv:2502.19110](https://arxiv.org/abs/2502.19110)) and C-RAG (Kang et al., [ICML 2024](https://proceedings.mlr.press/v235/kang24a.html)). Geotag applies calibrated back-off to two hierarchies at once (place and time) for retrieval-augmented, multimodal answers.

## Limitations

- The corpus is NYT only, 2010–2021, and the CLIP encoders are frozen, so absolute numbers are not comparable with the NewsRECON paper.
- The trust layer's bound is an average over calibration sets and assumes dev and test images are exchangeable; it says nothing about photos from other sources.
- Place names the gazetteer cannot resolve count as errors, and a gold answer coarser than the committed one is judged at the gold's level.
