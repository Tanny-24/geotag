# NewsVQA: architecture guide

## Big picture
- **Task.** Given a news photo with no caption, predict where (`Location:`) and when (`Date:`) it was taken (TARA benchmark).
- **Retrieval (NewsRECON-lite).** Frozen CLIP ViT-L/14 ranks 176k NYT articles (2010-2021) by image-text similarity; a small MLP re-ranks the top 50 by location match.
- **Reader.** Qwen2.5-VL-3B reads the photo plus the top-3 articles and answers in a fixed two-line format (QA), or first picks one of 4 clustered candidate events or "none" (MCQA).
- **Training (LiveVQA recipe).** LoRA on the language model only, 6k samples mixing QA / MCQA / QAR formats; 15% of samples carry only wrong-place evidence so the model learns to reject articles.
- **Evaluation.** NewsRECON metrics on 500 TARA test images: EM@1, Example-F1, Delta / CO-Delta, GREAT, plus blind-trust flips and MCQA calibration (ECE).

## Layout (golden path first)
```
nv/                 pipeline package, one module per stage (run as python -m nv.<stage>)
  config.py         paths (NV_DATA=data, NV_OUT=runs) and constants (TOPK=50, N_WINDOW=7, N_EVIDENCE=3)
  prepare.py        corpus filter + TARA normalisation  -> runs/prepared/
  embed.py          frozen CLIP embeddings              -> runs/emb/
  retrieve.py       bi-encoder top-50                   -> runs/retr/*_bi.npz
  cross_encoder.py  location re-ranker, alpha search    -> runs/retr/*_final.npz
  build_sft.py      data engine (QA/MCQA/QAR, noise)    -> runs/sft/
  train_lora.py     LoRA fine-tuning                    -> runs/ckpt/
  infer.py          batched generation                  -> runs/preds/<split>/<run>.jsonl
  evaluate.py       tables, flips, ECE                  -> runs/results/
  hierarchy.py      trust layer: calibrated back-off    -> runs/results/trust_<run>.json
  metrics.py, geo.py   NewsRECON metric code, offline gazetteer
  common.py, prompts.py  labels, clustering, prompt templates
scripts/            run_all.sh (full pipeline), resume.sh, extract_metrics.py
demo.py             Gradio chatbot
results/            committed summaries (tables, recall, stats, metrics_summary.json)
paper/              slides.tex + figs/
data/  runs/        git-ignored inputs and outputs
_archive/           earlier code, never deleted (see "Where things moved")
```

## Code-to-math map
Notation: image x, article a with headline h_a, abstract b_a and location string l_a; f_I, f_T are the frozen CLIP encoders, L2-normalised.

| Concept | Equation | Code |
|---|---|---|
| Bi-encoder score | s_bi(x,a) = f_I(x)ᵀ f_T(h_a ⊕ b_a) | `embed.py:34,60`, `retrieve.py:55` |
| Candidate set | A_K(x) = top-K_a s_bi(x,a), K = 50; train images never retrieve their own article | `retrieve.py:56-60` |
| Location head | s_loc(x,a) = MLP([i; t; i⊙t; \|i−t\|]), i = f_I(x), t = f_T("An image from l_a") | `cross_encoder.py:29-35` |
| Head loss | BCE: −[y log σ(s_loc) + (1−y) log(1−σ(s_loc))], 1 positive : 7 negatives per image | `cross_encoder.py:70,82,89` |
| Score fusion | s(x,a) = s_bi(x,a) · σ(s_loc(x,a))^α, α ∈ {0, .1, .25, .5, 1} picked on dev by R@1+R@5 → α = 0.1 | `cross_encoder.py:59-62,128-135` |
| Cross-fitting | train images scored by a head trained on the other half (2 folds) | `cross_encoder.py:140-146` |
| Location relevance | rel_loc(x,a) = 1[atom(x) ∈ atoms(a)] | `common.py:149-152` |
| Event relevance | rel_evt(x,a) = rel_loc(x,a) · 1[\|date(a) − date(x)\| ≤ 7 days] | `common.py:155-174` |
| Candidate events | group ranked articles by l_a; date = median of the largest ±7-day cluster | `common.py:189-207` |
| Noise injection | with prob. ρ = 0.15, evidence E ← 3 articles from ranks 10-50 with rel_loc = 0 | `build_sft.py:88-94`, `run_all.sh` |
| SFT objective | L(θ) = −Σ_{t ∈ answer} log p_θ(y_t \| y_<t, x, E); prompt tokens masked (−100) | `train_lora.py: make_batch` |
| LoRA | W = W₀ + (α_L / r)·BA, r = 64, α_L = 128, all LM projections, vision tower frozen | `train_lora.py: get_peft_model` |
| MCQA confidence | c(x) = max_k p_k / Σ_{k ≤ K+1} p_k, p_k = softmax prob. of letter k at the first generated token | `infer.py: FirstTokenLetterProbs`, `conf_dist` in `infer.py: main` |
| ECE | Σ_b (\|B_b\|/n)·\|acc(B_b) − conf(B_b)\|, 10 equal-width bins, correctness = location EM@1 | `evaluate.py:27-34` |
| GREAT_loc | 1 if EM, else max(0, 1 − d_km/1000) | `metrics.py:134-149` |
| GREAT_date | Σ_u w_u·max(0, 1 − \|g_u − p_u\|/τ_u) / Σ_u w_u over u ∈ {century, decade, year, month, day}; w = (1, 1, 1.25, 1.5, 1.5), τ = (exact, 50, 5, 6, 15) | `metrics.py:101-131` |
| GREAT | (GREAT_date + GREAT_loc)/2 | `evaluate.py:71` |
| Delta / CO-Delta | 1/(1 + \|Δyears\|), 1/(1 + d_km) | `metrics.py:144,158` |
| Blind-trust flips | C→W = #{i : EM_base(i) = 1 ∧ EM_run(i) = 0}, base = zero-shot 3B without evidence | `evaluate.py:78-90` |
| Hierarchy paths | location → (country, region, city) via the gazetteer; date → (year, month, day) | `hierarchy.py: loc_path, date_path` |
| Node confidence | k* = argmax_k p_k; c_l = Σ_k p_k·1[path_k[:l] = path_k*[:l]], so c_1 ≥ c_2 ≥ c_3 | `hierarchy.py: build` |
| Commit rule | answer k*'s node at depth l*(λ) = max{l : c_l ≥ λ}; no answer if c_1 < λ | `hierarchy.py: commit` |
| Conformal risk control | λ̂ = min{λ : (n·R̂_cal(λ) + 1)/(n + 1) ≤ ε}, R̂ = error of the committed node ⇒ E[error] ≤ ε on exchangeable test images | `hierarchy.py: crc` |

GREAT was introduced by Siingh et al. (GETReason, ACL 2025). `metrics.py` follows NewsRECON's implementation, which scores only the date and location parts (no event-similarity term); say so in the paper.

## Commands (run from the repo root; defaults NV_DATA=data, NV_OUT=runs)
| Step | Command |
|---|---|
| Corpus (CPU, needed once: `runs/prepared/corpus.jsonl` is missing) | `python -m nv.prepare` |
| CLIP embeddings | `python -m nv.embed` |
| Bi-encoder retrieval | `python -m nv.retrieve` |
| Location re-ranker | `python -m nv.cross_encoder` |
| Training / eval data | `python -m nv.build_sft --n_train 6000 --n_eval 500 --noise 0.15` |
| **Train** | `python -m nv.train_lora --out ckpt/lora_mixed --workers 4` |
| Inference | `python -m nv.infer --split test --fmt qa --name lora_mixed_qa --adapter ckpt/lora_mixed/final` |
| **Evaluate** | `python -m nv.evaluate --split test --baseline zs_noev` |
| Trust layer (needs the dev and test MCQA runs) | `python -m nv.hierarchy --name lora_mixed_mcqa` |
| Paper numbers | `python scripts/extract_metrics.py --src runs/results` |
| Everything | `bash scripts/run_all.sh 2>&1 \| tee runs/run_all.log` |
| Demo | `python demo.py --adapter ckpt/lora_mixed/final --share` |

Missing locally: TARA dev/test images (only `data/TARA_dataset/img/train/` is present) and the trained adapter. Both come from the Colab run.

## Where things moved (nothing was deleted)
| Before | Now |
|---|---|
| `NewsVQA/` (clone of the group repo) | `_archive/upstream/newsvqa_group_repo/` |
| `code/emnlp2026-newsrecon/` (NewsRECON with uncommitted Colab edits) | `_archive/upstream/newsrecon_colab_modified/` |
| `TARA/` (TARA repo) | `_archive/upstream/tara_repo/` |
| `colab_pipeline.ipynb` (targets code that is not in this folder) | `_archive/colab_pipeline_newsrecon.ipynb` |
| `outputs/`, `input_zip_nyt_final/` (empty) | `_archive/` |
| `datasets/TARA`, `datasets/5Pils-OOC` | `data/TARA_dataset`, `data/5Pils-OOC` |
| `NewsVQA/src/data/processed_articles` | `data/processed_articles` |
| `~/Downloads/MM-main` code, `newsvqa_runs/` (copied; originals untouched) | `nv/`, `scripts/`, `demo.py`, `runs/` |
| `~/Downloads/newsvqa_slides_latex` (copied) | `paper/` |
