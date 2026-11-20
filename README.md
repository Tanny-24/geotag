# NewsVQA-lite: NewsRECON retrieval + LiveVQA-style knowledge updating

Where and when was a news photo taken? We retrieve NYT articles for the image (a light NewsRECON
pipeline), then fine-tune Qwen2.5-VL-3B with LoRA on the three LiveVQA training formats (QA, MCQA, QAR).
Training targets come from weak relevance labels, so the model learns when to **trust or reject**
retrieved evidence. This targets NewsRECON's "blind trust" failure (Sec. 4.8.4).

```
image ─► CLIP ViT-L/14 (frozen) ─► top-50 NYT articles (headline+abstract) ─► location cross-encoder rerank
      ─► top-3 evidence + 4 clustered candidate events ─► Qwen2.5-VL-3B + LoRA ─► "Location: … / Date: …"
```

| NewsRECON / LiveVQA component | Here | Why |
|---|---|---|
| Bi-encoder (CLIP fine-tuned on Qwen news captions) | frozen CLIP-L/14 on headline+abstract | no captions available; abstract is the best text input before fine-tuning (NewsRECON Table 3) |
| Location cross-encoder | kept: MLP head on frozen CLIP, BCE, `s_bi·σ(s_loc)` | largest gain in NewsRECON (+10 pp R@1), trains in minutes |
| Event cross-encoder + clustering | replaced by ±7-day date clustering of candidate events | only +0.4 pp R@1 in NewsRECON |
| Corpus (Qwen filter, NYT+Guardian, 88k) | raw NYT 2010–2021, rule filter, 176k articles; TARA dev/test source articles removed | no time to run the Qwen filter |
| MLLM | Qwen2.5-VL-3B-Instruct (also used by LiveVQA) | fits the deadline and GPU |
| PEFT formats QA / MCQA / QAR | all three, mixed 30/40/30, LoRA r=64, lr 1e-4, 1 epoch, vision frozen | LiveVQA Sec. 4 recipe |
| Noise | 30% of training samples get only non-relevant evidence | teaches the model to reject evidence |

## 0. Setup on the server

```bash
git clone <this repo> NewsVQA-lite && cd NewsVQA-lite
conda create -n newsvqa python=3.10 -y && conda activate newsvqa
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121   # match the server's CUDA
pip install -r requirements.txt
# pre-download models (≈ 9 GB) so later steps never wait on the network
huggingface-cli download Qwen/Qwen2.5-VL-3B-Instruct
huggingface-cli download openai/clip-vit-large-patch14
```

Data layout on the server (`NV_DATA` and `NV_OUT` can live anywhere):

```
~/newsvqa_data/TARA_dataset/img/train/train/<idx>.png   (from Windows: src/data/TARA_dataset/img)
~/newsvqa_data/TARA_dataset/img/dev/<idx>.png
~/newsvqa_data/TARA_dataset/img/test/<idx>.png          (when available; otherwise use SPLIT=dev)
~/newsvqa_runs/prepared/{corpus.jsonl,tara_*.jsonl}     (from prepared.tar.gz, already built)
~/newsvqa_runs/geonames/geonames_results.json           (from prepared.tar.gz)
```

```bash
mkdir -p ~/newsvqa_runs && tar -xzf prepared.tar.gz -C ~/newsvqa_runs
# offline gazetteer for GREAT/CO-delta (≈3 MB); avoids the GeoNames API rate limit
cd ~/newsvqa_runs/geonames && wget https://download.geonames.org/export/dump/cities15000.zip \
   && unzip -o cities15000.zip && wget https://download.geonames.org/export/dump/countryInfo.txt \
   && wget https://download.geonames.org/export/dump/admin1CodesASCII.txt && cd -
```

## 1. Run

```bash
export NV_DATA=~/newsvqa_data NV_OUT=~/newsvqa_runs
tmux new -s nv
bash scripts/run_all.sh 2>&1 | tee run_all.log        # SPLIT=dev bash scripts/run_all.sh  if test images are missing
```

| step | command | time (24 GB GPU, rough) | output |
|---|---|---|---|
| (done) corpus + TARA normalisation | `python -m nv.prepare` | CPU, 3 min | `prepared/` |
| CLIP embeddings | `python -m nv.embed` | 10 min | `emb/` |
| bi-encoder retrieval | `python -m nv.retrieve` | 1 min | `retr/*_bi.npz`, `results/retrieval_bi_recall.json` |
| location cross-encoder | `python -m nv.cross_encoder` | 3 min | `retr/*_final.npz`, `results/retrieval_ce_recall.json` |
| data engine | `python -m nv.build_sft` | 1 min | `sft/train.jsonl`, `sft/{dev,test}_eval.jsonl`, `preds/*/retrieval_*.jsonl` |
| zero-shot runs | `python -m nv.infer --fmt qa_noev/qa/mcqa` | 3–6 min each | `preds/<split>/zs_*.jsonl` |
| LoRA training | `python -m nv.train_lora` | 30–60 min | `ckpt/lora_mixed/final` |
| fine-tuned runs | `python -m nv.infer --adapter …` | 3–6 min each | `preds/<split>/lora_*.jsonl` |
| scoring | `python -m nv.evaluate --split test` | 1 min | `results/test_results.md` |
| demo chatbot | `python demo.py --share` | – | Gradio link |

If training runs out of memory: `--bs 2 --accum 8`, or `--max_pixels 200704` → `--max_pixels 150000`.

## 2. What to report

* **Retrieval**: R@1/R@5 for location and event before and after the cross-encoder (`results/retrieval_*_recall.json`).
* **Main table** (`results/<split>_results.md`): date EM@1 / E-F1 / Δ / GREAT_date, location EM@1 / E-F1 / CO∆ /
  GREAT_loc, overall GREAT, for `retrieval_bi`, `retrieval_ce`, `retrieval_ce_cluster`, `zs_noev` (MLLM default),
  `zs_qa` (MLLM + NewsRECON evidence), `zs_mcqa`, and `lora_mixed_*` (ours).
* **Blind trust**: flips vs `zs_noev`, i.e. answers that were correct without evidence and became wrong with it.
  NewsRECON reports 6/50 such cases for the zero-shot MLLM; we want fewer for `lora_mixed_*` than for `zs_qa`.
* **Calibration**: `MCQA_ECE` from the option-letter probability (lower is better).

Caveats to state: different corpus (NYT only, rule-filtered), frozen CLIP, 3B model, 500-image evaluation subset,
offline GeoNames geocoding. Absolute numbers are not comparable to the NewsRECON paper; comparisons inside
the table are like for like.
