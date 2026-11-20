"""Central paths and hyper-parameters. Override any path with an environment variable.

    NV_DATA  folder that contains TARA_dataset/ (and processed_articles/ for prepare.py)
    NV_OUT   folder where every output (embeddings, retrieval, SFT data, checkpoints, results) goes
    NV_PREP  folder with prepare.py outputs (corpus.jsonl, tara_*.jsonl). Default: $NV_OUT/prepared
"""
import os

DATA = os.environ.get("NV_DATA", "data")
OUT = os.environ.get("NV_OUT", "runs")
PREP = os.environ.get("NV_PREP", os.path.join(OUT, "prepared"))

CLIP_MODEL = os.environ.get("NV_CLIP", "openai/clip-vit-large-patch14")
MLLM = os.environ.get("NV_MLLM", "Qwen/Qwen2.5-VL-3B-Instruct")

YEARS = [str(y) for y in range(2010, 2022)]   # TARA images go up to 2021 (as in NewsRECON)
TOPK = 50                                     # bi-encoder candidates
N_WINDOW = 7                                  # +/- days for event relevance (NewsRECON)
N_EVIDENCE = 3                                # articles shown to the MLLM (NewsRECON uses top-3)
N_OPTIONS = 4                                 # retrieved candidate events in MCQA (+ "none" option)


def p(*parts):
    path = os.path.join(OUT, *parts)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path


def tara_image(split, idx):
    """TARA images are stored as <idx>.png where idx is the line number in the jsonl file."""
    base = os.path.join(DATA, "TARA_dataset", "img", split)
    nested = os.path.join(base, split, f"{idx}.png")       # train images sit in img/train/train/
    return nested if os.path.exists(nested) else os.path.join(base, f"{idx}.png")
