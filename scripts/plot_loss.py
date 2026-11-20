"""Training loss vs epoch from the train_lora.py logs.

    python plot_loss.py gpu1.log gpu0.log      -> figs/loss_curve.pdf (+ .png)
    python plot_loss.py newsvqa_runs/ckpt/lora_mixed_rerun/loss_log.json newsvqa_runs/ckpt/lora_qa_rerun/loss_log.json

Parses lines like 'step 25/375  loss 0.6123 ...' (one point per 5 optimizer steps).
The x axis is in epochs (step / total steps; both runs are 1 epoch).
"""
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PAT = re.compile(r"^step (\d+)/(\d+)\s+loss ([0-9.]+)")


def parse(path):
    if path.endswith(".json"):          # loss_log.json from nv/train_lora.py (one entry per optimizer step)
        import json
        h = json.load(open(path))
        tot = h[-1]["step"]
        w = 5                           # 5-step moving average, same smoothing as the printed log
        out = []
        for i in range(len(h)):
            win = h[max(0, i - w + 1):i + 1]
            out.append((h[i]["step"] / tot, sum(x["loss"] for x in win) / len(win), tot))
        return out
    pts = []
    for line in open(path, encoding="utf-8", errors="ignore"):
        m = PAT.match(line.strip())
        if m:
            s, tot, loss = int(m.group(1)), int(m.group(2)), float(m.group(3))
            pts.append((s / tot, loss, tot))
    return pts


def main():
    runs = [(sys.argv[1], "Mixed QA/MCQA/QAR (6,000 samples)", "#1f5aa6", "-")]
    if len(sys.argv) > 2:
        runs.append((sys.argv[2], "QA only (1,769 samples)", "#8a8a8a", "--"))
    plt.rcParams.update({"font.size": 8, "font.family": "serif"})
    fig, ax = plt.subplots(figsize=(3.4, 2.2))
    for path, label, color, ls in runs:
        pts = parse(path)
        if not pts:
            print("no 'step ... loss' lines in", path)
            continue
        tot = pts[0][2]
        totals = {p[2] for p in pts}
        if len(totals) > 1:   # log holds more than one run: keep the last one
            pts = [p for p in pts if p[2] == pts[-1][2]]
        ax.plot([p[0] for p in pts], [p[1] for p in pts], ls, color=color, lw=1.2, label=f"{label}")
        print(f"{path}: {len(pts)} points, {tot} steps, loss {pts[0][1]:.3f} -> {pts[-1][1]:.3f}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Training loss (answer tokens)")
    ax.set_xlim(0, 1)
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.3, lw=0.5)
    ax.legend(frameon=False)
    fig.tight_layout()
    import os
    os.makedirs("figs", exist_ok=True)
    fig.savefig("figs/loss_curve.pdf")
    fig.savefig("figs/loss_curve.png", dpi=200)
    print("saved figs/loss_curve.pdf")


if __name__ == "__main__":
    main()
