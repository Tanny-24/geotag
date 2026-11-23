"""Geotag web app (Gradio): upload a news photo -> retrieve NYT articles -> the fine-tuned Qwen2.5-VL weighs the
candidate events -> the trust layer answers only as precisely as its calibrated confidence allows. Follow-up
questions are answered with the same photo and articles in context.

    python demo.py --adapter ckpt/lora_mixed/final --share [--examples 12,345,678]

Reads from $NV_OUT: prepared/corpus.jsonl, emb/corpus_{text,loc}.npy, ckpt/loc_head.pt, geonames/,
results/retrieval_ce_recall.json and results/trust_lora_mixed_mcqa.json (written by nv.hierarchy).
"""
import argparse
from datetime import date

import gradio as gr
import numpy as np
import torch
from transformers import CLIPModel, CLIPProcessor

from nv import config as C
from nv.common import CONTINENTS, candidate_events, evidence_block, load_corpus, read_json
from nv.cross_encoder import LocHead
from nv.geo import Geo
from nv.hierarchy import LEVELS, date_path, evidence_conf, features, loc_path, node_scores
from nv.infer import generate, load_model
from nv.prompts import LETTERS, build_prompt

EPS = {"5%": "0.05", "10%": "0.1", "20%": "0.2"}       # allowed error -> key in the trust-layer results


class Pipeline:
    def __init__(self, adapter):
        self.corpus = load_corpus()
        self.txt = torch.from_numpy(np.load(C.p("emb", "corpus_text.npy"))).cuda()
        self.loc = torch.from_numpy(np.load(C.p("emb", "corpus_loc.npy")).astype(np.float32)).cuda()
        self.clip = CLIPModel.from_pretrained(C.CLIP_MODEL, torch_dtype=torch.float16).cuda().eval()
        self.cproc = CLIPProcessor.from_pretrained(C.CLIP_MODEL)
        self.head = LocHead().cuda().eval()
        self.head.load_state_dict(torch.load(C.p("ckpt", "loc_head.pt")))
        self.alpha = read_json(C.p("results", "retrieval_ce_recall.json"))["alpha"]   # re-ranker weight picked on dev
        self.geo = Geo()
        dims = read_json(C.p("results", "trust_lora_mixed_mcqa.json"))["dims"]
        # evidence-model weights and the lambda calibrated for every allowed error, per dimension
        self.trust = {dim: (np.array(list(d["weights"]["evidence"].values())),
                            {e: d["eps"][e]["evidence"]["lambda"] for e in EPS.values()}) for dim, d in dims.items()}
        self.paths = {"location": (lambda o: loc_path(self.geo, o["loc"]), lambda a: loc_path(self.geo, a["loc_str"])),
                      "date": (lambda o: date_path(o["date"]), lambda a: date_path(a["date"]))}
        self.model, self.proc = load_model(adapter)

    @torch.no_grad()
    def retrieve(self, img, k=C.TOPK):
        px = self.cproc(images=img, return_tensors="pt")["pixel_values"].cuda().half()
        e = self.clip.visual_projection(self.clip.vision_model(pixel_values=px).pooler_output)
        e = torch.nn.functional.normalize(e.float(), dim=-1)
        bi, ids = (e.half() @ self.txt.T).float().topk(k)
        ce = torch.sigmoid(self.head(e.expand(k, -1), self.loc[ids[0]]))
        comb = bi[0] * ce.clamp_min(1e-6).pow(self.alpha)   # same fusion as cross_encoder.rerank (alpha=1 hurt on dev)
        order = comb.argsort(descending=True)
        return [self.corpus[int(ids[0, o])] for o in order]

    def answer(self, img):
        ranked = self.retrieve(img)
        options = candidate_events(ranked[:20])
        _, probs = generate(self.model, self.proc, [build_prompt("mcqa", ranked[:C.N_EVIDENCE], options)], [img], 8)
        p = probs[0][:len(options) + 1]
        return self.score(options, (p / p.sum()).tolist(), ranked)

    def score(self, options, p, ranked):
        """-> candidate events, their probabilities (last: "none"), and per dimension k*'s path with its calibrated
        score at each depth."""
        scores = {}
        for dim, (opt_fn, art_fn) in self.paths.items():
            pk, c, t, e = node_scores(p[:len(options)], [opt_fn(o) for o in options], [art_fn(a) for a in ranked[:20]])
            conf = np.array([c])
            scores[dim] = (pk, evidence_conf(self.trust[dim][0], features(conf, np.array([t]), np.array([e])), conf)[0])
        return {"options": options, "p": p, "scores": scores, "arts": ranked[:C.N_EVIDENCE]}

    def chat(self, img, arts, question, history):
        prompt = (f"This is a news photograph. Retrieved news articles:\n\n{evidence_block(arts)}\n\n"
                  + "".join(f"Q: {q}\nA: {a}\n" for q, a in history[-4:]) + f"Q: {question}\nA:")
        out, _ = generate(self.model, self.proc, [prompt], [img], 200)
        return out[0].strip()


def place(geo, loc, depth):
    """'Haiti' / 'Ouest, Haiti' / 'Port-au-Prince, Ouest, Haiti' for depth 1 / 2 / 3, same nodes as loc_path."""
    h = [x for x in geo.lookup(loc)[1] if x.lower() not in CONTINENTS]
    h = h[:2] + [h[-1]] if len(h) > 2 else h
    return ", ".join(reversed(h[:depth]))


def when(path, depth):
    y, m, d = (path[-1].split("-") + [None, None])[:3]
    if depth == 1:
        return y
    month = date(int(y), int(m), 1).strftime("%B %Y")
    return month if depth == 2 else f"{int(d)} {month}"


def render(P, res, eps_label):
    """Markdown for the answer at one allowed error."""
    e, options, p = EPS[eps_label], res["options"], res["p"]
    if not options:
        return "No candidate events were retrieved for this photo, so Geotag gives no answer."
    k = int(np.argmax(p[:len(options)]))
    lines = []
    for dim in ("location", "date"):
        pk, s = res["scores"][dim]
        lam = P.trust[dim][1][e]
        depth = int((s >= lam).sum())
        if not pk:
            lines.append(f"**{dim.capitalize()}:** no answer (the top candidate is not in the gazetteer)")
        elif depth == 0:
            lines.append(f"**{dim.capitalize()}:** not sure enough to say. Confidence in the {LEVELS[dim][0]} is "
                         f"{max(s[0], 0):.0%}, the threshold is {lam:.0%}.")
        else:
            value = place(P.geo, options[k]["loc"], depth) if dim == "location" else when(pk, depth)
            lines.append(f"**{dim.capitalize()}: {value}**, answered at {LEVELS[dim][depth - 1]} level "
                         f"(confidence {s[depth - 1]:.0%}, threshold {lam:.0%})")
    best = options[k]
    lines += ["", f"Model's top candidate in full detail: {best['loc']}, {best['date']} ({p[k]:.0%})",
              "", f"The thresholds are calibrated on held-out photos so that on average at most {eps_label} of photos "
              "get a wrong answer. Coarser answers are given when a finer one would not meet that bar.",
              "", "| Option | Location | Date | Probability |", "|---|---|---|---|"]
    lines += [f"| {LETTERS[i]} | {o['loc']} | {o['date']} | {p[i]:.0%} |" for i, o in enumerate(options)]
    lines.append(f"| {LETTERS[len(options)]} | none of these | | {p[len(options)]:.0%} |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default="ckpt/lora_mixed/final")
    ap.add_argument("--share", action="store_true")
    ap.add_argument("--examples", default="", help="comma list of TARA test indices shown as example photos")
    args = ap.parse_args()
    P = Pipeline(args.adapter)

    with gr.Blocks(title="Geotag") as demo:
        gr.Markdown("## Geotag: where and when was this news photo taken?\n"
                    "Retrieves NYT articles for the photo, lets a fine-tuned Qwen2.5-VL-3B weigh them, and answers "
                    "only as precisely as it can back up: city, region or country, or no answer.")
        state = gr.State({"img": None, "arts": [], "hist": [], "res": None})
        with gr.Row():
            with gr.Column():
                img = gr.Image(type="pil", label="News photo")
                eps = gr.Radio(list(EPS), value="10%", label="Allowed error")
                btn = gr.Button("Locate", variant="primary")
            ans = gr.Markdown()
        with gr.Accordion("Evidence: top-3 retrieved articles", open=False):
            ev = gr.Markdown()
        chat = gr.Chatbot(label="Ask about the photo or the event", type="messages")
        q = gr.Textbox(placeholder="e.g. What event is this? Who organised it?")
        if args.examples:
            gr.Examples([[C.tara_image("test", int(i))] for i in args.examples.split(",")], inputs=img)

        def run(image, eps_label, st):
            if image is None:
                return "Upload a photo first.", "", [], st
            res = P.answer(image)
            st = {"img": image, "arts": res["arts"], "hist": [], "res": res}
            evid = "\n\n".join(f"**{a['head']}** ({a['date']}, {a['loc_str']})  \n{a['abs']}  \n[{a['url']}]({a['url']})"
                               for a in res["arts"])
            return render(P, res, eps_label), evid, [], st

        def rerender(eps_label, st):
            return render(P, st["res"], eps_label) if st["res"] else ""

        def ask(question, hist, st):
            if st["img"] is None:
                return hist + [{"role": "user", "content": question},
                               {"role": "assistant", "content": "Upload a photo and press Locate first."}], "", st
            a = P.chat(st["img"], st["arts"], question, st["hist"])
            st["hist"].append((question, a))
            return hist + [{"role": "user", "content": question}, {"role": "assistant", "content": a}], "", st

        btn.click(run, [img, eps, state], [ans, ev, chat, state])
        eps.change(rerender, [eps, state], ans)
        q.submit(ask, [q, chat, state], [chat, q, state])
    demo.launch(share=args.share, server_name="0.0.0.0")


if __name__ == "__main__":
    main()
