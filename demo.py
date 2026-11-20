"""News image chatbot demo (Gradio): upload a news photo -> retrieve NYT articles -> fine-tuned Qwen2.5-VL
says where/when (MCQA over retrieved candidate events) and shows the evidence; follow-up questions are
answered with the same image + evidence in context.

    python demo.py --adapter ckpt/lora_mixed/final [--share]
"""
import argparse

import gradio as gr
import numpy as np
import torch
from transformers import CLIPModel, CLIPProcessor

from nv import config as C
from nv.common import candidate_events, evidence_block, load_corpus, parse_answer, read_json
from nv.cross_encoder import LocHead
from nv.infer import generate, load_model
from nv.prompts import LETTERS, build_prompt


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
        arts, options = ranked[:C.N_EVIDENCE], candidate_events(ranked[:20])
        out, probs = generate(self.model, self.proc, [build_prompt("mcqa", arts, options)], [img], 64)
        loc, date, letter = parse_answer(out[0])
        if letter and LETTERS.index(letter) < len(options):
            loc, date = loc or options[LETTERS.index(letter)]["loc"], date or options[LETTERS.index(letter)]["date"]
        conf = float(probs[0][:len(options) + 1].max() / probs[0][:len(options) + 1].sum())
        return loc, date, letter, conf, arts, options

    def chat(self, img, arts, question, history):
        prompt = (f"This is a news photograph. Retrieved news articles:\n\n{evidence_block(arts)}\n\n"
                  + "".join(f"Q: {q}\nA: {a}\n" for q, a in history[-4:]) + f"Q: {question}\nA:")
        out, _ = generate(self.model, self.proc, [prompt], [img], 200)
        return out[0].strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default="ckpt/lora_mixed/final")
    ap.add_argument("--share", action="store_true")
    args = ap.parse_args()
    P = Pipeline(args.adapter)

    with gr.Blocks(title="NewsVQA") as demo:
        gr.Markdown("## NewsVQA — where and when was this news photo taken?")
        state = gr.State({"img": None, "arts": [], "hist": []})
        with gr.Row():
            img = gr.Image(type="pil", label="News photo")
            with gr.Column():
                ans = gr.Markdown()
                btn = gr.Button("Contextualize", variant="primary")
        ev = gr.Markdown()
        chat = gr.Chatbot(label="Ask about the photo / event", type="messages")
        q = gr.Textbox(placeholder="e.g. What event is this? Who organised it?")

        def run(image, st):
            loc, date, letter, conf, arts, options = P.answer(image)
            st = {"img": image, "arts": arts, "hist": []}
            opts = "\n".join(f"- **{LETTERS[i]})** {o['loc']} — {o['date']}" for i, o in enumerate(options))
            evid = "\n\n".join(f"**{a['head']}** ({a['date']}, {a['loc_str']})  \n{a['abs']}  \n[{a['url']}]({a['url']})"
                               for a in arts)
            return (f"### 📍 {loc}\n### 🗓️ {date}\nchosen option **{letter}** (confidence {conf:.0%})\n\n{opts}",
                    "#### Evidence (top-3 retrieved articles)\n\n" + evid, [], st)

        def ask(question, hist, st):
            if st["img"] is None:
                return hist + [{"role": "user", "content": question},
                               {"role": "assistant", "content": "Upload a photo and press Contextualize first."}], "", st
            a = P.chat(st["img"], st["arts"], question, st["hist"])
            st["hist"].append((question, a))
            return hist + [{"role": "user", "content": question}, {"role": "assistant", "content": a}], "", st

        btn.click(run, [img, state], [ans, ev, chat, state])
        q.submit(ask, [q, chat, state], [chat, q, state])
    demo.launch(share=args.share, server_name="0.0.0.0")


if __name__ == "__main__":
    main()
