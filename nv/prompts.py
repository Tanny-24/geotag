"""Prompt templates shared by data building, inference and the demo.

Formats (LiveVQA Sec. 4: QA / MCQA / QAR), adapted to image contextualisation:
    qa_noev : image only (MLLM "default" setting in NewsRECON)
    qa      : image + top-3 retrieved articles  -> Location / Date      (NewsRECON "MLLM + NewsRECON" setting)
    mcqa    : image + top-3 articles + candidate events A-D from retrieval (hard negatives) + E "none"
              -> option letter, then Location / Date
    qar     : image + top-3 articles -> per-article verification, conflict resolution, then Location / Date
All outputs end with the same two lines, so a single parser scores every format.
"""
from .common import evidence_block

LETTERS = "ABCDE"
ANSWER_FMT = ("Location: <place, region, country>\n"
              "Date: <YYYY-MM-DD, or YYYY-MM / YYYY if unsure>")

INTRO = "This is a news photograph."
EVID_INTRO = ("Below are news articles retrieved for this photo. They may or may not describe the same event, "
              "so check them against what you see in the image.")


def build_prompt(fmt, arts=None, options=None):
    if fmt == "qa_noev":
        return f"{INTRO} Where and when was it taken?\nAnswer exactly in this format:\n{ANSWER_FMT}"
    ev = evidence_block(arts or [])
    head = f"{INTRO} {EVID_INTRO}\n\n{ev}\n\n"
    if fmt == "qa":
        return head + f"Where and when was the photo taken?\nAnswer exactly in this format:\n{ANSWER_FMT}"
    if fmt == "mcqa":
        opts = "\n".join(f"{LETTERS[k]}) Location: {o['loc']} | Date: {o['date']}" for k, o in enumerate(options))
        none = LETTERS[len(options)]
        return head + (f"Which candidate event matches the photo?\n{opts}\n"
                       f"{none}) None of the candidates; infer from the image.\n"
                       f"Reply with the option letter on the first line, then give the final answer:\n{ANSWER_FMT}")
    if fmt == "qar":
        return head + ("Check each article against the image, say which ones you trust or reject and why, "
                       f"then give the final answer in this format:\n{ANSWER_FMT}")
    raise ValueError(fmt)


def answer_lines(loc, date):
    return f"Location: {loc}\nDate: {date}"
