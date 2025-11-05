import re
from typing import List, Tuple
import jieba.posseg as pseg

SENT_SPLIT = re.compile(r"[。！？!?；;]\s*|\n+")
CODE_BLOCK = re.compile(r"```.*?```", re.S)
INLINE_CODE = re.compile(r"`[^`]+`")
LINK_IMG = re.compile(r"!\[[^\]]*\]\([^)]+\)|\[[^\]]*\]\([^)]+\)")
MD_MARKS = re.compile(r"^[#>\-\*\s]+", re.M)

def read_md_text(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        txt = f.read()
    txt = CODE_BLOCK.sub(" ", txt)
    txt = INLINE_CODE.sub(" ", txt)
    txt = LINK_IMG.sub(" ", txt)
    txt = MD_MARKS.sub("", txt)
    return txt

def extract_entities(sent: str) -> List[str]:
    keep = {"nr","ns","nt","nz","n"}
    seen, out = set(), []
    for w, f in pseg.cut(sent):
        if f in keep and len(w) >= 2 and w not in seen:
            seen.add(w); out.append(w)
    return out

def triples_from_sentence(sent: str) -> List[Tuple[str,str,str]]:
    ents = extract_entities(sent)
    rel = "RELATED"
    if "属于" in sent: rel = "BELONGS_TO"
    elif "位于" in sent: rel = "LOCATED_IN"
    elif "包含" in sent or "包括" in sent: rel = "INCLUDES"
    elif "是" in sent: rel = "IS_A"
    triples: List[Tuple[str,str,str]] = []
    if len(ents) >= 2:
        triples.append((ents[0], rel, ents[1]))
        for i in range(len(ents)-1):
            triples.append((ents[i], "RELATED", ents[i+1]))
    return triples

def md_to_triples(path: str):
    text = read_md_text(path)
    triples = []
    for s in filter(None, (seg.strip() for seg in SENT_SPLIT.split(text))):
        triples.extend(triples_from_sentence(s))
    return triples