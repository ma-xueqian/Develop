import os, re, json, hashlib
from typing import List, Dict, Any
from openai import OpenAI
import requests

# —— 文本清洗与分块 —— #
SENT_SPLIT = re.compile(r"[。！？!?；;]\s*|\n+")
CODE_BLOCK = re.compile(r"```.*?```", re.S)
INLINE_CODE = re.compile(r"`[^`]+`")
LINK_IMG   = re.compile(r"!\[[^\]]*\]\([^)]+\)|\[[^\]]*\]\([^)]+\)")
MD_MARKS   = re.compile(r"^[#>\-\*\s]+", re.M)

def read_md_clean(path: str) -> str:
    t = open(path, "r", encoding="utf-8", errors="ignore").read()
    t = CODE_BLOCK.sub(" ", t)
    t = INLINE_CODE.sub(" ", t)
    t = LINK_IMG.sub(" ", t)
    t = MD_MARKS.sub("", t)
    return t.strip()

def chunks(text: str, max_len=1200) -> List[str]:
    segs = [s.strip() for s in SENT_SPLIT.split(text) if s.strip()]
    out, cur = [], ""
    for s in segs:
        if len(cur) + len(s) + 1 > max_len:
            if cur: out.append(cur); cur=""
        cur = (cur + " " + s).strip()
    if cur: out.append(cur)
    return out

# —— 明确的抽取契约（s/p/o） —— #
SYS = "你是信息抽取助手。严格按要求输出 JSON，不要任何解释。"
USR = """请从给定文本中抽取知识图谱三元组，严格输出 JSON：
{
  "triples": [
    {"s": "主语(原文中的实体短语)", "p": "谓词短语(原文中的关系短语，尽量简短)", "o": "宾语(原文中的实体短语)", "confidence": 0.0, "snippet": "可选:承载该三元组的原句"}
  ]
}
要求：
- s 与 o 必须是原文中可直接找到的连续片段（允许把全/半角括号统一为()后匹配）。
- p 也尽量来自原文中的短谓词短语（如“符合…标准”“要求”“采用”“包含”“定义为”等）。
- 没有可抽取的三元组时返回 {"triples": []}。
- 不要输出多余字段，不要解释。
文本：
{chunk}
"""

def _key(text: str) -> str:
    m=hashlib.md5(); m.update(text.encode()); return m.hexdigest()

def _safe_json(s: str):
    try:
        return json.loads(s or "{}")
    except Exception:
        m = re.search(r"\{.*\}", s or "", re.S)
        return json.loads(m.group(0)) if m else {"triples":[]}

_LTP = None
def extract_ltp(chunk: str) -> List[Dict[str, Any]]:
    """
    使用 LTP 的语义角色标注（SRL）抽取三元组：
    A0 -> 主语，predicate -> 谓词，A1/A2 -> 宾语
    返回 [{s,p_raw,o,confidence,snippet}]
    """
    try:
        global _LTP
        if _LTP is None:
            from ltp import LTP
            model_name = os.getenv("LTP_MODEL", "LTP/small")
            _LTP = LTP(model_name)
        out = _LTP.pipeline([chunk], tasks=["srl"])
        triples: List[Dict[str, Any]] = []
        for pred in (out.srl[0] or []):
            p = (pred.get("predicate") or "").strip()
            args = pred.get("arguments") or []
            subj = None; objs: List[str] = []
            for role, text, *_ in args:
                if role == "A0" and subj is None:
                    subj = (text or "").strip()
                elif role in ("A1", "A2", "PAT", "CONT", "OBJ"):
                    if text: objs.append(text.strip())
            if subj and p and objs:
                for o in objs:
                    triples.append({"s": subj, "p_raw": p, "o": o,
                                    "confidence": None, "snippet": chunk})
        return triples
    except Exception:
        return []

def extract_openie(chunk: str) -> List[Dict[str, Any]]:
    """
    调用 OpenIE5，输出统一结构 [{s,p_raw,o,confidence,snippet}]
    环境变量：OPENIE_URL（默认 http://localhost:8000/extract）
    """
    url = os.getenv("OPENIE_URL", "http://localhost:8000/extract")
    r = requests.get(url, params={"text": chunk}, timeout=30)
    r.raise_for_status()
    payload = r.json()
    out: List[Dict[str, Any]] = []

    items = payload.get("extractions") if isinstance(payload, dict) and "extractions" in payload else payload
    if items is None and isinstance(payload, dict) and "sentences" in payload:
        items = payload["sentences"]

    def emit(s, p, o, conf=None, sent=None):
        s, p, o = (s or "").strip(), (p or "").strip(), (o or "").strip()
        if s and p and o:
            out.append({"s": s, "p_raw": p, "o": o, "confidence": conf, "snippet": sent or chunk})

    if isinstance(items, list):
        for e in items:
            if isinstance(e, dict) and "extraction" in e:
                ex = e["extraction"]
                s = ex.get("arg1", {}).get("text") if isinstance(ex.get("arg1"), dict) else ex.get("arg1")
                p = ex.get("rel", {}).get("text") if isinstance(ex.get("rel"), dict) else ex.get("rel")
                for a2 in (ex.get("arg2s") or []):
                    o = a2.get("text") if isinstance(a2, dict) else a2
                    emit(s, p, o, e.get("confidence"), e.get("sentence"))
            elif isinstance(e, dict) and ("rel" in e or "relation" in e):
                s = e.get("arg1") or e.get("subject")
                if isinstance(s, dict): s = s.get("text")
                p = e.get("rel") or e.get("relation")
                if isinstance(p, dict): p = p.get("text")
                arg2s = e.get("arg2s") or [e.get("arg2")]
                for a2 in (arg2s or []):
                    o = a2.get("text") if isinstance(a2, dict) else a2
                    emit(s, p, o, e.get("confidence"), e.get("sentence"))
            elif isinstance(e, dict) and "arguments" in e:
                args = e.get("arguments") or []
                if len(args) >= 2:
                    s = args[0].get("text"); o = args[1].get("text")
                    p = e.get("rel") or e.get("relation")
                    emit(s, p, o, e.get("confidence"), e.get("sentence"))
    return out

# —— 谓词归一（软归一：不丢 p_raw） —— #
REL_MAP = {
    # 协议/标准
    "协议":"COMPLIES_WITH","标准":"COMPLIES_WITH","规范":"COMPLIES_WITH","规约":"COMPLIES_WITH",
    "protocol":"COMPLIES_WITH","standard":"COMPLIES_WITH","itu":"COMPLIES_WITH","eia":"COMPLIES_WITH",
    "gb":"COMPLIES_WITH","iso":"COMPLIES_WITH","x.25":"COMPLIES_WITH",
    # 要求/参数
    "要求":"REQUIRES","规定":"REQUIRES","参数":"REQUIRES","必须":"REQUIRES","需要":"REQUIRES",
    "波特率":"REQUIRES","校验":"REQUIRES","停止位":"REQUIRES","帧":"REQUIRES","格式":"REQUIRES",
    # 定义/含义
    "定义":"DEFINED_AS","是":"DEFINED_AS","定义为":"DEFINED_AS",
    "含义":"CODE_MEANING","意思":"CODE_MEANING","表示":"CODE_MEANING","meaning":"CODE_MEANING",
    # 组成/包含/隶属
    "包含":"INCLUDES","包括":"INCLUDES","组成":"INCLUDES","属于":"PART_OF","部分":"PART_OF",
    # 使用/位置/别名
    "使用":"USES","采用":"USES","位于":"LOCATED_IN","在":"LOCATED_IN",
    "别名":"ALIAS","又称":"ALIAS","aka":"ALIAS"
}
def _normalize_rel(raw: str) -> str:
    # 在 openie/ltp 模式下，默认不做归一(type=OPEN)，避免丢语义；KG_CANONICALIZE=1 时再映射
    if os.getenv("EXTRACTOR", "llm").lower() in ("openie", "ltp") and os.getenv("KG_CANONICALIZE", "0") != "1":
        return "OPEN"
    if not raw: 
        return "RELATED"
    k = str(raw).strip().lower()
    for key, std in REL_MAP.items():
        if key in k:
            return std
    return "RELATED"

# —— 轻过滤与去重（只校验 s/o 可在原文找到） —— #
def _post_filter(raw_triples: List[Dict[str, Any]], chunk: str) -> List[Dict[str, Any]]:
    ch = chunk.replace("（","(").replace("）",")")
    out, seen = [], set()
    for t in raw_triples or []:
        s = (t.get("s") or t.get("subject") or t.get("head") or "").strip()
        p_raw = (t.get("p_raw") or t.get("p") or t.get("predicate") or t.get("relation") or "").strip()  # 修改点
        o = (t.get("o") or t.get("object") or t.get("tail") or "").strip()
        if not s or not o or not p_raw: 
            continue
        if s == o:
            continue
        s_n, o_n = s.replace("（","(").replace("）",")"), o.replace("（","(").replace("）",")")
        if (s_n not in ch) or (o_n not in ch):
            continue
        key = (s, p_raw, o)
        if key in seen: 
            continue
        seen.add(key)
        out.append({
            "s": s, "o": o,
            "p": _normalize_rel(p_raw),
            "p_raw": p_raw,
            "confidence": t.get("confidence"),
            "snippet": t.get("snippet")
        })
    return out

# —— 主入口：返回 [{s,o,p,p_raw,confidence,source,snippet}] —— #
def md_to_triples_llm_glm(path: str) -> List[Dict[str, Any]]:
    extractor = os.getenv("EXTRACTOR", "llm").lower()  # llm | openie | ltp
    cache_dir = os.getenv("KG_LLM_CACHE", "/opt/maxkb/data/kg/cache")
    bypass = os.getenv("KG_LLM_CACHE_BYPASS", "0") == "1"
    os.makedirs(cache_dir, exist_ok=True)

    client = None; model = None
    if extractor == "llm":
        client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL", "https://open.bigmodel.cn/api/paas/v4")
        )
        model = os.getenv("OPENAI_MODEL", "glm-4-flash")

    txt = read_md_clean(path)
    if not txt:
        return []

    results: List[Dict[str, Any]] = []
    for ck in chunks(txt, 1200):
        cf = os.path.join(cache_dir, f"{_key(extractor + ':' + ck)}.json")
        if (not bypass) and os.path.exists(cf):
            triples = json.load(open(cf, "r", encoding="utf-8"))
        else:
            if extractor == "openie":
                triples = extract_openie(ck)
            elif extractor == "ltp":
                triples = extract_ltp(ck)
            else:
                resp = client.chat.completions.create(
                    model=model, temperature=0,
                    response_format={"type": "json_object"},
                    messages=[{"role":"system","content":SYS},
                              {"role":"user","content":USR.format(chunk=ck)}]
                )
                data = _safe_json(resp.choices[0].message.content)
                triples = []
                for t in data.get("triples", []):
                    s = (t.get("s") or t.get("subject") or "").strip()
                    p_raw = (t.get("p") or t.get("predicate") or "").strip()
                    o = (t.get("o") or t.get("object") or "").strip()
                    if s and p_raw and o:
                        triples.append({"s":s,"p_raw":p_raw,"o":o,
                                        "confidence":t.get("confidence"),"snippet":t.get("snippet")})
            json.dump(triples, open(cf,"w",encoding="utf-8"), ensure_ascii=False)

        normed = _post_filter(triples, ck)
        for t in normed:
            t["source"] = path
        results.extend(normed)
    return results

def push_triples_to_neo4j(triples: List[Dict[str, Any]], uri: str, user: str, password: str, batch_size: int = 1000):
    from neo4j import GraphDatabase
    driver = GraphDatabase.driver(uri, auth=(user, password))
    cypher = """
    UNWIND $rows AS row
    MERGE (s:Entity {name: row.s})
    MERGE (o:Entity {name: row.o})
    MERGE (s)-[r:REL {p_raw: row.p_raw}]->(o)
    SET r.type = coalesce(row.p,'OPEN'),
        r.source = row.source,
        r.snippet = row.snippet,
        r.confidence = row.confidence
    """
    with driver.session() as sess:
        sess.run("CREATE CONSTRAINT IF NOT EXISTS FOR (e:Entity) REQUIRE e.name IS UNIQUE")
        for i in range(0, len(triples), batch_size):
            sess.run(cypher, rows=triples[i:i+batch_size]).consume()
    driver.close()

if __name__ == "__main__":
    import argparse, glob, csv
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="/opt/maxkb/data/kg/md/*.md", help="要抽取的 Markdown 文件通配符")
    ap.add_argument("--out", default="", help="可选：导出 CSV 路径，不需要可留空")
    ap.add_argument("--to-neo4j", action="store_true", help="将三元组直接写入 Neo4j")
    ap.add_argument("--uri", default=os.getenv("NEO4J_URI","bolt://neo4j:7687"))
    ap.add_argument("--user", default=os.getenv("NEO4J_USER","neo4j"))
    ap.add_argument("--password", default=os.getenv("NEO4J_PASSWORD","neo4j"))
    args = ap.parse_args()

    paths = sorted(glob.glob(args.glob))
    rows: List[Dict[str, Any]] = []
    for p in paths:
        rows.extend(md_to_triples_llm_glm(p))

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["s","p","o","p_raw","source","snippet","confidence"])
            w.writeheader(); w.writerows(rows)
        print(f"CSV written: {args.out} ({len(rows)})")

    if args.to_neo4j and rows:
        push_triples_to_neo4j(rows, args.uri, args.user, args.password)
        print(f"Pushed {len(rows)} triples to Neo4j")