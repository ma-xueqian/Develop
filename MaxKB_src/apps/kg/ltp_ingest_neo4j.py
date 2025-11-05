import os, glob, re
from typing import List, Dict, Any
from neo4j import GraphDatabase
from ltp import LTP

MAX_NAME_LEN = 18

def push_triples_to_neo4j(triples: List[Dict[str, Any]], uri: str, user: str, password: str, batch_size: int = 1000):
    driver = GraphDatabase.driver(uri, auth=(user, password))
    cypher = """
    UNWIND $rows AS row
    MERGE (s:Entity {name: row.s})
    MERGE (o:Entity {name: row.o})
    MERGE (s)-[r:REL {p_raw: row.p_raw}]->(o)
    SET r.type = coalesce(row.p,'OPEN'),
        r.caption = coalesce(row.p_raw, row.p),
        r.source = row.source,
        r.snippet = row.snippet,
        r.confidence = row.confidence
    """
    with driver.session() as sess:
        sess.run("CREATE CONSTRAINT IF NOT EXISTS FOR (e:Entity) REQUIRE e.name IS UNIQUE")
        for i in range(0, len(triples), batch_size):
            sess.run(cypher, rows=triples[i:i+batch_size]).consume()
    driver.close()

def clean_inline_math(s: str) -> str:
    s = re.sub(r'\$[^$]*\$', ' ', s)
    s = re.sub(r'\s+', ' ', s)
    return s.strip()

def canon_entity(s: str) -> str:
    if not s: return s
    s = clean_inline_math(s)
    s = re.sub(r'^(如果|若|当|在[^，,。；;]*时|在[^，,。；;]*情况下|对于|对)\s*[^，,。；;]*[，,]\s*', '', s)
    s = re.split(r'[，,；;]', s)[-1].strip()
    s = re.sub(r'[：:]+$', '', s)
    return s[:MAX_NAME_LEN]

def norm_value(o: str) -> str:
    if not o: return o
    o = o.strip()
    o = re.sub(r'\s*([~～\-至到])\s*', r'\1', o)
    return o[:max(24, MAX_NAME_LEN)]

def split_items(s: str) -> List[str]:
    parts = re.split(r'[、，,；;和或/]+', s)
    return [p.strip() for p in parts if p.strip()]

def add(rows: List[Dict[str, Any]], s: str, p: str, o: str, source: str, snippet: str):
    s = canon_entity(s)
    o = norm_value(o)
    if not s or not o: return
    rows.append({"s": s, "o": o, "p_raw": p, "p": "OPEN", "source": source, "snippet": snippet, "confidence": None})

def extract_regex(text: str, source: str) -> List[Dict[str, Any]]:
    triples=[]
    for raw in re.split(r'[。\n\r]+', text):
        line = clean_inline_math(raw)
        t = line.strip().strip('-—').strip()
        if not t: continue

        # 组成/构成
        m = re.match(r'^(.{1,50}?)[以由]\s*(.+?)\s*(组成|构成)', t)
        if m:
            s = canon_entity(m.group(1)); items = split_items(m.group(2))
            for it in items: add(triples, s, "组成", it, source, t)
            continue

        # 包含/包括
        m = re.match(r'^(.{1,50}?)(?:包含|包括)\s*(.+)$', t)
        if m:
            s = canon_entity(m.group(1)); items = split_items(m.group(2))
            for it in items: add(triples, s, "包含", it, source, t)
            continue

        # 采用/使用
        m = re.match(r'^(.{1,50}?)(?:采用|使用)\s*(.+)$', t)
        if m:
            add(triples, m.group(1), "采用", m.group(2), source, t); continue

        # 编码为
        m = re.match(r'^(.{1,50}?)(?:的)?编码为\s*([^；;。]+)$', t)
        if m:
            add(triples, m.group(1), "编码", m.group(2), source, t); continue

        # 长度为 N 字符
        m = re.match(r'^(.{1,50}?)(?:的)?长度为\s*(\d+)\s*个?字?符', t)
        if m:
            add(triples, m.group(1), "长度", f"{m.group(2)}字符", source, t); continue
        m = re.match(r'^(.{1,50}?)为?\s*(\d+)\s*个?字?符', t)
        if m:
            add(triples, m.group(1), "长度", f"{m.group(2)}字符", source, t); continue

        # 范围: a~b / a至b / a到b
        m = re.match(r'^(.{1,50}?)(?:的)?(?:范围|取值范围)?(?:为|是)?\s*([-\d.%]+[~～\-至到][-\d.%]+)', t)
        if m:
            add(triples, m.group(1), "范围", m.group(2), source, t); continue

        # 最大/最小
        m = re.match(r'^(.{1,50}?)(?:应)?(?:不超过|不得超过|不大于|≤)\s*([-\d.]+%?)', t)
        if m:
            add(triples, m.group(1), "最大值", m.group(2), source, t); continue
        m = re.match(r'^(.{1,50}?)(?:应)?(?:不少于|不小于|≥)\s*([-\d.]+%?)', t)
        if m:
            add(triples, m.group(1), "最小值", m.group(2), source, t); continue

        # 称为/叫做/名为
        m = re.match(r'^(.{1,50}?)(?:称为|叫做|名为)\s*([^；;。]+)', t)
        if m:
            add(triples, m.group(1), "称为", m.group(2), source, t); continue

        # 冒号“X：Y” -> 取值/定义
        m = re.match(r'^(.{1,50})[：:]\s*([^。；;]+)', t)
        if m:
            add(triples, m.group(1), "取值", m.group(2), source, t); continue

        # 普通“X 为 Y”
        m = re.match(r'^(.{1,50}?)\s*为\s*([^。；;]+)', t)
        if m:
            add(triples, m.group(1), "为", m.group(2), source, t); continue

    return triples

def extract_ltp(text: str, source: str, ltp: LTP) -> List[Dict[str, Any]]:
    triples=[]
    sents = [clean_inline_math(t) for t in re.split(r'[。；;.!?！？\n\r]+', text) if t.strip()]
    for sent in sents:
        try:
            out = ltp.pipeline([sent], tasks=["srl"])
        except Exception:
            continue
        srl_list = getattr(out, "srl", None) or []
        if not srl_list or not srl_list[0]:
            continue
        for pred in (srl_list[0] or []):
            p = (pred.get("predicate") or "").strip()
            subj=None; objs=[]
            for role, val, *_ in (pred.get("arguments") or []):
                if role=="A0" and subj is None: subj=(val or "").strip()
                elif role in ("A1","A2","PAT","CONT","OBJ") and val: objs.append(val.strip())
            if subj and p and objs:
                for o in objs:
                    add(triples, subj, p, o, source, sent)
    return triples

def extract_dep(text: str, source: str, ltp: LTP) -> List[Dict[str, Any]]:
    """
    依存句法抽取：谓词=HED词，主语=指向HED的SBV，宾语=指向HED的VOB/POB/CMP/DBL
    不依赖手写领域规则
    """
    triples=[]
    sents=[clean_inline_math(t) for t in re.split(r'[。；;.!?！？\n\r]+', text) if t.strip()]
    for sent in sents:
        try:
            out = ltp.pipeline([sent], tasks=['cws','pos','dep'])
        except Exception:
            continue
        if not out.dep: 
            continue
        words = out.cws[0]
        info  = out.dep[0]          # {'head':[...], 'label':[...]} 1-based
        head  = info['head']
        label = info['label']
        hed_idx = next((i for i,l in enumerate(label) if l=='HED'), None)
        if hed_idx is None: 
            continue
        p = words[hed_idx]

        # 找到直接依存到 HED 的主语与宾语
        sbj_idx = [i for i,l in enumerate(label) if l=='SBV' and head[i]==hed_idx+1]
        obj_idx = [i for i,l in enumerate(label) if l in ('VOB','POB','CMP','DBL') and head[i]==hed_idx+1]

        # 简单扩展主语短语：拼接其修饰（ATT/ADV/NUM/QUN）
        def phrase(i:int)->str:
            mods = [k for k in range(len(words)) if head[k]==i+1 and label[k] in ('ATT','ADV','NUM','QUN')]
            toks = sorted(mods+[i])
            return ''.join(words[t] for t in toks if words[t].strip())

        for si in sbj_idx:
            s = phrase(si)
            for oi in obj_idx:
                o = words[oi]
                add(triples, s, p, o, source, sent)
    return triples

def run(pattern: str, uri: str, user: str, password: str, model: str="LTP/small", extract: str="srl,regex"):
    ltp=LTP(model)
    use = {x.strip() for x in extract.split(',') if x.strip()}
    files=sorted(glob.glob(pattern))
    all_triples=[]
    for fp in files:
        with open(fp, "r", encoding="utf-8", errors="ignore") as f:
            txt=f.read()
        t=[]
        if 'regex' in use: t += extract_regex(txt, fp)
        if 'srl'   in use: t += extract_ltp(txt, fp, ltp)
        if 'dep'   in use: t += extract_dep(txt, fp, ltp)
        all_triples.extend(t)
        print(f"{os.path.basename(fp)} -> {len(t)} triples (extract={'+'.join(sorted(use))})")
    if all_triples:
        push_triples_to_neo4j(all_triples, uri, user, password)
        print(f"pushed {len(all_triples)} triples")
    else:
        print("no triples")

if __name__ == "__main__":
    import argparse
    ap=argparse.ArgumentParser()
    ap.add_argument("--glob", default="/opt/maxkb/data/kg/md/*.md")
    ap.add_argument("--uri", default="bolt://neo4j:7687")
    ap.add_argument("--user", default="neo4j")
    ap.add_argument("--password", default="neo4j_pass123")
    ap.add_argument("--model", default="LTP/small")
    ap.add_argument("--extract", default="srl,regex", help="可选: srl|dep|regex，逗号分隔；如只用SRL则传 srl")
    args=ap.parse_args()
    run(args.glob, args.uri, args.user, args.password, args.model, args.extract)