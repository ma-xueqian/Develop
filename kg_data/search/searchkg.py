# 加入 verbalize
import os, re, json, sys
# sys.path.append("/opt/maxkb/app/sandbox/python-packages")
sys.path.insert(0, "/opt/maxkb/app/sandbox/python-packages")  # 先于系统包
from neo4j import GraphDatabase

# --- 内联 KG 访问 ---
URI = os.getenv("NEO4J_URI", "bolt://neo4j:7687")
USER = os.getenv("NEO4J_USER", "neo4j")
PWD  = os.getenv("NEO4J_PASSWORD", "neo4j_pass123")
_driver = GraphDatabase.driver(URI, auth=(USER, PWD))

def kg_search_nodes(kw: str, limit: int = 30):
    """
    多策略实体召回（单函数版）：
    1) 原词双向包含（保留 .，只去空格和短横线）
    2) 抽取“代号锚点”(X.25, RS-232 等) 逐个补检
    3) 若结果过少，再做中文尾缀回退（末尾 1~4 个中文字符去掉再搜）
    排序优先：RAW > ANCHOR > SUFFIX；同级按规范化长度差与名称长度。
    返回：[{id,name}]
    """
    kw = (kw or "").strip()
    if not kw:
        return []

    # ------------ 内部工具 ------------
    def norm(s: str) -> str:
        return s.replace(" ", "").replace("-", "").upper() if s else ""

    anchor_re = re.compile(r"[A-Za-z0-9]+(?:[.\-][A-Za-z0-9]+)*")
    cn_suffix_re = re.compile(r"^(.{2,}?)([\u4e00-\u9fa5]{1,4})$")

    def strip_cn_suffix(s: str) -> str:
        m = cn_suffix_re.match(s.strip())
        return m.group(1) if m else s

    def extract_anchors(text: str):
        raw = anchor_re.findall(text or "")
        seen = set(); out = []
        for a in raw:
            u = a.upper()
            if len(a) >= 2 and u not in seen:
                seen.add(u); out.append(a)
            if len(out) >= 6:  # 限量
                break
        return out

    def search_raw(term: str, tag: str):
        with _driver.session() as sess:
            rows = sess.run("""
                WITH $kw AS KW_RAW, toUpper($kw) AS KW_UP
                WITH KW_RAW, KW_UP,
                     toUpper(REPLACE(REPLACE($kw,'-',''),' ','')) AS K
                MATCH (n) WHERE n.name IS NOT NULL
                WITH n, K, KW_RAW, KW_UP,
                     toUpper(REPLACE(REPLACE(n.name,'-',''),' ','')) AS N,
                     toUpper(n.name) AS N_UP
                WHERE N CONTAINS K OR K CONTAINS N OR N_UP CONTAINS KW_UP
                RETURN id(n) AS id, n.name AS name
                LIMIT $lim
            """, kw=term, lim=int(limit)).data()
        return [{"id": r["id"], "name": r["name"], "_m": tag} for r in rows]

    def search_anchor(a: str):
        with _driver.session() as sess:
            rows = sess.run("""
                WITH toUpper(REPLACE(REPLACE($a,'-',''),' ','')) AS A
                MATCH (n) WHERE n.name IS NOT NULL
                WITH n, A, toUpper(REPLACE(REPLACE(n.name,'-',''),' ','')) AS N
                WHERE N CONTAINS A OR A CONTAINS N
                RETURN id(n) AS id, n.name AS name
                LIMIT $lim
            """, a=a, lim=int(limit)).data()
        return [{"id": r["id"], "name": r["name"], "_m": "ANCHOR"} for r in rows]

    # ------------ 检索流程 ------------
    results = []

    # (1) 原词匹配
    results += search_raw(kw, "RAW")

    # (2) 锚点补检
    if len(results) < limit:
        for a in extract_anchors(kw):
            results += search_anchor(a)
            if len(results) >= limit:
                break

    # (3) 中文尾缀回退（仅在结果很少时触发）
    if len(results) < 3:
        kw2 = strip_cn_suffix(kw)
        if kw2 != kw:
            results += search_raw(kw2, "SUFFIX")

    # ------------ 去重与排序 ------------
    uniq = {}
    for r in results:
        if r["id"] not in uniq:
            uniq[r["id"]] = r

    def score(r):
        base = {"RAW": 0, "ANCHOR": 10, "SUFFIX": 20}.get(r["_m"], 15)
        gap = abs(len(norm(r["name"])) - len(norm(kw)))
        return (base, gap, len(r["name"]))

    ordered = sorted(uniq.values(), key=score)
    return [{"id": r["id"], "name": r["name"]} for r in ordered[:limit]]


def kg_neighbors(node):
    nid = node["id"] if isinstance(node, dict) else int(node)
    with _driver.session() as s:
        rows = s.run("""
            MATCH (n) WHERE id(n)=$id
            MATCH (n)-[r]-(m)
            RETURN {id:id(n),name:n.name} AS src,
                   {type: coalesce(r.type, r.name, r.rel, r.kind, r.category),
                    caption: r.caption, p_raw: r.p_raw,         // 取出关系文字
                    source:r.source, confidence:r.confidence, label:type(r)} AS rel,
                   {id:id(m),name:m.name} AS dst
            LIMIT 1000
        """, id=nid).data()
    return [{"from": r["src"], "rel": r["rel"], "to": r["dst"]} for r in rows]

def kg_find_tails(node, relation: str):
    """
    type精配 OR caption/p_raw 同义词匹配
    """
    nid = node["id"] if isinstance(node, dict) else int(node)
    R = str(relation).strip().upper()
    syns = sorted({s.upper() for s in REL_VALUE_SYNONYMS.get(R, set())})
    with _driver.session() as s:
        rows = s.run("""
            WITH $id AS nid, $R AS R, $syns AS SYN_LIST
            MATCH (n) WHERE id(n)=nid
            MATCH (n)-[r]-(t)
            WITH r, t, SYN_LIST,
                 coalesce(r.type, r.name, r.rel, r.kind, r.category) AS RT,
                 toUpper(coalesce(r.caption,"") + " " + coalesce(r.p_raw,"")) AS TXT
            WHERE (RT IS NOT NULL AND toUpper(RT)=R)
               OR (size(SYN_LIST) > 0 AND any(x IN SYN_LIST WHERE TXT CONTAINS x))
            RETURN id(t) AS id, t.name AS name, r.source AS source, r.confidence AS confidence
            LIMIT 400
        """, id=nid, R=R, syns=syns).data()
    return [{"id": r["id"], "name": r["name"], "source": r.get("source"), "confidence": r.get("confidence")} for r in rows]
# --- 内联 KG 访问结束 ---

ALLOW = {"DEFINED_AS","CODE_MEANING","ALIAS","INCLUDES","PART_OF","USES","LOCATED_IN","REQUIRES","PROHIBITS","COMPLIES_WITH","RELATED"}
ORDER = {"COMPLIES_WITH":0,"USES":1,"DEFINED_AS":2,"CODE_MEANING":3,"ALIAS":4,"INCLUDES":5,"PART_OF":6,"LOCATED_IN":7,"REQUIRES":8,"PROHIBITS":9,"RELATED":99}
DEFAULT_CONF = {"DEFINED_AS":0.85,"CODE_MEANING":0.85,"COMPLIES_WITH":0.8,"USES":0.8,"INCLUDES":0.75,"PART_OF":0.75,"ALIAS":0.8,"LOCATED_IN":0.7,"REQUIRES":0.7,"PROHIBITS":0.7,"RELATED":0.5}

SYN_REL = {
    "DEFINES":"DEFINED_AS","DEFINE":"DEFINED_AS","IS":"DEFINED_AS","IS_A":"DEFINED_AS",
    "MEANS":"CODE_MEANING","MEANING":"CODE_MEANING","REPRESENTS":"CODE_MEANING",
    "COMPLIES":"COMPLIES_WITH","COMPLIES_WITH":"COMPLIES_WITH","STANDARD":"COMPLIES_WITH",
    "USE":"USES","USES":"USES","USED_BY":"USES",
    "INCLUDE":"INCLUDES","INCLUDES":"INCLUDES","CONTAINS":"INCLUDES",
    "PART_OF":"PART_OF","BELONGS_TO":"PART_OF",
    "ALIAS":"ALIAS","AKA":"ALIAS","SYNONYM":"ALIAS",
    "LOCATED":"LOCATED_IN","LOCATED_IN":"LOCATED_IN",
    "REQUIRE":"REQUIRES","REQUIRES":"REQUIRES","MUST":"REQUIRES","SHALL":"REQUIRES",
    "PROHIBIT":"PROHIBITS","PROHIBITS":"PROHIBITS","FORBID":"PROHIBITS",
    "RELATED":"RELATED"
}
REL_VALUE_SYNONYMS = {
    "DEFINED_AS": {"DEFINED_AS","定义","是","定义为"},
    "CODE_MEANING": {"CODE_MEANING","含义","意思","表示"},
    "COMPLIES_WITH": {"COMPLIES_WITH","符合","依据","遵循","标准"},
    "USES": {"USES","使用","采用"},
    "INCLUDES": {"INCLUDES","包含","包括","组成","含有"},
    "PART_OF": {"PART_OF","属于","部分","是…的一部分"},
    "ALIAS": {"ALIAS","别名","又称","AKA"},
    "LOCATED_IN": {"LOCATED_IN","位于","在"},
    "REQUIRES": {"REQUIRES","要求","需要","必须"},
    "PROHIBITS": {"PROHIBITS","禁止","不得"},
    "RELATED": {"RELATED","相关"}
}

# ==== 新增：通用属性类别 ATTRIBUTE 支持 ====
ALLOW.update({"ATTRIBUTE"})
ORDER.update({"ATTRIBUTE": 10})
DEFAULT_CONF.update({"ATTRIBUTE": 0.75})

REL_VALUE_SYNONYMS["ATTRIBUTE"] = {
    "ATTRIBUTE","属性","参数","接口速率","速率","比特率","频率","电压","电流",
    "长度","范围","容量","带宽","版本","格式","温度","湿度","海拔","功耗",
    "KBPS","KBIT/S","KB/S","MB/S","BPS"
}

# 参数/属性关键词正则
_ATTR_PAT = re.compile(
    r"(属性|参数|接口速率|速率|比特率|频率|电压|电流|长度|范围|容量|带宽|版本|格式|温度|湿度|海拔|功耗|kbps|kbit/s|kb/s|mb/s|bps)",
    re.I
)

def classify_edge(rel_obj: dict) -> str:
    """
    关系语义分类：
    1) 原始 type 在 ALLOW 直接返回
    2) 否则 caption/p_raw 命中参数词 → ATTRIBUTE
    3) 其它 → RELATED
    """
    t = (rel_obj.get("type") or "").upper()
    if t in ALLOW:
        return t
    text = f"{rel_obj.get('caption') or ''} {rel_obj.get('p_raw') or ''}"
    if _ATTR_PAT.search(text):
        return "ATTRIBUTE"
    return "RELATED"

def normalize_relations(rels):
    out=[]
    for r in (rels or []):
        key = str(r).strip().upper()
        out.append(SYN_REL.get(key, key))
    return [r for r in out if r in ALLOW]

def extract_entities(q: str, intent: dict) -> list:
    # 先用上游给的实体
    ents = [e.strip() for e in (intent.get("entities") or []) if str(e).strip()]
    anchor_re = re.compile(r"[A-Za-z0-9]+(?:[.\-][A-Za-z0-9]+)*")
    if ents:
        # 从给定实体中追加“代号锚点”（如 X.25、RS-232、IEEE802.3）
        extra = set()
        for e in ents:
            for a in anchor_re.findall(e):
                if 2 <= len(a) <= 40:
                    extra.add(a)
        ents.extend(sorted(extra))
        # 去重保序并限量
        seen, out = set(), []
        for x in ents:
            if x not in seen:
                seen.add(x); out.append(x)
            if len(out) >= 8:
                break
        return out

    # 上游没给实体时，走原来的自动抽取
    code_tokens = re.findall(r"[A-Za-z][A-Za-z0-9\-\./]{1,}", q)
    zh_chunks = re.findall(r"[\u4e00-\u9fa5][\u4e00-\u9fa5A-Za-z0-9·\-\./（）()]{1,20}", q)
    cand = code_tokens + [c for c in zh_chunks if len(c) >= 2]
    seen=set(); out=[]
    for x in cand:
        x = x.strip(" ，。、《》“”\"'()（）[]")
        if x and x not in seen:
            seen.add(x); out.append(x)
        if len(out) >= 8: break
    return out or [q.strip()]

def map_relations(q: str, intent: dict) -> list:
    norm = normalize_relations(intent.get("candidate_relations") or [])
    text = q + " " + " ".join(intent.get("keywords") or [])
    needs_attr = bool(_ATTR_PAT.search(text))

    # 规则推断（原逻辑）
    rs=[]
    if re.search(r"(协议|标准|符合|依据|遵循|基于|规约)", q): rs+=["COMPLIES_WITH","USES"]
    if re.search(r"(定义|是|称为|叫做)", q): rs+=["DEFINED_AS"]
    if re.search(r"(表示|含义|意思|意为|meaning|代表)", q): rs+=["CODE_MEANING"]
    if re.search(r"(包含|包括|组成)", q): rs+=["INCLUDES","PART_OF"]
    if re.search(r"(使用|采用)", q): rs+=["USES"]
    if re.search(r"(位于|在)", q): rs+=["LOCATED_IN"]
    if re.search(r"(应当|应|必须|需要|要求|条件)", q): rs+=["REQUIRES"]
    if re.search(r"(不得|禁止|不应|严禁)", q): rs+=["PROHIBITS"]
    if needs_attr: rs+=["ATTRIBUTE"]
    heur = set(normalize_relations(rs))

    if norm:
        out = set(norm) | heur          # ← 与规则并集，避免被锁死
        if needs_attr: out.add("ATTRIBUTE")
        return [r for r in out if r in ALLOW]

    return list(heur or {"DEFINED_AS","CODE_MEANING","USES","COMPLIES_WITH","INCLUDES","PART_OF","ATTRIBUTE","RELATED"})


def expand_alias(seed, depth: int = 2):
    cluster={seed["id"]:seed}; frontier=[seed]
    for _ in range(max(0, depth)):
        nxt=[]
        for n in frontier:
            for e in (kg_neighbors(n) or []):
                if (e["rel"].get("type") or "").upper() in REL_VALUE_SYNONYMS["ALIAS"]:
                    m = e["to"] if e["from"]["id"]==n["id"] else e["from"]
                    if m["id"] not in cluster:
                        cluster[m["id"]]=m; nxt.append(m)
        frontier=nxt
    return list(cluster.values())

def parse_intent_value(x):
    if isinstance(x, dict): return x
    s = (x or "").strip()
    if not s: return {}
    if s.startswith("```"):
        s = s.strip("`"); s = re.sub(r"^json\s*", "", s, flags=re.I)
    try: return json.loads(s)
    except: return {}

def norm_conf(rel, value):
    try:
        return float(value)
    except Exception:
        return float(DEFAULT_CONF.get(rel, 0.8))

# 入口（仅两个输入）
# 入口（仅两个输入）
def handler(question, intent):
    q = str(question or "")
    intent_obj = parse_intent_value(intent)
    keys = intent_obj.get("keywords") or ["协议", "标准", "含义", "速率", "字段", "定义"]

    entities = extract_entities(q, intent_obj)
    rels = map_relations(q, intent_obj)

    seeds = []
    for e in entities:
        hits = kg_search_nodes(e, limit=30) or []
        if hits:
            seeds.extend(hits)
    seeds = list({n["id"]: n for n in seeds}.values())

    answers = []
    if seeds:
        for s in seeds:
            for n in expand_alias(s, 2):
                for r in rels:
                    # 先尝试精确匹配
                    tails = kg_find_tails(n, r)
                    for t in tails:
                        # 从 r 可以推断原始关系（因为 r 是标准化后的类别）
                        # 但更好的方式是从知识图谱中记录原始关系名
                        # 这里我们用 r 作为 fallback
                        answers.append({
                            "entity": n["name"],
                            "relation_raw": r,  # 或根据需要映射回中文名
                            "relation_category": r,
                            "value": t["name"],
                            "confidence": norm_conf(r, t.get("confidence")),
                            "source": t.get("source")
                        })

                    # 如果没找到，再尝试模糊匹配
                    if not tails:
                        for e in (kg_neighbors(n) or []):
                            rel_obj = e["rel"]
                            rt = classify_edge(rel_obj)
                            cap_all = f"{rel_obj.get('caption') or ''} {rel_obj.get('p_raw') or ''}".upper()
                            syns = {s.upper() for s in REL_VALUE_SYNONYMS.get(r, set())}
                            if rt == r or any(s in cap_all for s in syns):
                                t = e["to"] if e["from"]["id"] == n["id"] else e["from"]
                                # 获取原始关系字符串
                                raw_rel = (
                                    rel_obj.get("type") or
                                    rel_obj.get("name") or
                                    rel_obj.get("rel") or
                                    rel_obj.get("caption") or
                                    rel_obj.get("p_raw") or
                                    r  # fallback
                                )
                                answers.append({
                                    "entity": n["name"],
                                    "relation_raw": raw_rel,
                                    "relation_category": r,
                                    "value": t["name"],
                                    "confidence": norm_conf(r, rel_obj.get("confidence")),
                                    "source": rel_obj.get("source")
                                })

        if not answers:
            keys_tuple = tuple(keys)
            for s in seeds:
                for n in expand_alias(s, 2):
                    for e in (kg_neighbors(n) or []):
                        rel_obj = e["rel"]
                        rtype = classify_edge(rel_obj)
                        m = e["to"] if e["from"]["id"] == n["id"] else e["from"]
                        hit = sum(1 for k in keys_tuple if k in (m["name"] or ""))
                        if hit > 0:
                            conf = 0.40 + 0.05 * min(hit, 6)
                            raw_rel = (
                                rel_obj.get("type") or
                                rel_obj.get("name") or
                                rel_obj.get("rel") or
                                rel_obj.get("caption") or
                                rel_obj.get("p_raw") or
                                "关系"
                            )
                            answers.append({
                                "entity": n["name"],
                                "relation_raw": raw_rel,
                                "relation_category": rtype,
                                "value": m["name"],
                                "confidence": norm_conf(rtype, conf),
                                "source": e["rel"].get("source")
                            })

    # 去重（按 category + raw_rel + value）
    seen = set()
    uniq = []
    for a in answers:
        key = (a["relation_category"], a["relation_raw"], a["value"])
        if key not in seen:
            seen.add(key)
            uniq.append(a)

    # 按 ORDER 排序
    def sort_key(x):
        rel = x.get("relation_category")
        conf = x.get("confidence")
        conf_val = norm_conf(rel, conf)
        return (ORDER.get(rel, 50), -conf_val, str(x.get("value")))

    uniq.sort(key=sort_key)

    # === 聚合 + verbalization ===
    from collections import defaultdict
    groups = defaultdict(list)
    for a in uniq[:120]:
        # 聚合键：(实体, 语义类别, 原始关系名) -> 保证“FR接入的接口速率”和“FR接入的符合标准”分开
        key = (a["entity"], a["relation_category"], a["relation_raw"])
        groups[key].append(a["value"])

    def verbalize_by_category(entity, raw_rel, values, category):
        val_str = "、".join(sorted(set(values)))  # 去重 + 拼接
        if category == "ATTRIBUTE":
            return f"{entity}的{raw_rel}为{val_str}。"
        elif category == "COMPLIES_WITH":
            return f"{entity}符合{val_str}。"
        elif category == "USES":
            return f"{entity}使用{val_str}。"
        elif category == "INCLUDES":
            return f"{entity}包含{val_str}。"
        elif category in ("DEFINED_AS", "CODE_MEANING"):
            return f"{entity}定义为{val_str}。"
        elif category == "ALIAS":
            return f"{entity}又称{val_str}。"
        elif category == "LOCATED_IN":
            return f"{entity}位于{val_str}。"
        elif category == "PART_OF":
            return f"{val_str}是{entity}的一部分。"
        elif category == "REQUIRES":
            return f"{entity}要求{val_str}。"
        elif category == "PROHIBITS":
            return f"{entity}禁止{val_str}。"
        else:
            return f"{entity}的{raw_rel}是{val_str}。"

    verbalized_texts = []
    for (entity, category, raw_rel), values in groups.items():
        sentence = verbalize_by_category(entity, raw_rel, values, category)
        verbalized_texts.append(sentence)

    # 返回自然语言句子列表（供统一 rerank 使用）
    return verbalized_texts






    
    # # 新增：渲染成可直接放入提示词的文本
    # text = "\n".join(
    #     f"- {x['entity']} —{x['relation']}→ {x['value']}"
    #     for x in uniq[:120]
    # )

    # return {
    #     "kg_answers": {"best": (uniq[0] if uniq else None), "candidates": uniq},
    #     "text": text
    # }
    
    # return {"kg_answers": {"best": (uniq[0] if uniq else None), "candidates": uniq}}