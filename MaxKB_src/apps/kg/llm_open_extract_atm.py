# docker exec -it maxkb-dev bash -lc '
# export OPENAI_BASE_URL=https://open.bigmodel.cn/api/paas/v4
# export OPENAI_API_KEY=81d9a02a6a2c00c1e543e4eec620446f.1GB9GwtCMpeJMZuZ
# export OPENAI_MODEL=glm-4-flash
# python -m apps.kg.llm_open_extract_atm \
#   --glob "/opt/maxkb/data/kg/md/*.md" \
#   --uri "bolt://neo4j:7687" \
#   --user neo4j --password "neo4j_pass123"
# '


import os, re, glob, json, time
from typing import List, Dict
from neo4j import GraphDatabase
from openai import OpenAI
from tqdm import tqdm

# 标志行正则：支持行首 '——'/'—'/'-'/'•'/'·'/'一' 前缀；中文/英文括号；全角/半角冒号
FLAG_LINE = re.compile(
    r'^\s*(?:[—\-•·一]*\s*)?(?P<flag>[A-Z][A-Z0-9]{1,15})'
    r'(?:[（(](?P<alias>[^）)]{0,50})[）)])?\s*[:：]\s*(?P<body>.+?)\s*[；;。]?\s*$'
)

def extract_flag_lines(text: str) -> List[Dict]:
    rows=[]
    for raw in text.splitlines():
        line = raw.strip()
        m = FLAG_LINE.match(line)
        if not m: 
            continue
        flag  = (m.group("flag") or "").strip()
        alias = (m.group("alias") or "").strip(" 、，, ")
        body  = (m.group("body")  or "").strip()
        if not flag or not body:
            continue
        s = f"{flag} 标志"
        rows.append({"s": s, "p_raw": "用途", "o": body, "snippet": line})
        if alias:
            rows.append({"s": s, "p_raw": "别名", "o": alias, "snippet": line})
    return rows

SECTION_ONLY = re.compile(r'^\s*(\d+(?:\.\d+){1,6}|[一二三四五六七八九十]+|[A-Za-z])([.)、）])?\s*$')
def clean(s:str)->str:
    if not s: return ""
    s = re.sub(r'\$[^$]*\$', ' ', s)          # 去 LaTeX 公式
    s = re.sub(r'\s+', ' ', s).strip(' ：:;，,。')
    return s

def drop_noise(x:str)->bool:
    if not x: return True
    if SECTION_ONLY.match(x): return True      # 纯段落编号
    if len(x) <= 1: return True                # 单字符/噪声
    return False

FEWSHOT = r"""
从技术规范中文段落中抽取事实三元组，保持中文谓词，不做同义归一。忽略纯编号/小节标记。主语/宾语尽量是名词短语。
输出JSON: {"triples":[{"s":"主语","p":"谓词","o":"宾语","evidence":"原句"}...]}

示例1（RS-232接口）：
文本：
“RS232接入的要求如下：——通信协议：异步通信协议，符合ITU-T X.24；——接口速率：50 bit/s，75 bit/s，100 bit/s，300 bit/s，600 bit/s，1200 bit/s，2400 bit/s，4800 bit/s，9600 bit/s，19200 bit/s；——传输码：IA-5 码制时，包含7或8个数据位、1或2个停止位、无校验位；传输码为ITA-2 码制时，包含5个数据位、1.5个停止位、无校验位。”
期望triples：
- RS-232 接口 —通信协议→ 异步通信协议
- RS-232 接口 —符合→ ITU-T X.24
- RS-232 接口 —接口速率→ 50 bit/s，75 bit/s，100 bit/s，300 bit/s，600 bit/s，1200 bit/s，2400 bit/s，4800 bit/s，9600 bit/s，19200 bit/s
- IA-5 码制 —传输码→ 7或8个数据位、1或2个停止位、无校验位
- ITA-2 码制 —传输码→ 5个数据位、1.5个停止位、无校验位

示例2（电流环）：
文本：
“电流环接入……电压为直流±24V，电流为直流4mA±2mA；接口速率：50 bit/s，100 bit/s，300 bit/s，600 bit/s，1200 bit/s。”
期望：
- 电流环接入 —电压→ ±24V 直流
- 电流环接入 —电流→ 4mA±2mA 直流
- 电流环接入 —接口速率→ 50 bit/s，100 bit/s，300 bit/s，600 bit/s，1200 bit/s

示例3（地址长度与组成）：
文本：
“每个SITA收电地址由7个字符组成：前3位城市或机场代码，中2位部门代码，后2位网络用户代码（IATA定义）。”
期望：
- SITA 收电地址 —长度→ 7字符
- SITA 收电地址 —组成→ 城市或机场代码(3位)
- SITA 收电地址 —组成→ 部门代码(2位)
- SITA 收电地址 —组成→ 网络用户代码(2位)

示例4（AFTN地址）：
文本：
“收电地址标识为8位字符：第1-2位国家情报区；第3-4位通信中心；第5-7位A-Z；第8位组织细分或X填充。”
期望：
- AFTN 收电地址 —长度→ 8字符
- AFTN 收电地址 —组成→ 国家情报区(2位)
- AFTN 收电地址 —组成→ 通信中心(2位)
- AFTN 收电地址 —组成→ 组织细分(第8位)

示例5（以太网）：
文本：
“以太网接入……通信协议：符合IEEE802.3；网络协议：IPX/SPX、TCP/IP、UDP/IP；接口速率：10 Mbit/s，100 Mbit/s，1000 Mbit/s。”
期望：
- 以太网接口 —通信协议→ IEEE 802.3
- 以太网接口 —网络协议→ IPX/SPX、TCP/IP、UDP/IP
- 以太网接口 —接口速率→ 10 Mbit/s，100 Mbit/s，1000 Mbit/s

示例6（环境范围）：
文本：
“工作温度：0°C~40°C；相对湿度：20%~80%；设备供电：电压220V±20V，频率50Hz；极端海拔不超过5000m。”
期望：
- 工作温度 —范围→ 0°C~40°C
- 相对湿度 —范围→ 20%~80%
- 供电 —电压→ 220V±20V
- 供电 —频率→ 50Hz
- 环境 —海拔上限→ 5000m

示例7（标志行）：
文本：
“——COL（校对、核对）：在新的电报中对原来重要的电报进行校对时，在校对副本之前应使用COL标志；”
期望：
- COL 标志 —用途→ 在新的电报中对原来重要的电报进行校对时，在校对副本之前应使用COL标志
- COL 标志 —别名→ 校对、核对
"""

def call_llm(text: str, client: OpenAI, model: str) -> List[Dict]:
    prompt = FEWSHOT + "\n待抽取文本：\n" + text[:3500]
    msgs = [
        {"role":"system","content":"严格返回JSON，不要解释。"},
        {"role":"user","content": prompt}
    ]
    for _ in range(3):
        try:
            rsp = client.chat.completions.create(
                model=model, temperature=0.1, max_tokens=1400,
                response_format={"type":"json_object"}, messages=msgs
            )
            data = json.loads(rsp.choices[0].message.content or "{}")
            triples = data.get("triples", [])
            out=[]
            for t in triples:
                s = clean(t.get("s","")); p = clean(t.get("p","")); o = clean(t.get("o",""))
                ev = clean(t.get("evidence",""))
                if drop_noise(s) or not p or not o: continue
                out.append({"s":s, "p_raw":p, "o":o, "snippet":ev})
            return out
        except Exception:
            time.sleep(1.2)
    return []

def chunk(text: str, size=1400, overlap=180):
    sents = re.split(r'(?<=[。；;!！?？\n])', text)
    res, buf = [], ""
    for s in sents:
        if len(buf)+len(s) <= size: buf += s
        else:
            if buf.strip(): res.append(buf.strip())
            buf = (buf[-overlap:]+s) if overlap else s
    if buf.strip(): res.append(buf.strip())
    return res

def push(rows: List[Dict], uri: str, user: str, pwd: str, src: str):
    if not rows: return
    for r in rows: r["src"]=src
    drv = GraphDatabase.driver(uri, auth=(user, pwd))
    cypher = """
    UNWIND $rows AS r
    MERGE (s:Entity {name:r.s})
    MERGE (o:Entity {name:r.o})
    MERGE (s)-[e:REL {p_raw:r.p_raw}]->(o)
    SET e.type='OPEN', e.caption=r.p_raw, e.source=r.src, e.snippet=r.snippet
    """
    with drv.session() as s:
        s.run("CREATE CONSTRAINT IF NOT EXISTS FOR (e:Entity) REQUIRE e.name IS UNIQUE")
        s.run(cypher, rows=rows).consume()
    drv.close()

def run(pattern: str, uri: str, user: str, pwd: str, base: str, key: str, model: str):
    client = OpenAI(base_url=base, api_key=key)
    files = sorted(glob.glob(pattern))
    for fp in files:
        with open(fp, "r", encoding="utf-8", errors="ignore") as f:
            txt = f.read()

        # 先跑规则兜底（解决 COL/COR/PDM 这类行）
        rule_rows = extract_flag_lines(txt)
        all_rows, seen = [], set()
        for t in rule_rows:
            sig = (t["s"], t["p_raw"], t["o"])
            if sig in seen: 
                continue
            seen.add(sig); all_rows.append(t)

        # 再跑 LLM（补充其它关系）
        chunks = chunk(txt)
        t_file0 = time.time()
        with tqdm(total=len(chunks), desc=f"Extract {os.path.basename(fp)}", unit="chunk") as bar:
            avg, n = 0.0, 0
            for ck in chunks:
                t0 = time.time()
                triples = call_llm(ck, client, model)
                dt = time.time() - t0
                n += 1; avg = (avg*(n-1)+dt)/n
                bar.set_postfix(last_s=f"{dt:.2f}", avg_s=f"{avg:.2f}", triples=len(triples))
                bar.update(1)
                for t in triples:
                    sig = (t["s"], t["p_raw"], t["o"])
                    if sig in seen: 
                        continue
                    seen.add(sig); all_rows.append(t)

        push(all_rows, uri, user, pwd, fp)
        print(f"{os.path.basename(fp)} -> {len(all_rows)} triples | {len(chunks)} chunks | {time.time()-t_file0:.1f}s")


if __name__ == "__main__":
    import argparse
    ap=argparse.ArgumentParser()
    ap.add_argument("--glob", default="/home/mxq/Develop/kg_data/md/*.md")
    ap.add_argument("--uri", default="bolt://localhost:7687")
    ap.add_argument("--user", default="neo4j")
    ap.add_argument("--password", default="neo4j_pass123")
    ap.add_argument("--api_base", default=os.getenv("OPENAI_BASE_URL","http://localhost:11434/v1"))
    ap.add_argument("--api_key",  default=os.getenv("OPENAI_API_KEY","sk-xxx"))
    ap.add_argument("--model",    default=os.getenv("OPENAI_MODEL","qwen2.5-14b-instruct"))
    args=ap.parse_args()
    run(args.glob, args.uri, args.user, args.password, args.api_base, args.api_key, args.model)