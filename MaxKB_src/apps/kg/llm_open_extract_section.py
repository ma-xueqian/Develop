# docker exec -it maxkb-dev bash -lc '
# export OPENAI_BASE_URL=https://open.bigmodel.cn/api/paas/v4
# export OPENAI_API_KEY=81d9a02a6a2c00c1e543e4eec620446f.1GB9GwtCMpeJMZuZ
# export OPENAI_MODEL=glm-4-flash
# python -m apps.kg.llm_open_extract_section \
#   --glob "/opt/maxkb/data/kg/md/*.md" \
#   --uri "bolt://neo4j:7687" \
#   --user neo4j --password "neo4j_pass123"
# '


import os, re, glob, json, time
from typing import List, Dict
from neo4j import GraphDatabase
from openai import OpenAI
from tqdm import tqdm
from collections import defaultdict

# 标志行正则：支持行首 '——'/'—'/'-'/'•'/'·'/'一' 前缀；中文/英文括号；全角/半角冒号
FLAG_LINE = re.compile(
    r'^\s*(?:[—\-•·一]*\s*)?(?P<flag>[A-Z][A-Z0-9]{1,15})'
    r'(?:[（(](?P<alias>[^）)]{0,50})[）)])?\s*[:：]\s*(?P<body>.+?)\s*[；;。]?\s*$'
)

def should_create_entity_node(sec_id: str, title: str, child_count: int, content: str = "") -> bool:
    """
    综合判断是否应该为章节标题创建 Entity 节点
    """
    # 1. 基础过滤（保持不变）
    if not title or len(title.strip()) < 2:
        return False
    if re.search(r'(表|示例|注|图|附录|参考|bibliography|格式|组成|结构|定义|概述|范围)', title, re.IGNORECASE):
        return False  # 👈 新增：明确排除"格式"、"组成"等文档结构词
    depth = len(sec_id.split('.'))
    if depth >= 4:
        return False
        
    # 2. 技术关键词匹配（优化关键词集）
    TECH_CONCEPT_WORDS = {'接口', '协议', '同步', '异步', '物理', '传输', '编码', 
                         '校验', '速率', '帧', '电路', '网络', '地址', '标志', 
                         '类型', '标准', '要求', '规范', '参数', '属性', '电报'}
    
    STRUCTURE_WORDS = {'格式', '组成', '结构', '定义', '概述', '范围', '内容', '说明'}
    
    title_words = set(re.findall(r'[\u4e00-\u9fa5]{2,}', title))
    
    # 必须包含技术概念词，且不包含文档结构词
    has_tech = bool(title_words & TECH_CONCEPT_WORDS)
    has_struct = bool(title_words & STRUCTURE_WORDS)
    
    if has_tech and not has_struct:
        return True
        
    # 3. 结构特征判断（保持不变）
    if child_count >= 2:
        return True
        
    # 4. 内容特征判断（保持不变）
    if content and len(content) < 800 and re.search(r'包括|包含|如下', content):
        return True
        
    return False

def parse_sections(text: str):
    """
    解析 Markdown 文档中的章节结构（支持 # 5.3.1 FR接入 格式）
    """
    lines = text.splitlines()
    sections = []
    id_to_title = {}

    # ✅ 支持 # 开头 + 数字编号 + 空格 + 标题
    section_re = re.compile(r'^\s*#\s*(\d+(?:\.\d+){0,5})\s+(.+?)\s*$')

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # 跳过干扰项
        if re.match(r'^(表|示例|注|图|附录)\d+', line):
            continue

        m = section_re.match(line)
        if m:
            sec_id = m.group(1)
            title = m.group(2).strip()
            sections.append({"id": sec_id, "title": title})
            id_to_title[sec_id] = title

    # 构建父子关系
    section_rels = []
    for sec in sections:
        sec_id = sec["id"]
        if '.' in sec_id:
            parent_id = '.'.join(sec_id.split('.')[:-1])
            if parent_id in id_to_title:
                section_rels.append({"parent": parent_id, "child": sec_id})

    return sections, section_rels

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

⚠️ 关键指令：主语必须是实体名称（如“RS232接口”），谓词必须是技术参数（如“通信协议”、“接口速率”），尽量不要使用“应满足”等动词性谓词。

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
                if re.match(r'^\d+(?:\.\d+){1,5}$', s.strip()):
                    continue
                out.append({"s":s, "p_raw":p, "o":o, "snippet":ev})
            return out
        except Exception:
            time.sleep(1.2)
    return []

def chunk_with_section(text: str, size=1400, overlap=180):
    lines = text.splitlines()
    chunks = []
    current_section = ""
    buffer = ""
    section_re = re.compile(r'^\s*#\s*(\d+(?:\.\d+){0,5})\s+(.+?)\s*$')

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue

        m = section_re.match(stripped)
        if m:
            # 保存上一个 chunk（如果有）
            if buffer.strip():
                chunks.append((buffer.strip(), current_section))
            # 开始新 chunk：将章节行加入 buffer
            current_section = m.group(1)
            buffer = line  # 👈 关键：保留章节行
            continue

        if len(buffer) + len(line) <= size:
            buffer += "\n" + line
        else:
            if buffer.strip():
                chunks.append((buffer.strip(), current_section))
            buffer = line[-overlap:] + "\n" + line

    if buffer.strip():
        chunks.append((buffer.strip(), current_section))

    return chunks

def push_enhanced(llm_triples, section_nodes, section_rels, uri, user, pwd, src):
    if not llm_triples and not section_nodes:
        return

    drv = GraphDatabase.driver(uri, auth=(user, pwd))
    with drv.session() as session:
        # 创建唯一约束
        session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (e:Entity) REQUIRE e.name IS UNIQUE")
        session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (s:Section) REQUIRE s.id IS UNIQUE")

        # 1. 入库 LLM 三元组（保持你原有的逻辑）
        if llm_triples:
            for r in llm_triples:
                r["src"] = src
            session.run("""
            UNWIND $rows AS r
            MERGE (s:Entity {name: r.s})
            MERGE (o:Entity {name: r.o})
            MERGE (s)-[e:REL {p_raw: r.p_raw}]->(o)
            SET e.type = 'FACT',
                e.source = r.src,
                e.snippet = r.snippet
            """, rows=llm_triples)

        # 2. 入库 Section 节点
        if section_nodes:
            session.run("""
            UNWIND $rows AS r
            MERGE (s:Section {id: r.id})
            SET s.title = r.title,
                s.cleanTitle = r.clean_title,
                s.source = $src
            """, rows=section_nodes, src=src)

        # 3. 入库 Section 层级关系
        if section_rels:
            session.run("""
            UNWIND $rows AS r
            MATCH (p:Section {id: r.parent})
            MATCH (c:Section {id: r.child})
            MERGE (p)-[:HAS_SUBSECTION]->(c)
            """, rows=section_rels)
        
        # ✅ 4. 新增：建立 Entity -> Section 的 DEFINED_IN 关系
        defined_in_triples = [
            {"entity": r["s"], "section_id": r.get("section_id", "")}
            for r in llm_triples
            if r.get("section_id")
        ]
        if defined_in_triples:
            session.run("""
            UNWIND $rows AS r
            MATCH (e:Entity {name: r.entity})
            MATCH (s:Section {id: r.section_id})
            MERGE (e)-[:DEFINED_IN]->(s)
            """, rows=defined_in_triples)

        # ✅✅✅ 5. 新增：自动化创建有意义的章节Entity节点 + 反向INCLUDES关系
        if section_nodes and llm_triples:
            # 构建章节ID到子章节数量的映射
            child_count = defaultdict(int)
            for rel in section_rels:
                child_count[rel["parent"]] += 1
            
            # 收集所有section_id到标题的映射
            id_to_title = {s["id"]: s["title"] for s in section_nodes}
            
            # 判断哪些章节需要创建Entity节点
            meaningful_entities = []
            for sec in section_nodes:
                sec_id = sec["id"]
                title = sec["title"]
                count = child_count[sec_id]
                
                if should_create_entity_node(sec_id, title, count):
                    meaningful_entities.append({"name": title})
            
            # 入库有意义的Entity节点
            if meaningful_entities:
                session.run("""
                UNWIND $rows AS r
                MERGE (e:Entity {name: r.name})
                SET e.type = 'CATEGORY', e.source = $src
                """, rows=meaningful_entities, src=src)
            
            # 建立反向INCLUDES关系（章节Entity → 技术实体）
            if meaningful_entities:
                # 构建标题到是否存在Entity的映射
                title_to_entity = {e["name"] for e in meaningful_entities}
                
                includes_rels = []
                for r in llm_triples:
                    if r.get("section_id"):
                        sec_id = r["section_id"]
                        # 向上遍历所有祖先章节
                        current_id = sec_id
                        while current_id in id_to_title:
                            title = id_to_title[current_id]
                            if title in title_to_entity:
                                includes_rels.append({
                                    "category": title,
                                    "entity": r["s"],
                                    "source": r["src"]
                                })
                            if '.' in current_id:
                                current_id = '.'.join(current_id.split('.')[:-1])
                            else:
                                break
                
                # 去重（避免同一实体被多次关联）
                unique_includes = []
                seen = set()
                for rel in includes_rels:
                    key = (rel["category"], rel["entity"])
                    if key not in seen:
                        seen.add(key)
                        unique_includes.append(rel)
                
                if unique_includes:
                    session.run("""
                    UNWIND $rows AS r
                    MATCH (cat:Entity {name: r.category})
                    MATCH (e:Entity {name: r.entity})
                    MERGE (cat)-[rel:INCLUDES]->(e)
                    SET rel.source = r.source,
                        rel.type = 'CATEGORY_REL'
                    """, rows=unique_includes)

    drv.close()

def run(pattern: str, uri: str, user: str, pwd: str, base: str, key: str, model: str):
    client = OpenAI(base_url=base, api_key=key)
    files = sorted(glob.glob(pattern))
    
    # 定义章节正则（与 parse_sections 一致）
    section_re = re.compile(r'^\s*#\s*(\d+(?:\.\d+){0,5})\s+(.+?)\s*$')

    for fp in files:
        with open(fp, "r", encoding="utf-8", errors="ignore") as f:
            txt = f.read()

        # === 1. 规则抽取（标志行）===
        rule_rows = extract_flag_lines(txt)
        all_rows, seen = [], set()
        for t in rule_rows:
            sig = (t["s"], t["p_raw"], t["o"])
            if sig in seen: 
                continue
            seen.add(sig)
            # 为规则行也尝试分配章节（可选）
            t["section_id"] = ""  # 暂不处理，简单起见
            all_rows.append(t)

        # === 2. LLM 抽取（按章节精确分块）===
        lines = txt.splitlines()
        chunks_with_sec = []
        current_section = ""
        current_content_lines = []

        section_re = re.compile(r'^\s*#\s*(\d+(?:\.\d+){0,5})\s+(.+?)\s*$')

        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue

            # 检查是否为章节行
            m = section_re.match(stripped)
            if m:
                # 保存上一个章节（如果存在）
                if current_section and current_content_lines:
                    chunk_text = "\n".join(current_content_lines)
                    chunks_with_sec.append((chunk_text, current_section))
                    current_content_lines = []

                # 开始新章节
                current_section = m.group(1)
                current_content_lines.append(line)  # 保留章节行本身
                continue

            # 累积非章节行到当前章节
            if current_section:
                current_content_lines.append(line)

        # 保存最后一个章节
        if current_section and current_content_lines:
            chunk_text = "\n".join(current_content_lines)
            chunks_with_sec.append((chunk_text, current_section))

        # 处理每个块
        t_file0 = time.time()
        with tqdm(total=len(chunks_with_sec), desc=f"Extract {os.path.basename(fp)}", unit="chunk") as bar:
            avg, n = 0.0, 0
            for ck, sec_id in chunks_with_sec:
                t0 = time.time()
                triples = call_llm(ck, client, model)
                dt = time.time() - t0
                n += 1
                avg = (avg * (n - 1) + dt) / n
                bar.set_postfix(last_s=f"{dt:.2f}", avg_s=f"{avg:.2f}", triples=len(triples))
                bar.update(1)

                for t in triples:
                    sig = (t["s"], t["p_raw"], t["o"])
                    if sig in seen:
                        continue
                    seen.add(sig)
                    t["section_id"] = sec_id  # 👈 关键：附加章节ID
                    all_rows.append(t)

        # === 3. 解析章节结构 ===
        section_nodes, section_rels = parse_sections(txt)

        # === 4. 入库：实体 + 章节 + 关联 ===
        push_enhanced(
            llm_triples=all_rows,
            section_nodes=section_nodes,
            section_rels=section_rels,
            uri=uri, user=user, pwd=pwd, src=fp
        )

        print(f"{os.path.basename(fp)} -> "
              f"{len(all_rows)} facts | "
              f"{len(section_nodes)} sections | "
              f"{time.time() - t_file0:.1f}s")


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