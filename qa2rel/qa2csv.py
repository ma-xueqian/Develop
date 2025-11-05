# 运行方式：
# 单个文件：
# python3 /home/mxq/Develop/qa2rel/qa2csv.py --input "电报基础（33道）.docx" --out "/home/mxq/Develop/qa2csv/dev.csv"
# 整个目录(脚本会合并为一个CSV)：
# python3 /home/mxq/Develop/qa2rel/qa2csv.py --input "/path/题库目录" --out "/home/mxq/Develop/qa2csv/dev.csv"

# 将 .docx格式的问答集 转换成 只包含问题的.csv格式文本



import re, csv, argparse, sys
from pathlib import Path
from docx import Document

RX_OPTION  = re.compile(r'^\s*([A-HＡ-Ｈa-h])([\.．、\)])?\s+')
RX_ANSWER  = re.compile(r'^\s*(答案|正确答案|参考答案|答[:：])')
RX_EMPTY   = re.compile(r'^\s*$')

def clean(s):
    s=(s or '').replace('\t',' ').replace('\u3000',' ')
    s=s.replace('（','(').replace('）',')')
    return re.sub(r'\s+',' ',s).strip()

def parse_docx(p: Path):
    doc = Document(str(p))
    lines=[]
    for pg in doc.paragraphs:
        t=clean(pg.text)
        if t: lines.append(t)
    for tbl in getattr(doc, "tables", []):
        for row in tbl.rows:
            for cell in row.cells:
                t=clean(cell.text)
                if t: lines.append(t)

    questions=[]; cur=[]; in_opts=False
    for t in lines:
        if RX_EMPTY.match(t):  # 忽略空行
            continue

        if RX_ANSWER.match(t):  # 到“答案：”行 -> 收尾并开始下一题
            if cur:
                questions.append(clean(' '.join(cur)))
                cur=[]; in_opts=False
            continue

        if RX_OPTION.match(t):  # 选项行 -> 跳过，但标记处于选项区
            in_opts=True
            continue

        # 普通文本行
        if in_opts and cur:
            # 选项结束后出现的普通行，视为新题开始
            questions.append(clean(' '.join(cur)))
            cur=[t]; in_opts=False
            continue

        # 题干续行或新题的第一行
        cur.append(t)

    if cur:
        questions.append(clean(' '.join(cur)))

    # 去重与清洗
    out, seen = [], set()
    for q in questions:
        q = re.split(r'\s*(答案[:：]|A[.\、)].*$)' , q)[0].strip()
        if q and q not in seen:
            seen.add(q); out.append(q)
    return out

def collect(inp: Path):
    if inp.is_file() and inp.suffix.lower()==".docx": return [inp]
    if inp.is_dir(): return sorted(inp.rglob("*.docx"))
    return []

def main():
    ap=argparse.ArgumentParser("Extract questions from .docx to CSV")
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True)
    a=ap.parse_args()

    files=collect(Path(a.input))
    if not files: sys.exit("未找到 .docx")

    all_q=[]
    for f in files:
        qs=parse_docx(f)
        print(f"[OK] {f.name}: {len(qs)}")
        all_q+=qs

    with open(a.out,"w",encoding="utf-8-sig",newline="") as fw:
        w=csv.writer(fw); w.writerow(["question"])
        for q in all_q: w.writerow([q])
    print(f"Saved: {a.out} ({len(all_q)} rows)")

if __name__=="__main__":
    main()