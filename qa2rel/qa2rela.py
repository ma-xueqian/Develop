# 运行：
# python3 /home/mxq/Develop/qa2rel/qa2rela.py --csv /home/mxq/Develop/qa2rel/dev.csv --out_csv /home/mxq/Develop/qa2rel/dev_labeled.csv --out_yaml /home/mxq/Develop/MaxKB_src/apps/kg/relation_patterns.yml

# 从问答集的.csv文件中提取realtionship以辅助后续的三元组提取工作
import re, csv, argparse, yaml
from collections import Counter, defaultdict
KW = {
  "DEFINED_AS": ["是什么","是什么意思","是指","指的是","称为","叫做","含义"],
  "CODE_MEANING": ["的含义","含义是","表示","代表","缩写为","全称为","意思是"],
  "ALIAS": ["也称","又称","别名为"],
  "INCLUDES": ["由","组成","包括","包含"],
  "USES": ["用来","用于","使用","采用","基于"],
  "HAS_LEVEL": ["等级"],
  "RETENTION_LONG": ["长期保存期","长期保留期","长期保存"],
  "RETENTION_SHORT": ["短期保存期","短期保留期","短期保存"],
  "AVG_LENGTH": ["平均长度"],
  "MAX_LENGTH": ["最大长度"],
  "SPLIT_THRESHOLD": ["超过","会被拆分","拆分为"],
  "BELONGS_TO": ["属于","隶属","归属"],
}
REQ = {
  "CODE_MEANING": [r"[A-Za-z]"],                 # 必须含英文/代码
  "HAS_LEVEL": ["等级"],
  "RETENTION_LONG": ["天"],
  "RETENTION_SHORT": ["天","小时"],
  "AVG_LENGTH": ["字符"],
  "MAX_LENGTH": ["字符"],
  "SPLIT_THRESHOLD": ["字符"],
}
PRIORITY = ["CODE_MEANING","DEFINED_AS","ALIAS","INCLUDES","USES",
            "HAS_LEVEL","RETENTION_LONG","RETENTION_SHORT",
            "AVG_LENGTH","MAX_LENGTH","SPLIT_THRESHOLD","BELONGS_TO"]

def hit(q:str, keys): return sum(1 for k in keys if k in q)
def req_ok(q:str, reqs): 
    for r in reqs or []:
        if r.startswith("re:"): 
            if not re.search(r[3:], q): return False
        elif r.startswith("[") or "\\" in r:
            if not re.search(r, q): return False
        elif r not in q: 
            return False
    return True

def detect_rel(q:str):
    q = q.strip()
    best,score,matched=None,0,None
    for rel in PRIORITY:
        c = hit(q, KW.get(rel, []))
        if c==0: continue
        if not req_ok(q, REQ.get(rel)): continue
        if c>score or (c==score and PRIORITY.index(rel) < PRIORITY.index(best) if best else True):
            best,score,matched = rel,c,[k for k in KW[rel] if k in q]
    return best, matched or []

def main():
    ap=argparse.ArgumentParser("Label relations from QA questions (no answers)")
    ap.add_argument("--csv", required=True, help="输入：仅含 question 列的 CSV")
    ap.add_argument("--out_csv", default="dev_labeled.csv")
    ap.add_argument("--out_yaml", default="relation_patterns.yml")
    args=ap.parse_args()

    rows=[]
    with open(args.csv,"r",encoding="utf-8-sig") as f:
        rdr=csv.DictReader(f)
        for r in rdr:
            q=r.get("question","").strip()
            if not q: continue
            rel, keys = detect_rel(q)
            rows.append({"question":q,"relation":rel or "OTHER","keys":"|".join(keys)})

    # 写标注结果
    with open(args.out_csv,"w",encoding="utf-8-sig",newline="") as f:
        w=csv.DictWriter(f, fieldnames=["question","relation","keys"])
        w.writeheader(); w.writerows(rows)

    # 汇总到 YAML（供 relation_from_question 使用）
    rel2cnt=defaultdict(Counter)
    for r in rows:
        if r["relation"]!="OTHER":
            for k in r["keys"].split("|"):
                if k: rel2cnt[r["relation"]][k]+=1
    y={"relationships":[]}
    for rel in PRIORITY:
        if rel not in rel2cnt: continue
        pats=[k for k,_ in rel2cnt[rel].most_common()]
        item={"name":rel,"patterns":pats}
        if rel in REQ and REQ[rel]:
            # 规范成 re: 前缀，便于后续解析
            item["must_have"]=[("re:"+r) if (r.startswith("[") or "\\" in r) else r for r in REQ[rel]]
        y["relationships"].append(item)
    with open(args.out_yaml,"w",encoding="utf-8") as f:
        yaml.safe_dump(y,f,allow_unicode=True, sort_keys=False)

    print(f"Labeled: {len(rows)} -> {args.out_csv}")
    print(f"Patterns: {args.out_yaml}")

if __name__=="__main__":
    main()