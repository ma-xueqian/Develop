# 运行
# docker exec -it maxkb-dev bash -lc '
# export OPENAI_API_KEY=81d9a02a6a2c00c1e543e4eec620446f.1GB9GwtCMpeJMZuZ
# export OPENAI_BASE_URL=https://open.bigmodel.cn/api/paas/v4
# export OPENAI_MODEL=glm-4-flash
# python3 -m apps.kg.ingest_md_llm_cli --md "/opt/maxkb/data/kg/md/*.md" --uri bolt://neo4j:7687 --user neo4j --password neo4j_pass123
# '

import glob, argparse
from apps.kg.neo4j_client import Neo4jClient
from apps.kg.llm_to_kg_glm import md_to_triples_llm_glm

def main():
    ap = argparse.ArgumentParser("Use glm-4-flash to extract Markdown -> Neo4j")
    ap.add_argument("--md", default="/opt/maxkb/data/kg/md/*.md", help="容器内通配路径, 支持 ** 递归")
    ap.add_argument("--uri", default="bolt://neo4j:7687")
    ap.add_argument("--user", default="neo4j")
    ap.add_argument("--password", default="neo4j_pass123")
    a = ap.parse_args()

    files = sorted(glob.glob(a.md, recursive=True))
    if not files:
        print(f"未找到 Markdown：{a.md}")
        return

    cli = Neo4jClient(a.uri, a.user, a.password)
    cli.create_constraints()

    total_files, total_edges = 0, 0
    for f in files:
        T = md_to_triples_llm_glm(f)
        merged = cli.upsert_triples(T)
        print(f"[OK] {f} -> triples={len(T)}, merged={merged}")
        total_files += 1; total_edges += merged

    cli.close()
    print(f"Done. files={total_files}, merged={total_edges}")

if __name__ == "__main__":
    main()