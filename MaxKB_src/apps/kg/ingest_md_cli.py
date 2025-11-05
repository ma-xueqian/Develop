# 运行
# docker exec -it maxkb-dev bash -lc 'python3 -m apps.kg.ingest_md_cli --md "/opt/maxkb/data/kg/md/*.md" --uri bolt://neo4j:7687 --user neo4j --password neo4j_pass123'

import glob, argparse
from apps.kg.neo4j_client import Neo4jClient
from apps.kg.md_to_kg import md_to_triples

def main():
    ap = argparse.ArgumentParser(description="Batch import Markdown -> Neo4j")
    ap.add_argument("--md", default="/opt/maxkb/data/kg/md/*.md", help="容器内通配路径，支持 ** 递归")
    ap.add_argument("--uri", default="bolt://neo4j:7687")
    ap.add_argument("--user", default="neo4j")
    ap.add_argument("--password", default="neo4j_pass123")
    args = ap.parse_args()

    files = sorted(glob.glob(args.md, recursive=True))
    if not files:
        print(f"未找到 Markdown：{args.md}")
        return

    cli = Neo4jClient(uri=args.uri, user=args.user, password=args.password)
    cli.create_constraints()

    total_files, total_merged = 0, 0
    for f in files:
        triples = md_to_triples(f)
        merged = cli.upsert_triples(triples)
        print(f"[OK] {f} -> triples={len(triples)}, merged={merged}")
        total_files += 1; total_merged += merged

    cli.close()
    print(f"Done. files={total_files}, merged={total_merged}")

if __name__ == "__main__":
    main()