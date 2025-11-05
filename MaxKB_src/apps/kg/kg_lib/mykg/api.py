import os
from neo4j import GraphDatabase

# 用环境变量配置连接
URI = os.getenv("NEO4J_URI", "bolt://neo4j:7687")
USER = os.getenv("NEO4J_USER", "neo4j")
PWD  = os.getenv("NEO4J_PASSWORD", "neo4j_pass123")

_driver = GraphDatabase.driver(URI, auth=(USER, PWD))

def kg_search_nodes(kw: str, limit: int = 50):
    kw = (kw or "").strip()
    if not kw:
        return []
    with _driver.session() as s:
        rows = s.run("""
            MATCH (n:Entity)
            WHERE n.name CONTAINS $kw
            RETURN id(n) AS id, n.name AS name
            LIMIT $lim
        """, kw=kw, lim=limit).data()
    return [{"id": r["id"], "name": r["name"]} for r in rows]

def kg_neighbors(node: dict, depth: int = 1):
    nid = node["id"] if isinstance(node, dict) else int(node)
    with _driver.session() as s:
        rows = s.run("""
            MATCH (n) WHERE id(n)=$id
            MATCH (n)-[p:RELATION*1..$d]-(m)
            WITH n, p[0] AS r, m
            RETURN {id:id(n),name:n.name} AS src,
                   {type:r.type, source:r.source, confidence:r.confidence} AS rel,
                   {id:id(m),name:m.name} AS dst
            LIMIT 500
        """, id=nid, d=max(1, depth)).data()
    return [{"from": r["src"], "rel": r["rel"], "to": r["dst"]} for r in rows]

def kg_find_tails(node: dict, relation: str):
    nid = node["id"] if isinstance(node, dict) else int(node)
    with _driver.session() as s:
        rows = s.run("""
            MATCH (n) WHERE id(n)=$id
            MATCH (n)-[r:RELATION {type:$rel}]->(t)
            RETURN id(t) AS id, t.name AS name,
                   r.source AS source, r.confidence AS confidence
            LIMIT 200
        """, id=nid, rel=relation).data()
    return [{"id": r["id"], "name": r["name"],
             "source": r.get("source"), "confidence": r.get("confidence")} for r in rows]