from typing import Iterable, Tuple
from neo4j import GraphDatabase

class Neo4jClient:
    def __init__(self, uri="bolt://neo4j:7687", user="neo4j", password="neo4j_pass123"):
        # self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self._driver = GraphDatabase.driver(uri, auth=(user, password))
        self.driver = self._driver   # 兼容旧代码
        # self._db = database

    def close(self):
        self.driver.close()

    def create_constraints(self):
        with self._driver.session() as s:
            s.run("""
            CREATE CONSTRAINT entity_name IF NOT EXISTS
            FOR (n:Entity) REQUIRE n.name IS UNIQUE
            """)

    def upsert_triples(self, triples: list[dict]) -> int:
        if not triples:
            return 0
        cypher = """
        UNWIND $rows AS t
        WITH t WHERE t.s IS NOT NULL AND t.o IS NOT NULL
        MERGE (s:Entity {name: t.s})
        MERGE (o:Entity {name: t.o})
        MERGE (s)-[r:RELATION]->(o)
        SET r.type       = coalesce(t.p, 'RELATED'),   // 归一后的类型
            r.raw_type   = t.p_raw,                    // 原始短语
            r.source     = t.source,
            r.snippet    = t.snippet,
            r.confidence = coalesce(t.confidence, 0.8),
            r.updatedAt  = timestamp()
        """
        with self._driver.session() as s:
            s.run(cypher, rows=triples).consume()
        return len(triples)