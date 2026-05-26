"""Memory layer — четыре бэкенда под единым recall API.

- Episodic:  LanceDB
- Semantic:  Qdrant
- World:     Graphiti на Neo4j
- Procedural: интегрирован со skill-registry

См. `agent_architecture.html` § 13. v0 — задача #8.
"""
