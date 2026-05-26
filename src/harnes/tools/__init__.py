"""Tool layer — граница между агентом и миром.

Глобальный реестр тулов. Tool-layer pipeline: schema-validate → resolve
irreversibility → invoke → classify → retry → output-validate → Observation.

См. `agent_architecture.html` § 10. v0 — задача #7.
"""
