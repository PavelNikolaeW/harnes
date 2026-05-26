"""Goal subsystem — Goal-объекты, хранение и арбитраж.

Хранение: SQLite. Иерархия: tree + depends_on (DAG-lite).
Классы: task | inquiry | maintenance | standing | practice.

См. `agent_architecture.html` § 4. v0 — задача #5.
"""
