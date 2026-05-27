"""harnes — research wrapper around local LLMs.

Implements a meta-cycle over a ReAct loop for a universal autonomous agent.
See `agent_architecture.html` for the full architecture.

Имя агента, запускаемого внутри этой обвязки: **Irida** (Ἶρις, греческая богиня
радуги — посланница, мост между мирами). Отражает роль агента как моста между
распределениями задач, слоями памяти и тиерами моделей.
"""
__version__ = "0.0.1"

AGENT_NAME = "Irida"
PROJECT_NAME = "harnes"

__all__ = ["AGENT_NAME", "PROJECT_NAME", "__version__"]
