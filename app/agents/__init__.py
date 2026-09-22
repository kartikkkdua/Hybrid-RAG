"""LangGraph multi-agent layer: router → lookup/research → synthesizer → critic.

Imported lazily (langgraph is an optional extra) so the core app runs without it.
"""

__all__ = ["AgentRunner", "build_graph", "get_runner"]


def __getattr__(name):
    if name in __all__:
        from . import graph as _graph

        return getattr(_graph, name)
    raise AttributeError(name)
