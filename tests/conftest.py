"""Configuração compartilhada dos testes.

Os testes rodam offline: sem rede, sem API key, sem servidor Qdrant e sem
download de modelos. Dependências pesadas (sentence-transformers, langchain-groq,
langgraph) são substituídas por stubs em ``sys.modules`` antes dos imports.
"""

import sys
import types
from pathlib import Path

import pytest

# raiz do repositório no sys.path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _stub_module(name: str, **attrs) -> None:
    """Registra um módulo falso em ``sys.modules``.

    O stub é usado mesmo que o pacote real esteja instalado, para que os testes
    nunca importem torch/modelos nem dependam de rede.
    """
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    sys.modules[name] = module


class _Unavailable:
    """Placeholder para classes que nunca devem ser instanciadas nos testes."""

    def __init__(self, *args, **kwargs):
        raise RuntimeError(f"{type(self).__name__} não deve ser usado nos testes.")


class SentenceTransformer(_Unavailable):
    pass


class ChatGroq(_Unavailable):
    pass


def create_react_agent(*args, **kwargs):
    raise RuntimeError("create_react_agent não deve ser usado nos testes.")


_stub_module("sentence_transformers", SentenceTransformer=SentenceTransformer)
_stub_module("langchain_groq", ChatGroq=ChatGroq)
_stub_module("langgraph")
_stub_module("langgraph.prebuilt", create_react_agent=create_react_agent)


@pytest.fixture
def tools_module():
    """Módulo agent.tools com o estado compartilhado limpo a cada teste."""
    from agent import tools

    tools.set_document("")
    yield tools
    tools.set_document("")


@pytest.fixture
def fake_llm(monkeypatch, tools_module):
    """Substitui ``_llm_call`` e registra as chamadas feitas ao LLM."""

    class FakeLLM:
        def __init__(self):
            self.calls: list[dict] = []
            self.response = ""

        def __call__(self, system: str, user: str) -> str:
            self.calls.append({"system": system, "user": user})
            return self.response

    fake = FakeLLM()
    monkeypatch.setattr(tools_module, "_llm_call", fake)
    return fake
