"""Testes das ferramentas do agente e do estado compartilhado."""

import json
import sys
import types
from types import SimpleNamespace

import pytest


# ── estado compartilhado ───────────────────────────────────────────────

def test_set_document_resets_previous_results(tools_module):
    state = tools_module._state
    state["summary"] = "resumo antigo"
    state["entities"] = {"title": "antigo"}
    state["nlp_analysis"] = {"token_count": 10}
    state["literature_context"] = "contexto antigo"

    tools_module.set_document("novo texto")

    assert state == {
        "full_text": "novo texto",
        "summary": "",
        "entities": {},
        "nlp_analysis": {},
        "literature_context": "",
    }


def test_generate_summary_stores_result_and_truncates_text(tools_module, fake_llm):
    tools_module.set_document("A" * 10_000)
    fake_llm.response = "Resumo do paper."

    result = tools_module.generate_summary.invoke({})

    assert result == "Resumo do paper."
    assert tools_module._state["summary"] == "Resumo do paper."
    # apenas os primeiros 8000 caracteres são enviados ao LLM
    assert "A" * 8000 in fake_llm.calls[0]["user"]
    assert "A" * 8001 not in fake_llm.calls[0]["user"]


# ── parsing de JSON das entidades ──────────────────────────────────────

ENTITIES = {"title": "Attention Is All You Need", "authors": ["Vaswani"], "year": "2017"}


@pytest.mark.parametrize(
    "response",
    [
        json.dumps(ENTITIES),
        f"```json\n{json.dumps(ENTITIES)}\n```",
        f"Aqui está:\n```\n{json.dumps(ENTITIES)}\n```\nFim.",
    ],
    ids=["json-puro", "bloco-json", "bloco-generico"],
)
def test_extract_entities_parses_json_variants(tools_module, fake_llm, response):
    tools_module.set_document("texto do paper")
    fake_llm.response = response

    result = tools_module.extract_entities.invoke({})

    assert tools_module._state["entities"] == ENTITIES
    assert json.loads(result) == ENTITIES


def test_extract_entities_falls_back_to_raw_on_invalid_json(tools_module, fake_llm):
    tools_module.set_document("texto do paper")
    fake_llm.response = "isto não é JSON"

    result = tools_module.extract_entities.invoke({})

    assert result == "isto não é JSON"
    assert tools_module._state["entities"] == {"raw": "isto não é JSON"}


# ── análise NLP ────────────────────────────────────────────────────────

def test_analyze_nlp_without_spacy_model_returns_message(tools_module, monkeypatch):
    def fail_load(name):
        raise OSError("modelo não encontrado")

    monkeypatch.setitem(sys.modules, "spacy", types.SimpleNamespace(load=fail_load))
    tools_module._state["nlp_analysis"] = {}

    result = tools_module.analyze_nlp.invoke({})

    assert "spaCy model not found" in result
    assert tools_module._state["nlp_analysis"] == {}


def test_analyze_nlp_with_fake_pipeline(tools_module, monkeypatch):
    def tok(text, lemma=None, pos="NOUN", stop=False, punct=False):
        return SimpleNamespace(
            text=text, lemma_=lemma or text, pos_=pos, is_stop=stop, is_punct=punct
        )

    tokens = [
        tok("Transformers", lemma="transformer"),
        tok("use", pos="VERB"),  # 3 caracteres — conta como keyword
        tok("the", pos="DET", stop=True),
        tok("attention"),
        tok("attention"),
        tok(".", pos="PUNCT", punct=True),
        tok("is", pos="AUX", stop=True),
        tok("ok", pos="ADJ"),  # descartado por ter 2 caracteres
    ]
    ents = [
        SimpleNamespace(text="Google", label_="ORG"),
        SimpleNamespace(text="Google ", label_="ORG"),  # duplicata após strip
        SimpleNamespace(text="2017", label_="DATE"),
        SimpleNamespace(text="BLEU", label_="PRODUCT"),  # tipo ignorado
    ]

    class FakeDoc(list):
        pass

    doc = FakeDoc(tokens)
    doc.ents = ents
    doc.sents = iter(["s1", "s2"])

    monkeypatch.setitem(
        sys.modules, "spacy", types.SimpleNamespace(load=lambda name: (lambda text: doc))
    )
    tools_module.set_document("texto qualquer")

    result = json.loads(tools_module.analyze_nlp.invoke({}))

    assert result["sentence_count"] == 2
    assert result["token_count"] == len(tokens)
    assert result["named_entities"] == {"ORG": ["Google"], "DATE": ["2017"]}
    assert result["top_keywords"][0] == "attention"
    assert "ok" not in result["top_keywords"]
    assert "the" not in result["top_keywords"]
    assert tools_module._state["nlp_analysis"] == result


# ── literatura ─────────────────────────────────────────────────────────

def test_compare_with_literature_stores_search_result(tools_module, monkeypatch):
    class FakeSearch:
        def run(self, query):
            return f"resultados para {query}"

    utilities = types.ModuleType("langchain_community.utilities")
    utilities.DuckDuckGoSearchAPIWrapper = FakeSearch
    monkeypatch.setitem(sys.modules, "langchain_community", types.ModuleType("langchain_community"))
    monkeypatch.setitem(sys.modules, "langchain_community.utilities", utilities)

    result = tools_module.compare_with_literature.invoke({"query": "Transformers"})

    assert result == "resultados para Transformers"
    assert tools_module._state["literature_context"] == result


def test_compare_with_literature_handles_search_error(tools_module, monkeypatch):
    class BrokenSearch:
        def run(self, query):
            raise ConnectionError("sem rede")

    utilities = types.ModuleType("langchain_community.utilities")
    utilities.DuckDuckGoSearchAPIWrapper = BrokenSearch
    monkeypatch.setitem(sys.modules, "langchain_community", types.ModuleType("langchain_community"))
    monkeypatch.setitem(sys.modules, "langchain_community.utilities", utilities)

    result = tools_module.compare_with_literature.invoke({"query": "x"})

    assert result.startswith("Erro na busca web:")
    assert "sem rede" in result
    assert tools_module._state["literature_context"] == ""


# ── relatório final ────────────────────────────────────────────────────

def test_generate_report_consolidates_accumulated_state(tools_module, fake_llm):
    state = tools_module._state
    state["summary"] = "RESUMO-X"
    state["entities"] = {"title": "TITULO-Y"}
    state["nlp_analysis"] = {"token_count": 42}
    state["literature_context"] = "LITERATURA-Z"
    fake_llm.response = "# Relatório"

    result = tools_module.generate_report.invoke({})

    assert result == "# Relatório"
    prompt = fake_llm.calls[0]["user"]
    assert "Summary: RESUMO-X" in prompt
    assert '"title": "TITULO-Y"' in prompt
    assert '"token_count": 42' in prompt
    assert "Literature Context: LITERATURA-Z" in prompt
    for section in ["Executive Summary", "Key Entities", "Linguistic Analysis",
                    "Literature Context", "Main Insights"]:
        assert section in prompt


# ── chamada ao Groq ────────────────────────────────────────────────────

def test_llm_call_uses_groq_client(monkeypatch):
    from agent import tools

    captured = {}

    class FakeCompletions:
        def create(self, **kwargs):
            captured.update(kwargs)
            message = SimpleNamespace(content="resposta")
            return SimpleNamespace(choices=[SimpleNamespace(message=message)])

    class FakeGroq:
        def __init__(self, api_key=None):
            captured["api_key"] = api_key
            self.chat = SimpleNamespace(completions=FakeCompletions())

    monkeypatch.setattr(tools, "Groq", FakeGroq)
    monkeypatch.setattr(tools, "GROQ_API_KEY", "chave-teste")

    assert tools._llm_call("sys", "usr") == "resposta"
    assert captured["api_key"] == "chave-teste"
    assert captured["temperature"] == 0.0
    assert captured["messages"] == [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "usr"},
    ]
