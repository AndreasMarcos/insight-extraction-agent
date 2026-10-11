"""Testes da execução do agente e do parsing do resultado (LLM mockado)."""

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from agent import agent as agent_module
from agent import tools


class FakeAgent:
    def __init__(self, result=None, error=None):
        self.result = result
        self.error = error
        self.calls = []

    def invoke(self, payload, config=None):
        self.calls.append({"payload": payload, "config": config})
        if self.error:
            raise self.error
        return self.result


def _tool_call(name, args, call_id):
    return {"name": name, "args": args, "id": call_id, "type": "tool_call"}


def test_run_agent_parses_steps_and_state(monkeypatch):
    messages = [
        HumanMessage(content="Analyze"),
        AIMessage(content="", tool_calls=[_tool_call("generate_summary", {}, "1")]),
        ToolMessage(content="resumo", name="generate_summary", tool_call_id="1"),
        AIMessage(content="", tool_calls=[
            _tool_call("compare_with_literature", {"query": "T"}, "2"),
        ]),
        ToolMessage(content="x" * 500, name="compare_with_literature", tool_call_id="2"),
        AIMessage(content="Relatório final"),
    ]

    class StatefulAgent(FakeAgent):
        def invoke(self, payload, config=None):
            # simula o efeito colateral das ferramentas no estado compartilhado
            tools._state["summary"] = "resumo"
            tools._state["entities"] = {"title": "T"}
            return super().invoke(payload, config)

    fake = StatefulAgent(result={"messages": messages})
    monkeypatch.setattr(agent_module, "_get_agent", lambda: fake)

    report = agent_module.run_agent("paper.pdf", "texto completo")

    assert tools._state["full_text"] == "texto completo"
    assert report.filename == "paper.pdf"
    assert report.final_report == "Relatório final"
    assert report.summary == "resumo"
    assert report.entities == {"title": "T"}
    assert [s["tool"] for s in report.steps] == ["generate_summary", "compare_with_literature"]
    assert report.steps[1]["input"] == {"query": "T"}
    assert report.steps[0]["observation"] == "resumo"
    # observações são truncadas em 300 caracteres
    assert len(report.steps[1]["observation"]) == 300
    assert fake.calls[0]["config"] == {"recursion_limit": agent_module.AGENT_MAX_ITERATIONS}
    assert "paper.pdf" in fake.calls[0]["payload"]["messages"][0].content


def test_run_agent_returns_error_report_on_failure(monkeypatch):
    fake = FakeAgent(error=RuntimeError("rate limit"))
    monkeypatch.setattr(agent_module, "_get_agent", lambda: fake)

    report = agent_module.run_agent("paper.pdf", "texto")

    assert report.filename == "paper.pdf"
    assert report.final_report == "Erro ao executar o agente: rate limit"
    assert report.steps == []
    assert report.summary == ""


def test_run_agent_resets_state_from_previous_document(monkeypatch):
    tools._state["summary"] = "resumo de outro paper"
    fake = FakeAgent(result={"messages": []})
    monkeypatch.setattr(agent_module, "_get_agent", lambda: fake)

    report = agent_module.run_agent("novo.pdf", "novo texto")

    assert report.summary == ""
    assert report.final_report == ""
    assert report.steps == []
