import logging
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent
from agent.prompt import SYSTEM_PROMPT
from agent.tools import (
    generate_summary,
    extract_entities,
    analyze_nlp,
    compare_with_literature,
    generate_report,
    set_document,
)
from models import InsightReport
from config import GROQ_API_KEY, GROQ_MODEL, AGENT_MAX_ITERATIONS


logger = logging.getLogger(__name__)


_agent = None


def _get_agent():
    """Inicializa o agente uma única vez (singleton)."""
    global _agent
    if _agent is None:
        logger.info("Inicializando agente...")
        llm = ChatGroq(
            api_key=GROQ_API_KEY,
            model=GROQ_MODEL,
            temperature=0.0,
        )
        tools = [
            generate_summary,
            extract_entities,
            analyze_nlp,
            compare_with_literature,
            generate_report,
        ]
        _agent = create_react_agent(
            model=llm,
            tools=tools,
            prompt=SYSTEM_PROMPT,
        )
    return _agent


def run_agent(filename: str, full_text: str) -> InsightReport:
    logger.info("Analisando: '%s'", filename)

    set_document(full_text)
    agent = _get_agent()

    try:
        result = agent.invoke(
            {"messages": [HumanMessage(
                content=f"Analyze the document '{filename}' and generate a comprehensive insight report."
            )]},
            config={"recursion_limit": AGENT_MAX_ITERATIONS},
        )
    except Exception as e:
        logger.error("Erro ao executar o agente: %s", e)
        return InsightReport(
            filename=filename,
            final_report=f"Erro ao executar o agente: {e}",
        )

    messages = result.get("messages", [])
    final_answer = messages[-1].content if messages else ""

    steps = []
    for msg in messages:
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tool_call in msg.tool_calls:
                steps.append({
                    "tool": tool_call["name"],
                    "input": tool_call["args"],
                    "observation": "",
                })
        if hasattr(msg, "name") and msg.name:
            if steps and not steps[-1]["observation"]:
                steps[-1]["observation"] = str(msg.content)[:300]

    from agent.tools import _state

    return InsightReport(
        filename=filename,
        summary=_state.get("summary", ""),
        entities=_state.get("entities", {}),
        nlp_analysis=_state.get("nlp_analysis", {}),
        literature_context=_state.get("literature_context", ""),
        final_report=final_answer,
        steps=steps,
    )