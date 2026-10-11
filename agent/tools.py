# agent/tools.py

from langchain_core.tools import tool
from groq import Groq
from config import GROQ_API_KEY, GROQ_MODEL, SPACY_MODEL
import json
import logging


logger = logging.getLogger(__name__)


def _llm_call(system: str, user: str) -> str:
    """Chamada direta ao Groq para as ferramentas que precisam de LLM."""
    client = Groq(api_key=GROQ_API_KEY)
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.0,
    )
    return response.choices[0].message.content


# estado compartilhado entre ferramentas durante a execução do agente
_state: dict = {
    "full_text": "",
    "summary": "",
    "entities": {},
    "nlp_analysis": {},
    "literature_context": "",
}


def set_document(full_text: str) -> None:
    """Define o documento a ser analisado — chamado antes de run_agent."""
    _state["full_text"] = full_text
    _state["summary"] = ""
    _state["entities"] = {}
    _state["nlp_analysis"] = {}
    _state["literature_context"] = ""


@tool
def generate_summary(input: str = "") -> str:
    """Generate an executive summary of the document.
    Use this as the first step to understand what the document is about.
    No input required."""
    logger.info("Gerando resumo...")
    text = _state["full_text"][:8000]

    summary = _llm_call(
        system="You are an expert at summarizing scientific papers concisely and accurately.",
        user=f"Generate a concise executive summary (3-5 sentences) of this scientific paper:\n\n{text}"
    )
    _state["summary"] = summary
    return summary


@tool
def extract_entities(input: str = "") -> str:
    """Extract structured entities from the document including authors, institutions,
    methodology, datasets and key results. No input required."""
    logger.info("Extraindo entidades...")
    text = _state["full_text"][:8000]

    response = _llm_call(
        system="You are an expert at extracting structured information from scientific papers. Return only valid JSON.",
        user=f"""Extract the following entities and return ONLY a valid JSON object:
{{
  "title": "paper title",
  "authors": ["list of authors"],
  "institutions": ["list of institutions"],
  "year": "publication year",
  "area": "research area",
  "methodology": "main method or approach",
  "datasets": ["list of datasets"],
  "key_results": "main results in one sentence",
  "keywords": ["up to 10 keywords"]
}}

Text:
{text}"""
    )

    # remove markdown se presente
    text_clean = response.strip()
    if "```json" in text_clean:
        text_clean = text_clean.split("```json")[1].split("```")[0].strip()
    elif "```" in text_clean:
        text_clean = text_clean.split("```")[1].split("```")[0].strip()

    try:
        entities = json.loads(text_clean)
        _state["entities"] = entities
        return json.dumps(entities, indent=2)
    except json.JSONDecodeError:
        logger.warning("Resposta do LLM não é JSON válido — armazenando texto bruto.")
        logger.debug("Resposta bruta do LLM: %s", response)
        _state["entities"] = {"raw": response}
        return response


@tool
def analyze_nlp(input: str = "") -> str:
    """Run linguistic NLP analysis on the document using spaCy.
    Extracts named entities, top keywords and document statistics.
    No input required."""
    logger.info("Executando análise NLP...")
    import spacy
    from collections import Counter

    try:
        nlp = spacy.load(SPACY_MODEL)
    except OSError:
        logger.warning("Modelo spaCy '%s' não encontrado — análise NLP ignorada.", SPACY_MODEL)
        return "spaCy model not found. Run: python -m spacy download en_core_web_sm"

    text = _state["full_text"][:100_000]
    doc = nlp(text)

    # entidades por tipo
    relevant_types = {"PERSON", "ORG", "DATE", "GPE"}
    entities: dict[str, list[str]] = {}
    for ent in doc.ents:
        if ent.label_ not in relevant_types:
            continue
        if ent.label_ not in entities:
            entities[ent.label_] = []
        value = ent.text.strip()
        if value and value not in entities[ent.label_]:
            entities[ent.label_].append(value)

    # keywords
    relevant_pos = {"NOUN", "VERB", "ADJ"}
    words = [
        token.lemma_.lower()
        for token in doc
        if not token.is_stop
        and not token.is_punct
        and token.pos_ in relevant_pos
        and len(token.text) > 2
    ]
    top_keywords = [w for w, _ in Counter(words).most_common(15)]

    analysis = {
        "sentence_count": len(list(doc.sents)),
        "token_count": len(doc),
        "named_entities": entities,
        "top_keywords": top_keywords,
    }

    _state["nlp_analysis"] = analysis
    return json.dumps(analysis, indent=2)


@tool
def compare_with_literature(query: str) -> str:
    """Search the web to contextualize the paper within current literature.
    Input should be the paper title or main topic to search for."""
    logger.info("Comparando com literatura: '%s'", query)
    try:
        from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
        search = DuckDuckGoSearchAPIWrapper()
        result = search.run(query)
        _state["literature_context"] = result
        return result
    except Exception as e:
        logger.error("Erro na busca web: %s", e)
        return f"Erro na busca web: {e}"


@tool
def generate_report(input: str = "") -> str:
    """Consolidate all collected insights into a final structured report.
    Use this as the last step after running all other tools. No input required."""
    logger.info("Gerando relatório final...")

    context = f"""
Summary: {_state.get('summary', 'Not available')}

Entities: {json.dumps(_state.get('entities', {}), indent=2)}

NLP Analysis: {json.dumps(_state.get('nlp_analysis', {}), indent=2)}

Literature Context: {_state.get('literature_context', 'Not available')}
"""

    report = _llm_call(
        system="You are an expert research analyst. Generate clear, structured insight reports.",
        user=f"""Based on the following analysis, generate a comprehensive insight report with these sections:
1. Executive Summary
2. Key Entities & Metadata
3. Linguistic Analysis
4. Literature Context
5. Main Insights & Conclusions

Analysis data:
{context}"""
    )

    return report