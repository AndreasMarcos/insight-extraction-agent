# Exemplo de uso do Insight Extraction Agent.
# Ingere um artigo científico e executa análise completa via agente autônomo.
#
# Execução:
#   python -m examples.analyze_paper

import logging
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline import ingest_pdf, status
from agent import run_agent
from rag.store import delete_collection


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )

    # ── ingestão ──────────────────────────────────────────────────────
    print("=== Resetando índice ===")
    delete_collection()

    print("\n=== Ingestão ===")
    result = ingest_pdf("data/attention.pdf")
    print(f"  {result['filename']} — {result['pages']} páginas, {result['chunks']} chunks")
    status()

    # ── análise via agente ────────────────────────────────────────────
    print("\n=== Insight Extraction Agent ===")
    report = run_agent(
        filename=result["filename"],
        full_text=result["full_text"],
    )

    # ── exibe steps ───────────────────────────────────────────────────
    print("\n--- Raciocínio do Agente ---")
    for i, step in enumerate(report.steps, start=1):
        print(f"\n[{i}] Tool: {step['tool']}")
        print(f"     Input: {step['input']}")
        if step['observation']:
            print(f"     Observation: {step['observation']}...")

    # ── exibe seções do relatório ─────────────────────────────────────
    print("\n--- Resumo ---")
    print(report.summary)

    print("\n--- Entidades Extraídas ---")
    for key, value in report.entities.items():
        print(f"  {key}: {value}")

    print("\n--- Análise NLP ---")
    nlp = report.nlp_analysis
    print(f"  Sentenças : {nlp.get('sentence_count', 'N/A')}")
    print(f"  Tokens    : {nlp.get('token_count', 'N/A')}")
    print(f"  Keywords  : {', '.join(nlp.get('top_keywords', []))}")

    print("\n--- Contexto na Literatura ---")
    print(report.literature_context[:500] if report.literature_context else "N/A")

    print("\n--- Relatório Final ---")
    print(report.final_report)


if __name__ == "__main__":
    main()