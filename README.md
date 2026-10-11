# Insight Extraction Agent

Agente autônomo de extração de insights de artigos científicos, construído
com LangChain e LangGraph. Combina NLP clássico, extração via LLM e busca
na literatura para gerar relatórios estruturados de forma autônoma.

## Visão Geral

Este projeto demonstra a diferença fundamental entre um pipeline determinístico
e um agente autônomo — mesmo domínio, mesma stack, abordagem completamente
diferente. Enquanto um pipeline executa passos fixos, o agente decide
autonomamente quais análises realizar e em qual ordem.

```
Documento → Agente decide → Ferramentas → Relatório de Insights
```

## Comparação com Pipeline Determinístico

| | Document Intelligence Pipeline | Insight Extraction Agent |
|---|---|---|
| Fluxo | fixo e pré-definido | decidido pelo agente |
| Entidades | schema pré-definido | agente escolhe o que extrair |
| Literatura | não disponível | agente busca autonomamente |
| Output | metadata estruturado | relatório em linguagem natural |

## Arquitetura

```
insight-extraction-agent/
├── agent/
│   ├── tools.py          # ferramentas disponíveis para o agente
│   ├── prompt.py         # system prompt e diretrizes
│   └── agent.py          # definição e execução do agente
├── rag/
│   ├── embedder.py       # geração de embeddings
│   └── store.py          # integração com Qdrant
├── examples/
│   └── analyze_paper.py  # exemplo de uso completo
├── tests/                # testes unitários (offline, com mocks)
├── pipeline.py           # ingestão de documentos
├── models.py             # estruturas de dados
└── config.py             # configurações centralizadas
```

## Ferramentas do Agente

| Ferramenta | O que faz |
|---|---|
| `generate_summary` | Gera resumo executivo do documento |
| `extract_entities` | Extrai entidades estruturadas via LLM |
| `analyze_nlp` | Análise linguística com spaCy (NER, keywords, estatísticas) |
| `compare_with_literature` | Busca na web para contextualizar o paper |
| `generate_report` | Consolida todos os insights em relatório final |

## Exemplo de Raciocínio Autônomo

```
Documento: attention.pdf

[1] Tool: generate_summary
    → Resumo executivo do paper

[2] Tool: extract_entities
    → Autores, instituições, metodologia, datasets, resultados

[3] Tool: analyze_nlp
    → 365 sentenças, 8203 tokens, top keywords

[4] Tool: compare_with_literature("Attention Is All You Need")
    → Contexto e impacto na literatura atual

[5] Tool: generate_report
    → Relatório final estruturado em 5 seções
```

## Tecnologias

- `Python 3.10+`
- `LangChain 1.2` + `LangGraph 1.1` — orquestração do agente
- `Groq` — inferência LLM gratuita (`llama-3.3-70b-versatile`)
- `spaCy` — NLP clássico e NER (`en_core_web_sm`)
- `sentence-transformers` — embeddings (`all-MiniLM-L6-v2`)
- `Qdrant` — vector store local
- `DuckDuckGo` — busca web gratuita e sem API key
- `Docker` — execução local do Qdrant

## Como Executar

**1. Clone o repositório**
```bash
git clone https://github.com/AndreasMarcos/insight-extraction-agent.git
cd insight-extraction-agent
```

**2. Crie e ative o ambiente virtual**
```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux/Mac
source .venv/bin/activate
```

**3. Instale as dependências**
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

**4. Configure as variáveis de ambiente**
```bash
# copie o arquivo de exemplo e preencha sua chave
cp .env.example .env
```

```
GROQ_API_KEY=sua_chave_aqui
```

Obtenha uma API key gratuita em `https://console.groq.com`.

**5. Suba o Qdrant com Docker**
```bash
docker compose up -d
```

O `docker-compose.yml` sobe o container `qdrant-insight` na porta `6333`,
persistindo os dados em `./qdrant_storage`. Sem o Compose, use o equivalente:
```bash
docker run --name qdrant-insight -p 6333:6333 -v $(pwd)/qdrant_storage:/qdrant/storage qdrant/qdrant
```

**6. Execute o exemplo**
```bash
python -m examples.analyze_paper
```

## Uso

```python
from pipeline import ingest_pdf
from agent import run_agent

# indexa e processa o documento
result = ingest_pdf("data/paper.pdf")

# executa o agente
report = run_agent(
    filename=result["filename"],
    full_text=result["full_text"],
)

# acessa as seções do relatório
print(report.summary)
print(report.entities)
print(report.nlp_analysis)
print(report.literature_context)
print(report.final_report)

# raciocínio do agente passo a passo
for step in report.steps:
    print(f"Tool: {step['tool']}")
    print(f"Observation: {step['observation']}")
```

## Testes

```bash
pip install -r requirements-dev.txt
pytest
```

Os testes rodam totalmente offline: o cliente Groq, a busca web, o spaCy,
os embeddings e o Qdrant são substituídos por mocks. Não é necessário API key,
conexão com a internet nem Qdrant em execução.

## Decisões de Design

**Por que estado compartilhado entre ferramentas?**
As ferramentas precisam passar informação entre si — o `generate_report`
consolida o que `generate_summary`, `extract_entities` e `analyze_nlp`
encontraram. O `_state` funciona como memória de curto prazo do agente
durante a análise de um documento.

**Por que o agente e não um pipeline fixo?**
Um pipeline fixo sempre executa os mesmos passos independente do documento.
O agente pode adaptar sua estratégia — por exemplo, chamar `compare_with_literature`
com o título exato extraído pelo `extract_entities`, em vez de usar um termo genérico.

**Por que `temperature=0.0`?**
Análise de documentos requer precisão e reprodutibilidade. Temperatura zero
garante que o agente tome sempre as decisões mais prováveis e factuais.

## Limitações Conhecidas

**Rate limit do Groq** — o free tier tem limite de tokens por dia. Para
documentos longos ou múltiplos papers, o limite pode ser atingido rapidamente.

**Contexto limitado** — o LLM recebe os primeiros 8000 caracteres do documento.
Informações no final de papers longos podem não ser capturadas.

**Dependência de internet** — a ferramenta `compare_with_literature` requer
conexão ativa. Em caso de falha, o agente continua sem o contexto da literatura.

## Licença

Distribuído sob a licença MIT. Veja o arquivo [LICENSE](LICENSE).
