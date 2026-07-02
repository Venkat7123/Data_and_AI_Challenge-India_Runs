# India Runs Candidate Ranking

Track 1: Data and AI Challenge

Team: CodeCrew

This repository contains an offline candidate discovery and ranking pipeline for the Redrob India Runs challenge. It reads candidate profiles, retrieves likely matches for job-style queries, scores each profile with multiple hiring signals, and writes the required `submission.csv` with 100 ranked candidates.

## What The System Does

The main entry point is [rank.py](C:/Users/admin/Downloads/India_runs_data_and_ai_challenge/india-runs/rank.py). It supports two practical modes:

```bash
python rank.py --batch
python rank.py --query "senior python developer with fastapi and aws in pune"
```

Batch mode runs 20 predefined role queries across software engineering, data/ML, DevOps, cloud, mobile, security, QA, analytics, product, and SRE roles. Query mode runs the same ranking stack for one supplied job description or search phrase.

The output is always:

```text
candidate_id,rank,score,reasoning
```

## Methodology

The ranker combines retrieval, scoring, and consistency checks:

1. Query parsing extracts skills, experience requirements, seniority hints, location, and aliases from natural-language text.
2. Hybrid retrieval combines FAISS vector search with BM25 keyword search. The checked-in full index uses a fast local 384-dim hashing representation; the slower transformer builder can be used when MiniLM embeddings are available.
3. Reciprocal Rank Fusion merges vector and keyword candidates.
4. An optional cross-encoder reranker (`cross-encoder/ms-marco-MiniLM-L-6-v2`) refines query/profile relevance when the model is available.
5. Structured skill matching checks the candidate skill array using exact, alias, and fuzzy matching.
6. Experience and title matching penalize candidates whose seniority or role family does not fit the query.
7. Behavioral, career trajectory, and skill proficiency scores use Redrob-style platform signals such as recruiter response rate, profile completeness, verification, saved-by-recruiter count, GitHub activity, openness to work, and interview completion rate.
8. Honeypot detection applies a strong penalty to internally inconsistent profiles, including impossible company dates, unrealistic skill density, expert skills with zero usage years, and unexplained long gaps.

The final score is a weighted fusion configured in [configs/scoring_weights.yaml](C:/Users/admin/Downloads/India_runs_data_and_ai_challenge/india-runs/configs/scoring_weights.yaml).

## Scoring Weights

| Signal | Weight |
| --- | ---: |
| Semantic similarity | 0.20 |
| Skill match | 0.20 |
| Cross-encoder or RRF relevance | 0.15 |
| Behavioral score | 0.15 |
| Keyword match | 0.10 |
| Experience match | 0.08 |
| Career trajectory | 0.07 |
| Skill proficiency | 0.05 |
| Location match | 0.00 |
| Education match | 0.00 |

Location and education are modeled but currently have zero backend weight. The Gradio UI has separate recruiter-facing sliders for interactive re-ranking.

## Repository Layout

```text
.
|-- rank.py                         Main submission generation script
|-- submission.csv                  Generated ranked output
|-- submission_metadata.yaml        Hackathon metadata and methodology
|-- configs/
|   |-- scoring_weights.yaml        Ranking weights and listwise settings
|   |-- settings.yaml               App/search settings
|   `-- models.yaml                 Model configuration
|-- src/
|   |-- agents/                     Planner, executor, reflector, orchestrator
|   |-- search/                     FAISS, BM25, hybrid fusion, reranker
|   |-- matching/                   Skill, experience, behavior, confidence scoring
|   |-- language/                   Multilingual embeddings and language helpers
|   |-- fairness/                   Bias/anonymization utilities
|   |-- rationale/                  Candidate explanation generation
|   |-- extraction/                 Profile field extraction
|   |-- ingestion/                  Parsers and normalizers
|   |-- api/                        FastAPI routes
|   `-- ui/                         Gradio interface
|-- scripts/                        Index building and evaluation utilities
`-- tests/                          Unit and integration tests
```

## Setup

Use Python 3.10 or newer.

```bash
pip install -e .
```

For development extras:

```bash
pip install -e ".[dev]"
```

Prebuilt indexes are expected under `data/indexes/`:

```text
data/indexes/faiss_index.bin
data/indexes/faiss_id_map.json
data/indexes/bm25_index.pkl
data/indexes/offset_index.json
data/indexes/index_meta.json
```

For this workspace, the full candidate file is `data/candidates.jsonl`. Rebuild the practical full-dataset indexes with:

```bash
python scripts/build_fast_indexes.py --profiles ./data/candidates.jsonl
```

To rebuild the slower transformer-backed indexes instead:

```bash
python scripts/build_indexes.py --profiles ./data/candidates.jsonl --force
```

## Running

Generate a submission with the predefined strategic query set:

```bash
python rank.py --batch --out submission.csv
```

Run one query:

```bash
python rank.py --query "data engineer with spark airflow kafka python in pune" --out submission.csv
```

Interactive mode:

```bash
python rank.py
```

Launch the optional UI:

```bash
python src/ui/app.py
```

Launch the API:

```bash
python src/main.py
```

## Testing

```bash
python -m pytest tests/ -q
```

Useful focused tests:

```bash
python -m pytest tests/test_matching tests/test_search -q
python -m pytest tests/test_agents tests/test_integration -q
```

## Reproducibility Notes

- Ranking is designed to run CPU-only.
- The batch path skips cross-encoder inference and uses normalized hybrid/RRF relevance for speed and offline reliability.
- Candidate filling uses a fixed random seed when additional profiles are needed to reach 100 rows.
- Submission writing asserts 100 rows, unique candidate IDs, valid `CAND_` IDs, and complete rank coverage from 1 to 100.

## Current Submission Snapshot

The checked-in [submission.csv](C:/Users/admin/Downloads/India_runs_data_and_ai_challenge/india-runs/submission.csv) contains 100 rows with monotonically sorted scores. Each row includes a compact recruiter-style reason using current role, experience, skill count, and response-rate signal.
