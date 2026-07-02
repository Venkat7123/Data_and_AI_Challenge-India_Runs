# Redrob Hackathon: Intelligent Candidate Discovery & Ranking (Track 01)

## Team: CodeCrew

### Team Members
- **Venkatachalam S** (Team Lead & Core AI Ranking Engine) - venkatachalamsubramanian23@gmail.com
- **Ameen Basha A** (Problem Statement Research & AI Improvements) - ameenbashanawaz.123@gmail.com
- **Vasu Vigneshwaran P** (Feature Engineering & Validation) - vasuvignesh28@gmail.com
- **Samritha R** (Documentation, Evaluation & Presentation) - samritha0123sai@gmail.com

---

## 1. The Problem
Recruiters are tasked with finding the perfect fit from oceans of profiles, but traditional keyword filters fall short. They miss the hidden gems—candidates whose true potential, intent, and subtle behavioral signals are lost in the noise of keyword-stuffed profiles. We needed a smarter system that acts as an "AI recruiter", capable of looking beyond surface-level keywords to deeply understand context, predict relevance, and integrate multi-dimensional signals to deliver precise candidate shortlists.

## 2. Our Approach
To solve this, we avoided black-box neural inference at rank time in favor of a **transparent, fully offline multi-signal ranker**. This ensures high execution speed and interpretability.

Our strategy specifically inverts the "keyword-stuffer trap" by isolating the skills array and focusing on real semantic fit and career trajectories. The top results are genuine professionals in AI, ML, Search, NLP, and Recommendation systems, not off-domain candidates who padded their profiles with buzzwords.

## 3. Architecture & Pipeline
The ranking engine is designed for rapid processing and deterministic output (byte-identical reruns), operating entirely offline without the need for external API calls.

### Pipeline Stages
1. **Data Ingestion**: Reads large pools of candidates from a structured `.jsonl` format.
2. **Feature Extraction**: Parses candidate profiles to extract textual data (headline, summary, role descriptions) and structured metadata (experience, locations, education, skill assessments).
3. **Semantic Scoring (TF-IDF)**: Calculates cosine similarity between a meticulously crafted Job Description (JD) vector and the candidate's textual features, explicitly excluding raw skill arrays.
4. **Heuristic Evaluation**: Evaluates candidates across multiple specialized dimensions:
   - **Title/Career-Trajectory Fit**: The most decisive signal, identifying progressive growth in relevant domains.
   - **Experience-Band Fit**: Targets the "sweet spot" of 5-9 years.
   - **Skill Credibility**: Validates skills based on endorsements and usage duration, discarding "expert with 0 months" claims.
5. **Penalties & Behavioral Adjustments**:
   - Applies multipliers to down-weight disengaged or unreachable candidates.
   - Penalizes candidates with off-domain titles or irrelevant tech stacks (e.g., vision/speech without NLP/IR).
   - **Consistency Detector (Honeypot Filter)**: Identifies logically impossible profiles and forces them to the bottom.
6. **Aggregation & Output**: Aggregates the weighted sub-scores (summing to 1.0), ranks the candidates, and outputs a highly accurate shortlist to `submission.csv`.

---

## Reproducibility & Execution

Our solution efficiently processes ~100K candidates in ~1-3 minutes on a standard CPU.

### Environment Requirements
- **Platform**: Local workstation (Windows/Linux/macOS)
- **Python**: 3.10.11 (or compatible)
- **Hardware**: CPU-only (No GPU required), ~16GB RAM recommended
- **Network**: Completely offline (No external API calls during ranking)

### Running the Code
To run the end-to-end ranking pipeline:

```bash
python rank.py --candidates ./candidates.jsonl --out ./submission.csv
```
This command reads the candidate pool from `candidates.jsonl` and writes the ranked output to `submission.csv`. The entire process runs offline and does not require pre-computation.
