# Business Entity Resolution Pipeline — Reproduction Guide

This directory contains the self-contained, end-to-end pipeline for the ML Challenge 2026 Business Entity Resolution Challenge.

## Pipeline Architecture

The solution uses a **Multi-Pass Blocking + High-Precision Matching Engine with Global 1-to-1 Uniqueness Resolution**:

1. **Country Partitioning**:
   - Entities are grouped by `country` (open set: US, India, France).
   - Guarantees zero cross-country false merges and limits peak RAM consumption to ~2 GB (safe for 8 GB RAM machines).

2. **Multi-Pass Blocking (`src/blocking.py`)**:
   - **Compact Name Key**: Normalized ASCII alphanumeric strings without legal suffixes (`inc`, `llc`, `sarl`, `pvt ltd`).
   - **Domain Stem Key**: Brand stem extraction for web domains (e.g. `www.mybiz.com` → `mybiz`).
   - **Sorted Name Key**: Alphabetically sorted substantive tokens to handle word transpositions.
   - **Rare Name Tokens**: IDF-filtered distinctive tokens (document frequency $\le 100$).
   - **Compound Address Keys**: Pairings of street/building numbers with distinctive locality/street tokens.

3. **Multi-Regime Matching (`src/matching.py`)**:
   - Evaluates RapidFuzz string similarities, token Jaccard, address number verification, and multi-regime matching rules:
     - Exact compact name match
     - High string similarity with address consistency
     - Non-Latin script transliteration matching (Hindi, Tamil, etc.)
     - Synthetic alias resolution with address verification
     - Website domain stem matching
   - **Global 1-to-1 Uniqueness Resolution**: Assigns each target entity to at most ONE Source 1 entity (its argmax confidence score), eliminating double-merges and maximizing Macro $F_{0.5}$.

4. **Submission Formatting (`src/pipeline.py`)**:
   - Generates both `output/matching_results.tsv` and `output/candidate_pairs.tsv` conforming to all validator checks.

---

## Environment Setup

Requires Python 3.8+ (tested on Python 3.13 / 3.12).

```bash
# 1. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt
```

---

## End-to-End Execution

Run the complete pipeline from the project root (`student_resource/`):

```bash
python3 -m code.business_entity_resolution.src.pipeline \
    --data-dir dataset/test \
    --output-dir output \
    --matching-out output/matching_results.tsv \
    --candidate-out output/candidate_pairs.tsv
```

### Validate Outputs

Run the official competition submission validator to verify compliance:

```bash
python3 utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
```
