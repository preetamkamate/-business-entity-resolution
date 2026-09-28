# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** EntityResolvers  
**Team Members:** Team Lead & Research Engineers  
**Submission Date:** September 2026  

---

## 1. Executive Summary

We present an end-to-end, high-precision Entity Resolution (ER) framework specifically engineered for the macro-$F_{0.5}$ metric on multi-source commercial entity identity data across three distinct geographic regions (US, India, and France). Our pipeline integrates **strict country partitioning**, **multi-pass high-recall inverted blocking** (compact name normalization, web domain stem alignment, rarest-token-first document frequency indexing, and compound address-street anchoring), followed by a **multi-regime string similarity scoring engine** coupled with a **global 1-to-1 target uniqueness resolver**. On held-out validation data across US and India (10,000 entities with 34,733 ground-truth links), our method achieves an exceptional Macro $F_{0.5}$ score of **0.9367**, with **97.13%** precision, **88.64%** recall, **94.32%** candidate recall, and **92.79%** singleton accuracy while keeping average candidate pool size under 35 pairs per entity.

---

## 2. Methodology

### 2.1 Problem Analysis
Exploratory data analysis across 26.4 million records revealed key structural properties and noise modalities:
1. **Deduplicated Reference Source**: Source 1 serves as the canonical reference source. Crucially, empirical auditing of the ground truth proved that every target entity in Source 2 and Source 3 matches **at most one** Source 1 entity (0 duplicate targets across 7.64M ground truth links). This implies a strict 1-to-1 mapping from any target to its corresponding reference entity.
2. **Hard Geographic Partitioning**: Zero matches cross national borders. All test entities strictly belong to their respective country (`US`, `India`, or `France`).
3. **Regional Heterogeneity & Noise Modalities**:
   - **US**: Extensive legal suffix permutations (`Inc.`, `LLC`, `Corp.`, `L.L.C.`, `PC`), address abbreviation variants (`Rd`, `Road`, `St`, `Street`, `Hwy`), web domain aliases (`companyname.com`), and synthetic fantasy aliases (`Drexkor`, `Vantagebrix`, `Lumnylazeta`) paired with identical street addresses.
   - **India**: Cross-script transliterations (Devanagari, Tamil, Kannada, Bengali, Telugu, Gujarati, Marathi) where target names appear in regional alphabets while the corresponding addresses remain in Latin script with identical building/shop numbers (`Shop No. 13`, `H.No A-5`, `113/154`). High frequency of shared commercial addresses (e.g. DLF Towers, industrial estates, commercial complexes).
   - **France**: French legal entities (`SARL`, `SASU`, `SAS`, `SCI`, `EURL`, `SA`, `& Fils`, `& Frères`), French street terminology (`Rue`, `Boulevard`, `Allée`, `Avenue`, `bis`, `ter`), and accent variations (`Àmicale` vs `Amicale`).
4. **Metric Asymmetry ($F_{0.5}$)**:
   The evaluation metric $F_{0.5} = \frac{1.25 \times P \times R}{0.25 P + R}$ weights precision twice as heavily as recall. False merges on singletons completely drop the entity score from 1.0 to 0.0. Therefore, precision must take absolute precedence over aggressive over-matching.

### 2.2 Solution Strategy
**Approach Type:** Open-Set Priority-Tier Inverted Blocking + Precision-Focused Matching + Global 1-to-1 Uniqueness Constraint Resolution.

**Core Innovations:**
- **Open-Set Country Partitioning**: Partitioning dynamically on whatever countries appear in the incoming dataset (US, India, France, or any unseen country) eliminates 100% of cross-border false positives while keeping peak memory under 1.5 GB.
- **Rarest-Token-First Blocking**: Inverted candidate indexing sorts substantive name tokens by ascending document frequency before candidate insertion, preventing common words (e.g. *services*, *associates*, *barbershop*) from crowding out rare distinctive brand names within the per-entity candidate cap.
- **Strict House Number Extraction (`int(n) > 0`)**: Normalizes address numbers strictly greater than zero, eliminating phantom `'0'` house numbers from apartment/floor indicators (`Fl 0`, `Floor 0`) that previously generated false alias matches.
- **True Non-Latin Script Detection**: Uses Unicode category inspection (`not unicodedata.name(c, '').startswith('LATIN')`) to isolate genuine Indic scripts (Devanagari, Tamil, etc.) while preventing accented French/European Latin names (`née`, `Córporation`) from erroneously triggering transliteration matching rules.
- **Missing Target Address Restrictions**: For target records with missing or empty addresses, matches are restricted to distinctive names ($\ge 2$ tokens or compact length $\ge 10$), eliminating cross-city false merges on generic 1-word entities.
- **Continuous Similarity Tie-Breaking**: Incorporates fine-grained continuous RapidFuzz token sort similarities for both name and address into pair confidence scores, ensuring optimal tie-breaking during competitive global 1-to-1 assignment.
- **Global 1-to-1 Target Uniqueness Resolver**: When multiple Source 1 entities claim the same target record, the pipeline assigns the target exclusively to the Source 1 entity with the highest match confidence, completely preventing multi-claim false merges.

---

## 3. Candidate Generation (Blocking)

To reduce the $1.73 \times 10^6 \times 9.97 \times 10^6 \approx 1.7 \times 10^{13}$ pair Cartesian space to a tractable inference set, we deploy a multi-pass inverted blocking index per country:

- **Blocking Keys Used**:
  1. `compact_name`: Normalized alphanumeric name stripped of punctuation, legal suffixes (`inc`, `llc`, `pvt ltd`, `sarl`, `sasu`, etc.), and spaces (length $\ge 3$).
  2. `domain_stem`: Cleaned brand stem extracted from website URLs/domains (`www.xyz.com` $\to$ `xyz`).
  3. `sorted_name`: Alphabetically sorted substantive name tokens to resolve word-order transpositions.
  4. `first_two_tokens`: Ordered prefix tuple of the first two substantive name tokens.
  5. `rare_name`: Distinctive brand tokens whose country document frequency is $\le 150$, visited in rarest-first order.
  6. `num_street`: Compound tuple `(normalized_house_number, street_token)` pairing the building/flat number ($> 0$) with clean substantive street names (excluding high-frequency city/region tokens).
- **Candidate Pairs Generated**: Average of ~34.6 candidate targets per Source 1 entity across all 1.73M entities (reduction ratio $> 99.9997\%$).
- **Recall Retention**: Achieves **94.32%** candidate recall on held-out ground truth data (up from 86.13% in the initial baseline).

---

## 4. Matching Model

### Features Used:
- **Name Features**:
  - `compact_name_match`: Boolean indicator for exact normalized name match.
  - `name_token_sort_ratio`: RapidFuzz token sort similarity ratio $\in [0, 1]$.
  - `name_jaccard`: Jaccard similarity over substantive non-stopword tokens.
  - `domain_stem_alignment`: Stem containment between website domain names and business names.
  - `compact_substring`: Bidirectional substring containment for acronyms and compound words.
- **Address Features**:
  - `shared_numbers`: Set intersection of extracted building, plot, shop, and street numbers (strictly $> 0$).
  - `diff_numbers`: Boolean indicator detecting non-overlapping street numbers.
  - `addr_token_sort_ratio`: RapidFuzz token sort similarity over normalized address strings.
  - `shared_addr_tokens`: Overlap of distinctive locality/street tokens.
  - `addr_match_flag`: Boolean indicator requiring both numeric match and street token overlap.
- **Categorical & Contextual Features**:
  - `has_non_latin`: Flags true non-Latin regional script records for transliteration rules.
  - `is_alias`: Flags single fantasy brand words ($\ge 6$ chars) for synthetic alias matching.
  - `is_domain`: Flags URL/website targets.

### Decision Engine & Threshold Selection:
Pairs are evaluated across 5 precision-calibrated regimes:
1. **Regime 1 (Exact Compact Name & Domain)**: Confirmed if target address is missing (with distinctive name criteria) or shares house number + street tokens (score: 0.94–0.99). If addresses have non-overlapping house numbers and zero shared street tokens, score is penalized to 0.20.
2. **Regime 2 (Synthetic Alias)**: Single fantasy word with verified house number and address fuzz $\ge 0.70$ (score: 0.89 + tie-break).
3. **Regime 3 (Transliteration Match)**: Target name in true non-Latin script paired with exact street number and address fuzz $\ge 0.70$ (score: 0.91 + tie-break).
4. **Regime 4 (Multi-Field Name + Address)**: Substantive name similarity ($\ge 0.40$ Jaccard or substring) supported by verified house number and street tokens (score: 0.84–0.89 + tie-break).
5. **Regime 5 (High Name Jaccard / Fuzzy Name)**: RapidFuzz $\ge 0.85$ or Jaccard $\ge 0.75$ with non-contradictory address consistency (score: 0.87–0.89 + tie-break).

**Global Uniqueness & Threshold Selection**:
Every target candidate exceeding threshold $\tau = 0.78$ is globally assigned to its argmax S1 entity. S1 entities with no target candidates exceeding the threshold remain empty singletons.

---

## 5. Results & Error Analysis

- **Macro $F_{0.5}$ Score**: **0.9367** on held-out validation data (US + India; up from 0.8283 baseline and 0.8835 V2).
- **Precision**: **0.9713** (up from 0.8267 baseline and 0.8781 V2).
- **Recall**: **0.8864** (up from 0.8034 baseline and 0.8844 V2).
- **Candidate Recall**: **0.9432** (up from 0.8613 baseline and 0.9302 V2).
- **Singletons Accuracy**: **92.79%** correctly identified as zero-match entities (up from 75.92% baseline).
- **False Positives**: **911** (down from 4,250 in V2 — a **78.6%** drop in false merges).
- **False Negatives**: **3,946** (down from 3,983 in V2).
- **Common Remaining Errors**:
  - Highly generic 1-token business names situated in multi-tenant commercial complexes with ambiguous suite numbering.
  - Records where both name and address underwent simultaneous extreme phonetic corruption and OCR truncation.

---

## 6. Conclusion

We developed an accurate, memory-efficient Entity Resolution pipeline tailored for large-scale multi-country commercial datasets under the precision-heavy macro $F_{0.5}$ metric. By combining geographic isolation, rarest-token-first inverted blocking, multi-regime string similarity scoring with fine-grained tie-breaking, and global 1-to-1 uniqueness constraints, the solution achieves an outstanding validation Macro $F_{0.5}$ score of **0.9367** while running within an 8 GB RAM budget without external data lookups.

---

## Appendix

### A. Code Artefacts
The reproducible codebase is organized under `code/business_entity_resolution/`:
- `src/utils.py`: Text normalization, Unicode/ASCII cleaning, legal suffix removal, address number extraction (`int(n) > 0`), domain stem parsing, true non-Latin script detection.
- `src/blocking.py`: `MultiPassBlocker` implementing multi-pass inverted indexes with rarest-token-first priority candidate generation.
- `src/matching.py`: `EntityMatcher` implementing multi-regime feature scoring and global 1-to-1 uniqueness resolution.
- `src/pipeline.py`: Main CLI entry point orchestrating country-partitioned data loading, blocking, matching, and TSV emission.
- `requirements.txt`: Pinned environment dependencies.
- `README.md`: Step-by-step reproduction instructions.

**Entry point to reproduce output files:**
```bash
python3 code/business_entity_resolution/src/pipeline.py \
    --data-dir dataset/test \
    --output-dir output \
    --matching-out output/matching_results.tsv \
    --candidate-out output/candidate_pairs.tsv
```

### B. Additional Results & Validation Performance
| Configuration / Experiment | Blocking Recall | Candidates / S1 | Precision | Recall | Singleton Acc. | Macro $F_{0.5}$ |
|---|---|---|---|---|---|---|
| Baseline Name Overlap (US only) | 83.8% | 14.2 | 76.2% | 68.5% | 68.4% | 0.7410 |
| Baseline Pipeline (US + India, 10k entities) | 86.13% | 22.4 | 82.67% | 80.34% | 75.92% | 0.8283 |
| V2 Production Model ($\tau = 0.78, \text{cands}=60$) | 93.02% | 33.7 | 87.81% | 88.44% | 77.86% | 0.8835 |
| V3 Recall + Accent Latin Fix | 94.18% | 34.1 | 88.92% | 88.53% | 80.84% | 0.8922 |
| **V4 Final Pipeline (Rarest-Token + Address Strictness + Tie-Break)** | **94.32%** | **34.6** | **97.13%** | **88.64%** | **92.79%** | **0.9367** |
