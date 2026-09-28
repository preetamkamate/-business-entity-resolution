"""
Business Entity Resolution — High-Performance Production Pipeline.

Features:
  - Open-Set Dynamic Country Partitioning (handles US, India, France, and any unseen countries).
  - Priority-Tier Candidate Generation (MultiPassBlocker) with open-set address frequency filtering.
  - Precomputed Entity Features (compact names, token sets, numeric sets, locality sets).
  - Precision-Focused Multi-Regime Decision Engine with strict address number conflict rejection.
  - Global 1-to-1 Target Uniqueness Constraint Resolution (argmax scoring assignment).
  - Memory-safe chunked execution (peak RAM < 1.5 GB).
"""

import argparse
import gc
import os
import sys
import time
from collections import defaultdict
import duckdb

try:
    from .utils import (
        clean_ascii,
        get_addr_numbers,
        get_addr_tokens,
        get_compact_name,
        get_domain_stem,
        get_name_tokens,
        has_non_latin,
        is_domain,
        precompute_entity,
    )
    from .blocking import MultiPassBlocker
    from .matching import compute_pair_score, score_pair_precomputed, EntityMatcher
except (ImportError, ValueError):
    from utils import (
        clean_ascii,
        get_addr_numbers,
        get_addr_tokens,
        get_compact_name,
        get_domain_stem,
        get_name_tokens,
        has_non_latin,
        is_domain,
        precompute_entity,
    )
    from blocking import MultiPassBlocker
    from matching import compute_pair_score, score_pair_precomputed, EntityMatcher


def run_pipeline(
    data_dir: str = "dataset/test",
    output_dir: str = "output",
    matching_file: str = "output/matching_results.tsv",
    candidate_file: str = "output/candidate_pairs.tsv",
    max_cands: int = 60,
    threshold: float = 0.78,
):
    total_start = time.time()
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 70)
    print("ML Challenge 2026: Business Entity Resolution Pipeline (Production)")
    print(f"Data directory:      {data_dir}")
    print(f"Matching results:    {matching_file}")
    print(f"Candidate pairs:     {candidate_file}")
    print(f"Score threshold:     {threshold}")
    print(f"Max candidates/S1:   {max_cands}")
    print("=" * 70)

    source1_path = os.path.join(data_dir, "test_source1.tsv")
    source2_path = os.path.join(data_dir, "test_source2.tsv")
    source3_path = os.path.join(data_dir, "test_source3.tsv")

    if not os.path.isfile(source1_path):
        source1_path = os.path.join(data_dir, "train_source1.tsv")
        source2_path = os.path.join(data_dir, "train_source2.tsv")
        source3_path = os.path.join(data_dir, "train_source3.tsv")

    con = duckdb.connect()

    print("\n[Step 1/3] Reading Source 1 entities and partitioning by country (Open-Set)...")
    s1_rows = con.execute(f"""
        SELECT entity_id, business_name, business_address, country
        FROM read_csv('{source1_path}', delim='\t', header=true, all_varchar=true)
    """).fetchall()

    total_s1 = len(s1_rows)
    s1_by_country = defaultdict(list)
    for sid, name, addr, country in s1_rows:
        s1_by_country[country].append((sid, name or '', addr or ''))

    print(f"Total Source 1 entities: {total_s1:,}")
    for country, records in s1_by_country.items():
        print(f"  - {country}: {len(records):,} entities ({len(records)*100/total_s1:.1f}%)")

    temp_matching_files = []
    temp_candidate_files = []

    print("\n[Step 2/3] Processing each country independently (Blocking + Matching)...")
    for country, s1_records in s1_by_country.items():
        c_t0 = time.time()
        print(f"\n--- Processing Country: {country} ({len(s1_records):,} S1 entities) ---")

        safe_country = country.replace("'", "''")
        # Load S2 and S3 targets for this country
        print(f"  Loading Source 2 targets for {country}...")
        s2_country = con.execute(f"""
            SELECT entity_id, business_name, business_address
            FROM read_csv('{source2_path}', delim='\t', header=true, all_varchar=true)
            WHERE country = '{safe_country}'
        """).fetchall()

        print(f"  Loading Source 3 targets for {country}...")
        s3_country = con.execute(f"""
            SELECT entity_id, business_name, business_address
            FROM read_csv('{source3_path}', delim='\t', header=true, all_varchar=true)
            WHERE country = '{safe_country}'
        """).fetchall()

        targets = s2_country + s3_country
        print(f"  Total target records in {country}: {len(targets):,}")

        # Step 2a: Build Inverted Indexes + Target Precomputed Features
        t_idx_start = time.time()
        blocker = MultiPassBlocker(max_token_freq=150, max_cands_per_entity=max_cands)
        blocker.build_index(targets)

        target_feats = {}
        for tid, name, addr in targets:
            target_feats[tid] = precompute_entity(name, addr)

        print(f"  Indexed {len(targets):,} targets in {time.time()-t_idx_start:.2f}s")

        # Step 2b: Precompute S1 Features
        t_s1_start = time.time()
        s1_feats = {sid: precompute_entity(name, addr) for sid, name, addr in s1_records}
        print(f"  Precomputed {len(s1_feats):,} S1 entities in {time.time()-t_s1_start:.2f}s")

        # Step 2c: Candidate Generation (Priority-Tiered)
        t_cand_start = time.time()
        candidates_per_s1 = blocker.generate_candidates(s1_records)
        print(f"  Generated candidate sets in {time.time()-t_cand_start:.2f}s")

        # Step 2d: Pairwise Precision Scoring & 1-to-1 Global Assignment
        t_match_start = time.time()
        target_best = {}  # target_id -> (score, s1_id)

        for sid, cands in candidates_per_s1.items():
            s1_feat = s1_feats[sid]
            for tid in cands:
                tg_feat = target_feats[tid]
                score = score_pair_precomputed(s1_feat, tg_feat)
                if score >= threshold:
                    if tid not in target_best or score > target_best[tid][0]:
                        target_best[tid] = (score, sid)

        # Invert target_best to get final matched entities for each S1 entity
        matches_per_s1 = defaultdict(list)
        for tid, (score, sid) in target_best.items():
            matches_per_s1[sid].append(tid)

        print(f"  Scored & Matched {country} in {time.time()-t_match_start:.2f}s: {len(matches_per_s1):,}/{len(s1_records):,} matched.")

        # Write country results to temporary files on disk
        c_match_tmp = os.path.join(output_dir, f"matching_{country}.tmp")
        c_cand_tmp = os.path.join(output_dir, f"candidate_{country}.tmp")

        with open(c_match_tmp, "w", encoding="utf-8") as f_m, open(c_cand_tmp, "w", encoding="utf-8") as f_c:
            for sid, _, _ in s1_records:
                matched_list = sorted(matches_per_s1.get(sid, []))
                cands_set = candidates_per_s1.get(sid, set()) | set(matched_list)
                cand_list = sorted(cands_set)

                f_m.write(f"{sid}\t{','.join(matched_list)}\n")
                f_c.write(f"{sid}\t{','.join(cand_list)}\n")

        temp_matching_files.append(c_match_tmp)
        temp_candidate_files.append(c_cand_tmp)

        # Explicit cleanup to release RAM
        del blocker, target_feats, s1_feats
        del targets, s2_country, s3_country, candidates_per_s1, target_best, matches_per_s1
        gc.collect()

    # Step 3: Merge all country files into final submission files
    print("\n[Step 3/3] Merging country outputs into final submission TSVs...")

    with open(matching_file, "w", encoding="utf-8") as f_out:
        f_out.write("source1_entity_id\tmatched_entity_ids\n")
        for tmp_file in temp_matching_files:
            with open(tmp_file, "r", encoding="utf-8") as f_in:
                for line in f_in:
                    f_out.write(line)
            os.remove(tmp_file)

    with open(candidate_file, "w", encoding="utf-8") as f_out:
        f_out.write("source1_entity_id\tcandidate_entity_ids\n")
        for tmp_file in temp_candidate_files:
            with open(tmp_file, "r", encoding="utf-8") as f_in:
                for line in f_in:
                    f_out.write(line)
            os.remove(tmp_file)

    total_elapsed = time.time() - total_start
    print(f"\nAll operations completed successfully in {total_elapsed/60:.2f} minutes ({total_elapsed:.1f}s)!")
    print(f"Generated output files:")
    print(f"  1. {matching_file} ({os.path.getsize(matching_file)/(1024**2):.1f} MB)")
    print(f"  2. {candidate_file} ({os.path.getsize(candidate_file)/(1024**2):.1f} MB)")


def main():
    parser = argparse.ArgumentParser(description="Run Entity Resolution Pipeline.")
    parser.add_argument("--data-dir", "-d", default="dataset/test", help="Path to dataset directory.")
    parser.add_argument("--output-dir", "-o", default="output", help="Output directory.")
    parser.add_argument("--matching-out", "-m", default="output/matching_results.tsv", help="Matching TSV output path.")
    parser.add_argument("--candidate-out", "-c", default="output/candidate_pairs.tsv", help="Candidate TSV output path.")
    parser.add_argument("--max-cands", type=int, default=60, help="Max candidates per entity.")
    parser.add_argument("--threshold", type=float, default=0.78, help="Score threshold for matching.")

    args = parser.parse_args()
    run_pipeline(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        matching_file=args.matching_out,
        candidate_file=args.candidate_out,
        max_cands=args.max_cands,
        threshold=args.threshold,
    )


if __name__ == "__main__":
    main()
