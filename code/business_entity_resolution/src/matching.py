"""
Business Entity Resolution — High-Precision Matching Engine.

Computes string similarity features, evaluates multi-regime matching rules,
and resolves global target 1-to-1 uniqueness constraints to maximize Macro F_0.5.
"""

from collections import defaultdict
from rapidfuzz import fuzz

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
    )
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
    )



def compute_pair_score(s1_name: str, s1_addr: str, tg_name: str, tg_addr: str) -> float:
    """
    Compute a continuous match confidence score in [0.0, 1.0] for an (S1, Target) candidate pair.
    """
    c1 = get_compact_name(s1_name)
    c2 = get_compact_name(tg_name)
    exact_cname = (c1 == c2 and len(c1) >= 3)
    domain_match = False

    if is_domain(tg_name):
        dstem = get_domain_stem(tg_name)
        if dstem and len(dstem) >= 4:
            if dstem == c1 or (len(c1) >= 4 and (dstem in c1 or c1 in dstem)):
                domain_match = True

    n1 = set(get_name_tokens(s1_name))
    n2 = set(get_name_tokens(tg_name))

    nums1 = set(get_addr_numbers(s1_addr)) if s1_addr else set()
    nums2 = set(get_addr_numbers(tg_addr)) if tg_addr else set()
    shared_nums = nums1 & nums2
    has_shared_nums = bool(shared_nums)

    a1 = set(get_addr_tokens(s1_addr)) if s1_addr else set()
    a2 = set(get_addr_tokens(tg_addr)) if tg_addr else set()
    shared_atoks = a1 & a2
    has_shared_atoks = bool(shared_atoks)

    clean_n1 = clean_ascii(s1_name)
    clean_n2 = clean_ascii(tg_name)
    clean_a1 = clean_ascii(s1_addr)
    clean_a2 = clean_ascii(tg_addr)

    if s1_addr and tg_addr:
        diff_numbers = bool(nums1 and nums2 and not shared_nums)
        addr_fuzz = fuzz.token_sort_ratio(clean_a1, clean_a2) / 100.0
        name_fuzz = fuzz.token_sort_ratio(clean_n1, clean_n2) / 100.0
        tb = 0.005 * addr_fuzz + 0.005 * name_fuzz

        # 1. Exact Compact Name Match or Domain Match
        if exact_cname or domain_match:
            if has_shared_nums and has_shared_atoks:
                return 0.98 + tb
            if has_shared_nums and (has_shared_atoks or addr_fuzz >= 0.50):
                return 0.94 + tb
            if len(shared_atoks) >= 2:
                return 0.93 + tb
            if has_shared_atoks and addr_fuzz >= 0.55:
                return 0.91 + tb
            if not diff_numbers and addr_fuzz >= 0.65:
                return 0.89 + tb
            # Contradictory street address -> reject
            return 0.20

        # 2. Synthetic Fantasy Alias Match
        words2 = str(tg_name).strip().split()
        is_alias2 = (len(words2) == 1 and len(str(tg_name).strip()) >= 6)
        if is_alias2 and has_shared_nums and (has_shared_atoks or len(shared_nums) >= 2):
            if addr_fuzz >= 0.70:
                return 0.89 + tb

        # 3. True Non-Latin Transliteration Match (Hindi, Tamil, etc.)
        if has_non_latin(tg_name) and has_shared_nums and addr_fuzz >= 0.70:
            return 0.91 + tb

        # 4. Multi-Field Name + Address Match
        shared_name = n1 & n2
        name_jaccard = len(shared_name) / max(1, len(n1 | n2))
        cname_sub = (len(c1) >= 4 and len(c2) >= 4 and (c1 in c2 or c2 in c1))

        if has_shared_nums and has_shared_atoks:
            if name_jaccard >= 0.40 or cname_sub:
                return 0.89 + tb
            if name_fuzz >= 0.70 and (len(shared_name) >= 1 or addr_fuzz >= 0.60):
                return 0.84 + tb

        if not diff_numbers:
            if name_jaccard >= 0.75 and (has_shared_nums or has_shared_atoks):
                return 0.89 + tb
            if name_fuzz >= 0.85 and (has_shared_nums or has_shared_atoks):
                return 0.89 + tb
            if name_fuzz >= 0.90 and addr_fuzz >= 0.50:
                return 0.87 + tb

        return 0.0

    else:
        name_fuzz = fuzz.token_sort_ratio(clean_n1, clean_n2) / 100.0
        tb = 0.005 * name_fuzz
        if (len(n1) >= 2 or len(c1) >= 10) and (exact_cname or domain_match):
            return 0.94 + tb
        if len(n1) >= 2:
            shared_name = n1 & n2
            name_jaccard = len(shared_name) / max(1, len(n1 | n2))
            if name_jaccard >= 0.80:
                return 0.89 + tb
            if name_fuzz >= 0.90:
                return 0.89 + tb
        return 0.0


def score_pair_precomputed(s1_feat: tuple, tg_feat: tuple) -> float:
    """
    High-speed scoring engine using precomputed entity feature tuples:
    s1_feat: (cname, dstem, ntoks, nums, atoks, non_latin, is_alias, c_name, c_addr)
    tg_feat: (cname, dstem, ntoks, nums, atoks, non_latin, is_alias, c_name, c_addr)
    """
    c1, d1, n1, nums1, a1, non_latin1, is_alias1, name1, addr1 = s1_feat
    c2, d2, n2, nums2, a2, non_latin2, is_alias2, name2, addr2 = tg_feat

    exact_cname = (c1 == c2 and len(c1) >= 3)
    domain_match = bool(d2 and len(d2) >= 4 and (d2 == c1 or (len(c1) >= 4 and (d2 in c1 or c1 in d2))))
    shared_nums = nums1 & nums2
    shared_atoks = a1 & a2
    has_shared_nums = bool(shared_nums)
    has_shared_atoks = bool(shared_atoks)

    if addr1 and addr2:
        diff_numbers = bool(nums1 and nums2 and not shared_nums)
        addr_fuzz = fuzz.token_sort_ratio(addr1, addr2) / 100.0
        name_fuzz = fuzz.token_sort_ratio(name1, name2) / 100.0
        tb = 0.005 * addr_fuzz + 0.005 * name_fuzz

        # 1. Exact Compact Name Match or Domain Match
        if exact_cname or domain_match:
            if has_shared_nums and has_shared_atoks:
                return 0.98 + tb
            if has_shared_nums and (has_shared_atoks or addr_fuzz >= 0.50):
                return 0.94 + tb
            if len(shared_atoks) >= 2:
                return 0.93 + tb
            if has_shared_atoks and addr_fuzz >= 0.55:
                return 0.91 + tb
            if not diff_numbers and addr_fuzz >= 0.65:
                return 0.89 + tb
            # Contradictory street address -> reject
            return 0.20

        # 2. Synthetic Fantasy Alias Match
        if is_alias2 and has_shared_nums and (has_shared_atoks or len(shared_nums) >= 2):
            if addr_fuzz >= 0.70:
                return 0.89 + tb

        # 3. True Non-Latin Transliteration Match (Hindi, Tamil, etc.)
        if non_latin2 and has_shared_nums and addr_fuzz >= 0.70:
            return 0.91 + tb

        # 4. Multi-Field Name + Address Match
        shared_name = set(n1) & set(n2)
        name_jaccard = len(shared_name) / max(1, len(set(n1) | set(n2)))
        cname_sub = (len(c1) >= 4 and len(c2) >= 4 and (c1 in c2 or c2 in c1))

        if has_shared_nums and has_shared_atoks:
            if name_jaccard >= 0.40 or cname_sub:
                return 0.89 + tb
            if name_fuzz >= 0.70 and (len(shared_name) >= 1 or addr_fuzz >= 0.60):
                return 0.84 + tb

        if not diff_numbers:
            if name_jaccard >= 0.75 and (has_shared_nums or has_shared_atoks):
                return 0.89 + tb
            name_fuzz = fuzz.token_sort_ratio(name1, name2) / 100.0
            if name_fuzz >= 0.85 and (has_shared_nums or has_shared_atoks):
                return 0.89 + tb
            if name_fuzz >= 0.90 and addr_fuzz >= 0.50:
                return 0.87 + tb

        return 0.0

    else:
        name_fuzz = fuzz.token_sort_ratio(name1, name2) / 100.0
        tb = 0.005 * name_fuzz
        if (len(n1) >= 2 or len(c1) >= 10) and (exact_cname or domain_match):
            return 0.94 + tb
        if len(n1) >= 2:
            shared_name = set(n1) & set(n2)
            name_jaccard = len(shared_name) / max(1, len(set(n1) | set(n2)))
            if name_jaccard >= 0.80:
                return 0.89 + tb
            if name_fuzz >= 0.90:
                return 0.89 + tb
        return 0.0


class EntityMatcher:
    """High-precision entity matcher with thresholding and global uniqueness resolution."""

    def __init__(self, score_threshold: float = 0.78):
        self.score_threshold = score_threshold

    def match_candidates(
        self,
        s1_dict: dict[str, tuple[str, str]],
        target_dict: dict[str, tuple[str, str]],
        candidates_map: dict[str, set[str]],
    ) -> dict[str, set[str]]:
        """
        Score all candidate pairs and resolve 1-to-1 target uniqueness:
        s1_dict: {s1_id: (name, address)}
        target_dict: {target_id: (name, address)}
        candidates_map: {s1_id: set_of_candidate_ids}
        Returns: {s1_id: set_of_matched_target_ids}
        """
        target_best = {}  # target_id -> (best_score, best_s1_id)

        for s1_id, cands in candidates_map.items():
            s1_name, s1_addr = s1_dict[s1_id]
            for tid in cands:
                if tid not in target_dict:
                    continue
                tg_name, tg_addr = target_dict[tid]
                score = compute_pair_score(s1_name, s1_addr, tg_name, tg_addr)
                if score >= self.score_threshold:
                    if tid not in target_best or score > target_best[tid][0]:
                        target_best[tid] = (score, s1_id)

        # Build final predictions ensuring 1-to-1 uniqueness
        predictions = {s1_id: set() for s1_id in s1_dict}
        for tid, (score, s1_id) in target_best.items():
            predictions[s1_id].add(tid)

        return predictions
