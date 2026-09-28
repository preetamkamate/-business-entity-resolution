"""
Business Entity Resolution — Multi-Pass Candidate Generation (Blocking).

Constructs high-recall inverted index structures per country and generates
compact candidate sets for Source 1 entities.
"""

from collections import defaultdict

try:
    from .utils import (
        clean_ascii,
        get_compact_name,
        get_domain_stem,
        get_name_tokens,
        get_sorted_name,
        get_addr_numbers,
        get_addr_tokens,
        is_domain,
        unleet,
    )
except (ImportError, ValueError):
    from utils import (
        clean_ascii,
        get_compact_name,
        get_domain_stem,
        get_name_tokens,
        get_sorted_name,
        get_addr_numbers,
        get_addr_tokens,
        is_domain,
        unleet,
    )


class MultiPassBlocker:
    """Multi-pass blocking engine indexed over target records (Source 2 and Source 3)."""

    def __init__(self, max_token_freq: int = 150, max_cands_per_entity: int = 60):
        self.max_token_freq = max_token_freq
        self.max_cands_per_entity = max_cands_per_entity
        
        # Inverted index tables
        self.compact_name_idx = defaultdict(list)
        self.sorted_name_idx = defaultdict(list)
        self.first_two_idx = defaultdict(list)
        self.rare_name_idx = defaultdict(list)
        self.num_street_idx = defaultdict(list)
        self.token_freq = defaultdict(int)
        self.addr_freq = defaultdict(int)
        self.high_addr_freq = 50

    def build_index(self, targets: list[tuple[str, str, str]]):
        """
        Build multi-pass inverted indexes from a list of target tuples:
        targets: list of (entity_id, business_name, business_address)
        """
        # Step 1: Compute token document frequency to filter out overly frequent terms
        self.token_freq = defaultdict(int)
        self.addr_freq = defaultdict(int)
        for tid, name, addr in targets:
            for t in set(get_name_tokens(name)):
                self.token_freq[t] += 1
            for at in set(get_addr_tokens(addr)):
                self.addr_freq[at] += 1

        # Dynamic frequency threshold for high-frequency city/state/region address tokens (open-set)
        self.high_addr_freq = max(50, int(len(targets) * 0.005))

        # Step 2: Index each target record into multi-pass tables
        for tid, name, addr in targets:
            # Pass 1: Compact Name & Unleeted Compact Name
            cname = get_compact_name(name)
            if len(cname) >= 3:
                self.compact_name_idx[cname].append(tid)
                c_u = unleet(cname)
                if c_u != cname and any(c.isalpha() for c in c_u):
                    self.compact_name_idx[c_u].append(tid)

            # Pass 2: Domain Stem (if name is a website)
            if is_domain(name):
                dstem = get_domain_stem(name)
                if len(dstem) >= 3 and dstem != cname:
                    self.compact_name_idx[dstem].append(tid)
                    d_u = unleet(dstem)
                    if d_u != dstem and any(c.isalpha() for c in d_u):
                        self.compact_name_idx[d_u].append(tid)

            # Pass 3: Sorted Name (transposed words) & First Two Tokens Prefix
            ntoks = get_name_tokens(name)
            if len(ntoks) >= 2:
                sname = ' '.join(sorted(ntoks))
                self.sorted_name_idx[sname].append(tid)
                self.first_two_idx[(ntoks[0], ntoks[1])].append(tid)

            # Pass 4: Rare Name Tokens
            for t in set(ntoks):
                if self.token_freq[t] <= self.max_token_freq:
                    self.rare_name_idx[t].append(tid)

            # Pass 5: Address Number + Street Tokens (excluding high-frequency city/state tokens)
            if addr:
                nums = get_addr_numbers(addr)
                atoks = get_addr_tokens(addr)
                clean_atoks = [at for at in atoks if self.addr_freq[at] <= self.high_addr_freq]
                if nums and clean_atoks:
                    for n in list(nums)[:2]:
                        for at in clean_atoks[:3]:
                            self.num_street_idx[(n, at)].append(tid)

    def generate_candidates(self, s1_records: list[tuple[str, str, str]]) -> dict[str, set[str]]:
        """
        Generate candidate target IDs for a list of Source 1 records:
        s1_records: list of (entity_id, business_name, business_address)
        Returns: {s1_entity_id: set_of_candidate_ids}
        """
        candidates = {}

        for sid, name, addr in s1_records:
            cname = get_compact_name(name)
            dstem = get_domain_stem(name) if is_domain(name) else ''
            ntoks = get_name_tokens(name)
            nums = get_addr_numbers(addr) if addr else []
            atoks = get_addr_tokens(addr) if addr else []
            clean_a1 = [at for at in atoks if self.addr_freq.get(at, 0) <= self.high_addr_freq]

            # Priority Tier 1: Exact compact name & domain stem matches
            p1 = set(self.compact_name_idx.get(cname, []))
            if dstem and dstem in self.compact_name_idx:
                p1.update(self.compact_name_idx[dstem])

            # Priority Tier 2: Address number + street token matches
            p2 = set()
            if nums and clean_a1:
                for n in list(nums)[:2]:
                    for at in clean_a1[:3]:
                        p2.update(self.num_street_idx.get((n, at), []))

            # Priority Tier 3: Sorted name (word order invariance) + First Two Tokens
            p3 = set()
            if len(ntoks) >= 2:
                s_key = ' '.join(sorted(ntoks))
                if s_key in self.sorted_name_idx:
                    p3.update(self.sorted_name_idx[s_key])
                p_key = (ntoks[0], ntoks[1])
                if p_key in self.first_two_idx:
                    p3.update(self.first_two_idx[p_key])

            # Priority Tier-ordered candidate selection up to max_cands_per_entity
            cands = list(p1)
            for c in p2:
                if len(cands) >= self.max_cands_per_entity:
                    break
                if c not in p1:
                    cands.append(c)
            for c in p3:
                if len(cands) >= self.max_cands_per_entity:
                    break
                if c not in p1 and c not in p2:
                    cands.append(c)

            # Priority Tier 4: Rarest name tokens first (avoids flooding by common words)
            sorted_tokens = sorted(ntoks, key=lambda t: self.token_freq.get(t, 0))
            for t in sorted_tokens:
                if t in self.rare_name_idx:
                    for c in self.rare_name_idx[t]:
                        if len(cands) >= self.max_cands_per_entity:
                            break
                        if c not in p1 and c not in p2 and c not in p3 and c not in cands:
                            cands.append(c)
                if len(cands) >= self.max_cands_per_entity:
                    break

            candidates[sid] = set(cands)

        return candidates
