"""
Business Entity Resolution — Utility and Preprocessing Functions.

Handles text normalization, Unicode cleaning, legal suffix removal,
domain extraction, and tokenization for entity names and addresses across
US, India, and France.
"""

import re
import unicodedata

# Legal suffixes across US, India, and France
LEGAL_SUFFIXES = {
    # US / UK / International
    'inc', 'incorporated', 'corp', 'corporation', 'llc', 'ltd', 'limited', 'pvt', 'private',
    'co', 'company', 'services', 'enterprises', 'associates', 'group', 'holdings', 'solutions',
    'technologies', 'consulting', 'management', 'industries', 'partners', 'agency', 'center',
    'pllc', 'llp', 'pc', 'lp', 'dba', 'corp.', 'inc.', 'llc.', 'ltd.',
    # India
    'private limited', 'pvt ltd', 'pvt. ltd.', 'ltd.', 'llp', 'trust', 'society',
    'enterprises', 'marketing', 'exports', 'infrastructure', 'projects', 'trading',
    # France
    'sarl', 'sasu', 'sas', 'sci', 'eurl', 'sa', 'snc', 'scp', 'gie', 'selarl',
    'fils', 'freres', 'cie', 'societe', 'diffusion', 'distribution', 'international',
    'france', 'groupe'
}

# High-frequency address stopwords (road types, directions, unit types) across US, India, France
COMMON_ADDR_WORDS = {
    # English / US
    'and', 'the', 'of', 'in', 'at', 'on', 'for', 'with', 'by', 'an', 'a', 'to', 'from',
    'st', 'rd', 'dr', 'ave', 'blvd', 'street', 'road', 'drive', 'avenue', 'lane', 'ln',
    'ct', 'court', 'way', 'hwy', 'highway', 'fl', 'floor', 'unit', 'ste', 'suite', 'apt',
    'apartment', 'box', 'po', 'no', 'near', 'opp', 'plot', 'house', 'hno', 'shop', 'behind',
    'north', 'south', 'east', 'west', 'city', 'town', 'village', 'state', 'building', 'bldg',
    'pkwy', 'parkway', 'cir', 'circle', 'ter', 'terrace', 'pl', 'place', 'loop', 'trl', 'trail',
    # India specific
    'nagar', 'marg', 'chowk', 'gali', 'colony', 'enclave', 'vihar', 'bhavan', 'bhawan',
    'complex', 'towers', 'tower', 'sector', 'block', 'phase', 'cross', 'main', 'layout',
    'road', 'dist', 'district', 'taluk', 'post', 'po', 'opp', 'near', 'c/o', 'so',
    # France specific
    'rue', 'r', 'av', 'avenue', 'bd', 'boulevard', 'allée', 'allee', 'chemin', 'route',
    'imp', 'impasse', 'place', 'pl', 'cours', 'passage', 'residence', 'batiment', 'bat',
    'etage', 'porte', 'cedex', 'bis', 'ter', 'quater', 'sur', 'sous', 'des', 'les', 'aux'
}


def clean_ascii(s: str) -> str:
    """Normalize Unicode (NFKD) and convert to lowercase ASCII alphanumeric string."""
    if not s:
        return ''
    s = unicodedata.normalize('NFKD', str(s))
    s = s.encode('ascii', 'ignore').decode('ascii').lower()
    return s


def has_non_latin(s: str) -> bool:
    """Check if string contains truly non-Latin scripts (e.g. Devanagari, Tamil, etc.), excluding accented Latin."""
    if not s:
        return False
    for c in str(s):
        if ord(c) > 127:
            name = unicodedata.name(c, '')
            if not name.startswith('LATIN'):
                return True
    return False


def unleet(s: str) -> str:
    """Normalize leetspeak / OCR digit substitutions in business names."""
    if not s:
        return ''
    tr = str.maketrans('0134578$', 'oieastbs')
    return str(s).translate(tr)


def is_domain(s: str) -> bool:
    """Check if string appears to be a web URL or domain name."""
    if not s:
        return False
    s_lower = str(s).lower()
    return any(d in s_lower for d in ['.com', '.org', '.net', '.in', '.co', '.fr', '.io', 'www.'])


def get_domain_stem(s: str) -> str:
    """Extract clean brand stem from domain name (e.g. www.mybiz.com -> mybiz), stripping trailing legal words."""
    s = clean_ascii(s)
    s = re.sub(r'https?://|www\.', '', s)
    s = re.sub(r'\.(com|org|net|in|co|us|gov|io|fr).*', '', s)
    stem = re.sub(r'[^a-z0-9]', '', s)
    stem_u = unleet(stem)
    for s_cand in [stem_u, stem]:
        for suf in sorted(LEGAL_SUFFIXES, key=len, reverse=True):
            if len(suf) >= 3 and s_cand.endswith(suf):
                return s_cand[:-len(suf)]
    return stem


def get_name_tokens(s: str) -> list[str]:
    """Tokenize entity name, stripping domains, IDs, legal words, and short noise."""
    s_asc = clean_ascii(s)
    s_asc = re.sub(r'https?://\S+|www\.\S+|\.(com|org|net|in|co|us|gov|io|fr)', ' ', s_asc)
    s_asc = re.sub(r'\(id:\s*\d+\)', ' ', s_asc)
    words = re.findall(r'[a-z0-9]+', s_asc)
    return [w for w in words if w not in LEGAL_SUFFIXES and w not in COMMON_ADDR_WORDS and len(w) > 2]


def get_compact_name(s: str) -> str:
    """Concatenate substantive name tokens into a single alphanumeric string."""
    tokens = get_name_tokens(s)
    return ''.join(tokens)


def get_sorted_name(s: str) -> str:
    """Return sorted name tokens joined by space (invariant to word order)."""
    tokens = sorted(get_name_tokens(s))
    return ' '.join(tokens)


def get_addr_numbers(s: str) -> list[str]:
    """Extract and normalize all street / building / shop / flat numbers (strictly > 0)."""
    if not s:
        return []
    nums = re.findall(r'\d+', str(s))
    # Strip leading zeros, require positive number (>0) to avoid 'Fl 0' false matches, <= 7 digits
    return [str(int(n)) for n in nums if len(n) <= 7 and int(n) > 0]


def get_addr_tokens(s: str) -> list[str]:
    """Extract substantive address tokens, excluding common road and location stopwords."""
    s_asc = clean_ascii(s)
    words = re.findall(r'[a-z]+', s_asc)
    return [w for w in words if w not in COMMON_ADDR_WORDS and len(w) > 3]


def precompute_entity(name: str, addr: str) -> tuple:
    """
    Precompute standardized entity features for fast candidate generation and pairwise matching.
    Returns:
      (compact_name, domain_stem, name_tokens_list, addr_numbers_set, addr_tokens_set,
       has_non_latin, is_synthetic_alias, clean_name, clean_addr)
    """
    cname = get_compact_name(name)
    dstem = get_domain_stem(name) if is_domain(name) else ''
    ntoks = get_name_tokens(name)
    nums = set(get_addr_numbers(addr))
    atoks = set(get_addr_tokens(addr))
    non_latin = has_non_latin(name)
    words = str(name).strip().split()
    is_alias = (len(words) == 1 and len(str(name).strip()) >= 6)
    c_name = clean_ascii(name)
    c_addr = clean_ascii(addr)
    return (cname, dstem, ntoks, nums, atoks, non_latin, is_alias, c_name, c_addr)
