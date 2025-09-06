"""
data_generator.py
===================

This script produces a synthetic dataset for fine‑tuning a language model to
generate brandable domain names from short business descriptions.  The data
structure aligns with the instruction/response format required by the
homework assignment.  Each record contains an `instruction` field, an
`input` field with a natural‑language business description, and an
`output` dictionary containing up to three domain suggestions with
confidence scores and a status.  Requests containing disallowed or
inappropriate content (e.g. adult themes, violence, illegal activities)
are blocked and return an empty list of suggestions.

Key features:

* **Category‑driven vocabulary** – Common business categories
  (food, coffee, tech, health, legal, education, ecommerce) each supply
  tailored adjectives, business types, keywords and recommended TLDs.
  These lists are derived from the provided notebook and can easily be
  extended with additional categories or vocabularies.
* **Multiple generation algorithms** – Several simple algorithms generate
  base domain strings from business keywords: straightforward keyword
  concatenation, suffix addition and portmanteaus.  A random algorithm
  is chosen for each suggestion to promote variety.
* **Heuristic confidence scoring** – To avoid the heavy weight of
  embedding‑based scores, a lightweight heuristic assigns higher
  confidence to shorter, cleaner names that include relevant keywords and
  common TLDs.  Names containing digits or hyphens are penalised.
* **Safety handling** – The generator checks each description against
  a list of banned keywords.  If any banned term is found, the record
  is marked as `blocked` and no domain suggestions are returned.

Usage example (from a terminal):

```
python data_generator.py --num-records 500 --output-file dataset.jsonl
```

This will produce a JSONL file with 500 synthetic instruction/response
pairs.  The top‑level arguments allow control over the number of records
and output path.  By default, a small dataset of 200 records is
generated and printed to stdout if no output file is specified.
"""

import argparse
import json
import random
import re
import unicodedata
from typing import List, Dict, Tuple

# -----------------------------------------------------------------------------
# Configuration: domain categories, top‑level domains, locations and banned terms
# -----------------------------------------------------------------------------

# Domain extension groups mapped to lists of extensions.  These were
# extracted from the provided notebook.  You can extend or modify these
# groups to support additional TLDs.
TLD_GROUPS: Dict[str, List[str]] = {
    "common": [".com", ".net", ".org"],
    "tech": [".ai", ".tech", ".dev"],
    "ecommerce": [".store", ".shop", ".co"],
    "creative": [".studio", ".design", ".media"],
    "health": [".health", ".care", ".clinic"],
    "legal": [".legal", ".law"],
    "education": [".academy", ".edu"],
    "food": [".kitchen", ".menu"],
    "coffee": [".cafe", ".coffee"],

    # New extension groups for additional categories
    "travel": [".travel", ".vacations", ".tour"],
    "finance": [".finance", ".capital", ".fund"],
    "entertainment": [".tv", ".media", ".stream"],
    "sports": [".fitness", ".sport", ".team"],
    "manufacturing": [".engineering", ".factory", ".industry"],
}

# Categories with adjectives, business types, keywords and extension keys.  The
# keywords lists should favour nouns and descriptive terms relevant to the
# category.  Extension keys map into `TLD_GROUPS` to recommend suitable TLDs.
DOMAIN_CATEGORIES: Dict[str, Dict[str, List[str]]] = {
    "food": {
        "adjectives": ["organic", "local", "vegan", "healthy", "homemade"],
        "business_types": ["meal service", "snack bar", "vegan diner", "organic eatery"],
        "keywords": [
            "organic", "local", "meal", "plate", "vegan", "snack", "deli", "healthy",
            "prep", "bites", "grill", "menu", "dish", "eat", "serve", "flavor",
            "kitchen", "taste", "market", "chef"
        ],
        "extension_keys": ["common", "food"],
    },
    "coffee": {
        "adjectives": ["cozy", "artisan", "independent", "sustainable", "premium"],
        "business_types": ["coffee shop", "espresso bar", "cafe", "roastery"],
        "keywords": [
            "beans", "brew", "cafe", "grind", "latte", "espresso", "blend",
            "java", "mug", "steam", "barista", "pour", "aroma", "cup", "beanery",
            "drip", "grounds", "sips", "filter", "darkroast"
        ],
        "extension_keys": ["common", "coffee"],
    },
    "tech": {
        "adjectives": ["innovative", "cloud-based", "scalable", "cutting-edge", "intelligent"],
        "business_types": ["AI SaaS startup", "blockchain platform", "mobile app developer"],
        "keywords": [
            "cloud", "bot", "data", "ai", "stack", "code", "dev", "logic", "compute",
            "neural", "node", "cyber", "script", "deploy", "stream", "tensor", "model",
            "byte", "core", "matrix"
        ],
        "extension_keys": ["common", "tech"],
    },
    "health": {
        "adjectives": ["holistic", "mindful", "therapeutic", "gentle", "wellness-focused"],
        "business_types": ["yoga studio", "nutritionist", "wellness center"],
        "keywords": [
            "calm", "vital", "fit", "care", "wellness", "med", "yoga", "therapy", "balance",
            "relax", "mind", "body", "clinic", "heal", "energy", "breathe", "flow", "zen",
            "pulse", "restore"
        ],
        "extension_keys": ["common", "health"],
    },
    "legal": {
        "adjectives": ["trusted", "professional", "experienced", "reputable", "compliant"],
        "business_types": ["law firm", "legal consultancy", "compliance office"],
        "keywords": [
            "legal", "justice", "firm", "counsel", "law", "brief", "case", "court",
            "advocate", "barrister", "defense", "claim", "compliance", "rights", "ruling",
            "precedent", "trial", "witness", "verdict", "legaltech"
        ],
        "extension_keys": ["common", "legal"],
    },
    "education": {
        "adjectives": ["interactive", "accessible", "global", "self-paced", "innovative"],
        "business_types": ["online academy", "language school", "tutoring platform"],
        "keywords": [
            "learn", "teach", "edu", "academy", "school", "class", "study", "brain",
            "train", "instruct", "mentor", "pupil", "professor", "course", "lesson",
            "homework", "quiz", "read", "skills", "curriculum"
        ],
        "extension_keys": ["common", "education"],
    },
    "ecommerce": {
        "adjectives": ["sustainable", "ethical", "minimalist", "convenient", "affordable"],
        "business_types": ["online shop", "eco product store", "fashion boutique"],
        "keywords": [
            "shop", "cart", "store", "buy", "eco", "green", "market", "checkout",
            "sale", "goods", "pack", "brand", "vendor", "retail", "product",
            "online", "fashion", "style", "deal", "order"
        ],
        "extension_keys": ["common", "ecommerce"],
    },

    # Additional categories to broaden the dataset coverage
    "travel": {
        "adjectives": ["adventurous", "luxury", "budget-friendly", "exotic", "family"],
        "business_types": ["travel agency", "tour operator", "hostel", "vacation planner"],
        "keywords": [
            "travel", "tour", "stay", "trip", "voyage", "book", "journey",
            "hotel", "vacation", "flight", "tourism", "holiday", "cruise", "wander"
        ],
        "extension_keys": ["common", "travel"],
    },
    "finance": {
        "adjectives": ["secure", "transparent", "global", "dynamic", "trusted"],
        "business_types": ["investment firm", "digital bank", "credit union", "loan service"],
        "keywords": [
            "invest", "fund", "money", "bank", "credit", "loan", "wealth",
            "capital", "save", "finance", "funding", "broker", "trade", "stock"
        ],
        "extension_keys": ["common", "finance"],
    },
    "entertainment": {
        "adjectives": ["exciting", "streaming", "creative", "dynamic", "interactive"],
        "business_types": ["video streaming platform", "music label", "film studio", "gaming network"],
        "keywords": [
            "stream", "video", "music", "show", "play", "film", "movie",
            "media", "game", "cast", "live", "fun", "view", "watch", "listen"
        ],
        "extension_keys": ["common", "entertainment"],
    },
    "sports": {
        "adjectives": ["competitive", "athletic", "dynamic", "outdoor", "elite"],
        "business_types": ["fitness center", "sports equipment shop", "athletic club", "online sports magazine"],
        "keywords": [
            "sport", "fit", "gym", "team", "goal", "train", "athlete",
            "league", "coach", "match", "gear", "game", "tournament", "playoff", "run"
        ],
        "extension_keys": ["common", "sports"],
    },
    "manufacturing": {
        "adjectives": ["industrial", "efficient", "innovative", "sustainable", "precision"],
        "business_types": ["factory", "production company", "machinery supplier", "3D printing service"],
        "keywords": [
            "factory", "machine", "industrial", "assembly", "fabricate", "tool",
            "process", "mechanic", "equipment", "production", "plant", "manufacture",
            "gear", "automation", "robotics", "forge", "workshop"
        ],
        "extension_keys": ["common", "manufacturing"],
    },
}

# Locations and sub‑locations used to add geographic variety to descriptions.
LOCATIONS: List[str] = [
    "New York", "Austin", "San Francisco", "Los Angeles", "Berlin",
    "Beirut", "Brussels", "Montreal", "Paris"
]

SUB_LOCATIONS: List[str] = [
    "downtown", "midtown", "uptown", "old town", "main street",
    "the suburbs", "the business district"
]

# Words or phrases that indicate disallowed content.  If a generated
# description contains any of these (case‑insensitive), the request is
# rejected.  This list is by no means exhaustive; you should add
# additional terms depending on your safety requirements.
BANNED_TERMS: List[str] = [
    "adult", "porn", "sex", "violence", "violent", "weapon", "hate",
    "racist", "illegal", "scam", "fraud", "gambling", "drugs", "weapon"
]

# Relative weights for TLDs used in the heuristic scoring.  More trusted or
# widely recognised TLDs carry a small bonus.  Any TLD not listed
# implicitly receives a weight of 0.0.
TLD_WEIGHTS: Dict[str, float] = {
    "com": 0.05,
    "org": 0.03,
    "net": 0.02,
    "io": 0.03,
    "ai": 0.03,
    "tech": 0.02,
    "dev": 0.02,
    "store": 0.01,
    "shop": 0.01,
    "co": 0.01,
    "travel": 0.02,
    "vacations": 0.02,
    "tour": 0.02,
    "finance": 0.02,
    "capital": 0.02,
    "fund": 0.02,
    "tv": 0.02,
    "media": 0.02,
    "stream": 0.02,
    "fitness": 0.02,
    "sport": 0.02,
    "team": 0.02,
    "engineering": 0.02,
    "factory": 0.02,
    "industry": 0.02,
}

# Groups of categories considered similar for scoring purposes.  If a domain
# suggestion is generated using keywords from a category in the same group as
# the current category, a milder penalty is applied when no keyword match
# occurs.  Categories not listed together are considered dissimilar.
SIMILAR_GROUPS = [
    {"food", "coffee"},
    {"tech", "finance"},
    {"health", "sports"},
    {"entertainment", "travel"},
    # Other categories do not have explicit similarities and are treated as
    # standalone (e.g., manufacturing, education).  You can extend this list
    # with additional groupings if necessary.
]

# Suffixes used by some algorithms to create more memorable names.  These
# terms are appended to a base keyword to form the domain string before
# adding the TLD.
SUFFIXES: List[str] = [
    "hub", "lab", "spot", "world", "zone", "space", "center", "land",
    "point", "works", "shop", "mart", "services", "solutions", "factory"
]

# Numeric suffixes that can be appended to create names containing digits.  These
# suffixes represent common themes (e.g., 24-hour availability, 365 days,
# educational basics like 101).  They are only used when the algorithm
# ``numeric`` is selected.
NUMERIC_SUFFIXES: List[str] = [
    "24", "365", "247", "360", "101", "1", "7", "24x7"
]

# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------

def transliterate(text: str) -> str:
    """Remove accents/diacritics from a string and return ASCII‑only text."""
    normalized = unicodedata.normalize('NFD', text)
    return ''.join(c for c in normalized if unicodedata.category(c) != 'Mn')


def clean_keyword(word: str) -> str:
    """Normalize a keyword: lowercase, transliterate and strip non‑alphanumeric chars."""
    word = transliterate(word)
    word = word.lower()
    # Remove any character that is not a letter or digit
    word = re.sub(r'[^a-z0-9]', '', word)
    return word


def assemble_keywords(keywords: List[str], n: int) -> List[str]:
    """Pick up to n keywords randomly and clean them for domain assembly."""
    if not keywords:
        return []
    selected = random.sample(keywords, k=min(n, len(keywords)))
    return [clean_keyword(k) for k in selected if k]


def generate_base_keyword(
    keywords: List[str],
    algorithm: str,
    hyphen_prob: float = 0.3,
) -> Tuple[str, List[str]]:
    """
    Generate a base string for the domain using one of several algorithms.

    Returns a tuple of the generated base string and the list of keywords
    used to create it.  This information is useful for the heuristic
    scoring (e.g. recognising portmanteau words).

    * ``keyword_concat``: concatenate between 1 and 4 cleaned keywords.
      Occasionally join them with a hyphen to create longer domains.  The
      returned keyword list contains the keywords used.

    * ``suffix``: take a single cleaned keyword and append a random
      suffix from ``SUFFIXES``.  Only the keyword portion is returned in
      the list of used keywords.

    * ``portmanteau``: merge two keywords by slicing each roughly in
      half.  Both original keywords are returned in the list of used
      keywords so the scoring function can identify that the result is
      derived from them.

    If an unrecognised algorithm is passed, the function falls back to
    simple concatenation.
    """
    # Ensure there are keywords to choose from
    if not keywords:
        return "", []
    if algorithm == 'keyword_concat':
        num = random.randint(1, 4)
        parts = assemble_keywords(keywords, num)
        # Optionally insert a hyphen between two parts to form a longer name.
        if len(parts) > 1 and random.random() < hyphen_prob:
            base = parts[0] + '-' + ''.join(parts[1:])
        else:
            base = ''.join(parts)
        return base, parts
    elif algorithm == 'suffix':
        parts = assemble_keywords(keywords, 1)
        suffix = clean_keyword(random.choice(SUFFIXES))
        base = (parts[0] + suffix) if parts else suffix
        return base, parts
    elif algorithm == 'portmanteau':
        parts = assemble_keywords(keywords, 2)
        if len(parts) < 2:
            # Fallback to concatenation
            base = ''.join(parts)
            return base, parts
        split_point1 = max(1, len(parts[0]) // 2)
        split_point2 = max(1, len(parts[1]) // 2)
        base = parts[0][:split_point1] + parts[1][split_point2:]
        return base, parts
    else:
        # New algorithm: numeric – append a numeric suffix to generate names with digits
        if algorithm == 'numeric':
            parts = assemble_keywords(keywords, random.randint(1, 2))
            suffix = random.choice(NUMERIC_SUFFIXES)
            base = ''.join(parts) + suffix
            return base, parts
        # Fallback: simple concatenation
        parts = assemble_keywords(keywords, random.randint(1, 3))
        base = ''.join(parts)
        return base, parts


def choose_extension(extension_keys: List[str]) -> str:
    """
    Select a random top‑level domain from the provided extension keys.  Each
    key references a list of TLDs in TLD_GROUPS.  If no valid TLDs are
    found, default to ".com".
    """
    candidates: List[str] = []
    for key in extension_keys:
        candidates.extend(TLD_GROUPS.get(key, []))
    if not candidates:
        return ".com"
    return random.choice(candidates)


def heuristic_confidence(
    domain: str,
    expected_keywords: List[str],
    used_keywords: List[str],
    current_category: str,
    used_category: str,
) -> float:
    """
    Compute a confidence score for a domain name using the original heuristic
    and a similarity rule.

    * Baseline: start at 0.7.
    * Keyword matches: add 0.1 per match between ``used_keywords`` and
      ``expected_keywords`` (up to 0.3).  If there are no matches,
      apply a penalty based on category similarity: −0.25 if
      ``used_category`` is considered similar to ``current_category``, and
      −0.5 otherwise.
    * TLD weighting: add the weight associated with the domain’s extension
      from ``TLD_WEIGHTS``.
    * Length penalty: subtract 0.05 for bases 16–25 characters, or 0.1 for
      bases longer than 25 characters.
    * Digit penalty: subtract 0.2 if any digit appears in the base.
    * Hyphen penalty: subtract 0.1 if the base contains a hyphen.
    * Repetition penalty: subtract 0.05 if a character repeats three or
      more times consecutively.

    The score is clamped to the [0, 1] range and rounded to two decimals.
    """
    base, _, ext = domain.rpartition('.')
    score = 0.7
    # Compute keyword matches
    matches = 0
    if expected_keywords and used_keywords:
        expected_set = set(expected_keywords)
        for kw in used_keywords:
            if kw in expected_set:
                matches += 1
    if matches > 0:
        score += min(matches, 3) * 0.1
    else:
        # Apply penalty based on category similarity
        penalty = -0.5
        if current_category != used_category:
            for group in SIMILAR_GROUPS:
                if current_category in group and used_category in group:
                    penalty = -0.25
                    break
        score += penalty
    # TLD weighting
    score += TLD_WEIGHTS.get(ext, 0.0)
    # Length penalty
    base_len = len(base)
    if 16 <= base_len <= 25:
        score -= 0.05
    elif base_len > 25:
        score -= 0.1
    # Digit penalty
    if any(ch.isdigit() for ch in base):
        score -= 0.2
    # Hyphen penalty
    if '-' in base:
        score -= 0.1
    # Repetition penalty
    if re.search(r'(.)\1\1', base):
        score -= 0.05
    # Clamp and round
    score = max(0.0, min(score, 1.0))
    return round(score, 2)


def contains_banned_term(description: str) -> bool:
    """Check whether a description contains any banned term."""
    lowered = description.lower()
    for term in BANNED_TERMS:
        if term in lowered:
            return True
    return False


def generate_description(
    category: str,
    config: Dict[str, List[str]],
    mission_prob: float = 0.3,
    target_prob: float = 0.3,
) -> str:
    """
    Build a business description using an adjective, a business type and an
    optional location.  Location may include a city and/or a sub‑location
    selected at random.  Accents are preserved here; transliteration occurs
    later when assembling the domain.
    """
    adjective = random.choice(config["adjectives"])
    business_type = random.choice(config["business_types"])
    # Randomly decide whether to include a specific sub‑location
    location = random.choice(LOCATIONS)
    if random.random() < 0.5:
        # 50% chance to append a sub‑location
        sub_location = random.choice(SUB_LOCATIONS)
        location = f"{sub_location} {location}"
    description = f"A {adjective} {business_type} based in {location}"
    # Optionally append a mission or target audience phrase to increase complexity
    # These phrases add realism by simulating multi‑sentence descriptions.
    mission_phrases = [
        "We aim to provide high-quality services",
        "Our mission is to serve the community",
        "Dedicated to sustainability and innovation",
        "Empowering customers through technology",
        "Committed to exceptional customer experiences",
    ]
    target_phrases = [
        "for busy professionals",
        "for families",
        "for students",
        "for entrepreneurs",
        "for the local community",
    ]
    # mission_prob chance to add a mission phrase
    if random.random() < mission_prob:
        description += f". {random.choice(mission_phrases)}"
    # target_prob chance to add a target audience phrase
    if random.random() < target_prob:
        description += f" {random.choice(target_phrases)}."
    return description


def generate_suggestions(
    config: Dict[str, List[str]],
    expected_keywords: List[str],
    current_category: str,
    num_suggestions: int = 3,
    low_score_chance: float = 0.3,
    hyphen_prob: float = 0.3,
) -> Tuple[List[Dict[str, float]], List[str]]:
    """
    Produce a list of domain suggestions along with the bases used.  Each
    suggestion is generated using one of several algorithms.  With
    probability `low_score_chance`, a suggestion will be intentionally
    mismatched: its keywords and extensions are drawn from a different
    category to create low relevance and thus a lower confidence score.

    Args:
        config: configuration for the current category.
        expected_keywords: cleaned keywords associated with the current
            category.  These are used by the heuristic to gauge relevance.
        num_suggestions: number of domain suggestions to generate.
        low_score_chance: probability that a suggestion will use keywords
            and TLDs from another category.

    Returns:
        A tuple of (suggestions, used_keywords) where suggestions is a list
        of dictionaries with `domain` and `confidence`, and used_keywords
        records the base strings generated for each suggestion.
    """
    suggestions: List[Dict[str, float]] = []
    used_keywords: List[str] = []
    algorithms = ['keyword_concat', 'suffix', 'portmanteau', 'numeric']
    # Clean current category's keywords for reuse when generating matched names
    own_keywords = [clean_keyword(k) for k in config["keywords"]]
    own_extensions = config["extension_keys"]
    categories = list(DOMAIN_CATEGORIES.keys())
    for _ in range(num_suggestions):
        # Decide whether to deliberately produce a low‑score suggestion
        if random.random() < low_score_chance:
            # Mismatched: draw keywords and extension keys from a random other category
            other_categories = [c for c in categories if c != current_category]
            used_category = random.choice(other_categories)
            other_config = DOMAIN_CATEGORIES[used_category]
            keywords_source = [clean_keyword(k) for k in other_config["keywords"]]
            ext_keys = other_config["extension_keys"]
        else:
            used_category = current_category
            keywords_source = own_keywords
            ext_keys = own_extensions
        # Choose an algorithm and build the base name along with the keywords used
        alg = random.choice(algorithms)
        base, used_kw = generate_base_keyword(keywords_source, alg, hyphen_prob=hyphen_prob)
        tld = choose_extension(ext_keys)
        domain = base + tld
        # Use the expected keywords from the original category for relevance
        # Compute confidence using the expected keywords and the actual keywords selected,
        # and pass along the current and used category names for similarity calculation
        confidence = heuristic_confidence(domain, expected_keywords, used_kw, current_category, used_category)
        suggestions.append({
            "domain": domain,
            "confidence": confidence,
            "keywords_used": used_kw,
        })
        used_keywords.append(base)
    return suggestions, used_keywords


def generate_record(
    category: str,
    blocked_prob: float = 0.05,
    low_score_chance: float = 0.3,
    mission_prob: float = 0.3,
    target_prob: float = 0.3,
    hyphen_prob: float = 0.3,
    num_suggestions: int = 3,
) -> Dict[str, object]:
    """
    Create a single dataset record.  If the description contains a banned
    term, the record is marked as blocked; otherwise domain suggestions are
    generated.
    """
    config = DOMAIN_CATEGORIES[category]
    description = generate_description(category, config, mission_prob=mission_prob, target_prob=target_prob)
    # Randomly inject a banned term into some descriptions to produce blocked examples
    if random.random() < blocked_prob:
        # 5% chance to insert a banned term for safety examples
        banned = random.choice(BANNED_TERMS)
        description += f" offering {banned} services"
    # Check for banned content
    if contains_banned_term(description):
        output = {
            "suggestions": [],
            "status": "blocked",
            "message": "Request contains inappropriate content",
        }
    else:
        # Provide expected keywords for scoring: clean the current category keywords
        expected_keywords = [clean_keyword(k) for k in config["keywords"]]
        suggestions, _ = generate_suggestions(
            config,
            expected_keywords,
            current_category=category,
            num_suggestions=num_suggestions,
            low_score_chance=low_score_chance,
            hyphen_prob=hyphen_prob,
        )
        output = {
            "suggestions": suggestions,
            "status": "success",
        }
    return {
        "instruction": "Generate brandable domain names for the following business description.",
        "input": description,
        "output": output
    }


def generate_dataset(
    num_records: int,
    blocked_prob: float = 0.05,
    low_score_chance: float = 0.3,
    mission_prob: float = 0.3,
    target_prob: float = 0.3,
    hyphen_prob: float = 0.3,
    num_suggestions: int = 3,
) -> List[Dict[str, object]]:
    """
    Generate a dataset of the requested size.  The function attempts to
    distribute records evenly across categories to ensure balanced
    representation.  If ``num_records`` is not divisible by the number of
    categories, the remainder is filled by randomly selecting from the
    categories.
    """
    records: List[Dict[str, object]] = []
    categories = list(DOMAIN_CATEGORIES.keys())
    if not categories:
        return records
    count_per_category = num_records // len(categories)
    remainder = num_records % len(categories)
    # Generate an even number of records per category
    for category in categories:
        for _ in range(count_per_category):
            records.append(
                generate_record(
                    category,
                    blocked_prob=blocked_prob,
                    low_score_chance=low_score_chance,
                    mission_prob=mission_prob,
                    target_prob=target_prob,
                    hyphen_prob=hyphen_prob,
                    num_suggestions=num_suggestions,
                )
            )
    # Fill the remainder with randomly chosen categories
    if remainder > 0:
        extra_categories = random.choices(categories, k=remainder)
        for category in extra_categories:
            records.append(
                generate_record(
                    category,
                    blocked_prob=blocked_prob,
                    low_score_chance=low_score_chance,
                    mission_prob=mission_prob,
                    target_prob=target_prob,
                    hyphen_prob=hyphen_prob,
                    num_suggestions=num_suggestions,
                )
            )
    # Shuffle the dataset so categories are interleaved
    random.shuffle(records)
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a synthetic domain name dataset.")
    parser.add_argument("--num-records", type=int, default=200, help="Number of records to generate")
    parser.add_argument("--output-file", type=str, default="", help="Path to write JSONL output (prints to stdout if omitted)")
    parser.add_argument("--blocked-prob", type=float, default=0.05, help="Probability of injecting a banned term into a description")
    parser.add_argument("--low-score-chance", type=float, default=0.3, help="Probability of generating a mismatched suggestion")
    parser.add_argument("--mission-prob", type=float, default=0.3, help="Probability of appending a mission sentence to the description")
    parser.add_argument("--target-prob", type=float, default=0.3, help="Probability of appending a target-audience phrase to the description")
    parser.add_argument("--hyphen-prob", type=float, default=0.3, help="Probability of inserting a hyphen between concatenated keywords")
    parser.add_argument("--num-suggestions", type=int, default=3, help="Number of domain suggestions to generate per record")
    args = parser.parse_args()
    dataset = generate_dataset(
        args.num_records,
        blocked_prob=args.blocked_prob,
        low_score_chance=args.low_score_chance,
        mission_prob=args.mission_prob,
        target_prob=args.target_prob,
        hyphen_prob=args.hyphen_prob,
        num_suggestions=args.num_suggestions,
    )
    if args.output_file:
        with open(args.output_file, 'w', encoding='utf-8') as f:
            for record in dataset:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
    else:
        for record in dataset:
            print(json.dumps(record, ensure_ascii=False))


if __name__ == "__main__":
    main()