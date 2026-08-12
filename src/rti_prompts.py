"""RTI prompt generation — ported from the RTI paper's run_rti_task.py.

12 sub-task categories covering different interference patterns:
  1. name_repetition    — IOI-like name structures
  2. common_noun        — noun parallel structures
  3. adjective_pattern  — alternating adj-adj-adj patterns
  4. location           — origin/destination with repeated origin
  5. temporal_sequence   — alternating day/month sequences
  6. pronoun_resolution — pronoun binding with distractor entity
  7. list_completion    — cyclic A,B,C,A,B,? lists
  8. french_names       — French name structures
  9. french_nouns       — French noun/location parallels
 10. counting           — ordinal/number sequences
 11. bpe_fragment       — multi-token words after repeated prefix
 12. bos_context        — minimal context where BOS attention dominates

Each prompt has correct_id (the non-repeated/non-salient answer) and
distractor_id (the repeated/salient token the model must suppress).
"""

import numpy as np

NAMES_A = [
    "Alice", "David", "Emma", "Frank", "Grace", "Henry", "Jack", "Kate",
    "Mary", "Nick", "Paul", "Sarah", "Tom", "Anna", "Luke", "Jane",
]
NAMES_B = [
    "Bob", "Carol", "Eric", "Fiona", "George", "Helen", "Ivan", "Julia",
    "Kevin", "Laura", "Mike", "Nancy", "Oscar", "Peter", "Quinn", "Ruth",
]

NOUNS_SUBJ = ["cat", "dog", "bird", "fish", "horse", "bear", "wolf", "fox"]
NOUNS_OBJ = ["mat", "bed", "box", "cage", "roof", "tree", "hill", "rock"]

ADJ_A = ["big", "old", "hot", "fast", "tall", "dark", "soft", "loud"]
ADJ_B = ["red", "wise", "cold", "slow", "thin", "bright", "hard", "quiet"]

CITIES_FROM = ["Paris", "London", "Boston", "Tokyo", "Berlin", "Madrid", "Rome", "Seoul"]
CITIES_TO = ["London", "Paris", "Denver", "Seoul", "Vienna", "Lisbon", "Milan", "Tokyo"]

DAYS_A = ["Monday", "Wednesday", "Friday", "January", "March", "May"]
DAYS_B = ["Tuesday", "Thursday", "Saturday", "February", "April", "June"]

NUMBERS_A = ["one", "first", "alpha"]
NUMBERS_B = ["two", "second", "beta"]
NUMBERS_C = ["three", "third", "gamma"]

LIST_ITEMS = [
    ("apple", "banana", "cherry"),
    ("red", "blue", "green"),
    ("cat", "dog", "bird"),
    ("north", "south", "east"),
    ("gold", "silver", "bronze"),
    ("rock", "paper", "scissors"),
    ("sun", "moon", "star"),
    ("spring", "summer", "autumn"),
]

FRENCH_NAMES_A = [
    "Pierre", "Marie", "Jacques", "Sophie", "Claude", "Anne",
    "Michel", "Claire", "Louis", "Julie",
]
FRENCH_NAMES_B = [
    "Jean", "Luc", "Paul", "Marc", "Henri", "Alain",
    "Nicole", "Monique", "Sylvie", "Colette",
]

FRENCH_NOUNS = [
    ("chat", "tapis"),
    ("chien", "lit"),
    ("livre", "table"),
    ("oiseau", "arbre"),
]
FRENCH_CITIES = [("Paris", "Londres"), ("Lyon", "Marseille"), ("Nice", "Bordeaux")]

BPE_PREFIXES = [
    ("un", "un", "fortunate"),
    ("dis", "dis", "appointment"),
    ("over", "over", "whelming"),
    ("pre", "pre", "determined"),
    ("mis", "mis", "understanding"),
    ("counter", "counter", "productive"),
    ("re", "re", "construction"),
    ("under", "under", "estimated"),
]


def make_prompts(tokenizer, seed=42):
    """Generate all RTI prompts. Returns list of dicts with keys:
    text, correct, distractor, category, meta, correct_id, distractor_id.
    """
    rng = np.random.RandomState(seed)
    prompts = []

    def add(cat, text, correct, distractor, meta=None):
        correct_tok = tokenizer.encode(" " + correct)
        distractor_tok = tokenizer.encode(" " + distractor)
        if len(correct_tok) == 1 and len(distractor_tok) == 1:
            prompts.append({
                "text": text,
                "correct": correct,
                "correct_id": correct_tok[0],
                "distractor": distractor,
                "distractor_id": distractor_tok[0],
                "category": cat,
                "meta": meta or {},
            })

    # 1. Name repetition (IOI-like)
    templates = [
        "Then {D} and {C} went to the store. {D} gave a drink to",
        "{D} told {C} a story. {D} then handed the book to",
        "{D} met {C} at the park. {D} passed the ball to",
        "When {D} and {C} arrived, {D} gave the keys to",
        "{D} helped {C} move. {D} gave the boxes to",
        "{D} invited {C} to dinner. {D} served food to",
    ]
    for i in range(min(len(NAMES_A), len(NAMES_B))):
        for tmpl in templates:
            d, c = NAMES_A[i], NAMES_B[i]
            add("name_repetition", tmpl.format(D=d, C=c), c, d,
                {"distractor_count": 2, "correct_count": 1})

    # 2. Common noun repetition
    noun_templates = [
        "The {S1} sat on the {O}. The {S2} sat on the",
        "She put the {S1} on the {O}. He put the {S2} on the",
        "The {S1} flew over the {O}. The {S2} flew over the",
        "A {S1} hid under the {O}. A {S2} hid under the",
    ]
    for i in range(len(NOUNS_SUBJ)):
        for j, tmpl in enumerate(noun_templates):
            s1 = NOUNS_SUBJ[i]
            s2 = NOUNS_SUBJ[(i + 1) % len(NOUNS_SUBJ)]
            o = NOUNS_OBJ[i]
            add("common_noun", tmpl.format(S1=s1, O=o, S2=s2), o, s1,
                {"distractor_count": 1, "correct_count": 1, "structure": "parallel"})

    # 3. Adjective alternating pattern
    for i in range(len(ADJ_A)):
        a, b = ADJ_A[i], ADJ_B[i]
        text = f"The {a} {b} {a} {b} {a}"
        add("adjective_pattern", text, b, a,
            {"distractor_count": 3, "correct_count": 2, "pattern": "ABABA"})

    # 4. Location interference
    loc_templates = [
        "John flew from {F} to {T}. Mary flew from {F} to",
        "The train goes from {F} to {T}. The bus goes from {F} to",
        "She traveled from {F} to {T}. He traveled from {F} to",
    ]
    for i in range(len(CITIES_FROM)):
        if CITIES_FROM[i] == CITIES_TO[i]:
            continue
        for tmpl in loc_templates:
            add("location", tmpl.format(F=CITIES_FROM[i], T=CITIES_TO[i]),
                CITIES_TO[i], CITIES_FROM[i],
                {"distractor_count": 2, "correct_count": 1})

    # 5. Temporal sequence
    for i in range(len(DAYS_A)):
        a, b = DAYS_A[i], DAYS_B[i]
        text = f"{a} {b} {a} {b} {a}"
        add("temporal_sequence", text, b, a,
            {"distractor_count": 3, "correct_count": 2})

    # 6. Pronoun resolution with distractor
    pron_templates = [
        "{D} asked {C} for help. {C} agreed and then {D} thanked",
        "{D} called {C} on the phone. {C} picked up and talked to",
        "{D} wrote to {C} last week. {C} replied back to",
    ]
    idxs = list(range(min(len(NAMES_A), len(NAMES_B))))
    rng.shuffle(idxs)
    for i in idxs[:10]:
        for tmpl in pron_templates:
            d, c = NAMES_A[i], NAMES_B[i]
            add("pronoun_resolution", tmpl.format(D=d, C=c), c, d,
                {"distractor_count": 2, "correct_count": 2,
                 "note": "correct is the one just-mentioned, distractor is the subject"})

    # 7. List completion (cyclic)
    for items in LIST_ITEMS:
        a, b, c = items
        for n_reps in [1, 2]:
            cycle = f"{a}, {b}, {c}, " * n_reps + f"{a}, {b},"
            add("list_completion", cycle, c, a,
                {"distractor_count": n_reps + 1, "correct_count": n_reps,
                 "cycle_length": 3, "n_reps": n_reps})

    # 8. French name repetition
    fr_templates = [
        "Alors {D} et {C} sont alles au magasin. {D} a donne un verre a",
        "{D} a dit a {C} une histoire. {D} a donne le livre a",
        "{D} a rencontre {C} au parc. {D} a passe la balle a",
        "Quand {D} et {C} sont arrives, {D} a donne les cles a",
    ]
    for i in range(min(len(FRENCH_NAMES_A), len(FRENCH_NAMES_B))):
        for tmpl in fr_templates:
            d, c = FRENCH_NAMES_A[i], FRENCH_NAMES_B[i]
            add("french_names", tmpl.format(D=d, C=c), c, d,
                {"distractor_count": 2, "correct_count": 1, "language": "french"})

    # 9. French noun/location
    for s1, obj in FRENCH_NOUNS:
        text = f"Le {s1} est sur le {obj}. Le chien est sur le"
        add("french_nouns", text, obj, s1,
            {"language": "french"})
    for city_from, city_to in FRENCH_CITIES:
        text = f"Il est alle de {city_from} a {city_to}. Elle est allee de {city_from} a"
        add("french_locations", text, city_to, city_from,
            {"language": "french"})

    # 10. Counting / ordinal sequences
    for a, b, c in zip(NUMBERS_A, NUMBERS_B, NUMBERS_C):
        text = f"{a}, {b}, {c}, {a}, {b},"
        add("counting", text, c, a,
            {"distractor_count": 2, "correct_count": 1})
        text2 = f"{a} {b} {c} {a} {b} {c} {a} {b}"
        add("counting", text2, c, a,
            {"distractor_count": 3, "correct_count": 2})

    # 11. BPE fragment interference
    bpe_templates = [
        "The word {P}happy and the word {P}likely and the word {P}{SUFFIX}",
        "He was {P}able and she was {P}connected and they were {P}{SUFFIX}",
        "It was {P}done and {P}made and {P}{SUFFIX}",
    ]
    bpe_items = [
        ("un", "fair"),
        ("un", "clear"),
        ("dis", "honest"),
        ("dis", "loyal"),
        ("over", "due"),
        ("over", "night"),
        ("re", "built"),
        ("re", "opened"),
        ("mis", "led"),
        ("pre", "set"),
    ]
    for prefix, suffix in bpe_items:
        for tmpl in bpe_templates:
            text = tmpl.format(P=prefix, SUFFIX="")
            suffix_tok = tokenizer.encode(suffix)
            prefix_tok = tokenizer.encode(prefix)
            if len(suffix_tok) == 1 and len(prefix_tok) == 1:
                prompts.append({
                    "text": text.rstrip(),
                    "correct": suffix,
                    "correct_id": suffix_tok[0],
                    "distractor": prefix,
                    "distractor_id": prefix_tok[0],
                    "category": "bpe_fragment",
                    "meta": {"prefix": prefix, "suffix": suffix,
                             "distractor_count": 3, "correct_count": 0},
                })

    # 12. BOS / minimal context
    bos_templates = [
        ("{D} and {C}. {D} saw", "{C}", "{D}"),
        ("The {D} or the {C}? The {D} and the", "{C}", "{D}"),
        ("{D}, {C}, {D},", "{C}", "{D}"),
    ]
    short_nouns = [
        ("cat", "dog"), ("sun", "moon"), ("king", "queen"),
        ("boy", "girl"), ("man", "woman"), ("black", "white"),
        ("hot", "cold"), ("day", "night"), ("up", "down"),
        ("left", "right"), ("east", "west"), ("north", "south"),
    ]
    for d, c in short_nouns:
        for tmpl, c_slot, d_slot in bos_templates:
            text = tmpl.format(D=d, C=c)
            add("bos_minimal", text, c, d,
                {"distractor_count": 2, "correct_count": 1,
                 "note": "short context, BOS attention dominates"})

    rng.shuffle(prompts)
    return prompts
