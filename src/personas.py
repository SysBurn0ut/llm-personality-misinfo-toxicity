import random

# bipolar markers: (high, low)
# % means the intensity, not how many markers
BIG_FIVE = {
    "Openness": [
        ("curious", "incurious"),
        ("imaginative", "down-to-earth"),
        ("artistic", "uninterested in art"),
        ("having wide interests", "having narrow interests"),
        ("excitable", "unexcitable"),
        ("unconventional", "conventional"),
    ],
    "Conscientiousness": [
        ("efficient", "inefficient"),
        ("organized", "disorganized"),
        ("careful", "careless"),
        ("thorough", "superficial"),
        ("hardworking", "lazy"),
        ("deliberate", "impulsive"),
    ],
    "Extraversion": [
        ("sociable", "withdrawn"),
        ("forceful", "submissive"),
        ("energetic", "lethargic"),
        ("adventurous", "cautious"),
        ("enthusiastic", "unenthusiastic"),
        ("outgoing", "reserved"),
    ],
    "Agreeableness": [
        ("forgiving", "grudge-holding"),
        ("undemanding", "demanding"),
        ("warm", "cold"),
        ("flexible", "stubborn"),
        ("modest", "show-off"),
        ("sympathetic", "unsympathetic"),
    ],
    "Neuroticism": [
        ("tense", "relaxed"),
        ("irritable", "even-tempered"),
        ("discontented", "contented"),
        ("shy", "self-assured"),
        ("moody", "emotionally stable"),
        ("insecure", "self-confident"),
    ],
}

TRAITS = list(BIG_FIVE)

BANDS = [
    (0, 20, "very low"),
    (20, 40, "low"),
    (40, 60, "moderate"),
    (60, 80, "high"),
    (80, 101, "very high"),
]

QUALIFICATIONS = [
    ("No Education", 0),
    ("Primary School", 1),
    ("Middle School", 2),
    ("High School Diploma", 3),
    ("Professional Certificate", 4),
    ("Associate Degree", 5),
    ("Bachelor's Degree", 6),
    ("Master's Degree", 7),
    ("PhD", 8),
]


def _band(score: int) -> str:
    return next(label for lo, hi, label in BANDS if lo <= score < hi)


def sample_big5_poles(n, rng, margin=0):
    def pole(hi):
        if margin == 0:
            return 100 if hi else 0
        return rng.randint(100 - margin, 100) if hi else rng.randint(0, margin)
    cols = {}
    for t in TRAITS:
        flags = [True] * (n // 2) + [False] * (n - n // 2)
        rng.shuffle(flags)
        cols[t] = [pole(f) for f in flags]
    return [{t: cols[t][i] for t in TRAITS} for i in range(n)]


def sample_big5_lhs(n: int, rng: random.Random, granularity: int = 5):
    '''
    Latin Hypercube Sampling over the 5 traits.

    Each trait's [0,100] range is split into n strata, with one value drawn per
    stratum; the strata are then shuffled independently for each trait. Result:
    uniform coverage of every dimension and near-zero cross-correlations — the
    condition needed for the regression coefficients to be interpretable.
    '''
    columns = {}
    for trait in TRAITS:
        strata = [(i + rng.random()) * 100.0 / n for i in range(n)]
        rng.shuffle(strata)
        columns[trait] = [
            int(min(100, max(0, round(v / granularity) * granularity))) for v in strata
        ]
    return [{t: columns[t][i] for t in TRAITS} for i in range(n)]


def render_big5_prompt(scores: dict) -> str:
    #Personality block to insert into the agent's system message.
    lines = [
        "YOUR PERSONALITY (Big Five profile).",
        "Each dimension below is scored from 0 to 100.",
        "",
        "How to read a score:",
        "- The score is the INTENSITY with which the whole dimension shapes your "
        "behaviour. It is NOT the fraction of the listed markers that apply to you.",
        "- A high score means every marker of that dimension is expressed strongly "
        "and consistently. A low score means you consistently show the opposite of "
        "those markers. A score near 50 means you sit in the middle: the marker is "
        "only mildly present and the situation decides which way you lean.",
        "- All five dimensions are active at the same time and must all remain "
        "visible in how you reason. Do not let one dimension take over the others: "
        "a dimension weighs more than another only in proportion to the gap between "
        "their scores.",
        "- Never name these dimensions, scores or markers explicitly.",
        "",
    ]
    for trait, markers in BIG_FIVE.items():
        score = scores[trait]
        hi = ", ".join(h for h, _ in markers)
        lo = ", ".join(l for _, l in markers)
        lines.append(
            f"{trait}: {score}/100 ({_band(score)})\n"
            f"  - high end of this dimension: {hi}\n"
            f"  - low end of this dimension: {lo}"
        )
    return "\n".join(lines)


def build_personas(n: int, seed: int, country_alpha2="US", mode="continuous", margin=0):
    '''
    Generate n personas. Each persona is later instantiated on EVERY LLM
    (crossed design), so name/age/education/traits are identical across models
    and the model effect is separable from the persona effect.
    '''
    from names_dataset import NameDataset

    rng = random.Random(seed)
    scores = (sample_big5_poles(n, rng, margin) if mode == "poles"
              else sample_big5_lhs(n, rng))

    nd = NameDataset()
    pool = nd.get_top_names(n * 2, "Male", country_alpha2)[country_alpha2]["M"] + \
           nd.get_top_names(n * 2, "Female", country_alpha2)[country_alpha2]["F"]
    names = rng.sample(pool, k=n)

    personas = []
    for i in range(n):
        qual_label, qual_ord = rng.choice(QUALIFICATIONS)
        personas.append({
            "persona_id": i,
            "name": names[i],
            "age": rng.randrange(18, 65),
            "qualification": qual_label,
            "qualification_ord": qual_ord,
            "big5": scores[i],
            "big5_prompt": render_big5_prompt(scores[i]),
        })
    return personas
