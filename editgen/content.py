"""Contenus textuels aléatoires (mots-clés, questions, sous-titres, bandeaux)."""

import random

# (requête vidéo en anglais, sujet en français)
QUERY_POOL = [
    ("cooking", "la cuisine"), ("chef", "le chef"), ("kitchen", "la cuisine maison"),
    ("pasta", "les pâtes"), ("baking bread", "le pain maison"), ("vegetables", "les légumes"),
    ("restaurant", "le restaurant"), ("coffee", "le café"), ("dessert", "les desserts"),
    ("grill", "le barbecue"), ("street food", "la street food"), ("fruit", "les fruits"),
    ("people talking", "la discussion"), ("city", "la ville"), ("nature", "la nature"),
    ("office", "le bureau"), ("sport", "le sport"), ("travel", "le voyage"),
]

STOPWORDS = {"the", "and", "with", "from", "into", "for", "man", "woman", "person",
             "people", "her", "his", "their", "while", "close", "view", "shot", "video"}

GENERIC_KEYWORDS = {
    "fr": ["ASTUCE", "SECRET", "INCROYABLE", "FACILE", "RAPIDE", "À TESTER", "TOP 3",
           "ATTENTION", "LE BON GESTE", "ÉTAPE 1", "ÉTAPE 2", "RÉSULTAT", "BONUS",
           "NE FAITES PAS ÇA", "PRO TIP", "WOW", "MAISON", "EN 5 MIN"],
    "en": ["TIP", "SECRET", "AMAZING", "EASY", "QUICK", "MUST TRY", "TOP 3", "WARNING",
           "STEP 1", "STEP 2", "RESULT", "BONUS", "PRO TIP", "WOW", "HOMEMADE"],
}

QUESTIONS = {
    "fr": [
        "Tu connais le secret {de_topic} ?",
        "Pourquoi tout le monde se trompe sur {topic} ?",
        "Et toi, tu fais comment ?",
        "Vous saviez ça sur {topic} ?",
        "Quelle est l'erreur la plus fréquente ?",
        "On teste {topic} ensemble ?",
        "C'est vraiment si simple ?",
        "Combien de temps faut-il ?",
        "Tu préfères quelle version ?",
        "Le résultat va vous surprendre, non ?",
    ],
    "en": [
        "Do you know the secret of {topic}?",
        "Why does everyone get this wrong?",
        "How do you do it?",
        "Did you know this?",
        "Which one do you prefer?",
        "Is it really that easy?",
    ],
}

SPEECH = {
    "fr": [
        "Aujourd'hui on va parler de {topic} et je vais tout vous expliquer",
        "Regardez bien ce qui se passe à ce moment précis",
        "La plupart des gens ne savent pas qu'il faut faire comme ça",
        "C'est l'étape la plus importante alors restez jusqu'au bout",
        "Franchement je ne m'attendais pas à un résultat pareil",
        "Abonnez-vous si vous voulez la suite de la recette",
        "On laisse reposer quelques minutes et c'est prêt",
    ],
    "en": [
        "Today we are talking about {topic} and I will explain everything",
        "Watch closely what happens right here",
        "Most people do not know you have to do it this way",
        "This is the most important step so stay until the end",
        "Follow for part two of this video",
    ],
}

FIRST_NAMES = ["Marie", "Lucas", "Camille", "Hugo", "Léa", "Nora", "Yanis", "Inès", "Paul", "Sofia"]
LAST_NAMES = ["Dupont", "Martin", "Bernard", "Moreau", "Laurent", "Garcia", "Roux", "Fontaine"]
ROLES = {
    "fr": ["Cheffe cuisinière", "Chef pâtissier", "Spécialiste {de_topic}", "Journaliste",
           "Fondateur", "Nutritionniste", "Restauratrice", "Invité"],
    "en": ["Head Chef", "Pastry Chef", "{topic} expert", "Journalist", "Founder", "Guest"],
}

TOPIC_EN = {fr: en for en, fr in QUERY_POOL}


def de_topic(topic: str) -> str:
    """'le pain' -> 'du pain', 'les pâtes' -> 'des pâtes', 'la ville' -> 'de la ville'."""
    if topic.startswith("le "):
        return "du " + topic[3:]
    if topic.startswith("les "):
        return "des " + topic[4:]
    return "de " + topic


def _fmt(template: str, topic: str) -> str:
    return template.format(topic=topic, de_topic=de_topic(topic))


def pick_lang(rng: random.Random, lang: str) -> str:
    return rng.choice(["fr", "en"]) if lang == "mix" else lang


def topic_for(lang: str, topic_fr: str) -> str:
    return TOPIC_EN.get(topic_fr, topic_fr) if lang == "en" else topic_fr


def keyword(rng: random.Random, lang: str, extra_words: list[str]) -> str:
    useful = [w for w in extra_words if len(w) > 3 and w.lower() not in STOPWORDS]
    if useful and rng.random() < 0.5:
        word = rng.choice(useful)
    else:
        word = rng.choice(GENERIC_KEYWORDS[lang])
    return word.upper() if rng.random() < 0.7 else word.capitalize()


def question(rng: random.Random, lang: str, topic: str) -> str:
    return _fmt(rng.choice(QUESTIONS[lang]), topic)


def subtitle_chunks(rng: random.Random, lang: str, topic: str, n: int) -> list[str]:
    words = _fmt(rng.choice(SPEECH[lang]), topic).split()
    chunks, i = [], 0
    while i < len(words) and len(chunks) < n:
        k = rng.randint(2, 5)
        chunks.append(" ".join(words[i:i + k]))
        i += k
    return chunks


def lower_third(rng: random.Random, lang: str, topic: str) -> tuple[str, str]:
    name = f"{rng.choice(FIRST_NAMES)} {rng.choice(LAST_NAMES)}"
    role = _fmt(rng.choice(ROLES[lang]), topic)
    return name, role[0].upper() + role[1:]
