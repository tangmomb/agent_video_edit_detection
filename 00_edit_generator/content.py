"""Contenus textuels aléatoires (mots-clés, questions, sous-titres, bandeaux)."""

import random

# (requête vidéo en anglais, sujet en français)
QUERY_POOL = [
    ("cooking", "la cuisine"), ("chef", "le chef"), ("kitchen", "la cuisine maison"),
    ("pasta", "les pâtes"), ("baking bread", "le pain maison"), ("vegetables", "les légumes"),
    ("restaurant", "le restaurant"), ("coffee", "le café"), ("dessert", "les desserts"),
    ("grill", "le barbecue"), ("street food", "la street food"), ("fruit", "les fruits"),
    ("people talking", "la discussion"), ("people", "l'interview"), ("podcast", "le podcast"),
    ("city", "la ville"), ("nature", "la nature"),
    ("office", "le bureau"), ("sport", "le sport"), ("travel", "le voyage"),
]

STOPWORDS = {"the", "and", "with", "from", "into", "for", "man", "woman", "person",
             "people", "her", "his", "their", "while", "close", "view", "shot", "video"}

# Les contenus dépendent de la famille du thème (cuisine, parole, autre) : un bandeau d'interview
# annonce un invité ou une journaliste, pas une cheffe pâtissière.
THEMES = {
    "cuisine": {"cooking", "chef", "kitchen", "pasta", "baking bread", "vegetables", "restaurant",
                "coffee", "dessert", "grill", "street food", "fruit"},
    "parole": {"people talking", "people", "podcast"},
}

KEYWORDS = {
    "commun": {
        "fr": ["SECRET", "INCROYABLE", "ATTENTION", "BONUS", "WOW", "TOP 3", "À RETENIR"],
        "en": ["SECRET", "AMAZING", "WARNING", "BONUS", "WOW", "TOP 3", "KEY POINT"],
    },
    "cuisine": {
        "fr": ["ASTUCE", "FACILE", "RAPIDE", "À TESTER", "LE BON GESTE", "ÉTAPE 1", "ÉTAPE 2",
               "RÉSULTAT", "NE FAITES PAS ÇA", "PRO TIP", "MAISON", "EN 5 MIN"],
        "en": ["TIP", "EASY", "QUICK", "MUST TRY", "STEP 1", "STEP 2", "RESULT", "PRO TIP", "HOMEMADE"],
    },
    "parole": {
        "fr": ["EXCLU", "EN DIRECT", "TÉMOIGNAGE", "CASH", "LA VÉRITÉ", "RÉVÉLATION", "DÉBAT",
               "SON AVIS", "INTERVIEW", "ÉPISODE 12"],
        "en": ["EXCLUSIVE", "LIVE", "TRUE STORY", "THE TRUTH", "DEBATE", "HOT TAKE", "INTERVIEW",
               "EPISODE 12"],
    },
    "autre": {
        "fr": ["À VOIR", "INCONTOURNABLE", "TOP 5", "BON PLAN", "LE CONSEIL", "AVANT / APRÈS"],
        "en": ["MUST SEE", "TOP 5", "BEST TIP", "BEFORE / AFTER", "DON'T MISS"],
    },
}

QUESTIONS = {
    "commun": {
        "fr": ["Tu connais le secret {de_topic} ?", "Vous saviez ça sur {topic} ?",
               "Et toi, tu en penses quoi ?", "C'est vraiment si simple ?",
               "Le résultat va vous surprendre, non ?"],
        "en": ["Do you know the secret of {topic}?", "Did you know this?", "What do you think?",
               "Is it really that easy?"],
    },
    "cuisine": {
        "fr": ["Pourquoi tout le monde se trompe sur {topic} ?", "Et toi, tu fais comment ?",
               "Quelle est l'erreur la plus fréquente ?", "On teste {topic} ensemble ?",
               "Combien de temps faut-il ?", "Tu préfères quelle version ?"],
        "en": ["Why does everyone get this wrong?", "How do you do it?", "Which one do you prefer?"],
    },
    "parole": {
        "fr": ["Comment tout a commencé ?", "Quel a été le déclic ?", "Vous referiez pareil ?",
               "Quel conseil donneriez-vous ?", "Qu'est-ce qui vous a le plus surpris ?"],
        "en": ["How did it all start?", "Would you do it again?", "What advice would you give?",
               "What surprised you the most?"],
    },
    "autre": {
        "fr": ["Vous êtes déjà allés là ?", "Combien ça coûte vraiment ?", "Ça vaut le coup ?"],
        "en": ["Have you been there?", "How much does it really cost?", "Is it worth it?"],
    },
}

SPEECH = {
    "commun": {
        "fr": ["Aujourd'hui on va parler de {topic} et je vais tout vous expliquer",
               "Regardez bien ce qui se passe à ce moment précis",
               "Franchement je ne m'attendais pas à un résultat pareil"],
        "en": ["Today we are talking about {topic} and I will explain everything",
               "Watch closely what happens right here"],
    },
    "cuisine": {
        "fr": ["La plupart des gens ne savent pas qu'il faut faire comme ça",
               "C'est l'étape la plus importante alors restez jusqu'au bout",
               "Abonnez-vous si vous voulez la suite de la recette",
               "On laisse reposer quelques minutes et c'est prêt"],
        "en": ["Most people do not know you have to do it this way",
               "This is the most important step so stay until the end",
               "Follow for part two of this recipe"],
    },
    "parole": {
        "fr": ["Au début personne ne croyait vraiment à ce projet",
               "Je pense que c'est la question qu'on me pose le plus souvent",
               "Ce jour-là tout a changé pour moi et pour l'équipe",
               "Honnêtement je ne referais pas les choses de la même façon"],
        "en": ["At first nobody really believed in this project",
               "That is the question people ask me the most",
               "Honestly I would not do it the same way again"],
    },
    "autre": {
        "fr": ["On est arrivés tôt le matin et il n'y avait personne",
               "C'est l'endroit que je recommande à tout le monde"],
        "en": ["We got there early in the morning and nobody was around",
               "This is the place I recommend to everyone"],
    },
}

CHAPTER_KICKERS = {
    "fr": ["CHAPITRE {n}", "PARTIE {n}", "ÉTAPE {n}", "{n}.", "#{n}"],
    "en": ["CHAPTER {n}", "PART {n}", "STEP {n}", "{n}.", "#{n}"],
}
CHAPTER_TITLES = {
    "commun": {
        "fr": ["Tout savoir sur {topic}", "Le verdict", "Avant de commencer", "Le secret {de_topic}"],
        "en": ["All about {topic}", "The verdict", "Before we start"],
    },
    "cuisine": {
        "fr": ["La préparation", "Les ingrédients", "La cuisson", "Le résultat", "Les erreurs à éviter",
               "Mon astuce", "Le dressage", "On passe à table"],
        "en": ["Preparation", "The ingredients", "Cooking time", "The result", "Mistakes to avoid",
               "My secret tip", "Plating"],
    },
    "parole": {
        "fr": ["Les débuts", "Le déclic", "Les coulisses", "Les doutes", "Et maintenant ?",
               "Questions du public", "Le parcours"],
        "en": ["The early days", "Behind the scenes", "The turning point", "What's next?", "Q&A"],
    },
    "autre": {
        "fr": ["Le départ", "Sur place", "Le bilan", "Nos conseils", "Le budget"],
        "en": ["Getting there", "On site", "The wrap-up", "Our tips", "The budget"],
    },
}

FIRST_NAMES = ["Marie", "Lucas", "Camille", "Hugo", "Léa", "Nora", "Yanis", "Inès", "Paul", "Sofia"]
LAST_NAMES = ["Dupont", "Martin", "Bernard", "Moreau", "Laurent", "Garcia", "Roux", "Fontaine"]
ROLES = {
    "commun": {"fr": ["Spécialiste {de_topic}", "Fondateur", "Fondatrice"],
               "en": ["{topic} expert", "Founder"]},
    "cuisine": {"fr": ["Cheffe cuisinière", "Chef pâtissier", "Nutritionniste", "Restauratrice",
                       "Boulanger"],
                "en": ["Head Chef", "Pastry Chef", "Nutritionist", "Restaurant owner"]},
    "parole": {"fr": ["Invité", "Invitée", "Journaliste", "Animatrice", "Auteur", "Entrepreneuse",
                      "Psychologue", "Ancien sportif pro"],
               "en": ["Guest", "Journalist", "Host", "Author", "Entrepreneur", "Psychologist"]},
    "autre": {"fr": ["Guide", "Photographe", "Coach sportif", "Architecte", "Voyageuse"],
              "en": ["Guide", "Photographer", "Coach", "Architect", "Traveler"]},
}

TOPIC_EN = {fr: en for en, fr in QUERY_POOL}


def theme(topic: str) -> str:
    """Famille du thème (cuisine, parole, autre), pour un sujet en français ou en anglais."""
    query = TOPIC_EN.get(topic, topic)
    return next((name for name, queries in THEMES.items() if query in queries), "autre")


def _pool(table: dict, lang: str, topic: str) -> list[str]:
    """Contenus communs + ceux de la famille du thème."""
    return table["commun"][lang] + table[theme(topic)][lang]


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


def keyword(rng: random.Random, lang: str, extra_words: list[str], topic: str = "") -> str:
    useful = [w for w in extra_words if len(w) > 3 and w.lower() not in STOPWORDS]
    if useful and rng.random() < 0.5:
        word = rng.choice(useful)
    else:
        word = rng.choice(_pool(KEYWORDS, lang, topic))
    return word.upper() if rng.random() < 0.7 else word.capitalize()


def question(rng: random.Random, lang: str, topic: str) -> str:
    return _fmt(rng.choice(_pool(QUESTIONS, lang, topic)), topic)


def subtitle_chunks(rng: random.Random, lang: str, topic: str, n: int) -> list[str]:
    words = _fmt(rng.choice(_pool(SPEECH, lang, topic)), topic).split()
    chunks, i = [], 0
    while i < len(words) and len(chunks) < n:
        k = rng.randint(2, 5)
        chunks.append(" ".join(words[i:i + k]))
        i += k
    return chunks


def title_card(rng: random.Random, lang: str, topic: str) -> tuple[str | None, str]:
    """(surtitre ou None, texte principal) d'un carton plein écran : question ou chapitre."""
    if rng.random() < 0.4:
        return None, question(rng, lang, topic)
    kicker = rng.choice(CHAPTER_KICKERS[lang]).format(n=rng.randint(1, 5))
    title = _fmt(rng.choice(_pool(CHAPTER_TITLES, lang, topic)), topic)
    title = title[0].upper() + title[1:]
    if rng.random() < 0.5:
        return kicker, title
    return None, rng.choice([title, f"{kicker} {title}" if kicker[-1] == "." else f"{kicker} : {title}"])


def lower_third(rng: random.Random, lang: str, topic: str) -> tuple[str, str]:
    name = f"{rng.choice(FIRST_NAMES)} {rng.choice(LAST_NAMES)}"
    role = _fmt(rng.choice(_pool(ROLES, lang, topic)), topic)
    return name, role[0].upper() + role[1:]
