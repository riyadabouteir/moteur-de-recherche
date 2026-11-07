"""Prétraitement du texte : normalisation, mots vides, racinisation (français)."""
import re
import unicodedata
from functools import lru_cache

try:  # Racinisation Snowball française et anglaise.
    import snowballstemmer
    _STEMMER_FR = snowballstemmer.stemmer("french")
    _STEMMER_EN = snowballstemmer.stemmer("english")
except Exception:  # pragma: no cover - repli si la dépendance est absente
    _STEMMER_FR = _STEMMER_EN = None

_STOPWORDS_RAW = """
le la les l un une des du de d et ou mais donc ni car ce cet cette ces c qui que qu quoi dont où ou quand
comment pourquoi il elle ils elles on nous vous je j tu me m te t se s lui leur leurs mon ma mes ton ta tes
son sa ses notre nos votre vos à a au aux en dans par pour sur sous avec sans chez vers entre est sont suis
es sommes êtes être été était étaient sera seront ai as avons avez ont avoir avait eu fait faire plus moins
très si ne n pas y aussi alors ainsi comme cela ça ceci celui celle ceux tout tous toute toutes autre autres
même mêmes peut peuvent doit dois cette ici là puis après avant lors où
the and of to is are was were with that this from not
"""


def fold(text: str) -> str:
    """Minuscules + suppression des accents (œ -> oe, æ -> ae)."""
    text = unicodedata.normalize("NFKC", text or "").lower()
    text = text.replace("œ", "oe").replace("æ", "ae")
    text = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in text if not unicodedata.combining(ch))


STOPWORDS = frozenset(fold(w) for w in _STOPWORDS_RAW.split())
_STOPWORDS_EN_RAW = """
a an the and or but if then else when where why how all any both each few more most
other some no nor not only own same so than too very can will just should now
for in on at by as be been being have has had do does did of to from with into
over under between about against during before after above below up down out off
again further once here there this that these those it its we you he she they them
our your their is are was were am i me my mine ours yours theirs who whom which what
while because until unless also may might must shall would could
"""
STOPWORDS_EN = frozenset(_STOPWORDS_EN_RAW.split())

_WORD_RE = re.compile(r"\w+", re.UNICODE)


@lru_cache(maxsize=200_000)
def _stem_cached(word: str, language: str = "fr") -> str:
    """Racine du mot avec le stemmer correspondant à la langue du corpus."""
    stemmer = _STEMMER_EN if language == "en" else _STEMMER_FR
    if stemmer is not None:
        return fold(stemmer.stemWord(word))
    word = fold(word)
    suffixes = ("ing", "edly", "ed", "ies", "s") if language == "en" else ("ements", "ement", "ations", "ation", "ites", "ite", "eaux", "aux", "es", "s", "x", "e")
    for suffix in suffixes:
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[: -len(suffix)]
    return word


def normalize_word(word: str, language: str = "fr"):
    """Mot brut -> racine normalisée, ou None si c'est un mot vide / du bruit."""
    w = unicodedata.normalize("NFKC", word).lower().strip("_")
    stopwords = STOPWORDS_EN if language == "en" else STOPWORDS
    if not w or fold(w) in stopwords:
        return None
    # Les nombres isolés (IDs, dates, codes) ne polluent pas le vocabulaire.
    if w.isdigit() or len(w) < 2:
        return None
    # Écarter les fragments commençant par un chiffre (ex. 05p, 0speculation),
    # fréquents dans les exports et peu utiles pour la recherche textuelle.
    if w[0].isdigit() and any(ch.isalpha() for ch in w):
        return None
    return _stem_cached(w, language)


def analyze_words(text: str, language: str = "fr"):
    """Liste de (mot_minuscule, racine) pour chaque mot significatif."""
    out = []
    for m in _WORD_RE.finditer(text or ""):
        stem = normalize_word(m.group(), language)
        if stem:
            out.append((m.group().lower(), stem))
    return out


def tokenize(text: str, language: str = "fr") -> list[str]:
    """Texte -> liste de racines (sans accents, sans mots vides)."""
    return [stem for _, stem in analyze_words(text, language)]


def preprocess(text: str, language: str = "fr") -> str:
    """Compatibilité : texte nettoyé sous forme d'une chaîne de racines."""
    return " ".join(tokenize(text, language))
