"""Moteur de recherche hybride : BM25 + TF-IDF (cosinus) sur des passages.

Améliorations par rapport à un simple TF-IDF documentaire :
- racinisation française + suppression des accents et des mots vides ;
- découpage des documents en passages (meilleure précision sur les longs textes) ;
- score hybride : couverture des termes de la requête, BM25 et cosinus TF-IDF ;
- bigrammes (expressions) et bonus pour la phrase exacte ;
- tolérance aux fautes de frappe et aux préfixes ;
- extraits centrés sur la zone la plus pertinente, avec mise en évidence.
"""
from __future__ import annotations

import bisect
import difflib
import html
import re
from collections import Counter, defaultdict

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from core.preprocessor import analyze_words, fold, normalize_word, tokenize

CHUNK_WORDS = 120      # taille d'un passage (en mots)
CHUNK_OVERLAP = 30     # chevauchement entre passages
K1, B = 1.5, 0.75      # paramètres BM25
BIGRAM_WEIGHT = 0.5    # poids des expressions (bigrammes) dans la requête
W_COVERAGE, W_BM25, W_COSINE = 0.55, 0.25, 0.20
MIN_RELATIVE_SCORE = 0.30   # un résultat doit atteindre 30 % du meilleur score


def _identity(tokens):
    return tokens


def _chunk_spans(text: str):
    """Découpe un texte en passages chevauchants -> liste de (début, fin) en caractères."""
    spans = [m.span() for m in re.finditer(r"\S+", text)]
    if not spans:
        return []
    if len(spans) <= CHUNK_WORDS + CHUNK_OVERLAP:
        return [(spans[0][0], spans[-1][1])]
    step = CHUNK_WORDS - CHUNK_OVERLAP
    out = []
    for i in range(0, len(spans), step):
        j = min(i + CHUNK_WORDS, len(spans))
        out.append((spans[i][0], spans[j - 1][1]))
        if j == len(spans):
            break
    return out


class SearchIndex:
    """Index documentaire hybride BM25 / TF-IDF avec classement par passages."""

    def __init__(self, documents: dict[str, str]):
        self.names = list(documents.keys())
        self.original_texts = [documents[n] for n in self.names]
        # Le corpus Reuters-21578 est en anglais : utiliser les stopwords et
        # le stemmer anglais au lieu de raciniser ses articles en français.
        self.language = "en" if self.names and all(n.startswith("Reuters — ") for n in self.names) else "fr"

        chunk_tokens, self.chunk_doc, self.chunk_span = [], [], []
        surface: dict[str, Counter] = defaultdict(Counter)
        for d, text in enumerate(self.original_texts):
            for start, end in _chunk_spans(text):
                pairs = analyze_words(text[start:end], self.language)
                stems = [s for _, s in pairs]
                for word, stem in pairs:
                    surface[stem][word] += 1
                chunk_tokens.append(stems + [f"{a}_{b}" for a, b in zip(stems, stems[1:])])
                self.chunk_doc.append(d)
                self.chunk_span.append((start, end))
        self.chunk_doc = np.array(self.chunk_doc, dtype=np.int64)
        self.surface = {s: c.most_common(1)[0][0] for s, c in surface.items()}
        self.name_tokens = [set(tokenize(re.sub(r"\.[A-Za-z0-9]{2,4}\b", " ", n), self.language)) for n in self.names]

        # Les bigrammes qui n'apparaissent qu'une seule fois sont souvent des
        # combinaisons accidentelles/noms rares : on les écarte, tout en gardant
        # tous les unigrammes pour ne pas perdre les termes de recherche rares.
        bigram_df = Counter()
        for tokens in chunk_tokens:
            bigram_df.update({term for term in tokens if self._is_bigram(term)})
        self.discarded_rare_bigrams = sum(1 for df in bigram_df.values() if df < 2)
        chunk_tokens = [
            [term for term in tokens if not self._is_bigram(term) or bigram_df[term] >= 2]
            for tokens in chunk_tokens
        ]

        self.vectorizer = CountVectorizer(analyzer=_identity)
        tf = self.vectorizer.fit_transform(chunk_tokens).tocsr().astype(np.float32)
        self.tf_matrix = tf  # matrice Term Frequency conservée pour la visualisation
        self.vocab = self.vectorizer.vocabulary_
        self._unigrams = sorted(t for t in self.vocab if not self._is_bigram(t))

        n_chunks = max(tf.shape[0], 1)
        df = np.asarray((tf > 0).sum(axis=0)).ravel().astype(np.float32)
        self.idf = np.log(1.0 + (n_chunks - df + 0.5) / (df + 0.5)).astype(np.float32)
        self.idf_unseen = float(np.log(1.0 + (n_chunks + 0.5) / 0.5))

        dl = np.asarray(tf.sum(axis=1)).ravel()
        avgdl = max(float(dl.mean()) if len(dl) else 1.0, 1.0)
        bm25 = tf.copy()
        rows = np.repeat(np.arange(tf.shape[0]), np.diff(tf.indptr))
        norm = K1 * (1 - B + B * dl[rows] / avgdl)
        bm25.data = (tf.data * (K1 + 1) / (tf.data + norm)) * self.idf[tf.indices]
        self.bm25 = bm25.tocsr()
        self.presence = (tf > 0).astype(np.float32).tocsr()

        self.tfidf = TfidfTransformer(sublinear_tf=True)
        self.tfidf_matrix = self.tfidf.fit_transform(tf).tocsr()
        self.matrix = self.tfidf_matrix  # compatibilité avec l'ancienne API

    # ------------------------------------------------------------------ requête
    @staticmethod
    def _is_bigram(term: str) -> bool:
        return "_" in term

    def _expand(self, stem: str):
        """Termes proches d'un terme inconnu : préfixes puis fautes de frappe."""
        if len(stem) >= 4:
            i = bisect.bisect_left(self._unigrams, stem)
            found = []
            while i < len(self._unigrams) and self._unigrams[i].startswith(stem) and len(found) < 5:
                found.append((self._unigrams[i], 0.7))
                i += 1
            if found:
                return found
        if len(stem) >= 4:
            cands = [t for t in self._unigrams if abs(len(t) - len(stem)) <= 4 and t[0] == stem[0]]
            close = difflib.get_close_matches(stem, cands, n=2, cutoff=0.78)
            return [(t, 0.75 * difflib.SequenceMatcher(None, stem, t).ratio()) for t in close]
        return []

    def _parse_query(self, query: str, fuzzy: bool):
        """Retourne (poids des termes, termes inconnus, corrections affichables)."""
        pairs = analyze_words(query, self.language)
        weights: dict[str, float] = {}
        unknown_stems, corrections, seq = [], [], []
        for word, stem in pairs:
            if stem in self.vocab:
                weights[stem] = max(weights.get(stem, 0.0), 1.0)
                seq.append(stem)
                continue
            alts = self._expand(stem) if fuzzy else []
            if not alts:
                unknown_stems.append(stem)
                continue
            for term, w in alts:
                weights[term] = max(weights.get(term, 0.0), w)
            best = alts[0][0]
            seq.append(best)
            corrections.append((word, self.surface.get(best, best)))
        for a, b in zip(seq, seq[1:]):
            bg = f"{a}_{b}"
            if bg in self.vocab:
                weights[bg] = BIGRAM_WEIGHT
        return weights, unknown_stems, corrections

    # ------------------------------------------------------------------ recherche
    def search(self, query: str, top_k: int = 10, fuzzy: bool = True):
        return self.search_detailed(query, top_k, fuzzy)["results"]

    def search_detailed(self, query: str, top_k: int = 10, fuzzy: bool = True):
        empty = {"results": [], "corrections": [], "unknown": []}
        if not self.names or not tokenize(query, self.language):
            return empty
        weights, unknown_stems, corrections = self._parse_query(query, fuzzy)
        unknown_words = [w for w, s in analyze_words(query, self.language) if s in unknown_stems]
        if not weights:
            return {**empty, "unknown": unknown_words}

        n_terms = self.bm25.shape[1]
        qvec = np.zeros(n_terms, dtype=np.float32)
        uni_mask = np.zeros(n_terms, dtype=np.float32)
        total_mass = self.idf_unseen * len(unknown_stems)
        for term, w in weights.items():
            j = self.vocab[term]
            qvec[j] = w
            if not self._is_bigram(term):
                uni_mask[j] = w * self.idf[j]
                total_mass += w * self.idf[j]
        total_mass = max(total_mass, 1e-9)

        bm25_scores = self.bm25 @ qvec
        if not np.any(bm25_scores > 0):
            return {**empty, "corrections": corrections, "unknown": unknown_words}
        coverage = np.clip((self.presence @ uni_mask) / total_mass, 0.0, 1.0)
        q_tfidf = qvec * np.asarray(self.tfidf.idf_, dtype=np.float32)
        norm = np.linalg.norm(q_tfidf)
        cosine = (self.tfidf_matrix @ (q_tfidf / norm)) if norm > 0 else np.zeros_like(bm25_scores)

        chunk_scores = (
            W_COVERAGE * coverage
            + W_BM25 * bm25_scores / max(float(bm25_scores.max()), 1e-9)
            + W_COSINE * cosine
        )

        # bonus pour la phrase exacte (requêtes de plusieurs mots)
        phrase = " ".join(re.findall(r"\w+", fold(query)))
        if phrase.count(" ") >= 1:
            for idx in np.argsort(chunk_scores)[::-1][:100]:
                if chunk_scores[idx] <= 0:
                    break
                s, e = self.chunk_span[idx]
                folded = " ".join(re.findall(r"\w+", fold(self.original_texts[self.chunk_doc[idx]][s:e])))
                if phrase in folded:
                    chunk_scores[idx] = min(1.0, chunk_scores[idx] * 1.2 + 0.05)

        # agrégation par document : meilleur passage + petit bonus des suivants
        per_doc: dict[int, list] = defaultdict(list)
        for idx in np.flatnonzero(chunk_scores > 0):
            per_doc[int(self.chunk_doc[idx])].append((float(chunk_scores[idx]), int(idx)))
        query_stems = {t for t in weights if not self._is_bigram(t)}
        scored = []
        for d, items in per_doc.items():
            items.sort(reverse=True)
            score = items[0][0] + 0.1 * min(1.0, sum(s for s, _ in items[1:3]))
            if self.name_tokens[d] & query_stems:  # le titre/nom du fichier correspond
                score += 0.08 * len(self.name_tokens[d] & query_stems) / max(len(query_stems), 1)
            scored.append((min(score, 1.0), d, items[0][1]))
        scored.sort(reverse=True)
        if not scored:
            return {**empty, "corrections": corrections, "unknown": unknown_words}

        threshold = MIN_RELATIVE_SCORE * scored[0][0]
        results = []
        for score, d, best_chunk in scored:
            if score < threshold:
                break
            snippet = self._snippet(d, best_chunk, query_stems)
            results.append({
                "name": self.names[d],
                "score": score,
                "snippet": snippet,
                "snippet_html": self._highlight(snippet, query_stems, self.language),
            })
            if len(results) >= top_k:
                break
        return {"results": results, "corrections": corrections, "unknown": unknown_words}

    # ------------------------------------------------------------------ extraits
    def _snippet(self, doc: int, chunk: int, stems: set[str], width: int = 420) -> str:
        text = self.original_texts[doc]
        c_start, c_end = self.chunk_span[chunk]
        hits = [m.start() for m in re.finditer(r"\w+", text[c_start:c_end]) if normalize_word(m.group(), self.language) in stems]
        hits = [c_start + h for h in hits]
        if hits:  # fenêtre contenant le plus de termes de la requête
            best = max(hits, key=lambda h: sum(1 for x in hits if h <= x < h + width))
            start = max(c_start, best - 80)
        else:
            start = c_start
        end = min(len(text), max(start + width, min(c_end, start + width)))
        if start > 0:  # ne pas couper un mot
            nxt = re.search(r"\s", text[start:end])
            start = start + nxt.end() if nxt and start != c_start else start
        if end < len(text):
            prv = text.rfind(" ", start, end)
            end = prv if prv > start + width // 2 else end
        excerpt = re.sub(r"\s+", " ", text[start:end]).strip()
        return ("…" if start > 0 else "") + excerpt + ("…" if end < len(text) else "")

    @staticmethod
    def _highlight(snippet: str, stems: set[str], language: str = "fr") -> str:
        out, last = [], 0
        for m in re.finditer(r"\w+", snippet):
            out.append(html.escape(snippet[last:m.start()]))
            word = m.group()
            if normalize_word(word, language) in stems:
                out.append(f"<mark>{html.escape(word)}</mark>")
            else:
                out.append(html.escape(word))
            last = m.end()
        out.append(html.escape(snippet[last:]))
        return "".join(out)
