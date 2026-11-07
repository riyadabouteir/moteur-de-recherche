"""Évalue un moteur sur le corpus de test : P@1, MRR. Usage : python tests/evaluate.py [chemin_projet]"""
import sys, os, importlib
root = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_corpus import DOCS, QUERIES
from core.indexer import SearchIndex

idx = SearchIndex(DOCS)
p1 = mrr = n = ok_empty = 0
for q, expected in QUERIES:
    res = [r["name"] for r in idx.search(q, top_k=10)]
    if expected is None:
        ok_empty += (not res); print(f"{'OK ' if not res else 'KO '} {q!r:35} -> {res[:2]} (attendu: aucun)"); continue
    n += 1
    rank = res.index(expected) + 1 if expected in res else None
    p1 += rank == 1; mrr += 1 / rank if rank else 0
    print(f"{'OK ' if rank == 1 else 'KO '} {q!r:35} -> rang {rank}")
print(f"\nPrécision@1 = {p1/n:.0%}   MRR = {mrr/n:.3f}   requêtes vides ok = {ok_empty}/1")
