import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core.indexer import SearchIndex
from core.serializer import load_documents, parse_csv
from eval_corpus import DOCS, QUERIES


def test_accuracy_on_corpus():
    idx = SearchIndex(DOCS)
    hits = total = 0
    for q, expected in QUERIES:
        res = idx.search(q)
        if expected is None:
            assert res == []
            continue
        total += 1
        hits += bool(res) and res[0]["name"] == expected
    assert hits / total >= 0.95


def test_csv_semicolon_rows_and_accents():
    data = "id;titre;texte\n1;Café;Le café noir\n2;Thé;Le thé vert\n".encode("cp1252")
    docs = load_documents(data, "t.csv", "rows")
    assert len(docs) == 2 and any("Café" in k for k in docs)
    idx = SearchIndex(docs)
    assert "ligne 1" in idx.search("cafe")[0]["name"]


def test_csv_comma_utf8_bom_and_file_mode():
    data = "﻿a,b\nx,y z\n,\n1,2\n".encode("utf-8")
    h, rows = parse_csv(data, "x.csv")
    assert h == ["a", "b"] and len(rows) == 2
    assert list(load_documents(data, "x.csv", "file")) == ["x.csv"]


def test_long_document_passage_snippet():
    filler = " ".join(["lorem ipsum dolor"] * 400)
    docs = {"long.txt": filler + " La fusée Ariane décolle de Kourou. " + filler, "autre.txt": filler}
    res = SearchIndex(docs).search("fusée Ariane")
    assert res[0]["name"] == "long.txt" and "Ariane" in res[0]["snippet"] and "<mark>" in res[0]["snippet_html"]


def test_html_is_escaped():
    res = SearchIndex({"a.txt": "bonjour <script>alert(1)</script> monde"}).search("bonjour")
    assert "<script>" not in res[0]["snippet_html"]


def test_empty_cases():
    assert SearchIndex({"a.txt": "bonjour"}).search("le la les") == []
    assert SearchIndex({"a.txt": "bonjour"}).search("zzzzqqq") == []


def test_reuters_csv_uses_content_and_ignores_technical_ids():
    data = (
        ",newID,oldID,title,author,data_split,isTopicsAvailable,document_date,dateline,people,places,orgs,exchanges,topic_body,topics\n"
        "0,123,999,COCOA MARKET,,TRAIN,YES,26-FEB-1987, SALVADOR,Feb 26 - ,[],['usa'],[],[],Cocoa prices rose sharply,['cocoa']\n"
    ).encode("utf-8")
    docs = load_documents(data, "reuters21578_news.csv", "rows")
    assert len(docs) == 1
    content = next(iter(docs.values())).lower()
    assert "cocoa market" in content and "cocoa prices" in content
    assert "123" not in content and "999" not in content and "train" not in content


def test_english_preprocessing_removes_stopwords_and_numeric_noise():
    from core.preprocessor import tokenize
    tokens = tokenize("The cocoa prices rose by 05p in 1987", language="en")
    assert "the" not in tokens and "by" not in tokens
    assert "05p" not in tokens and "1987" not in tokens
    assert "cocoa" in tokens and "price" in tokens


def test_reuters_index_uses_english_pipeline():
    docs = {"Reuters — COCOA MARKET [ligne 1]": "The cocoa prices are rising quickly."}
    idx = SearchIndex(docs)
    assert idx.language == "en"
    assert idx.search("cocoa prices")
