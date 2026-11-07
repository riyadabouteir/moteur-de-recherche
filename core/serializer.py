"""Lecture des fichiers (TXT, MD, PDF, DOCX, CSV/TSV) et sérialisation de l'index."""
import ast
import csv
import io
import re
from io import BytesIO

SUPPORTED_EXTENSIONS = ["txt", "md", "pdf", "docx", "csv", "tsv"]
CSV_EXTENSIONS = {"csv", "tsv"}
MAX_CSV_ROWS = 50_000
_TITLE_HINTS = ("titre", "title", "nom", "name", "sujet", "subject", "objet", "libelle", "libellé")


def get_extension(filename: str) -> str:
    return filename.lower().rsplit(".", 1)[-1] if "." in filename else ""


def decode_bytes(data: bytes) -> str:
    """Décode un fichier texte en détectant l'encodage courant (UTF-8, UTF-16, Windows-1252)."""
    if data.startswith((b"\xff\xfe", b"\xfe\xff")):
        return data.decode("utf-16")
    for encoding in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise ValueError("Encodage du fichier texte non reconnu.")


def parse_csv(data: bytes, filename: str = "fichier.csv"):
    """Retourne (en-têtes, lignes) d'un CSV, avec détection du séparateur et de l'encodage."""
    text = decode_bytes(data)
    if not text.strip():
        raise ValueError("Fichier CSV vide.")
    sample = text[:20_000]
    delimiter = "\t" if get_extension(filename) == "tsv" else None
    if delimiter is None:
        try:
            delimiter = csv.Sniffer().sniff(sample, delimiters=",;\t|").delimiter
        except csv.Error:
            first = sample.splitlines()[0] if sample.splitlines() else ""
            delimiter = max(",;\t|", key=first.count)
    csv.field_size_limit(10_000_000)
    reader = csv.reader(io.StringIO(text, newline=""), delimiter=delimiter)
    rows = [r for r in reader if any(c.strip() for c in r)]
    if not rows:
        raise ValueError("Fichier CSV vide.")
    headers = [h.strip() or f"colonne_{i + 1}" for i, h in enumerate(rows[0])]
    seen: dict[str, int] = {}
    for i, h in enumerate(headers):  # noms de colonnes uniques
        seen[h] = seen.get(h, 0) + 1
        if seen[h] > 1:
            headers[i] = f"{h}_{seen[h]}"
    return headers, rows[1:]


def _row_to_text(headers, row) -> str:
    return ". ".join(f"{h} : {v.strip()}" for h, v in zip(headers, row) if v and v.strip())


def csv_to_documents(data: bytes, filename: str, mode: str = "rows") -> dict[str, str]:
    """Transforme un CSV en documents.

    mode="rows" : une ligne = un document (recommandé pour des tableaux de données) ;
    mode="file" : tout le fichier = un seul document.
    """
    headers, rows = parse_csv(data, filename)
    if mode == "file":
        text = "\n".join(t for t in (_row_to_text(headers, r) for r in rows) if t)
        return {filename: text} if text.strip() else {}
    title_col = next((i for i, h in enumerate(headers) if h.lower() in _TITLE_HINTS), None)
    docs: dict[str, str] = {}
    for n, row in enumerate(rows[:MAX_CSV_ROWS], start=1):
        text = _row_to_text(headers, row)
        if not text:
            continue
        label = f" — {row[title_col].strip()[:50]}" if title_col is not None and title_col < len(row) and row[title_col].strip() else ""
        docs[f"{filename} · ligne {n}{label}"] = text
    return docs



def _reuters_list(value: str) -> str:
    """Convertit une colonne de liste Reuters (ex. ['usa', 'cocoa']) en texte propre."""
    value = (value or "").strip()
    if not value:
        return ""
    try:
        parsed = ast.literal_eval(value)
        if isinstance(parsed, (list, tuple, set)):
            return " ".join(str(item) for item in parsed if str(item).strip())
    except (ValueError, SyntaxError):
        pass
    return value


def _reuters_csv_to_documents(data: bytes, filename: str) -> dict[str, str] | None:
    """Import spécialisé Reuters-21578 : ignore les identifiants et champs techniques.

    Retourne None si le CSV ne possède pas le schéma Reuters attendu.
    """
    headers, rows = parse_csv(data, filename)
    normalized = {re.sub(r"[^a-z0-9]", "", h.lower()): i for i, h in enumerate(headers)}
    required = {"title", "topicbody"}
    if not required.issubset(normalized):
        return None

    # Champs de contenu utiles. Les ID, le split TRAIN/TEST, les dates techniques
    # et le drapeau isTopicsAvailable ne sont volontairement pas indexés.
    text_fields = ("title", "author", "dateline", "people", "places",
                   "orgs", "exchanges", "topic_body", "topics")
    list_fields = {"people", "places", "orgs", "exchanges", "topics"}
    docs: dict[str, str] = {}
    for row_number, row in enumerate(rows, start=1):
        parts = []
        for field in text_fields:
            key = re.sub(r"[^a-z0-9]", "", field.lower())
            pos = normalized.get(key)
            if pos is None or pos >= len(row):
                continue
            value = (row[pos] or "").strip()
            if not value:
                continue
            if field in list_fields:
                value = _reuters_list(value)
            if value:
                parts.append(value)
        content = "\n".join(parts).strip()
        if not content:
            continue
        title_pos = normalized.get("title")
        title = row[title_pos].strip() if title_pos is not None and title_pos < len(row) else ""
        label = title[:90] if title else f"Article {row_number}"
        # Le numéro de ligne garantit l'unicité même si deux titres sont identiques.
        docs[f"Reuters — {label} [ligne {row_number}]"] = content
    return docs


def extract_text(data: bytes, filename: str) -> str:
    """Extrait le texte d'un fichier TXT, MD, PDF, DOCX ou CSV/TSV."""
    extension = get_extension(filename)
    if extension in ("txt", "md"):
        return decode_bytes(data)
    if extension in CSV_EXTENSIONS:
        return "\n".join(csv_to_documents(data, filename, mode="file").values())
    if extension == "pdf":
        from pypdf import PdfReader
        reader = PdfReader(BytesIO(data))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    if extension == "docx":
        from docx import Document
        doc = Document(BytesIO(data))
        parts = [p.text for p in doc.paragraphs]
        for table in doc.tables:  # le contenu des tableaux Word est aussi indexé
            for row in table.rows:
                parts.append(" | ".join(c.text.strip() for c in row.cells))
        return "\n".join(parts)
    raise ValueError("Format non pris en charge.")


def load_documents(data: bytes, filename: str, csv_mode: str = "rows") -> dict[str, str]:
    """Point d'entrée : fichier -> {nom_du_document: texte}."""
    if get_extension(filename) in CSV_EXTENSIONS:
        # Détection automatique du corpus Reuters-21578 : indexer le contenu
        # éditorial plutôt que les identifiants et métadonnées de stockage.
        if get_extension(filename) == "csv":
            reuters_docs = _reuters_csv_to_documents(data, filename)
            if reuters_docs is not None:
                return reuters_docs
        return csv_to_documents(data, filename, csv_mode)
    text = extract_text(data, filename)
    return {filename: text} if text.strip() else {}


def serialize_index(index, path: str):
    """Enregistre l'index sur le disque."""
    import joblib
    joblib.dump(index, path)


def load_index(path: str):
    import joblib
    return joblib.load(path)
