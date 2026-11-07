# Moteur de recherche documentaire — Streamlit

Application de recherche dans une collection de documents, avec un moteur **hybride BM25 + TF-IDF** optimisé pour le français.

## Fonctionnalités
- Import de plusieurs fichiers **TXT, MD, PDF, DOCX, CSV et TSV**.
- **CSV / TSV** : détection automatique du séparateur (`,` `;` tab `|`) et de l'encodage (UTF-8, UTF-8 BOM, Windows-1252, UTF-16). Deux modes : *une ligne = un document* (recommandé) ou *fichier entier = un document*.
- Contenu des tableaux Word (DOCX) également indexé.
- Recherche tolérante : accents ignorés, pluriels/conjugaisons (racinisation française), mots vides retirés, fautes de frappe corrigées, préfixes.
- Classement par passages, bonus pour la phrase exacte et pour le titre/nom de fichier.
- Extraits centrés sur la zone pertinente, **mots recherchés surlignés**, score de pertinence en %.
- Export des résultats en CSV, retrait de fichiers individuels, nombre de résultats réglable.
- Explorateur interactif des matrices TF, BM25, TF-IDF et présence, avec sélection des lignes/colonnes et export CSV.
- Consultation et export du vocabulaire et des valeurs IDF.

## Précision
`python tests/evaluate.py` mesure la précision@1 sur un petit corpus français (24 requêtes : formes proches, sans accents, fautes de frappe, mots vides) :

| Version | Précision@1 | MRR |
|---|---|---|
| Ancienne (TF-IDF brut) | 83 % | 0,86 |
| Nouvelle (hybride) | 100 % | 1,00 |

Corpus petit : à titre indicatif. Ajoutez vos propres requêtes dans `tests/eval_corpus.py`.

## Installation
```bash
python -m venv .venv
source .venv/bin/activate        # Windows : .venv\Scripts\activate
pip install -r requirements.txt
```

## Lancement
```bash
streamlit run app.py
```

## Utilisation
1. Importez vos fichiers (exemple fourni : `sample_data/articles.csv`).
2. Choisissez le mode CSV, puis cliquez sur **Indexer les documents**.
3. Ouvrez **Explorer les matrices de l’index** pour visualiser les matrices TF, BM25, TF-IDF et présence. Vous pouvez choisir la portion affichée et télécharger les tableaux en CSV.
4. Saisissez une requête et validez (Entrée ou **Rechercher**).

## Tests
```bash
python -m pytest -q tests
```

## Structure
- `core/preprocessor.py` : normalisation, mots vides, racinisation.
- `core/indexer.py` : index hybride, requêtes, extraits.
- `core/serializer.py` : lecture des fichiers (dont CSV) et sauvegarde d'index.
- `controller/admin_controller.py` : import et indexation.
- `app.py` : interface Streamlit.

Limite : les racines sont françaises ; les textes anglais restent cherchables (correspondance exacte) mais sans lemmatisation.


## Prétraitement spécialisé Reuters-21578

Le fichier `reuters21578_news.csv` est détecté automatiquement grâce à ses colonnes `title` et `topic_body`. L'import indexe le titre, le corps de l'article, la dateline et les entités/topics disponibles. Les identifiants `newID`/`oldID`, le split `TRAIN`/`TEST`, le drapeau de disponibilité des topics et la date technique ne sont pas indexés.

Pour ce corpus anglais, le moteur sélectionne automatiquement les stopwords et le stemmer anglais. Les nombres isolés et les fragments alphanumériques commençant par un chiffre sont écartés. Les bigrammes qui n'apparaissent que dans un seul passage sont supprimés, tandis que les unigrammes rares sont conservés. Le panneau **Explorer les matrices** indique le nombre de bigrammes rares écartés. Les autres fichiers CSV gardent le comportement d'import générique.

### Exploration des matrices : mode pertinent

Dans « Explorer les matrices de l'index », le menu « Sélection des colonnes » propose deux modes :
- **Pertinent** : classe les termes selon le nombre de passages affichés dans lesquels ils ont une valeur non nulle, puis montre les termes les plus présents.
- **Original** : conserve l'ordre alphabétique du vocabulaire et permet de choisir la première colonne.

Le classement est calculé directement sur la matrice creuse et uniquement sur les lignes sélectionnées. Il ne modifie pas l'index ni les résultats de recherche.
