import csv
import io

import numpy as np
import pandas as pd
import streamlit as st

from controller.admin_controller import AdminController
from core.serializer import SUPPORTED_EXTENSIONS

st.set_page_config(page_title="Moteur de recherche", page_icon="🔎", layout="wide")
controller = AdminController()

st.title("🔎 Moteur de recherche documentaire")
st.caption("Importez vos documents (TXT, PDF, DOCX, CSV…), construisez un index et recherchez leur contenu.")

st.session_state.setdefault("documents", {})
st.session_state.setdefault("sources", {})
st.session_state.setdefault("search_index", None)
st.markdown("<style>mark{background:#ffe58a;padding:0 2px;border-radius:3px}</style>", unsafe_allow_html=True)

with st.sidebar:
    st.header("📁 Ajouter des documents")
    uploaded = st.file_uploader(
        "Choisissez des fichiers",
        type=SUPPORTED_EXTENSIONS,
        accept_multiple_files=True,
        help="Formats acceptés : TXT, MD, PDF, DOCX, CSV et TSV.",
    )
    csv_label = st.radio(
        "Fichiers CSV / TSV",
        ["Une ligne = un document", "Fichier entier = un document"],
        help="« Une ligne = un document » donne des résultats plus précis pour les tableaux de données.",
    )
    csv_mode = "rows" if csv_label.startswith("Une ligne") else "file"

    if st.button("Indexer les documents", type="primary", use_container_width=True):
        if not uploaded:
            st.warning("Veuillez d'abord importer au moins un document.")
        else:
            with st.spinner("Lecture et indexation…"):
                added, errors = controller.ingest(
                    uploaded, st.session_state.documents, st.session_state.sources, csv_mode
                )
                if st.session_state.documents:
                    st.session_state.search_index = controller.build_index(st.session_state.documents)
            if added:
                st.success(f"{added} document(s) ajouté(s) et indexé(s).")
            for err in errors:
                st.error(err)

    st.divider()
    st.subheader("Gestion de l'index")
    if st.session_state.sources:
        to_remove = st.multiselect("Retirer des fichiers", list(st.session_state.sources))
        if to_remove and st.button("Retirer la sélection", use_container_width=True):
            for src in to_remove:
                controller.remove_source(src, st.session_state.documents, st.session_state.sources)
            st.session_state.search_index = (
                controller.build_index(st.session_state.documents) if st.session_state.documents else None
            )
            st.rerun()
    if st.button("Effacer tous les documents", use_container_width=True):
        st.session_state.documents = {}
        st.session_state.sources = {}
        st.session_state.search_index = None
        st.rerun()
    st.caption("Le contenu des documents reste dans la session Streamlit en cours.")

    st.divider()
    st.subheader("Réglages de recherche")
    top_k = st.slider("Nombre de résultats", 3, 30, 10)
    fuzzy = st.toggle("Tolérer les fautes de frappe", value=True)

left, right = st.columns([2, 1])
with left:
    st.subheader("Recherche")
    with st.form("search_form", border=False):  # la touche Entrée lance la recherche
        query = st.text_input("Votre requête", placeholder="Ex. : intelligence artificielle, apprentissage automatique…")
        search_clicked = st.form_submit_button("Rechercher", type="primary")

with right:
    st.subheader("État de l'index")
    st.metric("Documents indexés", len(st.session_state.documents))
    st.metric("Fichiers sources", len(st.session_state.sources))
    total_words = sum(len(t.split()) for t in st.session_state.documents.values())
    st.metric("Mots dans la collection", f"{total_words:,}".replace(",", " "))

if st.session_state.documents:
    with st.expander("Consulter les documents indexés", expanded=False):
        items = list(st.session_state.documents.items())
        for name, content in items[:50]:
            st.markdown(f"**{name}** — {len(content.split())} mots")
            st.caption(content[:350].replace("\n", " ") + ("…" if len(content) > 350 else ""))
        if len(items) > 50:
            st.caption(f"… et {len(items) - 50} autres documents.")


# -----------------------------------------------------------------------------
# Exploration des matrices construites pendant l'indexation
# -----------------------------------------------------------------------------
if st.session_state.search_index is not None:
    index = st.session_state.search_index
    st.divider()
    with st.expander("🧮 Explorer les matrices de l'index", expanded=False):
        st.markdown(
            "Les lignes représentent les passages (*chunks*) des documents. "
            "Les colonnes représentent les termes du vocabulaire (mots racinisés "
            "et expressions de deux mots). Pour garder l'application réactive, "
            "seule la portion choisie est convertie en tableau dense."
        )
        if index.language == "en":
            st.info(
                "Profil Reuters/anglais détecté : les identifiants techniques, "
                "les dates de stockage et les champs TRAIN/TEST ne sont pas indexés. "
                "Le moteur utilise les stopwords et la racinisation anglaise. "
                f"{index.discarded_rare_bigrams:,} bigrammes présents une seule fois "
                "ont été écartés."
            )
        else:
            st.caption("Profil français par défaut : stopwords et racinisation française.")

        matrix_options = {
            "Term Frequency (TF)": index.tf_matrix,
            "BM25": index.bm25,
            "TF-IDF": index.tfidf_matrix,
            "Présence (0/1)": index.presence,
        }
        selected_matrix_name = st.selectbox(
            "Matrice à afficher", list(matrix_options.keys()), key="matrix_choice"
        )
        selected_matrix = matrix_options[selected_matrix_name]
        n_rows, n_cols = selected_matrix.shape
        terms = index.vectorizer.get_feature_names_out()

        metric_cols = st.columns(3)
        metric_cols[0].metric("Dimensions", f"{n_rows} × {n_cols}")
        metric_cols[1].metric("Valeurs non nulles", f"{selected_matrix.nnz:,}".replace(",", " "))
        density = selected_matrix.nnz / (n_rows * n_cols) if n_rows and n_cols else 0.0
        metric_cols[2].metric("Densité", f"{density:.2%}")

        if n_rows == 0 or n_cols == 0:
            st.info("Cette matrice est vide : aucun passage ou terme à afficher.")
        else:
            st.caption("Choisissez les passages et la manière de sélectionner les colonnes.")
            control_cols = st.columns(4)
            with control_cols[0]:
                row_start = st.number_input(
                    "Première ligne (passage)", min_value=0,
                    max_value=n_rows - 1, value=0, step=1, key="matrix_row_start"
                )
            with control_cols[1]:
                row_count = st.number_input(
                    "Nombre de lignes", min_value=1,
                    max_value=min(100, n_rows), value=min(15, n_rows), step=1,
                    key="matrix_row_count"
                )
            with control_cols[2]:
                col_count = st.number_input(
                    "Nombre de colonnes", min_value=1,
                    max_value=min(100, n_cols), value=min(15, n_cols), step=1,
                    key="matrix_col_count"
                )
            with control_cols[3]:
                column_mode = st.selectbox(
                    "Sélection des colonnes",
                    ["Pertinent (plus de valeurs non nulles)", "Original (ordre du vocabulaire)"],
                    key="matrix_column_mode",
                    help=(
                        "Le mode pertinent choisit, parmi les passages affichés, les termes "
                        "présents dans le plus grand nombre de passages."
                    ),
                )

            row_end = min(int(row_start) + int(row_count), n_rows)
            selected_rows = selected_matrix[int(row_start):row_end]

            if column_mode.startswith("Pertinent"):
                # Classe les termes selon le nombre de passages sélectionnés où ils apparaissent.
                # On travaille directement sur la matrice creuse : aucune conversion dense complète.
                column_counts = np.asarray(selected_rows.getnnz(axis=0)).ravel()
                ranked_cols = np.argsort(-column_counts, kind="stable")
                selected_cols = ranked_cols[:int(col_count)]
                visible_counts = column_counts[selected_cols]
                st.caption(
                    f"Mode pertinent : {int(np.count_nonzero(column_counts)):,} termes apparaissent "
                    f"dans les passages sélectionnés. Les {len(selected_cols)} termes les plus présents sont affichés."
                )
            else:
                col_start = st.number_input(
                    "Première colonne (ordre du vocabulaire)", min_value=0,
                    max_value=n_cols - 1, value=0, step=1, key="matrix_col_start"
                )
                selected_cols = np.arange(
                    int(col_start), min(int(col_start) + int(col_count), n_cols)
                )
                visible_counts = np.asarray(selected_rows[:, selected_cols].getnnz(axis=0)).ravel()

            dense_part = selected_rows[:, selected_cols].toarray()
            visible_terms = terms[selected_cols]

            row_labels = []
            for chunk_id in range(int(row_start), row_end):
                doc_id = int(index.chunk_doc[chunk_id])
                doc_name = index.names[doc_id]
                start, end = index.chunk_span[chunk_id]
                row_labels.append(f"Passage {chunk_id + 1} | {doc_name} | caractères {start}-{end}")

            matrix_df = pd.DataFrame(dense_part, columns=visible_terms, index=row_labels)
            st.dataframe(matrix_df, use_container_width=True)
            if column_mode.startswith("Pertinent"):
                with st.expander("Voir le nombre de passages contenant chaque terme affiché"):
                    counts_df = pd.DataFrame({
                        "Terme": visible_terms,
                        "Nombre de passages non nuls": visible_counts,
                    })
                    st.dataframe(counts_df, use_container_width=True, hide_index=True)
            st.download_button(
                "⬇️ Télécharger cette portion de matrice (CSV)",
                matrix_df.to_csv(index=True).encode("utf-8-sig"),
                file_name=f"matrice_{selected_matrix_name.lower().replace(' ', '_').replace('/', '-')}.csv",
                mime="text/csv",
                key="download_matrix_slice",
            )

        st.divider()
        st.subheader("Vocabulaire et valeurs IDF")
        idf_df = pd.DataFrame({"Terme": terms, "IDF": index.idf})
        st.caption(f"{len(idf_df)} termes dans le vocabulaire. Les termes avec un IDF élevé sont plus rares dans les passages.")
        st.dataframe(idf_df.head(100), use_container_width=True, hide_index=True)
        if len(idf_df) > 100:
            st.caption("Les 100 premiers termes sont affichés. Téléchargez le vocabulaire complet ci-dessous.")
        st.download_button(
            "⬇️ Télécharger le vocabulaire complet et les IDF (CSV)",
            idf_df.to_csv(index=False).encode("utf-8-sig"),
            file_name="vocabulaire_idf.csv",
            mime="text/csv",
            key="download_vocabulary_idf",
        )

if search_clicked:
    if not query.strip():
        st.warning("Saisissez une requête.")
    elif st.session_state.search_index is None:
        st.info("Importez et indexez d'abord des documents.")
    else:
        found = st.session_state.search_index.search_detailed(query, top_k=top_k, fuzzy=fuzzy)
        results = found["results"]
        st.divider()
        st.subheader(f"Résultats pour : « {query} »")
        for typed, fixed in found["corrections"]:
            if typed.lower() != fixed.lower():
                st.caption(f"🔧 « {typed} » a été interprété comme « {fixed} ».")
        if found["unknown"]:
            st.caption("Termes absents de la collection : " + ", ".join(f"« {w} »" for w in found["unknown"]))
        if not results:
            st.info("Aucun résultat pertinent trouvé. Essayez avec d'autres mots-clés.")
        else:
            st.write(f"{len(results)} résultat(s) pertinent(s)")
            for rank, result in enumerate(results, start=1):
                with st.container(border=True):
                    c1, c2 = st.columns([5, 1])
                    with c1:
                        st.markdown(f"**{rank}. {result['name']}**")
                    with c2:
                        st.metric("Pertinence", f"{result['score']:.0%}")
                    st.markdown(result["snippet_html"], unsafe_allow_html=True)  # texte échappé, seuls <mark> ajoutés
            buf = io.StringIO()
            writer = csv.writer(buf, delimiter=";")
            writer.writerow(["rang", "document", "pertinence", "extrait"])
            for rank, r in enumerate(results, start=1):
                writer.writerow([rank, r["name"], f"{r['score']:.3f}", r["snippet"]])
            st.download_button(
                "⬇️ Télécharger les résultats (CSV)",
                buf.getvalue().encode("utf-8-sig"),
                file_name="resultats_recherche.csv",
                mime="text/csv",
            )
elif not st.session_state.documents:
    st.info("Importez des documents depuis le panneau latéral pour commencer.")
