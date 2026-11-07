from core.indexer import SearchIndex
from core.serializer import load_documents


class AdminController:
    """Couche de contrôle entre l'interface et le moteur."""

    def ingest(self, files, documents: dict, sources: dict, csv_mode: str = "rows"):
        """Lit des fichiers importés et met à jour `documents` / `sources` en place.

        files : objets avec .name et .getvalue() (st.file_uploader).
        Retourne (nombre_de_documents_ajoutés, liste_d_erreurs).
        """
        added, errors = 0, []
        for file in files:
            try:
                new_docs = load_documents(file.getvalue(), file.name, csv_mode)
            except Exception as exc:
                errors.append(f"{file.name} : {exc}")
                continue
            if not new_docs:
                errors.append(f"{file.name} : aucun texte exploitable.")
                continue
            self.remove_source(file.name, documents, sources)  # ré-import = remplacement
            documents.update(new_docs)
            sources[file.name] = list(new_docs)
            added += len(new_docs)
        return added, errors

    @staticmethod
    def remove_source(source: str, documents: dict, sources: dict):
        for name in sources.pop(source, []):
            documents.pop(name, None)

    def build_index(self, documents):
        if not documents:
            raise ValueError("La collection de documents est vide.")
        return SearchIndex(documents)
