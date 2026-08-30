def extract_sources(documents):
    """
    Extrait les sources uniques des documents utilisés
    par le RAG.

    Une source est définie par :
        fichier + page
    """

    sources = []
    seen_sources = set()

    for doc in documents:

        filename = doc.metadata.get(
            "filename",
            "Inconnu"
        )

        page = doc.metadata.get(
            "page",
            "?"
        )

        source_key = (
            filename,
            page
        )

        if source_key in seen_sources:
            continue

        seen_sources.add(source_key)

        sources.append({
            "filename": filename,
            "page": page
        })

    return sources