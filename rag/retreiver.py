class Retriever:
    """
    Retriever basé sur FAISS.

    Pipeline :

    Question
        ↓
    FAISS
        ↓
    fetch_k candidats
        ↓
    Filtre par distance
        ↓
    Suppression des doublons
        ↓
    Maximum de chunks par page
        ↓
    Top K
    """

    def __init__(
        self,
        vector_store,
        fetch_k=20,
        distance_threshold=1.44,
        max_per_source=2
    ):

        if fetch_k <= 0:
            raise ValueError(
                "fetch_k doit être supérieur à 0."
            )

        if distance_threshold <= 0:
            raise ValueError(
                "distance_threshold doit être positif."
            )

        if max_per_source <= 0:
            raise ValueError(
                "max_per_source doit être supérieur à 0."
            )

        self.vector_store = vector_store
        self.fetch_k = fetch_k
        self.distance_threshold = distance_threshold
        self.max_per_source = max_per_source

        print(
            "[RETRIEVER] "
            f"fetch_k={fetch_k}, "
            f"threshold={distance_threshold}, "
            f"max_per_source={max_per_source}"
        )

    # =========================================================
    # RETRIEVAL PRINCIPAL
    # =========================================================

    def retrieve(self, question, k=5):

        if not question or not question.strip():
            return []

        if k <= 0:
            return []

        question = question.strip()

        # -----------------------------------------------------
        # 1. Recherche FAISS
        # -----------------------------------------------------

        results = (
            self.vector_store
            .similarity_search_with_score(
                query=question,
                k=self.fetch_k
            )
        )

        selected_documents = []

        seen_chunks = set()
        source_count = {}

        # -----------------------------------------------------
        # 2. Filtrage
        # -----------------------------------------------------

        for doc, distance in results:

            distance = float(distance)

            content = doc.page_content.strip()

            if not content:
                continue

            # -------------------------------------------------
            # Filtre de distance
            # -------------------------------------------------

            if distance > self.distance_threshold:
                continue

            filename = doc.metadata.get(
                "filename",
                "Inconnu"
            )

            page = doc.metadata.get(
                "page",
                "?"
            )

            # -------------------------------------------------
            # Suppression des chunks identiques
            # -------------------------------------------------

            normalized_content = " ".join(
                content.split()
            )

            chunk_key = (
                filename,
                page,
                normalized_content
            )

            if chunk_key in seen_chunks:
                continue

            seen_chunks.add(chunk_key)

            # -------------------------------------------------
            # Maximum de chunks par page
            # -------------------------------------------------

            source_key = (
                filename,
                page
            )

            count = source_count.get(
                source_key,
                0
            )

            if count >= self.max_per_source:
                continue

            source_count[source_key] = count + 1

            # -------------------------------------------------
            # Stocker la distance
            # -------------------------------------------------

            doc.metadata[
                "retrieval_distance"
            ] = distance

            selected_documents.append(doc)

            # -------------------------------------------------
            # Top K final
            # -------------------------------------------------

            if len(selected_documents) >= k:
                break

        # -----------------------------------------------------
        # DEBUG
        # -----------------------------------------------------

        print(
            f"[RETRIEVER FINAL] "
            f"{len(selected_documents)} document(s) sélectionné(s)"
        )

        return selected_documents

    # =========================================================
    # RECHERCHE BRUTE POUR DEBUG UNIQUEMENT
    # =========================================================

    def retrieve_with_score(
        self,
        question,
        k=10
    ):

        if not question or not question.strip():
            return []

        if k <= 0:
            return []

        return (
            self.vector_store
            .similarity_search_with_score(
                query=question.strip(),
                k=k
            )
        )