import re


class RAGChain:

    def __init__(
        self,
        retriever,
        chatbot,
        prompt_builder
    ):

        self.retriever = retriever
        self.chatbot = chatbot
        self.prompt_builder = prompt_builder


    def answer_question(
        self,
        question,
        k=5
    ):

        if not question or not question.strip():

            return (
                "Veuillez entrer une question.",
                [],
                []
            )

        question = question.strip()


        # ==================================================
        # 1. RETRIEVAL
        # ==================================================

        documents = self.retriever.retrieve(
            question,
            k=k
        )


        # ==================================================
        # 2. DEBUG
        # ==================================================

        print("\n" + "=" * 60)
        print("DEBUG RAG")
        print("=" * 60)

        print(
            "Question :",
            question
        )

        print(
            "Nombre de documents sélectionnés :",
            len(documents)
        )


        for i, doc in enumerate(
            documents,
            start=1
        ):

            print(
                f"\n--- DOCUMENT {i} ---"
            )

            print(
                "Fichier :",
                doc.metadata.get(
                    "filename",
                    "Inconnu"
                )
            )

            print(
                "Page :",
                doc.metadata.get(
                    "page",
                    "?"
                )
            )

            print(
                "Distance :",
                doc.metadata.get(
                    "retrieval_distance",
                    "?"
                )
            )

            print(
                "Contenu :"
            )

            print(
                doc.page_content[:500]
            )


        # ==================================================
        # 3. AUCUN DOCUMENT
        # ==================================================

        if not documents:

            print(
                "\nAucun document pertinent trouvé."
            )

            return (
                "Je n'ai trouvé aucune information "
                "correspondante dans la documentation "
                "technique d'ELCS Research.",
                [],
                []
            )


        # ==================================================
        # 4. CONSTRUCTION DU PROMPT
        # ==================================================

        prompt = self.prompt_builder.build_prompt(
            question,
            documents
        )


        # ==================================================
        # 5. LLM
        # ==================================================

        raw_answer = self.chatbot.generate(
            prompt
        )


        # ==================================================
        # 6. EXTRACTION DES SOURCES
        # ==================================================

        used_source_numbers = self.extract_source_numbers(
            raw_answer
        )


        # ==================================================
        # 7. NETTOYAGE DE LA RÉPONSE
        # ==================================================

        answer = self.clean_answer(
            raw_answer
        )


        # ==================================================
        # 8. DOCUMENTS RÉELLEMENT UTILISÉS
        # ==================================================

        used_documents = []

        for number in used_source_numbers:

            index = number - 1

            if 0 <= index < len(documents):

                used_documents.append(
                    documents[index]
                )


        # Si le LLM n'a donné aucune source,
        # on ne prétend pas savoir quelles sources
        # ont réellement été utilisées.
        if not used_documents:

            print(
                "[RAG] Aucune source explicitement "
                "identifiée par le LLM."
            )


        print(
            f"[RAG] {len(used_documents)} source(s) "
            "explicitement utilisée(s)"
        )


        return (
            answer,
            documents,
            used_documents
        )


    # ======================================================
    # EXTRACTION DES NUMÉROS DE SOURCES
    # ======================================================

    def extract_source_numbers(
        self,
        answer
    ):

        match = re.search(
            r"SOURCES?\s*:\s*([0-9,\s]+)",
            answer,
            re.IGNORECASE
        )

        if not match:
            return []

        numbers = []

        for value in match.group(1).split(","):

            value = value.strip()

            if value.isdigit():

                number = int(value)

                if number not in numbers:

                    numbers.append(number)

        return numbers


    # ======================================================
    # NETTOYAGE DE LA RÉPONSE
    # ======================================================

    def clean_answer(
        self,
        answer
    ):

        answer = re.sub(
            r"\n*SOURCES?\s*:\s*[0-9,\s]+",
            "",
            answer,
            flags=re.IGNORECASE
        )

        return answer.strip()