class PromptBuilder:

    def build_prompt(
        self,
        question,
        documents
    ):

        context = ""

        for i, doc in enumerate(
            documents,
            start=1
        ):

            filename = doc.metadata.get(
                "filename",
                "Inconnu"
            )

            page = doc.metadata.get(
                "page",
                "?"
            )

            context += f"""
[SOURCE {i}]
Fichier : {filename}
Page : {page}

Contenu :
{doc.page_content}

-------------------------
"""


        prompt = f"""
Tu es un assistant spécialisé dans la documentation
technique d'ELCS Research.

Réponds à la question uniquement à partir des
informations présentes dans les sources fournies.

Si les informations nécessaires ne sont pas présentes
dans les sources, indique que tu n'as pas trouvé
l'information dans la documentation.

Ne crée aucune information qui n'est pas présente
dans les sources.

QUESTION :
{question}

DOCUMENTATION :

{context}

INSTRUCTIONS IMPORTANTES :

1. Réponds directement à la question.
2. Utilise uniquement la documentation fournie.
3. Si plusieurs sources sont nécessaires pour construire
   la réponse, utilise-les toutes.
4. À la fin de ta réponse, indique uniquement les numéros
   des sources réellement utilisées.

Format obligatoire :

SOURCES: 1, 3

Si une seule source est utilisée :

SOURCES: 1

Si aucune source ne permet de répondre :

SOURCES:
"""

        return prompt