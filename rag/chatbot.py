from langchain_ollama import ChatOllama


class Chatbot:
    """
    Interface avec le modèle LLM utilisé
    pour générer les réponses.
    """

    def __init__(
        self,
        model="llama3.2",
        temperature=0
    ):

        self.llm = ChatOllama(
            model=model,
            temperature=temperature
        )

        print(
            f"[CHATBOT] Modèle chargé : {model}"
        )

    def generate(self, prompt):

        if not prompt or not prompt.strip():
            raise ValueError(
                "Le prompt ne peut pas être vide."
            )

        response = self.llm.invoke(
            prompt
        )

        return response.content