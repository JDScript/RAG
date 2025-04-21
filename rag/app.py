import dataclasses
from pathlib import Path
import sys

import gradio as gr
import ollama

sys.path.append(str(Path()))

from rag.infrastructure.retriever import ContextRetriever
from rag.settings import settings

MODEL_NAME = ""

ollama_client = ollama.Client()

retriever = ContextRetriever()


@dataclasses.dataclass
class ChatbotInstance:
    system: str
    model: str
    history: list = dataclasses.field(default_factory=list)

    def append_user_message(self, content: str):
        if not self.history:
            self.history.append({"role": "system", "content": self.system})

        self.history.append({"role": "user", "content": content})
        return self.history

    def request_response(self):
        assert self.history[-1]["role"] == "user", "Last message must be from user"

        # Retrieve context
        context = retriever.retrieve_context(self.history[-1]["content"])
        self.history[-1]["content"] = context + "\n" + self.history[-1]["content"]

        self.history.append({"role": "assistant", "content": ""})

        response = ollama_client.chat(
            model=self.model,
            messages=self.history,
            stream=True,
        )

        for chunk in response:
            if chunk.message.role == "assistant":
                self.history[-1]["content"] += chunk.message.content
                yield self.history


def append_user_message(state: ChatbotInstance, message: str):
    state.append_user_message(message)
    return state.history, ""


def chat(state: ChatbotInstance):
    yield from state.request_response()


def get_chatbot_instance():
    return ChatbotInstance(
        system="You are a helpful assistant. Please respond to user's question according to the video context.",
        model=settings.VLM_MODEL,
    )


with gr.Blocks() as demo:
    # State
    instance = get_chatbot_instance()
    state = gr.State(instance)

    gr.Markdown("# Video RAG")
    chatbot = gr.Chatbot(type="messages")
    msg = gr.Textbox()
    msg.submit(append_user_message, [state, msg], [chatbot, msg], queue=False).then(chat, [state], [chatbot])

if __name__ == "__main__":
    demo.launch()
