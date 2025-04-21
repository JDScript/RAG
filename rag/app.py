import dataclasses
from pathlib import Path
import sys

import gradio as gr
import ollama
from ollama import Message

sys.path.append(str(Path()))

from rag.infrastructure.retriever import ContextRetriever
from rag.settings import settings
from rag.utils import get_video_clip

gr.set_static_paths(paths=[Path.cwd().absolute() / "temp"])

MODEL_NAME = ""
ollama_client = ollama.Client()
retriever = ContextRetriever()


class ChatMessage(Message):
    is_context: bool = False


@dataclasses.dataclass
class ChatbotInstance:
    system: str
    model: str
    history: list[ChatMessage] = dataclasses.field(default_factory=list)

    def to_gradio_chat_message(self):
        return [
            {"role": message.role, "content": message.content}
            if not message.is_context
            else {
                "role": "assistant",
                "content": message.content,
                "metadata": {"title": "Context", "status": "done" if message.content else "pending"},
            }
            for message in self.history
        ]

    def append_user_message(self, content: str):
        if not self.history:
            self.history.append(ChatMessage(role="system", content=self.system))

        self.history.append(ChatMessage(role="user", content=content))
        return self.to_gradio_chat_message(), ""

    def request_response(self):
        assert self.history[-1].role == "user", "Last message must be from user"

        self.history.append(
            ChatMessage(
                role="user",
                content="",
                is_context=True,
            )
        )
        yield self.to_gradio_chat_message()

        # Retrieve context
        caption_context, image_context = retriever.retrieve_context(self.history[-1]["content"])

        self.history[-1].content = "\n\n".join([chunk.to_context() for chunk in caption_context])
        yield self.to_gradio_chat_message()

        self.history.append(ChatMessage(role="assistant", content=""))
        response = ollama_client.chat(
            model=self.model,
            messages=self.history,
            stream=True,
        )

        for chunk in response:
            if chunk.message.role == "assistant":
                self.history[-1].content += chunk.message.content
                yield self.to_gradio_chat_message()

        # Process video clip
        video_id = caption_context[0].video_id
        start_ms = caption_context[0].start_ms
        end_ms = caption_context[0].end_ms
        self.history.append(
            ChatMessage(
                role="assistant",
                content="",
                is_context=True,
            )
        )
        yield self.to_gradio_chat_message()
        video_clip_path = get_video_clip(video_id, start_ms, end_ms)
        self.history[
            -1
        ].content = f"<video src='/gradio_api/file={video_clip_path}' controls width='640' height='360'></video>"
        yield self.to_gradio_chat_message()


def append_user_message(state: ChatbotInstance, message: str):
    return state.append_user_message(message)


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
