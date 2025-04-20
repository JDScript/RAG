import gradio as gr

gr.load_chat("http://localhost:11434/v1/", model="gemma3:4b", token="***").launch()
