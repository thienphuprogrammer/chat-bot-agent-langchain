import uuid
from typing import Any, Optional, Dict

import gradio as gr
from dotenv import load_dotenv

from backend import MemoryTypes
from backend.src import Bot
from backend.src.core.models import ModelTypes
from backend.src.utils import CacheTypes

load_dotenv()


class BaseGradioUI:
    def __init__(
            self,
            bot: Bot = None,
            bot_memory: Optional[MemoryTypes] = None,
            bot_model: Optional[ModelTypes] = None,
            bot_cache: Optional[CacheTypes] = None,
    ):
        # self.bot = bot if bot is not None else Bot(memory=bot_memory, model=bot_model, cache=bot_cache)
        self._conversation_id = None
        self._uploaded_files: Dict[str, str] = {}  # Store uploaded files temporarily

    @staticmethod
    def create_conversation_id():
        return str(uuid.uuid4())

    def user_state(self, message: str, chat_history: Any, conversation_id, uploaded_file):
        if not conversation_id:
            conversation_id = self.create_conversation_id()

        if uploaded_file:
            self._uploaded_files[conversation_id] = uploaded_file

        return "", chat_history + [[message, None]], conversation_id

    def respond(self, conversation_id, chat_history):
        message = chat_history[-1][0]
        uploaded_file = self._uploaded_files.get(conversation_id, None)

        # Send the file and message to the bot
        result = self.bot.predict(sentence=message, conversation_id=conversation_id)
        chat_history[-1][-1] = result.message
        return chat_history

    def start_demo(self, port=8000, debug=False, share=True):
        with gr.Blocks() as demo:
            conversation_id_state = gr.State("")
            gr.Markdown("""<h1><center> Bot MiVa </center></h1>""")
            chatbot = gr.Chatbot(label="Assistant", height=400)

            with gr.Row():
                message = gr.Textbox(show_label=False,
                                     placeholder="Enter your prompt and press enter",
                                     visible=True)

            with gr.Row():
                file_upload = gr.UploadButton(label="Upload PDF", file_types=[".pdf"])

            btn_refresh = gr.ClearButton(components=[message, chatbot, file_upload],
                                         value="Refresh the conversation history")

            def clear_user_state():
                return {conversation_id_state: ""}

            message.submit(
                self.user_state,
                inputs=[message, chatbot, conversation_id_state, file_upload],
                outputs=[message, chatbot, conversation_id_state],
                queue=False
            ).then(
                self.respond,
                inputs=[conversation_id_state, chatbot],
                outputs=[chatbot]
            )
            btn_refresh.click(clear_user_state, outputs=conversation_id_state)
        demo.queue()
        demo.launch(debug=debug, server_port=port, share=share)


if __name__ == "__main__":
    # bot = Bot(
    #     memory=MemoryTypes.CUSTOM_MEMORY,
    #     model=ModelTypes.LLAMA_OLLAMA,
    #     cache=None,
    # )
    demo = BaseGradioUI(
        # bot=bot
    )
    demo.start_demo()
