# -*- coding: utf-8 -*-

import logging

import gradio as gr
from dotenv import dotenv_values

from modules.gpt3.dialogue_tab import create_tab
from modules.log import get_logger


logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    level=logging.DEBUG
)
logger = get_logger(__name__)

CONFIG = dotenv_values('.env')


with gr.Blocks(title='AIGC Prompt WebUI') as webui_display:
    tab = create_tab(CONFIG)


webui_display.launch(server_name=CONFIG['server_ip'], server_port=int(CONFIG['server_port']))