# -*- coding: utf-8 -*-

import re
import csv
import json

import openai
import gradio as gr

from modules.prompt import GPT3Prompt
from modules.log import get_logger


logger = get_logger(__name__)


def get_introduction():
    with open('doc/md/gpt3_introduction.md', 'r', encoding='utf-8') as f:
        introduction = f.read()
    return introduction

def get_examples():
    examples = []
    with open('examples/gpt3_example.csv', 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for items in reader:
            if len(items) == 3:
                examples.append(items)

        logger.info('Loading example succeeded %s', len(examples))
    return examples

## 组件回调函数
def key_change(key):
    openai.api_key = key

def parse_completion(completion: dict) -> str:
    if completion.get('choices') is None:
        raise Exception('ChatGPT API returned no choices')

    if len(completion['choices']) == 0:
        raise Exception('ChatGPT API returned no choices')

    if completion['choices'][0].get('text') is None:
        raise Exception('ChatGPT API returned no text')

    response = re.sub(r'<\|im_end\|>', '', completion['choices'][0]['text'])

    completion['choices'][0]['text'] = response
    return response

def ask_gpt3(
        input,
        prologue,
        example_query,
        example_response,
        max_tokens,
        temperature,
        history,
        com_gpt_engine,
        request: gr.Request
    ):

    logger.debug('accept request from %s user-agent %s', request.client.host, request.headers['user-agent'])
    logger.debug('query %s', input)

    prompter = GPT3Prompt(
        max_token=max_tokens, prologue=prologue,
        example_query=example_query,
        example_response=example_response
    )
    prompt, history = prompter.encode(input, history, 'User')

    log_req_info = {
        'prologue': prologue,
        'example_query': example_query,
        'example_response': example_response,
        'max_tokens': max_tokens,
        'temperature': temperature,
        'input': input,
        'history': history,
    }

    logger.info('OpenAI Request From %s data : %s', request.client.host, json.dumps(log_req_info, ensure_ascii=False))

    try:
        completion = openai.Completion.create(
            engine=com_gpt_engine,
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens - len(prompt),
            stop=['\n\n\n'],
            stream=False
        )
        response = parse_completion(completion)
    except Exception as e:
        logger.error(e, exc_info=True)
        return input, history, history

    history = history + [[input, response]]
    return '', history, history


def create_tab(CONFIG):

    OPENAI_API_KEY = CONFIG['OPENAI_API_KEY']
    if OPENAI_API_KEY:
        openai.api_key = OPENAI_API_KEY

    with gr.Tab('Dialog Robot(GPT-3)', css='#warning_button {color: red}') as tab:
        with gr.Accordion('点击查看详细说明', open=False):
            gr.Markdown(get_introduction())

        with gr.Row():
            com_openai_key = gr.Textbox(
                label='OpenAI KEY',
                value=CONFIG['OPENAI_API_KEY'],
                lines=1,
                max_lines=1,
                interactive=True,
                type='password',
            )
            com_openai_key.change(key_change, inputs=com_openai_key)

            com_gpt_engine = gr.Textbox(
                label='GPT Engine',
                value=CONFIG['GPT_ENGINE'],
                lines=1,
                max_lines=1,
                interactive=True,
            )

        with gr.Row():
            with gr.Column():
                prologue = gr.Textbox(
                    label='prologue',
                    value='You are ChatGPT, a large language model trained by OpenAI. Respond conversationally. Do not answer as the user.',
                    lines=6,
                    max_lines=20,
                    interactive=True,
                )

                example_query = gr.Textbox(
                    label='query example',
                    value='Hello',
                    lines=1,
                    max_lines=2,
                    interactive=True,
                )

                example_response = gr.Textbox(
                    label='response example',
                    value='Hello! How can I help you today?',
                    lines=1,
                    interactive=True,
                )

                with gr.Row():
                    temperature = gr.Slider(
                        0, 1,
                        value=0.5,
                        label='temperature',
                        interactive=True,
                    )

                    max_tokens = gr.Slider(
                        500, 4000,
                        value=4000,
                        label='max_tokens',
                        interactive=True,
                    )

            with gr.Column():
                state = gr.State([])
                chatbot = gr.Chatbot()
                txt = gr.Textbox(show_label=False, lines=2, placeholder='Enter text and press enter').style(container=False)

                txt.submit(
                    ask_gpt3,
                    [txt, prologue, example_query, example_response, max_tokens, temperature, state, com_gpt_engine],
                    [txt, chatbot, state],
                    show_progress=True
                )

                with gr.Row():
                    clear_button = gr.Button(value='Clear dialogue', elem_id='warning_button')
                    clear_button.click(lambda *args: (None, []), inputs=[chatbot, state], outputs=[chatbot, state])

                    submit_button = gr.Button(value='Submit')
                    submit_button.click(
                        ask_gpt3,
                        inputs=[txt, prologue, example_query, example_response, max_tokens, temperature, state, com_gpt_engine],
                        outputs=[txt, chatbot, state],
                        show_progress=True
                    )

        gr.Examples(
            get_examples(),
            [prologue, example_query, example_response],
            [prologue, example_query, example_response],
            fn = lambda *args: args,
            cache_examples=False
        )
        
    return tab