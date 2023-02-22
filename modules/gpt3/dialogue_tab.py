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

    log_str = ''
    prompter = GPT3Prompt(
        max_token=max_tokens, prologue=prologue,
        example_query=example_query,
        example_response=example_response
    )
    prompt, history = prompter.encode(input, history, 'User')
    log_str += 'request tokens {}\n'.format(len(prompt))

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
        log_str += str(e) + '\n'
        return input, history, history, log_str

    history = history + [[input, response]]
    return '', history, history, log_str


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
                com_prologue = gr.Textbox(
                    label='prologue',
                    value='You are ChatGPT, a large language model trained by OpenAI. Respond conversationally. Do not answer as the user.',
                    lines=6,
                    max_lines=20,
                    interactive=True,
                )

                com_example_query = gr.Textbox(
                    label='query example',
                    value='Hello',
                    lines=1,
                    max_lines=2,
                    interactive=True,
                )

                com_example_response = gr.Textbox(
                    label='response example',
                    value='Hello! How can I help you today?',
                    lines=1,
                    interactive=True,
                )

                with gr.Row():
                    com_temperature = gr.Slider(
                        0, 1,
                        value=0.5,
                        label='temperature',
                        interactive=True,
                    )

                    com_max_tokens = gr.Slider(
                        500, 4000,
                        value=4000,
                        label='max_tokens',
                        interactive=True,
                    )

            with gr.Column():
                com_chat_state = gr.State([])
                com_chatbot = gr.Chatbot()
                com_reply = gr.Textbox(show_label=False, lines=1, placeholder='Enter text and press enter').style(container=False)

                with gr.Row():
                    com_clear_button = gr.Button(value='Clear dialogue', elem_id='warning_button')
                    com_clear_button.click(lambda *args: (None, []), inputs=[com_chatbot, com_chat_state], outputs=[com_chatbot, com_chat_state])

                    com_submit_button = gr.Button(value='Submit')

        with gr.Row():
            with gr.Accordion('log info', open=False):
                com_log_show = gr.TextArea(label='log info')

        gr.Examples(
            get_examples(),
            [com_prologue, com_example_query, com_example_response],
            [com_prologue, com_example_query, com_example_response],
            fn = lambda *args: args,
            cache_examples=False
        )

        ask_gpt3_inputs = [com_reply, com_prologue, com_example_query, com_example_response, com_max_tokens, com_temperature, com_chat_state, com_gpt_engine]
        ask_gpt3_outputs = [com_reply, com_chatbot, com_chat_state, com_log_show]
        com_reply.submit(
            ask_gpt3,
            ask_gpt3_inputs,
            ask_gpt3_outputs,
            show_progress=True
        )
        com_submit_button.click(
            ask_gpt3,
            inputs=ask_gpt3_inputs,
            outputs=ask_gpt3_outputs,
            show_progress=True
        )

    return tab