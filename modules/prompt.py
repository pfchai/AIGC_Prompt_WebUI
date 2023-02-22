# -*- coding: utf-8 -*-

from datetime import date

import tiktoken


ENCODER = tiktoken.get_encoding('gpt2')

HEADER_PROMPT_TEMPLATE = '''{prologue} {date}

{user}: {example_query}
ChatGPT: {example_response} <|im_end|>


'''

ROUND_PROMPT_TEMPLATE = '''{user}: {query}


ChatGPT: {response}<|im_end|>
'''

QUERY_TEMPLATE = '''{user}: {query}


ChatGPT:'''


class GPT3Prompt():
    def __init__(
            self, max_token=4000, buffer: int = None, prologue=None,
            example_query='Hello', example_response='Hello! How can I help you today?'
    ):
        buffer = buffer or 800
        self.max_token = max_token - buffer
        self.buffer = buffer
        self.default_prologue = prologue or (
            'You are ChatGPT, a large language model trained by OpenAI. '
            'Respond conversationally. Do not answer as the user.'
        )
        self.example_query = example_query
        self.example_response = example_response

    def update(self, max_token=None, prologue=None):
        if max_token is not None:
            self.max_token = max_token - self.buffer

        if prologue is not None:
            self.default_prologue = prologue

    def encode_header(self, user: str) -> str:
        return HEADER_PROMPT_TEMPLATE.format(
            prologue=self.default_prologue,
            date=str(date.today()),
            user=user,
            example_query=self.example_query,
            example_response=self.example_response,
        )

    def encode_history_round(self, query: str, response: str = None, user: str = 'User') -> str:
        return ROUND_PROMPT_TEMPLATE.format(
            user=user,
            query=query,
            response=response
        )

    def encode_history(self, history: list, user: str = 'User') -> str:
        return ''.join([
            self.encode_history_round(query, response, user)
            for [query, response] in history
        ])

    def encode_query(self, query: str, user: str = 'User') -> str:
        return QUERY_TEMPLATE.format(
            user=user,
            query=query
        )

    def encode(self, query: str, history: list = [], user: str = 'User'):
        prompt = self.encode_header(user)
        prompt += self.encode_history(history, user)
        prompt += self.encode_query(query, user)

        prompt = ENCODER.encode(prompt)
        if (len(prompt) > self.max_token) and (len(history) > 0):
            return self.encode(query, history[1:], user)

        return prompt, history


if __name__ == '__main__':
    prompt = GPT3Prompt()

    query = '帮我写一段快速排序的代码'
    history = []
    pt, _ = prompt.encode(query, history)
    print(pt)

    history = [['你好', '你好，请问有什么可以帮你的吗']]
    pt, _ = prompt.encode(query, history)
    print(len(pt))

    history = [['你好', '你好，请问有什么可以帮你的吗'] for _ in range(200)]
    pt, _ = prompt.encode(query, history)
    print(len(pt))
