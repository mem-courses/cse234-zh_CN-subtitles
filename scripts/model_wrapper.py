import os
import abc
import json
import openai
import hashlib
from typing import List


class ModelWrapper(abc.ABC):
    def __init__(self, model_name: str):
        self.model_name = model_name

    @abc.abstractmethod
    def generate(self, prompt: str, max_tokens: int, temperature: float, top_p: float, repetition_penalty: float) -> str:
        pass


class OpenAIWrapper(ModelWrapper):
    def __init__(self,
                 model_name: str,
                 api_key: str,
                 base_url: str,
                 max_tokens: int,
                 temperature: float,
                 top_p: float,
                 ):
        super().__init__(model_name)
        self.api_key = api_key
        self.base_url = base_url
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p

        self.model_hash = hashlib.md5(f'{model_name}-{max_tokens}-{temperature}-{top_p}'.encode()).hexdigest()
        self.cache_dir = f'./output/{self.model_name}-{self.model_hash[0:8]}'
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)

    def generate(self, messages: List[dict], stream: bool = True) -> str:
        client = openai.OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )

        if stream:
            content = ""
            for chunk in client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                stream=True
            ):
                if chunk.choices[0].delta.content is not None:
                    # print(chunk.choices[0].delta.content, end='')
                    content += chunk.choices[0].delta.content
                else:
                    # print('None content:', chunk)
                    pass
            return content
        else:
            response = client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
            )
            return response.choices[0].message.content

    def send(self, message: str | List[dict], stream: bool = True, use_cache=True):
        message_hash = hashlib.md5(message.encode() if isinstance(message, str) else json.dumps(message).encode()).hexdigest()
        cache_file = os.path.join(self.cache_dir, f'{message_hash}.txt')

        if use_cache:
            if os.path.exists(cache_file):
                with open(cache_file, 'r', encoding='utf-8') as f:
                    return f.read()

        if isinstance(message, str):
            message = [{'role': 'system', 'content': "You are a helpful assistant."},
                        {"role": "user", "content": message}]
        output = self.generate(message, stream)

        with open(cache_file, 'w+', encoding='utf-8') as f:
            f.write(output)
        return output
