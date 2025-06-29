import os
from config import *
from model_wrapper import OpenAIWrapper

SYSTEM_PROMPT = '''
你是一个中文字幕翻译专家，每轮对话中，我会想你提供需要翻译的英文字幕，请你制作对应的中文字幕，并满足格式要求。
#### 要求
1.只按照格式输出中文翻译的结果，不输出其他内容
2.每一行都以- 开头，就类似于我下面输入的形式
3.我将给你{len}条字幕，他们共同组成了一段对话，但你不能擅自将他们合并
你给我的输出也应该是{len}条字幕，且完成了中文翻译的结果
'''.strip()
ADDITIONAL_PROMPT = '''
注意: 再次强调一定不要把多条字幕合并到一条中，必须遵循输入中的划分方式.
输入给你的是多少行数据，你输出的也应该是多少行数据
'''.strip()

model = OpenAIWrapper(
    model,
    api_key,
    base_url,
    max_tokens=16384,
    temperature=0.3,
    top_p=0.95,
)


def build(source_text):
    source = []
    for item in source_text.split('\n\n'):
        if item.strip() == '':
            continue
        lines = item.split('\n')
        data = [
            lines[1].split(' --> ')[0],
            lines[1].split(' --> ')[1],
            ' '.join(lines[2:]).replace('  ', ' ').strip(),
        ]
        if len(source) > 0 and data[2][0].islower() and len(source[-1][2]) + len(data[2]) < 100:
            source[-1][1] = data[1]
            source[-1][2] = source[-1][2].strip() + ' ' + data[2].strip()
        else:
            source.append(data)

    STEP = 10
    for l in range(0, len(source), STEP):
        r = min(len(source), l + STEP)
        data = source[l:r]
        prompt = ''
        for line in data:
            prompt += '- ' + line[2] + '\n'
        message = [
            {
                'role': 'system',
                'content': SYSTEM_PROMPT.replace('{len}', str(len(data))),
            },
            {
                'role': 'user',
                'content': prompt.strip(),
            },
        ]
        print(prompt)

        for retries in range(10):
            try:
                print(f'Translating {l} to {r}, retries: {retries}.')
                if retries == 1:
                    message[0]['content'] += '\n' + ADDITIONAL_PROMPT
                response = model.send(message, use_cache=retries == 0)
                print(response)
                translated = response.strip().split('\n')
                assert len(translated) == len(data)
                for i in range(len(translated)):
                    assert translated[i].startswith('- ')
                    data[i].append(translated[i][2:].strip())
                print(data)
                break
            except Exception as e:
                print(e)
                continue
            
        for line in data:
            if len(line) == 3:
                line.append('')

    response = ''
    for i in range(len(source)):
        line = source[i]
        response += f'{i + 1}\n'
        response += f'{line[0]} --> {line[1]}\n'
        response += f'{line[3]}\n'
        response += f'{line[2]}\n'
        response += '\n'
    return response.strip()


if __name__ == '__main__':
    source_dir = os.path.join(os.path.dirname(__file__), '../data/source')
    target_dir = os.path.join(os.path.dirname(__file__), '../data/dest')
    for source in os.listdir(source_dir):
        source_file = os.path.join(source_dir, source)
        with open(source_file, 'r', encoding='utf-8') as f:
            data = f.read()
        output = build(data)
        target_file = os.path.join(target_dir, source)
        with open(target_file, 'w', encoding='utf-8') as f:
            f.write(output)
