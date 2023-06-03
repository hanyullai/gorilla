import os
import json
from pathlib import Path

def encode_question(question, file_path):
    if 'torchhub' in file_path:
        api_name = 'torchhub'
    elif 'huggingface' in file_path:
        api_name = 'huggingface'
    elif 'tensorflowhub' in file_path:
        api_name = 'tensorhub'
    else:
        print(file_path)
        raise
    """Encode multiple prompt instructions into a single string."""

    question = question.replace('\\n', '\n')
    prompt = question + "\nWrite a python program in 1 to 2 lines to call API in " + api_name + ".\n\nThe answer should follow the format: {<<<domain>>> $DOMAIN, <<<api_call>>>: $API_CALL, <<<api_provider>>>: $API_PROVIDER, <<<explanation>>>: $EXPLANATION, <<<code>>>: $CODE}. Here are the requirements:\n1. The $API_CALL should have only 1 line of code that calls api.\n2. The $API_PROVIDER should be the programming framework used.\n3. $EXPLANATION should be a step-by-step explanation.\n4. The $CODE is the python code.\n5. Do not repeat the format in your answer."
    
    return prompt


def process_jsonl_file(filename):
    # 读取文件内容
    with open(filename, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # 逐行处理json数据
    for i, line in enumerate(lines):
        try:
            data = json.loads(line)
        except ValueError as e:
            print(f"Error parsing line {i} in file {filename}: {e}")
            continue

        # 更新text字段
        if 'text' in data:
            data['text'] = encode_question(data['text'], filename)

        # 写入更新后的数据
        lines[i] = json.dumps(data, ensure_ascii=False) + '\n'

    # 写回文件
    with open(filename, 'w', encoding='utf-8') as f:
        f.writelines(lines)
        
def process_directory(input_dir):
    # 递归处理目录下所有jsonl文件
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            # 只处理扩展名为jsonl的文件
            if file.endswith('.jsonl'):
                file_path = os.path.join(root, file)
                process_jsonl_file(file_path)

input_dir = '/workspace/hanyu/hanyu/gorilla/eval/eval-data/questions2'
process_directory(input_dir)