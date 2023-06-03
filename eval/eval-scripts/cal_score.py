import sys
import os, shutil
import signal
import subprocess
import platform
import time
import json

def run_cmd(cmd_string, timeout=3600):
    print("命令为：" + cmd_string)
    p = subprocess.Popen(cmd_string, stderr=subprocess.STDOUT, stdout=subprocess.PIPE, shell=True, close_fds=True,
                         start_new_session=True)
 
    format = 'utf-8'
    if platform.system() == "Windows":
        format = 'gbk'
 
    try:
        (msg, errs) = p.communicate(timeout=timeout)
        ret_code = p.poll()
        if ret_code:
            code = 1
            msg = "[Error]Called Error ： " + str(msg.decode(format))
        else:
            code = 0
            msg = str(msg.decode(format))
    except subprocess.TimeoutExpired:
        # 注意：不能只使用p.kill和p.terminate，无法杀干净所有的子进程，需要使用os.killpg
        p.kill()
        p.terminate()
        os.killpg(p.pid, signal.SIGTERM)
 
        # 注意：如果开启下面这两行的话，会等到执行完成才报超时错误，但是可以输出执行结果
        # (outs, errs) = p.communicate()
        # print(outs.decode('utf-8'))
 
        code = 1
        msg = "[ERROR]Timeout Error : Command '" + cmd_string + "' timed out after " + str(timeout) + " seconds"
    except Exception as e:
        code = 1
        msg = "[ERROR]Unknown Error : " + str(e)
 
    return code, msg

models = [
    'chatglm-6b', 
    'dolly',
    'alpaca',
    'koala',
    'vicuna',
    'bloomz-7b',
    'moss',
    'mt0-xxl'
]

response_dir = '/workspace/hanyu/hanyu/gorilla/eval/outputs2_copy'
result = {}

if __name__ == '__main__':
    for model in models:
        result[model] = {}
        for (script, split) in [('hf', 'huggingface'), ('tf', 'tensorflowhub'), ('th', 'torchhub')]:
            result[model][split] = {}
            for testset in [f'questions_{split}_0_shot', f'questions_{split}_bm25', f'questions_{split}_gpt_index', f'questions_{split}_oracle']:
                response_path = response_dir + '/' + model + '/' + split + '/' + testset + '/output.jsonl'
                cmd = f'python ast_eval_{script}.py --api_dataset ../../data/api/{split}_api.jsonl --apibench ../../data/apibench/{split}_eval.json --llm_responses /workspace/hanyu/hanyu/gorilla/eval/outputs2_copy/{model}/{split}/{testset}/output.jsonl'
                code, msg = run_cmd(cmd)
                print(msg)
                acc, hal = msg.split('\n')[:2]
                acc = acc.split('  ')[-1]
                hal = hal.split('  ')[-1]
                result[model][split][testset] = {'acc': acc, 'hal': hal}
                
    print(result)
