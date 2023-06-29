import sys
import os, shutil
import signal
import subprocess
import platform
import time
import json

def run_cmd(cmd_string, timeout=36000):
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

cmds = [
    ('chatglm', 'python get_llm_responses.py --model chatglm --api_key 666 --output_file %s --question_data %s --api_name %s')
]

output_dir = '/home/lhy/gorilla/eval/outputs'
question_dir = '/home/lhy/gorilla/eval/eval-data/questions'

if __name__ == '__main__':
    run_cmds = ''
    for (model, cmd) in cmds:
        for split in ['huggingface', 'tensorflowhub', 'torchhub']:
            for testset in [f'questions_{split}_0_shot', f'questions_{split}_bm25', f'questions_{split}_gpt_index', f'questions_{split}_oracle']:
                question_path = question_dir + '/' + split + '/' + testset + '.jsonl'
                output_path = output_dir + '/' + model + '/' + split + '/' + testset
                if not os.path.exists(output_path):
                    os.makedirs(output_path)
                if not os.path.exists(output_path + '/output.jsonl'):
                    if run_cmds:
                        run_cmds += ' & '
                    run_cmds += cmd % (output_path + '/output.jsonl', question_path, split)
    print(run_cmds)
    code, msg = run_cmd(run_cmds)
