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

def merge_files(dir_path):
    """
    合并目录下的所有文件并删除原文件
    """
    # 获取目录下所有文件
    file_list = os.listdir(dir_path)
    # 按字典序排序
    file_list.sort()
    # 合并文件
    with open(os.path.join(dir_path, 'output.jsonl'), 'w', encoding='utf-8') as f:
        for file_name in file_list:
            file_path = os.path.join(dir_path, file_name)
            if os.path.isfile(file_path):
                with open(file_path, 'r', encoding='utf-8') as f2:
                    f.write(f2.read())
                # 删除文件
                os.remove(file_path)

cmds = [
    ('chatglm-6b', 'python parrallel_chatglm-6b.py --model-id chatglm-6b_v1.1 --model-path /workspace/xuyifan/checkpoints/chatglm-6b-v1.1/chatglm_6b_new --question-file %s --answer-file %s --batch-size 4'),
    ('dolly', 'python parrallel_dolly.py --model-id dolly-v2-12b --model-path /workspace/xuyifan/checkpoints/dolly-v2-12b --question-file %s --answer-file %s --batch-size 3'),
    ('koala', 'python parrallel_koala.py --model-id koala-13B --model-path /workspace/xuyifan/checkpoints/koala-13B-HF --question-file %s --answer-file %s --batch-size 3'),
    ('vicuna', 'python parrallel_vicuna.py --model-id vicuna-13B --model-path /workspace/xuyifan/checkpoints/vicuna/13B --question-file %s --answer-file %s --batch-size 4'),
    ('alpaca', 'python parrallel_alpaca.py --model-id alpaca-7B --model-path /workspace/xuyifan/checkpoints/stanford_alpaca --question-file %s --answer-file %s --batch-size 4'),
    ('mt0-xxl', 'python parrallel_mt0_xxl.py --model-id mt0-xxl --model-path /workspace/xuyifan/checkpoints/mt0-xxl-mt --question-file %s --answer-file %s --batch-size 10'),
    ('bloomz-7b', 'python parrallel_bloomz.py --model-id bloomz-7b --model-path /workspace/xuyifan/checkpoints/bloomz-7b --question-file %s --answer-file %s --batch-size 4'),
    # ('belle-7b', 'python parrallel_belle.py --model-id BELLE-7B-2M --model-path /workspace/xuyifan/checkpoints/BELLE-7B-2M --question-file bbh_new/data/prompts_en_bbh.jsonl --answer-file bbh_new/outputs/BELLE-7B-2M.jsonl --batch-size 4'),
    ('moss', 'python parrallel_moss.py --model-id moss-moon-003-sft --model-path /workspace/xuyifan/checkpoints/moss-moon-003-sft --question-file %s --answer-file %s --batch-size 3'),
    ('oasst', 'python parrallel_oasst.py --model-id oasst --model-path /workspace/xuyifan/checkpoints/oasst-sft-4-pythia-12b-epoch-3.5 --question-file %s --answer-file %s --batch-size 4')
]

output_dir = '/workspace/hanyu/hanyu/gorilla/eval/outputs'
question_dir = '/workspace/hanyu/hanyu/gorilla/eval/eval-data/questions_prompt'

if __name__ == '__main__':
    for (model, cmd) in cmds:
        for split in ['huggingface', 'tensorflowhub', 'torchhub']:
            for testset in [f'questions_{split}_0_shot', f'questions_{split}_bm25', f'questions_{split}_gpt_index', f'questions_{split}_oracle']:
                question_path = question_dir + '/' + split + '/' + testset + '.jsonl'
                output_path = output_dir + '/' + model + '/' + split + '/' + testset
                if not os.path.exists(output_path):
                    os.makedirs(output_path)
                if not os.path.exists(output_path + '/output.jsonl'):
                    code, msg = run_cmd(cmd % (question_path, output_path + '/tmp.jsonl'))
                    print(msg)
                    time.sleep(3)
                    merge_files(output_path)
