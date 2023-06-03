import os
import json

PATH = "/workspace/hanyu/hanyu/gorilla/eval/outputs2_copy"  # 目录路径

for root, dirs, files in os.walk(PATH):  # 遍历目录下的所有文件
    for filename in files:
        if filename.endswith(".jsonl"):  # 判断文件是否为jsonl格式
            with open(os.path.join(root, filename), "r", encoding="utf-8") as f:  # 打开文件
                new_lines = []  # 存放修改后的每行json
                for line in f:  # 读取每行json
                    data = json.loads(line.strip())  # 将json字符串转为字典类型
                    data["query"] = data.pop("text", "")  # 将text字段改名为query
                    data["text"] = data.pop("prediction", "")  # 将prediction字段改名为text
                    data["model_id"] = data.pop("model", "")  # 将model字段改名为model_id
                    data["answer_id"] = "None"
                    data['metadata'] = {}
                    new_lines.append(json.dumps(data, ensure_ascii=False))  # 将修改后的字典再转为json字符串 ，加入新的行列表中
            with open(os.path.join(root, filename), "w", encoding="utf-8") as f:  # 写入修改后的新内容到文件中
                f.write("\n".join(new_lines))