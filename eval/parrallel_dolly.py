import argparse
import os
import sys
import torch
import torch.distributed as dist
from torch import nn
from torch.multiprocessing import Process
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, data
import jsonlines
import random
from tqdm import tqdm
import torch.multiprocessing as mp
from instruct_pipeline import InstructionTextGenerationPipeline

mp.set_start_method('spawn', force=True)


def set_device(rank):
    device = torch.device(f"cuda:{rank}")
    return device

def inference_dolly(model, tokenizer, prompts):
    generate_text = InstructionTextGenerationPipeline(model=model, tokenizer=tokenizer)

    with torch.no_grad():
        try:
            outputs = generate_text(prompts)
        except Exception as e:
            print(f"exception inference {e}")
            return [""] * len(prompts)
    
    outputs = [output[0]["generated_text"] for output in outputs]
    
    return outputs

def generate_with_dataloader(model_path, model_id, answer_file, rank, world_size, data, batch_size):

    os.environ["MASTER_PORT"] = "12356"
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in range(world_size))
    
    device = set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    
    torch.manual_seed(random.randint(1,100))

    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True).to(device)
    model = model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, padding_side="left")
    # tokenizer.pad_token = tokenizer.unk_token 

    print("initialize on gpu {rank}".format(rank=rank))

    with jsonlines.open("{answer}-{rank}.jsonl".format(answer=answer_file[:-6], rank = rank),'a') as f:
        batch_num = (len(data) + batch_size - 1) // batch_size
        for i in tqdm(range(batch_num)):
            questions = data[i*batch_size : min((i+1)*batch_size, len(data))]
            prompts = [bat["text"] for bat in questions]
            outputs = inference_dolly(model, tokenizer, prompts)

            for (bat, output) in zip(questions, outputs):
                bat["prediction"] = output
                bat["model"] = model_id
                f.write(bat)
            
        f.close()

    dist.destroy_process_group()

def main(model_path, model_id, question_file, answer_file, batch_size):
    world_size = torch.cuda.device_count()
    if world_size < 2:
        print("This example requires at least 2 GPUs.")
        sys.exit(1)

    ques_jsons = []

    with jsonlines.open(question_file, "r") as f:
        for doc in f:
            ques_jsons.append(doc)
 
    processes = []
    chunk_size = (len(ques_jsons) + world_size - 1) // world_size
    for gpu_rank in range(world_size):
        rank = gpu_rank
        p = Process(target=generate_with_dataloader, args=(model_path, model_id, answer_file, rank, world_size, ques_jsons[gpu_rank*chunk_size : min((gpu_rank+1)*chunk_size, len(ques_jsons))], batch_size))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--model-id", type=str, required=True)
    parser.add_argument("--question-file", type=str, required=True)
    parser.add_argument("--answer-file", type=str, default="answer.jsonl")
    parser.add_argument("--batch-size", type=int, default=1)
    args = parser.parse_args()

    main(args.model_path, args.model_id, args.question_file, args.answer_file, args.batch_size)
