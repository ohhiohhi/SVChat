"""
This is a demo for using CogVLM2 
Make sure you have installed pretained model (https://huggingface.co/THUDM/cogvlm2-llama3-chat-19B)
"""
import os
import math
import json
import argparse
import torch

from tqdm import tqdm
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


parser = argparse.ArgumentParser()
parser.add_argument("--quant", choices=[4, 8], type=int, default=0, help='Enable 4-bit or 8-bit precision loading')
parser.add_argument("--from_pretrained", type=str, default="THUDM/cogvlm2-llama3-chat-19B", help='pretrained ckpt')
parser.add_argument("--image_folder", type=str, default="xxxxxx")
parser.add_argument("--question_file", type=str, default="/xxxxxxxxxxxxx.jsonl")
parser.add_argument("--answers_file", type=str, default="xxxxxxxxxxxxxxxxxxxxx.jsonl")
parser.add_argument("--num_chunks", type=int, default=1)
parser.add_argument("--chunk_idx", type=int, default=0)
args = parser.parse_args()

MODEL_PATH = args.from_pretrained
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
TORCH_TYPE = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[
    0] >= 8 else torch.float16


if 'int4' in MODEL_PATH:
    args.quant = 4

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True
)

# Check GPU memory
if torch.cuda.is_available() and torch.cuda.get_device_properties(0).total_memory < 48 * 1024 ** 3 and not args.quant:
    print("GPU memory is less than 48GB. Please use cli_demo_multi_gpus.py or pass `--quant 4` or `--quant 8`.")
    exit()

print("========Use torch type as:{} with device:{}========\n\n".format(TORCH_TYPE, DEVICE))

def get_model_name_from_path(model_path):
    model_path = model_path.strip("/")
    model_paths = model_path.split("/")
    if model_paths[-1].startswith('checkpoint-'):
        return model_paths[-2] + "_" + model_paths[-1]
    else:
        return model_paths[-1]
    
if args.quant == 4:
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=TORCH_TYPE,
        trust_remote_code=True,
        quantization_config=BitsAndBytesConfig(load_in_4bit=True),
        low_cpu_mem_usage=True
    ).eval()
elif args.quant == 8:
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=TORCH_TYPE,
        trust_remote_code=True,
        quantization_config=BitsAndBytesConfig(load_in_8bit=True),
        low_cpu_mem_usage=True
    ).eval()
else:
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=TORCH_TYPE,
        trust_remote_code=True
    ).eval().to(DEVICE)

model_name = get_model_name_from_path(MODEL_PATH)

# eval model
def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]

def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

def eval_model(args):
    result = []
    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]  # json转dict
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)  # 分块
    for line in tqdm(questions):  # 获取question文件内容
        idx = line["question_id"]

        # load_image
        image_file = line["image"]
        image_path = os.path.join(args.image_folder, f'{image_file}')
        image = Image.open(image_path).convert('RGB')

        history = []
        query = line["text"]

        input_by_model = model.build_conversation_input_ids(tokenizer, query=query, history=history, images=[image])

        inputs = {
            'input_ids': input_by_model['input_ids'].unsqueeze(0).to(DEVICE),
            'token_type_ids': input_by_model['token_type_ids'].unsqueeze(0).to(DEVICE),
            'attention_mask': input_by_model['attention_mask'].unsqueeze(0).to(DEVICE),
            'images': [[input_by_model['images'][0].to(DEVICE).to(TORCH_TYPE)]] if image is not None else None,
        }

        # add any transformers params here.
        gen_kwargs = {
            "max_new_tokens": 2048,
            "pad_token_id": 128002,
            "do_sample": False,
            "temperature": 0.9
        }
        
        
        with torch.no_grad():
            outputs = model.generate(**inputs, **gen_kwargs)
            outputs = outputs[:, inputs['input_ids'].shape[1]:]
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = response.split("</s>")[0]

            print("\nCog:", response)

            result.append({
                "question_id":idx,
                "image": line["image"],
                "text":query,
                "answer":response,
                "model_id": model_name
            })

    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    with open(answers_file, 'w', encoding='utf-8') as ans_file: 
        for data in result:
            ans_file.write(json.dumps(data, ensure_ascii=False))
            ans_file.write('\n')

        ans_file.flush()
    ans_file.close()


if __name__ == "__main__":
    eval_model(args)

