import os
import json

dataset_path = "xxxxxxxxxxxxxxx"
target_json = "xxxxxxxxxxxxxx.jsonl"

def image_load(dataset_path):
    image_files = os.listdir(dataset_path)
    image_files.sort()
    # image_files = image_file[:2]
    return image_files


def json_gen():
    result = []
    question_id = 0
    image_name = image_load(dataset_path)
    text = "xxxxxxx"

    for image in image_name:
        
        result.append({
            "question_id":question_id,
            "image": image,
            "text":text
        })
        question_id += 1

    with open(target_json, 'w', encoding='utf-8') as json_file: 
        for data in result:
            json_file.write(json.dumps(data, ensure_ascii=False))
            json_file.write('\n')

    print("end")

json_gen()

