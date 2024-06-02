import os
import json
import base64
from IPython.display import Image, display
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from openai import OpenAI
import httpx


dataset_path = 'xxxxxxx'
json_path = 'xxxxxx.jsonl'
SUPPORTED_IMAGE_FORMATS = ('.png', '.jpg', '.jpeg', '.webp', '.bmp', '.gif', '.tiff', '.tif')


# Initializing OpenAI client - see https://platform.openai.com/docs/quickstart?context=python
client = OpenAI(
    base_url = 'xxxxxxx',
    api_key = 'sk-xxxxxxxxxx',
    http_client = httpx.Client(
        base_url = 'xxxxxxxxxxxxxx',
        follow_redirects = True,
        ),
    )

# Loading dataset
def image_load(dataset_path):
    filename = []
    image_path = []
    try:
        for root, dirs, _ in os.walk(dataset_path):
            for dir in dirs:
                image_dir = os.path.join(root, dir)
                image_files = os.listdir(image_dir)
                image_files.sort()
                image_files = image_files[:2]
    
                for image_filename in image_files:
                    if image_filename.endswith(SUPPORTED_IMAGE_FORMATS):
                        image_path.append(os.path.join(root, dir, image_filename))
                        filename.append(os.path.splitext(image_filename)[0])
    except Exception as e:
        print(f"Error loading images: {e}")

    # Assuming the function should return the paths and filenames
    return filename, image_path


def save_json(prompt, json_path):
    with open(json_path, 'a', encoding='utf-8') as json_file:
        for data in prompt:
            json_file.write(json.dumps(data, ensure_ascii=False))
            json_file.write('\n')

# The prompt here has been modified, and we'll upload a new version soon!
system_prompt = '''
    You are an agent specialized in tagging street view images with relevant keywords that could be used to generate detailed descriptions for the images. 

    furniture items, decorative items, or furnishings；search for these items on a marketplace.
    
    You will be provided with an image and your goal is to extract keywords for only the item specified. 
    Keywords should be concise and in lower case. 
    
    Keywords can describe things like:

    - Geographic location, e.g. " suburban", "city", "mountain", "underpass", "seaside"
    - Road type, e.g. "highway", "alleyway", "footpath", "intersection", "fork"
    - Terrain, e.g. "curves", "uphill", "downhill", "flat"
    - Weather conditions, e.g. "snowy", "foggy", "rainy", "sunny", "cloudy", "blue sky", "sunset"
    - Time of day, e.g. "daytime", "evening", "late afternoon", "early morning"
    - Architectural style, e.g. "gothic", "classical", "minimalist", "modern"
    - Building type, e.g. "single house", "high-rise office", "low-rise residential"
    - Vegetation, e.g. "trees", "lawn", "flower beds“
    - Traffic facilities, e.g. "safety islands", "traffic signals", "street lights", "traffic signs", "road markings", ""pedestrian crossings", "number of lanes"
    - Traffic conditions, e.g. "type of vehicle", "traffic flow", "pedestrian density", "direction of travel"
        
    1-2 adjectives can be added before the keywords to describe the object, the adjectives can describe the material, style, colour, quantity etc. If a very distinctive feature appears in the image but is not among the given keywords, choose the keyword you think is the most appropriate for the description

    Return keywords in the format of an array of strings, like this:
    ["flat terrain", "right-hand curve", "3 lanes", "5 trucks", "green light", "left lawn ", "modern style residential building"]

'''

def analyze_image(image):
    with open(image, "rb") as image_file:
        image_base64 = base64.b64encode(image_file.read()).decode('utf-8')

    response = client.chat.completions.create(
    model="gpt-4-vision-preview",
    messages=[
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": 
            [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}
                },
            ],
        },
    ],
        max_tokens=300,
        top_p=0.1
    )

    return response.choices[0].message.content


# Testing with a few example
def image_label(image_path):
    result = []

    for image in image_path:
        result.append({
            "image_path": image,
            'label':analyze_image(image)
        })

        return result


# Describing images with GPT-4V
describe_system_prompt = '''
    You are a system generating descriptions for different elements on street view images.
    Provided with an image and a title, you will describe the main object that you see in the image, giving details but staying concise.
    You can describe unambiguously what the object is, and its number, form, material, color and style, etc., if clearly identifiable.
    If there are multiple items depicted, refer to the title to understand which item you should describe.
'''

def describe_image(image, label):
    with open(image, "rb") as image_file:
        image_base64 = base64.b64encode(image_file.read()).decode('utf-8')

    response = client.chat.completions.create(
    model="gpt-4-vision-preview",
    temperature=0.2,
    messages=[
        {
            "role": "system",
            "content": describe_system_prompt
        },
        {
            "role": "user",
            "content":
            [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}
                },
            ],
        },
        {
            "role": "user",
            "content": label
        }
    ],
    max_tokens=300,
    )

    return response.choices[0].message.content


# Turning descriptions into captions
caption_system_prompt = '''
Your goal is to generate short, descriptive captions for images of street view based on an image description.
You will be provided with a description of an item image and you will output a caption that captures the most important information about the item.
Your generated caption should be short (1 sentence), and include the most relevant information about the item.
The most important information could be: weather, lanes, cars, the type of the item, the style (if mentioned), the material if especially relevant and any distinctive features.
'''

few_shot_examples = [
    {
        "description": "The street picture showcases a bustling urban area during winter, with snow partially covering the ground and roads. Vehicles, including cars and trucks, are navigating a curved road, with a prominent traffic sign indicating a mandatory direction to the right. Overhead traffic lights with directional arrows manage the flow of traffic. In the background, several high-rise apartment buildings dominate the skyline, and various colorful advertisements are visible, including one highlighting a product priced at 99. The overcast sky adds to the cold and wintry atmosphere of the scene.",
        "caption": "A winter urban scene with snow, vehicles, a right-turn sign, traffic lights, high-rise buildings, and ads, under an overcast sky."
    },
    {
        "description": "xxxxxxxxxxxxxxxxxxxxxxxxxxxx.",
        "caption": "xxxxxxxx"
    },
    {
        "description": "xxxxxxxxxxxxxxxxxxxxxxxxxxxx.",
        "caption": "xxxxxxxx"
    }
]

formatted_examples = [[{
    "role": "user",
    "content": ex['description']
},
{
    "role": "assistant", 
    "content": ex['caption']
}]
    for ex in few_shot_examples
]

formatted_examples = [i for ex in formatted_examples for i in ex]

def caption_image(description, model="gpt-4-turbo-preview"):
    messages = formatted_examples
    messages.insert(0, 
        {
            "role": "system",
            "content": caption_system_prompt
        })
    messages.append(
        {
            "role": "user",
            "content": description
        })
    response = client.chat.completions.create(
    model=model,
    temperature=0.2,
    messages=messages
    )

    return response.choices[0].message.content

# testing on a few examples
def image_caption(label):
    result = []

    for image, label in label:
        img_description = describe_image(image, label)
        img_caption = caption_image(img_description)
        result.append({
            "image_path": image,
            'label':label,
            'description':img_description,
            'prompt':img_caption
        })

        return result
    

_, image_path = image_load(dataset_path)
label = image_label(image_path)
print(label)
prompt = image_caption(label)
        
save_json(prompt, json_path)
