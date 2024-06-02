import os
import json
import base64
from IPython.display import Image, display
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from openai import OpenAI
import httpx

'''
读取数据集的每张图片（注：查看4v的输入图像限制条件
提取keyword
caption：
    初始化api并调用接口，设置参数，定义模型的system、user信息，输入图像，
    利用keyword，生成caption结果
    记录信息
读取json文件并将caption的结果写入文件中

'''

# 文件路径和初始化信息
dataset_path = 'C:\\Users\\W\\Desktop\\selected_image'
json_path = 'C:\\Users\\W\\Desktop\\selected_image.jsonl'
SUPPORTED_IMAGE_FORMATS = ('.png', '.jpg', '.jpeg', '.webp', '.bmp', '.gif', '.tiff', '.tif')


# Initializing OpenAI client - see https://platform.openai.com/docs/quickstart?context=python
client = OpenAI(
    base_url = 'https://api.gpts.vin',
    api_key = 'sk-h0t34CLEp3KiuqVu83148777725243769dFfF1Eb00748fA5',
    http_client = httpx.Client(
        base_url = 'https://api.gpts.vin',
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


# def save_json(prompt, json_path):
#     # 将label， prompt合并
#     with open(json_path, 'a', encoding='utf-8') as json_file:
#         for data in prompt:
#             json_file.write(json.dumps(data, ensure_ascii=False))
#             json_file.write('\n')  # 写入换行符，确保每个数据占一行

# Extract keywords
# system_prompt = '''
#     You are an agent specialized in tagging images of furniture items, decorative items, or furnishings with relevant keywords that could be used to search for these items on a marketplace.
    
#     You will be provided with an image and the title of the item that is depicted in the image, and your goal is to extract keywords for only the item specified. 
    
#     Keywords should be concise and in lower case. 
    
#     Keywords can describe things like:
#     - Item type e.g. 'sofa bed', 'chair', 'desk', 'plant'
#     - Item material e.g. 'wood', 'metal', 'fabric'
#     - Item style e.g. 'scandinavian', 'vintage', 'industrial'
#     - Item color e.g. 'red', 'blue', 'white'
    
#     Only deduce material, style or color keywords when it is obvious that they make the item depicted in the image stand out.

#     Return keywords in the format of an array of strings, like this:
#     ['desk', 'industrial', 'metal']
    
# '''

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
    # 这两行是在做什么，得到什么结果
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
        # { # 不需要内容
        #     "role": "user",
        #     "content": title
        # }
    ],
        max_tokens=300,
        top_p=0.1
    )

    return response.choices[0].message.content


# Testing with a few example
def image_label(image_path):
    result = []

    for image in image_path:
    # for index, ex in examples.iterrows():
    #     url = ex['primary_image']
    #     img = Image(url=url)
    #     display(img)
        # result = analyze_image(url, ex['title'])
        result.append({
            "image_path": image,
            'label':analyze_image(image)
        })

        return result

# 结果只要json文件，无需对图片有改变
_, image_path = image_load(dataset_path)
label = image_label(image_path)
print(label)
prompt = image_caption(label)
        
save_json(prompt, json_path)

# Looking up existing keywords:Using embeddings to avoid duplicates (synonyms) and/or match pre-defined keywords
def get_embedding(value, model="text-embedding-3-large"): # Feel free to change the embedding model here
    embeddings = client.embeddings.create(
      model=model,
      input=value,
      encoding_format="float"
    )
    return embeddings.data[0].embedding

# Existing keywords
keywords_list = ['industrial', 'metal', 'wood', 'vintage', 'bed']
df_keywords = pd.DataFrame(keywords_list, columns=['keyword'])
df_keywords['embedding'] = df_keywords['keyword'].apply(lambda x: get_embedding(x))
df_keywords

def compare_keyword(keyword):
    embedded_value = get_embedding(keyword)
    df_keywords['similarity'] = df_keywords['embedding'].apply(lambda x: cosine_similarity(np.array(x).reshape(1,-1), np.array(embedded_value).reshape(1, -1)))
    most_similar = df_keywords.sort_values('similarity', ascending=False).iloc[0]
    return most_similar

def replace_keyword(keyword, threshold = 0.6):
    most_similar = compare_keyword(keyword)
    if most_similar['similarity'] > threshold:
        print(f"Replacing '{keyword}' with existing keyword: '{most_similar['keyword']}'")
        return most_similar['keyword']
    return keyword

# Example keywords to compare to our list of existing keywords
example_keywords = ['bed frame', 'wooden', 'vintage', 'old school', 'desk', 'table', 'old', 'metal', 'metallic', 'woody']
final_keywords = []

for k in example_keywords:
    final_keywords.append(replace_keyword(k))
    
final_keywords = set(final_keywords)
print(f"Final keywords: {final_keywords}")



'''
---------------------------caption_核心code-----------------------------
    image_path = os.path.join(image_dir, filename)  # 获取图片
    caption = run_openai_api(image_path, prompt, api_key, api_url, quality, timeout)  # caption
'''
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
Your goal is to generate short, descriptive captions for images of furniture items, decorative items, or furnishings based on an image description.
You will be provided with a description of an item image and you will output a caption that captures the most important information about the item.
Your generated caption should be short (1 sentence), and include the most relevant information about the item.
The most important information could be: the type of the item, the style (if mentioned), the material if especially relevant and any distinctive features.
'''

few_shot_examples = [
    {
        "description": "This is a multi-layer metal shoe rack featuring a free-standing design. It has a clean, white finish that gives it a modern and versatile look, suitable for various home decors. The rack includes several horizontal shelves dedicated to organizing shoes, providing ample space for multiple pairs. Above the shoe storage area, there are 8 double hooks arranged in two rows, offering additional functionality for hanging items such as hats, scarves, or bags. The overall structure is sleek and space-saving, making it an ideal choice for placement in living rooms, bathrooms, hallways, or entryways where efficient use of space is essential.",
        "caption": "White metal free-standing shoe rack"
    },
    {
        "description": "The image shows a set of two dining chairs in black. These chairs are upholstered in a leather-like material, giving them a sleek and sophisticated appearance. The design features straight lines with a slight curve at the top of the high backrest, which adds a touch of elegance. The chairs have a simple, vertical stitching detail on the backrest, providing a subtle decorative element. The legs are also black, creating a uniform look that would complement a contemporary dining room setting. The chairs appear to be designed for comfort and style, suitable for both casual and formal dining environments.",
        "caption": "Set of 2 modern black leather dining chairs"
    },
    {
        "description": "This is a square plant repotting mat designed for indoor gardening tasks such as transplanting and changing soil for plants. It measures 26.8 inches by 26.8 inches and is made from a waterproof material, which appears to be a durable, easy-to-clean fabric in a vibrant green color. The edges of the mat are raised with integrated corner loops, likely to keep soil and water contained during gardening activities. The mat is foldable, enhancing its portability, and can be used as a protective surface for various gardening projects, including working with succulents. It's a practical accessory for garden enthusiasts and makes for a thoughtful gift for those who enjoy indoor plant care.",
        "caption": "Waterproof square plant repotting mat"
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
