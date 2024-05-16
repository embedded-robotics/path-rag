# This program will generate the query .json files which will need to be input into LLaVA-Med to generate descriptions or answers
# for all the images in PathVQA and all the relevant patches

import os
import pandas as pd
import pickle
from tqdm import tqdm
import json

PVQA_DATA_PATH = "pvqa"
HISTO_PATCH_SAVE_PATH = "histo_image_patch"
LLAVA_MED_QUERY_PATH = os.path.join("files", "query")

def get_path_vqa_open_data(data_path : str = "pvqa"):
    # Reading the PathVQA dataset
    qas_train_path = os.path.join(data_path, "qas", "test", "test_qa.pkl")
    with open(qas_train_path, 'rb') as file:
        pvqa_qas = pickle.load(file)

    # Getting all open-ended images
    qas_general = [qas for qas in pvqa_qas if qas['answer'] != 'yes' and qas['answer'] != 'no']
    img_general = [qas['image']  for qas in qas_general]
    
    return qas_general, img_general

def generate_image_files_direct(qas_general: list, img_general: list):
    # Generating all the info for images with original questions
    question = []
    idx = 0
    for i in range(0, len(qas_general)):
        question.append({"question_id": idx, "image": img_general[i]+'.jpg', "text": qas_general[i]+"\n<image>"})
        idx = idx+1

    # Writing each dictionary as a JSON object on a new line
    img_direct_path = os.path.join(LLAVA_MED_QUERY_PATH, 'image_direct.jsonl')
    with open(img_direct_path, 'w') as file:
        for item in question:
            json_line = json.dumps(item)  # Convert dictionary to JSON string
            file.write(json_line + '\n') 

def generate_patch_files_direct(qas_general: list, img_general: list):
    # Generating all the info for patches with original questions
    question = []
    idx = 0
    for i in range(0, len(qas_general)):
        question.append({"question_id": idx, "image": img_general[i]+"/1.png", "text": qas_general[i]+"\n<image>"})
        idx = idx+1
        question.append({"question_id": idx, "image": img_general[i]+"/2.png", "text": qas_general[i]+"\n<image>"})
        idx = idx+1
        question.append({"question_id": idx, "image": img_general[i]+"/3.png", "text": qas_general[i]+"\n<image>"})
        idx = idx+1

    # Writing each dictionary as a JSON object on a new line
    patch_direct_path = os.path.join(LLAVA_MED_QUERY_PATH, 'patch_direct.jsonl')
    with open(patch_direct_path, 'w') as file:
        for item in question:
            json_line = json.dumps(item)  # Convert dictionary to JSON string
            file.write(json_line + '\n') 
        
def generate_image_files_description(img_general: list):
    # Getting the description info for all the images
    question = []
    idx = 0
    for i in img_general:
        question.append({"question_id": idx, "image": i+'.jpg', "text": "Describe the following image in detail.\n<image>"})
        idx = idx+1

    # Writing each dictionary as a JSON object on a new line
    img_desc_path = os.path.join(LLAVA_MED_QUERY_PATH, 'image_description.jsonl')
    with open(img_desc_path, 'w') as file:
        for item in question:
            json_line = json.dumps(item)  # Convert dictionary to JSON string
            file.write(json_line + '\n') 

def generate_patch_files_description():
    # Getting all the directories of generated patches
    patch_dir_list = os.listdir(HISTO_PATCH_SAVE_PATH)
    patch_dir_list = patch_dir_list.sort()

    # Getting the description info for top 3 patches
    question = []
    idx = 0
    for i in patch_dir_list:
        question.append({"question_id": idx, "image": i+"/1.png", "text": "Describe the following image in detail.\n<image>"})
        idx = idx+1
        question.append({"question_id": idx, "image": i+"/2.png", "text": "Describe the following image in detail.\n<image>"})
        idx = idx+1
        question.append({"question_id": idx, "image": i+"/3.png", "text": "Describe the following image in detail.\n<image>"})
        idx = idx+1
        
    # Writing each dictionary as a JSON object on a new line
    patch_desc_path = os.path.join(LLAVA_MED_QUERY_PATH, 'patch_description.jsonl')
    with open(patch_desc_path, 'w') as file:
        for item in question:
            json_line = json.dumps(item)  # Convert dictionary to JSON string
            file.write(json_line + '\n')

if __name__ == "__main__":
    
    # Get all the data from PathVQA
    qas_general, img_general = get_path_vqa_open_data(PVQA_DATA_PATH)
    
    # Generate Image File with Answers
    generate_image_files_direct(qas_general, img_general)

    # Generate Patch File with Answers
    generate_patch_files_direct(qas_general, img_general)

    # Generate Image File with Descriptions
    generate_image_files_description(img_general)

    # Generate Patch File with Answers
    generate_patch_files_description()
    