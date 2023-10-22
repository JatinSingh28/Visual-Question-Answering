import json
import pandas as pd
import numpy as np

with open("./data/Annotations/train.json",'r', encoding='utf-8') as file:
    data = json.load(file)

df = pd.DataFrame(columns = ['image','question','answerable', 'answer_type','answers'])
for i in data:
    question = i['question']
    is_answerable =  i['answerable']
    image = i['image']
    image_array = np.array(image)
    answer_type = i['answer_type']
    answers = [(ans['answer_confidence'], ans['answer']) for ans in i['answers']]
    lis = [image,question,is_answerable,answer_type,answers]
    df = pd.concat([pd.DataFrame([lis], columns=df.columns), df], ignore_index=True)

print(df.head())
df.to_csv("ImageQA.csv")