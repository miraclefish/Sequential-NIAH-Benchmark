import re
import json
import tqdm
import random
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import logging

logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

sys.path.append('/Users/felix/Documents/GitProjects/Sequential_NIAH_Bench')
from utils import LLM_API

def cut_str_by_needles(needles, input_content):
    for needle in needles:
        input_content = input_content.replace(f"\n{needle}\n", '[CHAT_SEP]')
    text_list = input_content.split('[CHAT_SEP]')
    return text_list

def check_raw_long_text(text_list, raw_text):
    """
    Check if the raw text is the same as the concatenated text list
    """
    cat_text = ''.join(text_list)
    return cat_text == raw_text



def gen_syn_semantic_noise(llm, needles, question):
    system_prompt_1 = """You are skilled at crafting sentences according to requirements. Below, you will be provided with a set of sentences, each containing specific event information. You need to write a new sentence based on the following requirements:
Requirements:
1. The surname of the protagonist in your sentence must be the same as the one in the provided sentence, but the given name must be different.
2. The date in your sentence must not duplicate the one in the provided sentence but should be relatively close.
3. The event information in your sentence must not repeat the one in the example sentence; please create a new event.
4. Use the same language as the example sentence for your writing.
5. The format of your sentence must match the format of the provided example sentence.
Please output in JSON format:
```json
{
    "name": "Modified name",
    "date": "A non-repeating date close to the example",
    "event": "Non-repeating event information",
    "sentence": "Your complete sentence, formatted to match the example"
}
```
"""
    system_prompt_2 = """You are skilled at crafting sentences according to requirements. Below, you will be provided with a question and a set of sentences, each containing specific event information. You need to write a new sentence based on the following requirements:  
Requirements:  
1. The protagonist in your sentence must match the one in the question and the provided sentences.  
2. Your sentence should not answer the given question; instead, write an irrelevant response—for example, describing the protagonist's personality, hobbies, or work, but avoid including time-related information.  
3. Use the same language as the example sentence for your writing.  
Please output in JSON format:  
```json
{
    "name": "Protagonist's name",
    "reason": "Reason for the irrelevant response",
    "sentence": "Your irrelevant sentence"
}
```
"""
    query_prompt_1 = """
<sentence>
{needles_str}
</sentence>

Please output the sentence you have written according to the requirements.
"""
    query_prompt_2 = """<question>
{question}
</question>

<sentence>
{needles_str}
</sentence>

Please output the sentence you have written according to the requirements.
"""

    needles_str = '\n\n'.join([f"[{i}]. {n}" for i, n in enumerate(needles, 1)])
    query_1 = query_prompt_1.format(needles_str=needles_str)
    query_2 = query_prompt_2.format(needles_str=needles_str, question=question)
    noise_needle_1 = llm.call_qa_task(system_prompt_1, query_1, rd=True, json=True)
    assert noise_needle_1['name'] in noise_needle_1['sentence'], f"Name not in sentence: {noise_needle_1['name']} not in {noise_needle_1['sentence']}"
    noise_needle_2 = llm.call_qa_task(system_prompt_2, query_2, rd=True, json=True)
    return noise_needle_1['sentence'], noise_needle_2['sentence']

def gen_tkg_semantic_noise(llm, needles, ett_a, ett_b):
    system_prompt_1 = """You are skilled at crafting sentences according to requirements. Below, you will be provided with two entities and a set of sentences, each describing a real event or relationship between the two entities during a certain period.  
You need to write a new sentence based on the following requirements:  
1. The format of your sentence should be similar to the provided sentences.  
2. You need to describe a real event or relationship between Entity A and another Entity C.  
3. You must first identify a real Entity C before writing the sentence.  
4. Use the same language as the example sentences for your writing.  

Please output in JSON format:  
```json
{
    "entity_A": "Entity A",
    "entity_C": "Entity C",
    "sentence": "The sentence you have written"
}
```
"""
    system_prompt_2 = """You are skilled at crafting sentences based on requirements. Below, you will be provided with two entities and a set of sentences, each describing a real-world event or relationship between the two entities during a specific period.  

Your task is to compose a new sentence adhering to the following rules:  
1. The format of your sentence should closely resemble the provided examples.  
2. Your sentence should describe a real-world event or relationship between Entity B and another Entity C.  
3. You must first identify a real-world Entity C before composing the sentence.  
4. Use language consistent with the example sentences.  

Output in JSON format:  
```json
{
    "entity_B": "Entity B",
    "entity_C": "Entity C",
    "sentence": "Your composed sentence"
}
```
"""
    query_prompt = """<entities>
Entity A: {ett_a}
Entity B: {ett_b}
</entities>
<sentence>
{needles_str}
</sentence>

Please output the sentence you have written according to the requirements.
"""
    
    needles_str = '\n\n'.join([f"[{i}]. {n}" for i, n in enumerate(needles, 1)])
    query = query_prompt.format(needles_str=needles_str, ett_a=ett_a, ett_b=ett_b)
    noise_needle_1 = llm.call_qa_task(system_prompt_1, query, rd=True, json=True)
    noise_needle_2 = llm.call_qa_task(system_prompt_2, query, rd=True, json=True)
    return noise_needle_1['sentence'], noise_needle_2['sentence']

def gen_tkg_semantic_noise_finance(llm, needles, entity, question):
    system_prompt = """You are skilled at crafting sentences based on requirements. Below, you will be provided with an entity, a question, and a set of sentences, each describing multiple answers corresponding to that entity for the given question.  

You need to compose a new sentence according to the following requirements:  
1. The entity in your sentence must match the entity in the question and the provided sentences.  
2. The format of your sentence should be similar to that of the provided sentences.  
3. The time described in your sentence must fall outside the scope of the question and should not duplicate any of the example sentences.  
4. You may fabricate facts appropriately.  
5. Please write in the same language as the example sentences.  

Output in JSON format:  
```json
{
    "entity": "entity name",
    "date": "the date of the event you composed",
    "sentence": "the complete sentence you composed, outside the question's scope"
}
```
"""
    query_prompt = """<entity>
{entity}
</entity>
<question>
{question}
</question>
<sentence>
{needles_str}
</sentence>

Please output the sentence you have written according to the requirements.
"""
    needles_str = '\n\n'.join([f"[{i}]. {n}" for i, n in enumerate(needles, 1)])
    query = query_prompt.format(needles_str=needles_str, entity=entity, question=question)
    noise_needle = llm.call_qa_task(system_prompt, query, rd=True, json=True)
    return noise_needle['sentence']


def gen_open_semantic_noise(llm, question, answer, examples):
    system_prompt_1 = """You are skilled at crafting sentences based on requirements. Below, you will be provided with a question and a set of answers.  

You need to compose a new sentence according to the following requirements:  
1. The content of your sentence should be related to the subject in the question.  
2. Your sentence should not answer the question—you must write a sentence that is irrelevant to the question.  
3. Your sentence must not repeat any of the given examples.  
4. Please write in the same language as the example sentences.

Example：
<question>
How to properly cook pasta?
</question>
<answer_items>
Here are the steps to properly cook delicious pasta:
1. Choose a pot and fill it with water: Select a sufficiently large pot and add enough water to ensure that the pasta can float freely in the water. Generally, you need 1 liter of water for every 100 grams of pasta.
2. Heat the water to a boil: Place the pot on the stove and bring the water to a boil over high heat.
3. Add salt: After the water boils, add an appropriate amount of salt. Typically, you add about 10 grams of salt per liter of water. Salt not only flavors the pasta but also raises the boiling point of the water, making the pasta cook faster.
4. Add the pasta: Grab the middle of the pasta and place it into the boiling water. You can gently stir to prevent the pasta from sticking together.
5. Cook the pasta: Cook the pasta according to the package instructions, usually 8-12 minutes. You can occasionally stir to prevent it from sticking.
6. Test the pasta: Close to the end of the recommended cooking time, take out a piece of pasta, cool it down, and taste it to see if it has reached your preferred texture. The pasta should not be too soft; it is recommended to cook it until it is 'al dente' (firm to the bite).
7. Drain the pasta: After the pasta is cooked, drain it but reserve some of the cooking water for the sauce preparation.
8. Mix with sauce: Mix the pasta with the prepared sauce. If the sauce is too dry, you can add some reserved pasta water to better integrate and coat the pasta with the sauce.
9. Plate and enjoy: Serve the mixed pasta onto a plate and optionally sprinkle some cheese, black pepper, or fresh herbs according to your preference, then enjoy your delicious pasta.
</answer_items>
【new_sentence(json)】
{
    "entity": "pasta",
    "reason": "The origin and spread of pasta are related to the <pasta> mentioned in the question, but this information does not directly answer the question.",
    "sentence": "Pasta originated in Italy and spread worldwide through trade, exploration, and cultural exchange."
}
"""
    query_prompt = """<question>
{question}
</question>
<answer_items>
{needles_str}
</answer_items>

EEE
【new_sentence(json)】
"""
    example_prompt = """
<example>
{example}
</example>"""
    if examples:
        example = '\n'.join(examples)
        example = example_prompt.format(example=example)
    else:
        example = ""
    query = query_prompt.format(question=question, needles_str=answer).replace('EEE', example)
    noise_needle = llm.call_qa_task(system_prompt_1, query, rd=True, json=True)
    # noise_needle_2 = llm.call_qa_task(system_prompt_2, query, rd=True, json=True)
    return noise_needle['sentence']


def insert_after_random_period(input_text, text, noise_needle, lang, start_idx=0):
    # 找到所有句号的位置（包括中文句号和英文句号）
    if lang == "zh":
        periods = [i for i, char in enumerate(text) if char in ['。']]
    elif lang == "en":
        periods = [i for i, char in enumerate(text) if char in ['.']]
    if not periods:
        # 如果没有找到句号，直接返回原字符串
        raise ValueError("No periods found in string")
    # 随机选择一个句号的位置
    random_period_index = random.choice(periods)
    # 在选定的句号后插入短字符串
    result_1 = input_text[:random_period_index + start_idx + 1] + f"\n{noise_needle}\n" + input_text[random_period_index + start_idx + 1:]
    result_2 = text[:random_period_index + 1] + f"\n{noise_needle}\n" + text[random_period_index + 1:]
    return result_1, result_2

def insert_needles_to_text_new(input_text_list, text_list, needles, noise_needles, start_index, lang):

    assert len(input_text_list) == len(text_list), f'input_text_list and text_list not match'
    assert len(text_list) == len(needles) + 1, f'text_list and needles not match'

    str_list_input = []
    str_list_text = []
    noise_insert_id_list = random.sample(range(len(text_list)), len(noise_needles))
    # noise_insert_id_list = [0]
    # noise_dict = {0: noise_needles[0]}
    noise_dict = {index: noise_needle for index, noise_needle in zip(noise_insert_id_list, noise_needles)}
    needles = needles + [""]
    for insert_id, (sub_input_text, sub_text, sub_needle) in enumerate(zip(input_text_list, text_list, needles)):

        if insert_id in noise_insert_id_list:
            if insert_id == 0:
                sub_input_text, sub_text = insert_after_random_period(sub_input_text, sub_text, noise_dict[insert_id], lang, start_idx=start_index)
            else:
                sub_input_text, sub_text = insert_after_random_period(sub_input_text, sub_text, noise_dict[insert_id], lang, start_idx=0)

        str_list_input.append(sub_input_text)
        str_list_text.append(sub_text)

        if sub_needle != "":
            str_list_input.append(sub_needle)
            str_list_text.append(sub_needle)

    input_text_with_needles = '\n'.join(str_list_input)
    text_with_needles = '\n'.join(str_list_text)
    return input_text_with_needles, text_with_needles

def load_QA_source():
    root_path = Path("data/infer_data/add_data/add_QA")
    file_list = list(root_path.glob("*.jsonl"))
    file_list = [file for file in file_list if file.stem.split('_')[-1] in ["en", "zh"]]
    file_list.sort()

    QA_dict = {}
    md5_list = []
    for file in file_list:
        with open(file, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                md5 = data['md5']
                md5_list.append(md5)
                QA_dict[md5] = data
    print(f"{len(set(md5_list))}/{len(md5_list)} QA loaded.")
    return QA_dict

if __name__ == '__main__':

    random.seed(42)

    file = Path('data/test_noise_data/NIAH_test_sample_200.jsonl')
    output_file = Path('data/test_noise_data/NIAH_test_with_semantic_noise.jsonl')
    outfile = open(output_file, 'w', encoding='utf-8')
    sample_num = 9

    QA_dict = load_QA_source()

    llm = LLM_API(model_name="deepseek-v3", api_type="openai")

    with open(file, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f, 1):
            # if idx <= 122:
            #     continue
            data = json.loads(line)
            md5 = data['md5']
            meta_QA = data['meta_QA']
            needles = meta_QA['needles']
            lang = data['lang']
            anchor_needles = needles.copy()
            text_with_needles = data['text_with_needles']
            text_list = cut_str_by_needles(needles, text_with_needles)
            if check_raw_long_text(text_list, data['meta_text']['raw_text']):
                raw_input_content = data['input']
                input_text_list = cut_str_by_needles(needles, raw_input_content)

                start_idx = raw_input_content.find(text_with_needles)

                if meta_QA['ppl'] == "syn":
                    # Noise for synthetic-temporal order needles
                    # 1. Provide a noisy diary entry with an inconsistent name.  
                    # 2. Provide an event description (non-diary format) where the surname matches but the given name does not.  
                    # 3. Provide one example for each of the above.
                    # continue
                    question = meta_QA['question']
                    noise_needle_1, noise_needle_2 = gen_syn_semantic_noise(llm, needles, question)
                    insert_needle_list = [
                        [noise_needle_1],
                        [noise_needle_2],
                        [noise_needle_1, noise_needle_2]
                    ]
                    noise_type_list = [
                        "syn_1",
                        "syn_2",
                        "syn_1+2"
                    ]
                    logging.warning(f"[{idx:03d}] SYN [{meta_QA['md5']}]: \nQ->[{meta_QA['question']}]\n\nAnswer:\n{meta_QA['answer']}\n\nNoise_1: {noise_needle_1}\nNoise_2: {noise_needle_2}\n\n")
                    pass
                elif meta_QA['ppl'] == "tkg":
                    # Noise for real-temporal order needles
                    # 1. Provide an event description related to Entity A but unrelated to Entity B  
                    # 2. Provide an event description related to Entity B but unrelated to Entity A  
                    #3. Provide one example for each of the above  
                    # continue
                    raw_QA = QA_dict[meta_QA['md5']]
                    if 'entity_A' in raw_QA['meta']:
                        # continue
                        ett_a = raw_QA['meta']['entity_A']
                        ett_b = raw_QA['meta']['entity_B']
                        noise_needle_1, noise_needle_2 = gen_tkg_semantic_noise(llm, needles, ett_a, ett_b)
                        insert_needle_list = [
                            [noise_needle_1],
                            [noise_needle_2],
                            [noise_needle_1, noise_needle_2]
                        ]
                        noise_type_list = [
                            "tkg_1",
                            "tkg_2",
                            "tkg_1+2"
                        ]
                        logging.warning(f"[{idx:03d}] TKG [{meta_QA['md5']}]: \nQ->[{meta_QA['question']}]\n\nAnswer:\n{meta_QA['answer']}\n\nNoise_1: {insert_needle_list[0][0]}\nNoise_2: {insert_needle_list[1][0]}\n\n")
                    else:
                        # continue
                        entity = raw_QA['meta']['entity']
                        question = raw_QA['meta']['raw_question']
                        insert_needle_list = []
                        example_needle = list(needles)
                        noise_needle = ""
                        for i in range(2):
                            example_needle = example_needle + [noise_needle] if noise_needle != "" else example_needle
                            noise_needle = gen_tkg_semantic_noise_finance(llm, example_needle, entity, question)
                            insert_needle_list.append([noise_needle])
                        insert_needle_list.append([insert_needle_list[0][0], insert_needle_list[1][0]])
                        logging.warning(f"[{idx:03d}] TKG - Finance [{meta_QA['md5']}]: \nQ->[{meta_QA['question']}]\n\nAnswer:\n{meta_QA['answer']}\n\nNoise_1: {insert_needle_list[0][0]}\nNoise_2: {insert_needle_list[1][0]}\n\n")
                    pass
                elif meta_QA['ppl'] == "open":
                    # Noise for real-logical order needles
                    raw_QA = QA_dict[meta_QA['md5']]
                    question = raw_QA['question']
                    answer = raw_QA['answer']
                    insert_needle_list = []
                    examples = []
                    for i in range(3):
                        noise_needle = gen_open_semantic_noise(llm, question, answer, examples)
                        insert_needle_list.append([noise_needle])
                        examples.append(noise_needle)
                    noise_type_list = [
                        "open_1",
                        "open_2",
                        "open_3"
                    ]
                    logging.warning(f"[{idx:03d}] OPEN [{meta_QA['md5']}]: \nQ->[{meta_QA['question']}]\n\nAnswer:\n{meta_QA['answer']}\n\nNoise_1: {insert_needle_list[0][0]}\nNoise_2: {insert_needle_list[1][0]}\nNoise_3: {insert_needle_list[2][0]}\n\n")
                    pass

                for i, (insert_needles, noise_type) in enumerate(zip(insert_needle_list, noise_type_list), 1):

                    input_content, text_with_needles = insert_needles_to_text_new(input_text_list, text_list, needles, insert_needles, start_idx, lang)

                    data['input'] = input_content
                    data['text_with_needles'] = text_with_needles
                    data['meta_QA']['needles'] = needles
                    data['meta_QA']['insert_needles'] = insert_needles
                    data['md5'] = f"{md5}_{i+1}"
                    data['noise_type'] = noise_type

                    outfile.write(json.dumps(data, ensure_ascii=False) + '\n')
                    outfile.flush()

    outfile.close()
