import json
import random
import tqdm
from pathlib import Path
import re
import hashlib
random.seed(1024)

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

import tiktoken
encoding = tiktoken.encoding_for_model("gpt-4")

def gen_pair_md5(_input, _output):
    data = _input + _output
    return hashlib.md5(data.encode(encoding='UTF-8')).hexdigest()

def insert_strings_to_random_position(long_text, insert_strings, lang='zh'):
    if lang == "zh":
        period_positions = [m.start() for m in re.finditer('。', long_text)]
    else:
        period_positions = [m.start() for m in re.finditer('\.', long_text)]

    if not period_positions:
        if lang == 'zh':
            raise ValueError("No [。] in the long text. ")
        elif lang == 'en':
            raise ValueError("No [.] in the long text. ")
        
    if len(period_positions) <= 80:
        raise ValueError("The length of period_positions is too short. ")
    
    num_strings = len(insert_strings)
    insert_position = random.sample(period_positions, num_strings)
    insert_position.sort()
    insert_position = [-1] + insert_position + [len(long_text)-1]
    shard_text = []
    for s, e in zip(insert_position[:-1], insert_position[1:]):
        shard_text.append(long_text[s+1:e+1])

    assert len(shard_text) == num_strings + 1
    new_text = ''
    for raw_shard_text, insert_str in zip(shard_text[:-1], insert_strings):
        new_text += raw_shard_text + '\n' + insert_str + '\n'
    new_text += shard_text[-1]
    return new_text

def build_data(QA, long_text, lang):

    if QA['meta']['source'] == "open":
        assert len(QA['needles']) == len(QA['meta']['items'])

    question = QA['question']
    answer = QA['answer']
    needles = QA['needles']
    raw_needles = needles.copy()
    raw_text = long_text['text']

    random.shuffle(needles)

    text_with_needles = insert_strings_to_random_position(raw_text, needles, lang)

    data = {}

    if lang == 'zh':
        if random.choices([0, 1], weights=[0, 100])[0]:
            input_content = random.choice(["文档：","文档：\n","文档:","文档:\n",""])+ \
                text_with_needles + "\n\n" +random.choice(["问题：","问题：\n","问题:","问题:\n","回答以下问题：\n"]) + question
        else:
            input_content = random.choice(["问题：","问题：\n","问题:","问题:\n","根据下面的文档回答问题：\n"]) + \
                question + "\n\n" + random.choice(["文档：","文档：\n","文档:","文档:\n"]) + text_with_needles
            
    elif lang == 'en':
        if random.choices([0, 1], weights=[0, 100])[0]:
            input_content = random.choice(["Document: ","Document: \n","Document:","Document:\n",""])+ \
                text_with_needles + "\n\n" +random.choice(["Question: ","Question: \n","Question:","Question:\n","Please answer the question:\n"]) + question
        else:
            input_content = random.choice(["Question: ","Question: \n","Question:","Question:\n","Answer the questions according to the following documents:\n"]) + \
                question + "\n\n" + random.choice(["Document: ","Document: \n","Document:","Document:\n"]) + text_with_needles

    data['md5'] = gen_pair_md5(input_content, answer)  
    data['length'] = len(encoding.encode(input_content))
    data['lang'] = lang
    data['num_needles'] = len(needles)

    data['input'] = input_content
    data['output'] = answer
    data['text_with_needles'] = text_with_needles

    data['meta_QA'] = {}
    data['meta_QA']['md5'] = QA['md5']
    data['meta_QA']['question'] = question
    data['meta_QA']['answer'] = answer
    data['meta_QA']['raw_needles'] = raw_needles
    data['meta_QA']['needles'] = needles
    data['meta_QA']['source'] = QA['meta']['source']
    data['meta_QA']['ppl'] = QA['meta']['ppl']

    data['meta_text'] = {}
    data['meta_text']['md5'] = long_text['md5']
    data['meta_text']['raw_text'] = raw_text
    if 'source' in long_text:
        data['meta_text']['source'] = long_text['source']
    else:
        data['meta_text']['source'] = 'syn_by_mix_source'
    data['meta_text']['length'] = long_text['length']

    return data


def prepare_data(lang):

    QA_file_path = Path('data/source/QA')
    QA_file_list = [QA_file_path / f'QA_{lang}.jsonl']

    QA_pool = {}
    for file_path in QA_file_list:
        with open(file_path, 'r', encoding='utf-8') as f:
            for i, line in tqdm.tqdm(enumerate(f), desc=f"Load: {file_path.stem}"):
                data = json.loads(line)
                md5 = data['md5']
                if md5 not in QA_pool:
                    QA_pool[md5] = data
                else:
                    print(f"Duplication QA: {md5}")

    long_text_path = Path('data/source/LongText')
    long_text_file_list = [long_text_path / f'doc_{lang}.jsonl']
    count = 0
    for file in long_text_file_list:
        with open(file, 'r', encoding='utf-8') as f:
            for line in tqdm.tqdm(f):
                count += 1
    
    assert count == len(QA_pool), f'QA and Long_text_bank not match'

    return QA_pool, long_text_file_list

def read_jsonl_files(*file_paths):
    for file_path in file_paths:
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                yield json.loads(line)

def NIAH_PPL(lang):

    # 1. load QA and Long text
    QA_pool, long_text_file_list = prepare_data(lang)
    QA_keys = list(QA_pool.keys())
    random.shuffle(QA_keys)
    long_text_iterator = read_jsonl_files(*long_text_file_list)

    # 2. prepare output file
    output_file = Path(f'data/example/NIAH_{lang}.jsonl')
    outfile = open(output_file, 'w', encoding='utf-8')

    error_QA_file = Path(f'data/example/NIAH_{lang}_error_QA.jsonl')
    error_long_file = Path(f'data/example/NIAH_{lang}_error_LongText.jsonl')
    error_QA = open(error_QA_file, 'w', encoding='utf-8')
    error_long = open(error_long_file, 'w', encoding='utf-8')
    
    # 3. generate data
    error_count = 0
    for i, (QA_key, long_text) in enumerate(zip(QA_keys, long_text_iterator)):
        try:
            QA = QA_pool[QA_key]

            # NIAH synthetic pipelines
            data = build_data(QA, long_text, lang)

            # write to file
            outfile.write(json.dumps(data, ensure_ascii=False) + '\n')
            outfile.flush()
            # logging.info(f"Done {i+1} | QA: {QA['md5']} | LONG: {long['md5']}")
        except Exception as e:

            # 4. write error to file
            logging.info(f"Error {e} | QA: {QA['md5']} | LONG: {long_text['md5']}")
            error_QA.write(json.dumps(QA, ensure_ascii=False) + '\n')
            error_QA.flush()
            error_long.write(json.dumps(long_text, ensure_ascii=False) + '\n')
            error_long.flush()
            error_count += 1

    outfile.close()
    error_QA.close()
    error_long.close()

    if error_count > 0:
        logging.info(f"Error count: {error_count}")
    else:
        error_QA_file.unlink()
        error_long_file.unlink()

    return None

if __name__ == '__main__':

    # 1. Generate english data
    NIAH_PPL('en')

    # 2. Generate chinese data
    NIAH_PPL('zh')