import re
import json
import tqdm
import random
import pandas as pd
import numpy as np
from pathlib import Path

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

def insert_needles_to_text(text_list, needles):

    assert len(text_list) == len(needles) + 1, f'text_list and needles not match'

    str_list = []
    for sub_text, sub_needle in zip(text_list[:-1], needles):
        str_list.append(sub_text)
        str_list.append(sub_needle)
    
    str_list.append(text_list[-1])
    text_with_needles = '\n'.join(str_list)
    return text_with_needles

def check_moving_idx_box(str_list, lang):

    total_str = ''.join(str_list)
    if lang == "zh":
        period_positions = [m.start() for m in re.finditer('。', total_str)]
    else:
        period_positions = [m.start() for m in re.finditer('\.', total_str)]

    cut_idx_list = []
    current_length = 0
    for sub_str in str_list[:-1]:
        current_length += len(sub_str)
        cut_idx_list.append(current_length - 1)

    indices = [period_positions.index(idx) for idx in cut_idx_list]

    index_boxes = {}
    for i, (id, cut_id) in enumerate(zip(indices, cut_idx_list)):
        if index_boxes:
            # 当前box的最小值
            min_current = period_positions[max(id-3, 0)]
            # 上一个box的最大值
            max_pre = index_boxes[cut_idx_list[i-1]][-1]
            if max_pre < min_current:
                # 如果没有重叠，直接添加box
                index_boxes[cut_id] = period_positions[max(id-3, 0):id+3+1]
            else:
                # 临界id
                bound_id = (max(id-3, 0) + (indices[i-1] + 3 + 1)) // 2
                #如果重叠了，上一个box调整最大值
                index_boxes[cut_idx_list[i-1]] = period_positions[max(indices[i-1]-3, 0):bound_id+1]
                #当前box调整最小值
                index_boxes[cut_id] = period_positions[bound_id+1:id+3]

        else:
            index_boxes[cut_id] = period_positions[max(id-3, 0):id+3+1]

    for key, value in index_boxes.items():
        if len(value) <= 1:
            continue
        index_boxes[key] = [idx for idx in index_boxes[key] if idx != key]
    
    return index_boxes, cut_idx_list

def gen_moved_text(text_list, cut_idx_list, needles):

    text_str = ''.join(text_list)
    cut_idx_list = [-1] + cut_idx_list + [len(text_str) - 1]

    shard_text = []
    for s, e in zip(cut_idx_list[:-1], cut_idx_list[1:]):
        shard_text.append(text_str[s+1:e+1])
    
    assert len(shard_text) == len(needles) + 1
    new_text = ''
    for raw_shard_text, insert_str in zip(shard_text[:-1], needles):
        new_text += raw_shard_text + '\n' + insert_str + '\n'
    new_text += shard_text[-1]
    return new_text


def make_little_moving(needles, input_text_list, text_list, lang, start_idx):
    """
    Make a little moving to the needles locations
    """
    assert len(input_text_list) == len(needles) + 1, f'input_text_list and needles not match'
    assert len(text_list) == len(needles) + 1, f'text_list and needles not match'

    # input_text_moving_boxes, raw_input_text_cut_idx_list = check_moving_idx_box(input_text_list, lang)
    text_moving_boxes, raw_text_cut_idx_list = check_moving_idx_box(text_list, lang)

    new_input_text_cut_idx_list = []
    new_text_cut_idx_list = []
    for text_moving_box in text_moving_boxes.values():

        chosen_id = random.choice(np.arange(len(text_moving_box)))
        new_text_cut_idx_list.append(text_moving_box[chosen_id])
        new_input_text_cut_idx_list.append(text_moving_box[chosen_id] + start_idx)

    new_input_text = gen_moved_text(input_text_list, new_input_text_cut_idx_list, needles)
    new_text = gen_moved_text(text_list, new_text_cut_idx_list, needles)
    
    return new_input_text, new_text


def make_large_moving(needles, input_text_list, text_list, lang, start_idx):
    """
    Make a large moving to the needles locations
    """
    assert len(input_text_list) == len(needles) + 1, f'input_text_list and needles not match'
    assert len(text_list) == len(needles) + 1, f'text_list and needles not match'

    text_str = ''.join(text_list)
    if lang == "zh":
        period_positions = [m.start() for m in re.finditer('。', text_str)]
    else:
        period_positions = [m.start() for m in re.finditer('\.', text_str)]

    num_strings = len(needles)
    insert_position = random.sample(period_positions, num_strings)
    insert_position.sort()
    insert_position_for_input = [idx + start_idx for idx in insert_position]
    
    new_input_text = gen_moved_text(input_text_list, insert_position_for_input, needles)
    new_text = gen_moved_text(text_list, insert_position, needles)
    
    return new_input_text, new_text




if __name__ == '__main__':

    random.seed(1024)

    file = Path('data/example/NIAH_en.jsonl')
    output_file = file.parent / f'{file.stem}_noise.jsonl'
    outfile = open(output_file, 'w', encoding='utf-8')
    sample_num = 9
    
    with open(file, 'r', encoding='utf-8') as f:
        for line in tqdm.tqdm(f):
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

                for i in range(sample_num):

                    if i < 3:
                        # make Tiny Movement
                        input_content, text_with_needles = make_little_moving(needles, input_text_list, text_list, lang, start_idx)
                        noise_type = 'little_moving'
                    elif i < 6:
                        # make Significant Movement
                        input_content, text_with_needles = make_large_moving(needles, input_text_list, text_list, lang, start_idx)
                        noise_type = 'large_moving'
                    else:
                        # make Reordering
                        random.shuffle(needles)
                        while anchor_needles == needles:
                            random.shuffle(needles)
                        input_content = insert_needles_to_text(input_text_list, needles)
                        text_with_needles = insert_needles_to_text(text_list, needles)
                        noise_type = 'order'

                    data['input'] = input_content
                    data['text_with_needles'] = text_with_needles
                    data['meta_QA']['needles'] = needles
                    data['md5'] = f"{md5}_{i+1}"
                    data['noise_type'] = noise_type

                    outfile.write(json.dumps(data, ensure_ascii=False) + '\n')

    outfile.close()
