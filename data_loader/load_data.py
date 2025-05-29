import os.path
from functools import partial
from pathlib import Path

import pandas as pd

from utils.utils import *
from data_loader.prompts import *
from utils.interact_with_llm import interact_with_llm


def load_raw_data(file_path, file_type, save_path=None, start_idx=0, end_idx=-1):
    """
    Load raw data.

    Supported data types:
    1. Triplet format:
       file_path: data.jsonl, format: [{'idx': 0, 'triplet': [subject, relation, object]}]
       file_type: 'triplet'
    2. Free text format:
       file_path: data.jsonl, format: [{'idx': 0, 'text': text}]
       file_type: 'text'

    Returns:
        A unified format:
        [{'idx': 0, 'text': text, 'knowledges': [{'text': text, 'triplet': [subject, relation, object]}]}]
    """

    raw_data = read_file(file_path)
    data = []

    if file_type == 'triplet':
        for raw_d in raw_data:
            data.append({
                'idx': raw_d['idx'],
                'text': None,
                'knowledges': [{'text': None, 'triplet': raw_d['triplet']}]
            })
    elif file_type == 'text':
        for raw_d in raw_data:
            data.append({
                'idx': raw_d['idx'],
                'text': raw_d['text'],
                'knowledges': None
            })

    end_idx = len(data) if end_idx == -1 else end_idx
    data = data[max(start_idx, 0):end_idx]

    if save_path:
        write_file(save_path, data)

    return data


def data_preprocess(args, file_path, save_path=None):
    """
    Preprocess raw data:
    1. If `knowledges` is None, extract triplets from the original `text`;
    2. If `text` under `knowledges` is None, generate fluent natural language from triplets.
    """
    # Init.
    args = args.augmentation

    # Step 0: Load input data
    data = read_file(file_path)
    if not os.path.exists(args.data.middle_result_dir):
        os.makedirs(args.data.middle_result_dir, exist_ok=True)

    # Step 1: Extract triplets from natural language text
    if args.data.data_type == 'text':
        # Extract triplets
        relation_type_to_keys = {
            'role_attribute': ['role', 'attribute', 'value'],
            'role_inter': ['role', 'relation', 'role'],
        }
        relation_type_to_prompt_template = {
            'role_attribute': prompt_template_role_attribute_en,
            'role_inter': prompt_template_role_inter_en,
        }

        def parse_response_to_triplets(response, relation_type):
            output_lines = []
            raw_lines = response.replace('|'.join(relation_type_to_keys[relation_type]), '').strip().split('\n')
            raw_lines_set = {line for line in raw_lines if len(line.split('|')) == 3}

            for raw_line in raw_lines_set:
                parts = raw_line.split('|')
                output_lines.append([parts[0], parts[1], parts[2]])

            return {'knowledges': output_lines}

        # UIE
        results = []
        for relation_type in relation_type_to_keys.keys():
            prompts = [
                {
                    'prompt': relation_type_to_prompt_template[relation_type].format(text=d['text']),
                    'info': {'d_idx': d_idx}
                }
                for d_idx, d in enumerate(data)
            ]
            middle_path_nl2kn = os.path.join(args.data.middle_result_dir,
                                             f'nl2kn_{Path(file_path).stem}_{relation_type}.jsonl')
            results += interact_with_llm(prompts,
                                         args.model.name,
                                         args.model.mode,
                                         partial(parse_response_to_triplets, relation_type=relation_type),
                                         middle_path_nl2kn,
                                         **args.inference.__dict__)

        # Update extracted triplets into the original data
        data.sort(key=lambda x: x['idx'])
        data_idxs = [x['idx'] for x in data]
        for result in results:
            d_idx = result['info']['d_idx']
            data_d_idxs = find_all_indices_sorted(data_idxs, d_idx)
            if len(data_d_idxs):
                data_idx = data_d_idxs[0]
                triplets = [{'text': None, 'triplet': k} for k in result['knowledges']]
                if data[data_idx]['knowledges'] is None:
                    data[data_idx]['knowledges'] = triplets
                else:
                    data[data_idx]['knowledges'] += triplets

    # Step 2: Convert triplets to natural language
    prompt_template_en = (
        "{prefix}Given a factual triplet ({s}, {r}, {o}), i.e {s}'s {r} is {o}, "
        "rewrite it into a fluent natural language text without any additional information:"
    )
    prompt_template_zh = (
        "{prefix}给定一个事实三元组 (s, r, o)，即 s 的 r 是 o，将其重写为流畅的自然语言文本，"
        "请务必遵循下面的输出格式，不要添加任何说明信息。\n\n事实三元组：({s}, {r}, {o})\n自然语言文本："
    )
    lang_to_prompt_template = {
        'en': prompt_template_en,
        'zh': prompt_template_zh
    }

    prompts = [
        {
            'prompt': lang_to_prompt_template[args.language].format(
                prefix=args.message,
                s=k['triplet'][0], r=k['triplet'][1], o=k['triplet'][2]
            ),
            'info': {'d_idx': d_idx, 'k_idx': k_idx}
        }
        for d_idx, d in enumerate(data)
        for k_idx, k in enumerate(d['knowledges'])
    ]
    middle_path_kn2nl = os.path.join(args.data.middle_result_dir, f'kn2nl_{Path(file_path).stem}.jsonl')
    results = interact_with_llm(prompts,
                                args.model.name,
                                args.model.mode,
                                middle_path=middle_path_kn2nl,
                                **args.inference.__dict__)

    # Update natural language text back to knowledge
    data.sort(key=lambda x: x['idx'])
    data_idxs = [x['idx'] for x in data]
    for result in results:
        d_idx, k_idx = result['info']['d_idx'], result['info']['k_idx']
        data_d_idxs = find_all_indices_sorted(data_idxs, d_idx)
        if len(data_d_idxs):
            data_idx = data_d_idxs[0]
            data[data_idx]['knowledges'][k_idx]['text'] = result['text']

    if save_path:
        write_file(save_path, data)

    return data
