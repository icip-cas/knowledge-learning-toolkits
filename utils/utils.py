import json
import os
import pandas as pd
import numpy as np
import cn2an
import re
import bisect
import yaml
from argparse import Namespace

# 定义颜色常量
RED = "\033[91m"
GREEN = "\033[92m"
RESET = "\033[0m"  # 重置为默认颜色


def dict_to_namespace(d):
    """
    Recursively convert a dictionary into a Namespace.
    """
    ns = Namespace()
    for key, value in d.items():
        setattr(ns, key, dict_to_namespace(value) if isinstance(value, dict) else value)
    return ns


def load_config_with_namespace(config_path):
    """
    Load YAML config and convert to nested Namespace object.
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        config_dict = yaml.safe_load(f)
    return dict_to_namespace(config_dict)


def find_all_indices_sorted(lst, element):
    # 使用二分查找找到目标元素第一次出现的位置
    left_index = bisect.bisect_left(lst, element)

    # 如果元素不在列表中，返回空列表
    if left_index == len(lst) or lst[left_index] != element:
        return []

    # 使用二分查找找到目标元素最后一次出现的位置
    right_index = bisect.bisect_right(lst, element)

    # 返回从 left_index 到 right_index 之间的所有索引
    return list(range(left_index, right_index))


def number_to_chinese(number):
    if '%' in str(number):
        percent_value = float(str(number).replace('%', ''))
    else:
        percent_value = float(number)

    # 将整数部分和小数部分分开处理
    integer_part = int(percent_value)
    decimal_part = int((percent_value - integer_part) * 100)

    # 转换整数部分
    integer_part_cn = cn2an.an2cn(integer_part)

    # 处理小数部分
    if decimal_part > 0:
        decimal_part_cn = cn2an.an2cn(decimal_part)
        result = f"{integer_part_cn}点{decimal_part_cn}"
    else:
        result = f"{integer_part_cn}"

    # 如果百分数，加上百分之
    if '%' in str(number):
        result = f'百分之{result}'

    return result


def is_name(s):
    return bool(re.match(r'^[A-Z][a-z]+(?: [A-Z][a-z]+)*$', s))


def read_file(file_name, split_str=None, fraction=None):
    if 'jsonl' in file_name:
        datas = []
        with open(file_name, 'r', encoding='utf-8') as f:
            # Read part of file
            if fraction:
                total_lines = sum(1 for _ in f)
                f.seek(0)  # 回到文件开头
                read_lines = int(total_lines * fraction)
                for i, line in enumerate(f):
                    if i >= read_lines:
                        break
                    data = json.loads(line)
                    datas.append(data)
            else:
                lines = f.readlines()
                for line in lines:
                    data = json.loads(line)
                    datas.append(data)
        return datas
    elif 'json' in file_name:
        with open(file_name, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    elif 'xlsx' in file_name:
        data = pd.read_excel(file_name)
        return data
    elif 'csv' in file_name:
        data = pd.read_csv(file_name)
        return data
    elif 'parquet' in file_name:
        data = pd.read_parquet(file_name)
        return data
    else:
        with open(file_name, 'r', encoding='utf-8') as f:
            data = f.read()
        if split_str:
            elements = data.split(split_str)
            elements = [e.strip() for e in elements if e.strip()]
            return elements
        else:
            return data


def write_file(file_name, data, split_str=None):
    if type(data) is list:
        lists = data
        if 'jsonl' in file_name:
            with open(file_name, 'w', encoding='utf-8') as f:
                for element in lists:
                    json.dump(element, f, ensure_ascii=False)
                    f.write('\n')
        else:
            split_str = '\n' if not split_str else split_str
            with open(file_name, 'w', encoding='utf-8') as f:
                for element in lists:
                    f.write(str(element))
                    f.write(split_str)
    elif type(data) is dict:
        with open(file_name, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False)
    elif isinstance(data, pd.DataFrame):
        if 'csv' in file_name:
            data.to_csv(file_name, index=False)
        elif 'xlsx' in file_name:
            data.to_excel(file_name, index=False)
        elif 'parquet' in file_name:
            data.to_parquet(file_name, index=False)
    else:
        with open(file_name, 'w', encoding='utf-8') as f:
            f.write(str(data))


def clean_wrong_lines(path):
    if 'jsonl' in path:
        with open(path, 'rb') as f:
            decode_lines = []
            for i, line in enumerate(f, 1):  # 逐行读取并记录行号
                try:
                    # 尝试解码每一行
                    decode_line = line.decode('utf-8')
                    decode_lines.append(decode_lines.append(eval(eval(decode_line))))
                except UnicodeDecodeError as e:
                    print(f"Error at line {i}")
                    print(f"Last line: {decode_lines[-1]}")
                    # 打印出错误行，或者将错误信息保存到日志
                    print(f"Error details: {e}")
        write_file(path, decode_lines)
    else:
        print(f'Not implement clean wrong lines for file type {path.split(".")[-1]}')


def walk(path, only_file_name=False):
    filenames = []
    for root, dirs, files in os.walk(path):
        for file in files:
            filename = os.path.join(root, file)
            if only_file_name:
                filename = file
            filenames.append(filename)
    return filenames


def create_folder(folder_path):
    if not os.path.exists(folder_path):
        # 如果不存在，则创建文件夹
        os.makedirs(folder_path)
        print(f"Create '{folder_path}'!")


def clean_text(text):
    # 使用正则表达式清除表情符号，但保留文字和标点符号
    cleaned_text_with_punctuation = re.sub(r'[^\w\s\u4e00-\u9fff#，,。.！!？?；;：:、\'‘’\"“”(（)）<《>》\[【\]】〈〉{}·—…-]', '',
                                           text)

    return cleaned_text_with_punctuation


def is_empty_pd_value(value):
    # 检查空字符串
    if isinstance(value, str):
        if value.strip() == "":
            return True
    # 检查空列表
    elif isinstance(value, list):
        if len(value) == 0:
            return True
    # 检查空字典
    elif isinstance(value, dict):
        if len(value) == 0:
            return True
    # 检查 numpy 数组是否为空
    elif isinstance(value, np.ndarray):
        if value.size == 0:
            return True
    # 检查是否为 None 或 NaN
    elif pd.isna(value):
        return True
    return False
