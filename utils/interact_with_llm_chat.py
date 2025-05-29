import yaml
from openai import OpenAI
from openai.types.chat import ChatCompletion
from tqdm import tqdm
from multiprocessing.pool import ThreadPool
from functools import partial
import os.path
import random
import json
import concurrent.futures
import openai


def _get_openai_response(
    client: openai.Client,
    model: str,
    user_prompt: str,
    n: int = 1,
    stops: list = [],
    **paras
) -> ChatCompletion:
    """
    Send a prompt to the OpenAI API and get the response.

    Args:
        client (openai.Client): OpenAI API client.
        model (str): Model name.
        user_prompt (str): User query.
        system_prompt (str, optional): System-level instruction. Defaults to "".
        max_tokens (int, optional): Max number of tokens in the response.
        temperature (float, optional): Sampling temperature. Defaults to 1.
        n (int, optional): Number of completions. Defaults to 1.
        stops (list, optional): Stop words or phrases.
        timeout (int, optional): Timeout for the request. Defaults to 20.
        **kwargs: Other keyword arguments.

    Returns:
        ChatCompletion: OpenAI API chat completion result.
    """
    messages = []
    system_prompt = paras.get("system_prompt", "")
    if len(system_prompt) > 0:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})

    # answer
    allowed_keys = {
        'max_tokens',
        'temperature',
        'top_p',
        'frequency_penalty',
        'presence_penalty',
        'logit_bias',
        'user',
        'seed',
        'timeout'
    }
    filtered_paras = {k: v for k, v in paras.items() if k in allowed_keys}
    ret = client.chat.completions.create(
        model=model,
        messages=messages,
        n=n,
        stop=stops,
        **filtered_paras
    )

    if ret.choices is None:
        raise ValueError("No response from OpenAI")
    return ret


def get_openai_response(model,
                        prompt,
                        api_key,
                        api_base_pool,
                        **paras):
    """
    Randomly select an API endpoint and call the model.

    Args:
        model (str): Model name.
        prompt (str): Prompt text.

    Returns:
        str: Model response.
    """
    client = OpenAI(api_key=api_key,
                    base_url=random.choice(api_base_pool))

    ret = _get_openai_response(client=client,
                               model=model, 
                               user_prompt=prompt,
                               **paras)
    
    return ret.choices[0].message.content


def call_model(
    model,
    prompt,
    api_key,
    api_base_pool,
    max_try_time=3,
    timeout=30,
    **paras
):
    """
    Call the model with retry and timeout mechanism.

    Args:
        model (str): Model name.
        prompt (dict): A dictionary containing 'prompt' and 'info'.
        max_try_time (int, optional): Maximum number of retries.
        timeout (int, optional): Timeout for each attempt.

    Returns:
        dict: A dictionary indicating status, info, and examples.
    """
    call_model_func = lambda model, prompt: get_openai_response(
        model, prompt, api_key, api_base_pool, **paras
    )
    success_status_func = lambda x, info: {'status': 'success', 'info': info, 'examples': [x]}
    failed_status_func = lambda info: {'status': 'failed', 'info': info, 'examples': None}

    _prompt = prompt['prompt']
    basic_info = prompt['info']
    try_times = 1

    while try_times < max_try_time + 1:
        try:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(call_model_func, model, _prompt)
                response = future.result(timeout=timeout)

            return success_status_func(response, basic_info)
        except concurrent.futures.TimeoutError:
            print(f"[Timeout] {timeout}s exceeded on try {try_times}")
            try_times += 1
        except Exception as e:
            print(f"[Error] Failed to generate data: {e}")
            try_times += 1

    return failed_status_func(basic_info)


def multi_thread_call_and_write(
        prompts,
        model='DeepSeek-R1',
        paras=None,
        output_file_path='output.jsonl',
        config_path='config.yaml',
        output_parser=lambda x: {'text': x},
        write_mode=None,
        output=True
):
    """
    Multithreaded batch request and write to file.

    Args:
        prompts (list): A list of dictionaries, each containing a 'prompt' string
            and an 'info' dictionary with metadata.
        model (str, optional): Name of the model to invoke. Defaults to 'DeepSeek-R1'.
        output_file_path (str, optional): Path to the output `.jsonl` file.
        config_path (str, optional): Path to the model configuration YAML file.
        output_parser (function, optional): Function that parses a single model output
            into a dictionary. Defaults to a function returning {'text': output}.
        write_mode (str, optional): File write mode: 'w' to overwrite or 'a' to append.
            Defaults to 'a' if file exists, otherwise 'w'.
        output (bool, optional): Whether to display a progress bar using `tqdm`.

    Raises:
        ValueError: If the specified model is not found in the configuration file.

    Returns:
        None
    """
    # load model config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)['models']
        if model not in config:
            raise ValueError(f"Model {model} not found in config")

    # call model
    if not paras:
        paras = {
            'system_prompt': "",
            'max_new_tokens': 1024,
            'temperature': 0.2,
            'max_try_time': 3,
            'timeout': 30,
            'threads': 10
        }
    partial_model_call = partial(
        call_model,
        model,
        api_key=config[model]['api']['api_key'],
        api_base_pool=config[model]['api']['api_base_pool'],
        **paras
    )

    # answer
    if not write_mode:
        write_mode = 'a' if os.path.exists(output_file_path) else 'w'
    with open(output_file_path, write_mode, encoding='utf-8') as f:
        pool = ThreadPool(paras['threads'])
        iterator = pool.imap_unordered(partial_model_call, prompts, chunksize=1)

        if output:
            iterator = tqdm(iterator, total=len(prompts), mininterval=1, maxinterval=10)

        for response in iterator:
            if response['status'] == 'success':
                try:
                    examples = response['examples']
                    basic_info = response['info']
                    for example in examples:
                        f.write(json.dumps({'info': basic_info, **output_parser(example)}, ensure_ascii=False) + '\n')
                    f.flush()
                except Exception as e:
                    print(f'[Write Error] {e}')
                    continue
