from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import sys
from tqdm import tqdm
from utils.utils import *


def load_model(model_id, gpu_id=None):
    """
    Load tokenizer and model.

    Args:
        model_id (str): Path or name of the pretrained model.
        gpu_id (int, optional): GPU device ID. If None, use automatic device mapping.

    Returns:
        tokenizer (AutoTokenizer): Loaded tokenizer.
        model (AutoModelForCausalLM): Loaded causal language model.
    """
    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token

    # load model
    if gpu_id:
        device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            use_safetensors=True,
            device_map=None
        )
        model.to(device)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            use_safetensors=True,
            device_map="auto",
        )

    return tokenizer, model


def generate_text(messages, tokenizer, model, history=[], chat=False, **paras):
    """
    Generate responses based on input messages and optional history.

    Args:
        messages (list): List of input prompts.
        tokenizer (Tokenizer): Tokenizer for input encoding.
        model (Model): Language model for generation.
        history (list, optional): List of (user, assistant) message tuples for context.
        temperature (float, optional): Sampling temperature.
        max_new_tokens (int): Maximum number of tokens to generate.
        system (str, optional): Optional system prompt to prepend.
        chat (bool): Whether to use chat-style prompt formatting.

    Returns:
        list: Generated responses.
    """
    # Construct input ids.
    if chat:
        # construct conversations
        conversations = []
        for message in messages:
            conversation = []
            # add system message
            system = paras.get("system_prompt", "")
            if system:
                conversation.append({"role": "system", "content": system})
            # add history
            for user, assistant in zip(*history):
                conversation.extend([{"role": "user", "content": user}, {"role": "assistant", "content": assistant}])
            # add message
            conversation.append({"role": "user", "content": message})

            conversations.append(conversation)

        inputs = tokenizer.apply_chat_template(conversations, return_tensors="pt", padding=True, return_dict=True)
        input_ids = inputs["input_ids"].to(model.device)
        attention_mask = inputs["attention_mask"].to(model.device)
    else:
        encoding = tokenizer.batch_encode_plus(messages, return_tensors="pt", padding=True, truncation=True, add_special_tokens=False)
        input_ids = encoding["input_ids"].to(model.device)
        attention_mask = encoding["attention_mask"].to(model.device)

    responses = []
    # get paras
    allowed_keys = {
        'max_new_tokens',
        'temperature',
    }
    filtered_paras = {k: v for k, v in paras.items() if k in allowed_keys}
    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>"),
        tokenizer.convert_tokens_to_ids('<|endoftext|>'),
    ]
    terminators = [t for t in terminators if t]
    generate_kwargs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "eos_token_id": terminators,  # Specify tokens to stop generation
        **filtered_paras
    }
    outputs = model.generate(**generate_kwargs)
    for i, output in enumerate(outputs):
        response = tokenizer.decode(output, skip_special_tokens=True)
        response = response.replace(messages[i], '')
        responses.append(response)
    return responses


def generate_text_batch(data,
                        batch_size,
                        model_name,
                        chat=False,
                        save_step=10,
                        start_idx=-1,
                        end_idx=-1,
                        save_path='result.csv',
                        log_path='log.log',
                        config_path='config.yaml',
                        **paras):
    """
    Batch generation of model responses.

    Args:
        data (DataFrame): Input DataFrame with a column named 'prompt'.
        batch_size (int): Batch size for inference.
        model_id (str): Huggingface model ID or local path.
        gpu_id (int, optional): GPU device ID. If None, use device_map='auto'.
        temperature (float): Sampling temperature.
        max_new_tokens (int): Max new tokens to generate per message.
        system (str): Optional system prompt.
        chat (bool): Whether to use chat-style formatting.
        save_path (str): Path to save the results.
        save_step (int): Save frequency.
        start_idx (int): Start index for subrange processing.
        end_idx (int): End index for subrange processing.
        log_path (str): File to save logs.

    Returns:
        DataFrame: Updated DataFrame with generated responses in 'pred' column.
    """
    if log_path:
        log_file = open(log_path, 'w')
        sys.stdout = log_file
        sys.stderr = log_file

    # load model config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)['models']
        if model_name not in config:
            raise ValueError(f"Model {model_name} not found in config")

    try:
        # Load model
        tokenizer, model = load_model(config[model_name]['local_path'], paras.get('gpu_id', None))

        # Split data
        if start_idx != -1 or end_idx != -1:
            save_path = save_path.replace('.parquet', f'_{start_idx}_{end_idx}.parquet')
        end_idx = len(data) if end_idx == -1 else end_idx
        start_idx = 0 if start_idx == -1 else start_idx
        data = data[max(start_idx, 0):end_idx]
        data = data.reset_index(drop=True)

        # Reload result
        if os.path.exists(save_path):
            data = read_file(save_path)
        else:
            data['pred'] = None

        # Infer
        print(f'There are {len(data[~data["pred"].apply(lambda x: type(x) is str)])} messages to response.')
        for row_idx in tqdm(range(0, len(data), batch_size)):
            # Check if these cases are answered
            is_answer = True
            for i in range(batch_size):
                answer = data.loc[min(row_idx + i, len(data) - 1), 'pred']
                if (type(answer) is float and np.isnan(answer)) or answer is None:
                    is_answer = False
                    break

            # If not answered, start inferring
            if not is_answer:
                messages = data[row_idx:row_idx + batch_size]['prompt'].tolist()
                responses = generate_text(messages, tokenizer, model, chat=chat, **paras)
                for response_idx, response in enumerate(responses):
                    data.loc[row_idx + response_idx, 'pred'] = response
                    if ((row_idx + response_idx) % save_step == 0) and save_path:
                        write_file(save_path, data)
        if save_path:
            write_file(save_path, data)
        return data
    finally:
        if log_path:
            log_file.close()
            sys.stdout = sys.__stdout__
            sys.stderr = sys.__stderr__