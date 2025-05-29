from utils.utils import *
from utils.interact_with_llm import interact_with_llm
from data_augmentation.prompt_hub import *
import random


def run_augment(args):
    """
    Main function to run data augmentation based on the specified arguments.
    The augmentation can be done using an API model or a local model, with either basic or random styles.

    Args:
        args: An argparse.Namespace object with all necessary parameters.
    """
    # Read and preprocess input data
    args = args.augmentation
    data = preprocess_data(args)

    # Check if augmentation results already exist
    if args.data.result_path is None:
        args.data.result_path = args.data.data_path.replace('.json', f'_augment_{args.strategy}.json')
    data, write_mode = filter_augmented_data(data, args.data.result_path)

    # Data Augmentation
    messages = []
    for idx in range(0, len(data), args.inference.batch_size):
        batch = data[idx: idx + args.inference.batch_size]
        styles = get_styles(args.strategy, args.factor)
        for style in styles:
            inputs = build_inputs(batch, style)
            messages += build_prompts(style, inputs, args.message)
    interact_with_llm(messages,
                      args.model.name,
                      args.model.mode,
                      BasicPrompt().parse_response,
                      args.data.result_path,
                      **args.inference.__dict__)


def preprocess_data(args):
    """
    Load and preprocess the input data depending on the type.

    Args:
        args: An argparse.Namespace object.

    Returns:
        A list of dictionaries representing the input examples.
    """
    args = args.data
    data = read_file(args.data_path)

    # range
    args.end_idx = len(data) if args.end_idx == -1 else args.end_idx
    data = data[max(args.start_idx, 0):args.end_idx]

    # reformat
    new_data = []
    for d in data:
        for i, k in enumerate(d['knowledges']):
            new_data.append({'idx': f"{d['idx']}-{i}", 'text': k['text']})
    data = new_data

    return data


def filter_augmented_data(data, result_path):
    """
    Filters out already augmented data based on the result file.

    Args:
        data: A list of input examples.
        result_path: Path to the augmentation result file.

    Returns:
        A tuple (filtered_data, write_mode).
    """
    if os.path.exists(result_path):
        existing = read_file(result_path)
        existing_idxs = set(d['info']['idx'] for d in existing)
        filtered = [d for d in data if d['idx'] not in existing_idxs]
        print(f"Found {len(existing_idxs)} augmented. {len(filtered)} pending.")
        return filtered, 'a'
    else:
        print(f"{len(data)} pending.")
        return data, 'w'


def get_styles(method, factor):
    """
    Determines augmentation styles to use based on the method.

    Args:
        method: Augmentation method ('basic' or 'random').
        factor: Number of augmentations per sample.

    Returns:
        A list of selected styles.
    """
    if method == 'basic':
        return ['basic'] * factor
    elif method == 'random':
        styles = list(style_hub.keys())
        return random.choices(styles, k=factor)


def build_inputs(batch, style):
    """
    Constructs inputs from a batch of data for a specific augmentation style.

    Args:
        batch: A list of input samples.
        style: Augmentation style.

    Returns:
        A list of dictionaries suitable for prompt construction.
    """
    return [{'nl': d['text'], 'info': {'idx': d['idx'], 'style': style}} for d in batch]


def build_prompts(style, inputs, addition_message):
    """
    Builds prompts based on the style and inputs.

    Args:
        style: Augmentation style.
        inputs: A list of formatted input examples.
        addition_message: Additional context or instruction to add to the prompt.

    Returns:
        A list of prompt messages.
    """
    if style == 'basic':
        return BasicPrompt().build_prompt(inputs, addition_message=addition_message)
    else:
        return prompt_hub[style_hub[style]](style=style).build_prompt(inputs, addition_message=addition_message)
