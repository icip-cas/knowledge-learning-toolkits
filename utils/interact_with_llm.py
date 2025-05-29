import pandas as pd

from utils.interact_with_llm_base import generate_text_batch
from utils.interact_with_llm_chat import multi_thread_call_and_write
from utils.utils import *


def interact_with_llm(prompts,
                      model_name,
                      model_mode,
                      output_parser=lambda x: {'text': x},
                      middle_path=None,
                      **paras):
    # Filter out already processed samples
    if not middle_path:
        middle_path = f'middle_result_{model_name}_{model_mode}.jsonl'
    middle_path = middle_path.replace('.jsonl', '.parquet') if model_mode == 'local' else middle_path

    if model_mode == 'remote':
        # check
        if os.path.exists(middle_path):
            middle_data = read_file(middle_path)
            processed_infos = {str(d['info']) for d in middle_data}
            prompts = [p for p in prompts if str(p['info']) not in processed_infos]
            print(f"Processed data exists. {len(processed_infos)} processed, {len(prompts)} pending.")

        multi_thread_call_and_write(prompts,
                                    model_name,
                                    paras,
                                    middle_path,
                                    output_parser=output_parser)
        results = read_file(middle_path)
    elif model_mode == 'local':
        df = pd.DataFrame(prompts)
        generate_text_batch(df,
                            model_name=model_name,
                            save_path=middle_path,
                            log_path=None,
                            **paras)
        # parse
        df = read_file(middle_path)
        results = []
        for ridx, row in df.iterrows():
            result = {'info': row['info'], **output_parser(row['pred'])}
            results.append(result)
        middle_path = middle_path.replace('.parquet', '.jsonl')
        write_file(middle_path, results)
    else:
        raise AssertionError(f'Model mode {model_mode} is not supported.')
    return results
