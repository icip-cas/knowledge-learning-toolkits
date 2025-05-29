from data_loader.load_data import load_raw_data, data_preprocess
from data_augmentation.augment import run_augment
from evaluator.generate_test_cases import generate_test_cases
from utils.utils import *


def main():
    args = load_config_with_namespace('config.yaml')

    # Data load
    print(f'{GREEN}[Start] Load data...{RESET}')
    args_data = args.augmentation.data
    handled_data_path = args_data.data_path.replace('.jsonl', '_handled.jsonl')
    data = load_raw_data(args_data.data_path,
                         args_data.data_type,
                         handled_data_path,
                         args_data.start_idx,
                         args_data.end_idx)
    print(f'{GREEN}[Done] Load data.{RESET}')

    # Data preprocess
    print(f'{GREEN}[Start] Preprocess data...{RESET}')
    data = data_preprocess(args, handled_data_path, handled_data_path)
    print(f'{GREEN}[Done] Preprocess data.{RESET}')

    # Data augmentation
    print(f'{GREEN}[Start] Augmenting data...{RESET}')
    args.augmentation.data.data_path = handled_data_path
    run_augment(args)
    print(f'{GREEN}[Done] Augmenting data.{RESET}')

    # Test cases generation
    generate_test_cases(args)


if __name__ == '__main__':
    main()