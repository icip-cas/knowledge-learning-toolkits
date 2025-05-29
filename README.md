# Knowledge-Learning-Toolkits

- An implementation for Memorizing is Not Enough: Deep Knowledge Injection Through Reasoning (ACL 2025).
- Please contact @Ruoxi Xu (ruoxi2021@iscas.ac.cn) for questions and suggestions.

## Key Innovation

Moves beyond simple memorization to enable deep knowledge reasoning through automated data augmentation and multi-level evaluation.

- Diversified Data Augmentation
- Multi-level Test Cases Generation 
  - Recall-level: Basic knowledge recall 
  - Extraction-level: Knowledge extraction from complex contexts 
  - Reasoning-level: Logical inference and knowledge application

## Project Structure

```
knowledge-learning-toolkits
├── data_augumentation  # Knowledge augmentation module
├── ── augment.py  # Data augmentation
├── ── prompt_hub.py  # Defines prompts for data generation and functions for parsing returned results
├── data_loader  # Data preprocessing module
├── ── load_data.py  # Reads and preprocesses data; converts structured and unstructured data into a unified format: [{'idx': 0, 'text': text, 'knowledges': [{'text': text, 'triplet': [subject, relation, object]}]}]
├── evaluator  # Evaluation module
├── ── generate_test_cases.py  # Automatically constructs test cases
├── trainer  # Train module
├── utils  # Common utility functions
├── config.yaml  # Configuration settings
└── main.py  # Main entry script
```

## Quick Start

### Data Preparation

The toolkit currently accepts two types of data formats:
- For data in the form of triples, it should be stored in a .jsonl file, with each entry in the following format:
{'idx': 0, 'triplet': [subject, relation, object]}
- For free-text data, it should also be stored in a .jsonl file, with each entry in the following format:
{'idx': 0, 'text': text}

### Run Pipeline

```bash
python main.py
```

### Citation
If you find this project helpful, please use the following to cite it:
```
@article{xu2025memorizing,
  title={Memorizing is Not Enough: Deep Knowledge Injection Through Reasoning},
  author={Xu, Ruoxi and Ji, Yunjie and Cao, Boxi and Lu, Yaojie and Lin, Hongyu and Han, Xianpei and He, Ben and Sun, Yingfei and Li, Xiangang and Sun, Le},
  journal={arXiv preprint arXiv:2504.00472},
  year={2025}
}
```