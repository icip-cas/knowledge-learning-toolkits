import json

"""
    Prompt construction and response parsing functions for data augmentation.
    Each augmentation strategy should implement both build_prompt and model_call_and_parse_response.
"""


class BasicPrompt:
    def __init__(self, prompt=None):
        self.prompt = (
            "Rewrite the above text without changing, adding, or removing any factual information.\n"
            "Provide only one version of the rewritten text without any additional information:"
        )
        if prompt:
            self.prompt = prompt

    def build_prompt(self, orig_data, icl_data=None, addition_message="", prompt_save_path=None):
        """
        Build prompts for data generation based on original input, few-shot examples, and additional instructions.
        Save the generated prompts if a save path is provided.

        Parameters:
            orig_data (list): List of dictionaries in the format {'nl': '', 'info': {}}.
            icl_data (list): In-context learning examples (not used in this implementation).
            addition_message (str): Extra message appended to the prompt.
            prompt_save_path (str): Path to save the generated prompts.

        Returns:
            list: List of prompts with corresponding meta-information.
        """
        messages = []
        for line in orig_data:
            data_nl = line['nl']
            info = line['info']
            message = f'{self.prompt} {addition_message}\nOriginal text: {data_nl}\nRewritten text:'
            messages.append({'prompt': message, 'info': info})

        if prompt_save_path:
            with open(prompt_save_path, 'w', encoding='utf-8') as f:
                for prompt in messages:
                    f.write(json.dumps(prompt, ensure_ascii=False) + '\n')

        return messages

    def parse_response(self, response: str):
        """
        Parse the model's response.

        Parameters:
            response (str): The response returned by the language model.

        Returns:
            dict: Dictionary containing the parsed rewritten text.
        """
        return {'text': response}


class TextGenrePrompt(BasicPrompt):
    """
    Genre-based rewriting (e.g., textbook, news, paper, etc.)
    """
    def __init__(self, prompt=None, style='textbook', lang='en'):
        super(TextGenrePrompt, self).__init__(prompt)
        self.style = style

        self.style2prompt = {
            'textbook': 'Rewrite the text in a clear, structured, textbook-style format, without changing, adding, or removing any factual information. Provide only one version of the rewritten text without any additional information.',
            'news': 'Rewrite the text in a news style, without changing, adding, or removing any factual information. Provide only one version of the rewritten text without any additional information.',
            'paper': 'Rewrite the text in a paper style, without changing, adding, or removing any factual information. Provide only one version of the rewritten text without any additional information.',
            'lyrics': 'Rewrite the text in the form of lyrics, without changing, adding, or removing any factual information. Provide only one version of the rewritten text without any additional information.',
            'dialogue': 'Rewrite the text as a dialogue between two or more characters, without changing, adding, or removing any factual information. Provide only one version of the rewritten text without any additional information.',
            'speech': 'Rewrite the text in the form of a speech, without changing, adding, or removing any factual information. Provide only one version of the rewritten text without any additional information.',
            'story': 'Rewrite the text in a narrative story format, without changing, adding, or removing any factual information. Provide only one version of the rewritten text without any additional information.',
            'summary': 'Rewrite the text as a concise summary, without changing, adding, or removing any factual information. Provide only one version of the rewritten text without any additional information.'
        }

        self.style2prompt_zh = {
            'textbook': '将上述文本以清晰、结构化的教科书风格格式重写，不改变、添加或删除任何事实信息。\n仅提供一个版本的重写文本，不包含任何附加信息：',
            'news': '将上述文本以新闻风格重写，不改变、添加或删除任何事实信息。\n仅提供一个版本的重写文本，不包含任何附加信息：',
            'paper': '将上述文本以论文风格重写，不改变、添加或删除任何事实信息。\n仅提供一个版本的重写文本，不包含任何附加信息：',
            'lyrics': '将上述文本以歌词形式重写，不改变、添加或删除任何事实信息。\n仅提供一个版本的重写文本，不包含任何附加信息：',
            'dialogue': '将上述文本以两个或多个角色之间的对话形式重写，不改变、添加或删除任何事实信息。\n仅提供一个版本的重写文本，不包含任何附加信息：',
            'speech': '将上述文本以演讲形式重写，不改变、添加或删除任何事实信息。\n仅提供一个版本的重写文本，不包含任何附加信息：',
            'story': '将上述文本以叙述故事的形式重写，不改变、添加或删除任何事实信息。\n仅提供一个版本的重写文本，不包含任何附加信息：',
            'summary': '将上述文本以简明摘要的形式重写，不改变、添加或删除任何事实信息。\n仅提供一个版本的重写文本，不包含任何附加信息：'
        }

        if not prompt:
            self.prompt = self.style2prompt[self.style] if lang == 'en' else self.style2prompt_zh[self.style]


class TextTypePrompt(BasicPrompt):
    """
    Sentence-type rewriting (e.g., Q&A, exclamation).
    """
    def __init__(self, prompt=None, style='qa', lang='en'):
        super(TextTypePrompt, self).__init__(prompt)
        self.style = style

        self.style2prompt = {
            'qa': 'Rewrite the text into Q&A format without altering, adding, or omitting any of the factual information originally conveyed. Provide only one version of the rewritten text without any additional information.',
            'exclamation': 'Rewrite the facts into exclamatory format without changing, adding, or deleting any of the original information conveyed. Provide only one version of the rewritten text without any additional information.',
        }

        self.style2prompt_zh = {
            'qa': '将上述文本重写为问答格式，不改变、添加或省略任何原始传达的事实信息。\n仅提供一个版本的重写文本，不包含任何附加信息：',
            'exclamation': '将上述事实重写为感叹句格式，不改变、添加或删除任何原始传达的信息。\n仅提供一个版本的重写文本，不包含任何附加信息：',
        }

        if not prompt:
            self.prompt = self.style2prompt[self.style] if lang == 'en' else self.style2prompt_zh[self.style]


class TextFormalityPrompt(BasicPrompt):
    """
    Formality-level rewriting (formal vs. informal).
    """
    def __init__(self, prompt=None, style='informal', lang='en'):
        super(TextFormalityPrompt, self).__init__(prompt)
        self.style = style

        self.style2prompt = {
            'informal': 'Rewrite the text in an informal, conversational style. Do not change, add, or delete any factual expressions of the original information. Provide only one version of the rewritten text without any additional information.',
            'formal': 'Rewrite the text in a formal manner without altering, adding, or deleting any of the original information conveyed. Provide only one version of the rewritten text without any additional information.',
        }

        self.style2prompt_zh = {
            'informal': '将上述文本以非正式、对话的风格重写。不要改变、添加或删除任何原始信息的事实表达。\n仅提供一个版本的重写文本，不包含任何附加信息：',
            'formal': '将上述文本以正式的方式重写，不改变、添加或删除任何原始传达的信息。\n仅提供一个版本的重写文本，不包含任何附加信息：',
        }

        if not prompt:
            self.prompt = self.style2prompt[self.style] if lang == 'en' else self.style2prompt_zh[self.style]


class TextSentimentPrompt(BasicPrompt):
    """
    Sentiment-based rewriting (positive vs. negative).
    """
    def __init__(self, prompt=None, style='positive', lang='en'):
        super(TextSentimentPrompt, self).__init__(prompt)
        self.style = style
        self.style2prompt = {
            'positive': 'Rewrite the text into one that conveys positive emotions. Do not alter, add, or remove any factual expressions from the original message. Provide only one version of the rewritten text without any additional information.',
            'negative': 'Rewrite the text into one that conveys negative emotions. Do not alter, add, or remove any factual expressions from the original message. Provide only one version of the rewritten text without any additional information.'
        }

        self.style2prompt_zh = {
            'positive': '将上述文本重写为传达积极情感的内容。不要改变、添加或删除原始信息中的任何事实表达。\n仅提供一个版本的重写文本，不包含任何附加信息：',
            'negative': '将上述文本重写为传达消极情感的内容。不要改变、添加或删除原始信息中的任何事实表达。\n仅提供一个版本的重写文本，不包含任何附加信息：'
        }

        if not prompt:
            self.prompt = self.style2prompt[self.style] if lang == 'en' else self.style2prompt_zh[self.style]


# Prompt strategy registry
prompt_hub = {
    'text_genre': TextGenrePrompt,
    'text_type': TextTypePrompt,
    'text_formality': TextFormalityPrompt,
    'text_sentiment': TextSentimentPrompt,
}

# Prompt style reverse lookup for convenience
style_hub = {}
for dim_name, dim_class in prompt_hub.items():
    dim_styles = list(dim_class().style2prompt.keys())
    for style in dim_styles:
        style_hub[style] = dim_name
