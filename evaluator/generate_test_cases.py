import os.path
from string import Template
import random
from utils.utils import *
from utils.interact_with_llm import interact_with_llm


class KnowledgeGraph:
    """封装知识图谱操作"""

    def __init__(self, data_path: str):
        self.data = self._load_data(data_path)
        self.knowledges = self._build_knowledge_df()
        self.cmp_knowledges = self._prepare_comparison_knowledge()
        self.cmb_knowledges = self._prepare_composition_knowledge()

    def _load_data(self, data_path: str):
        """加载原始数据文件"""
        # 实现read_file函数读取数据
        return read_file(data_path)

    def _build_knowledge_df(self) -> pd.DataFrame:
        """构建知识DataFrame"""
        knowledges = []
        for d in self.data:
            text_idx = d['idx']
            source = d['source'] if 'source' in d.keys() else 'basic'
            for k_idx, k in enumerate(d['knowledges']):
                subject, relation, object = k['triplet']
                knowledge = [text_idx, k_idx, subject, relation, object, source]
                knowledges.append(knowledge)

        df = pd.DataFrame(knowledges, columns=["text_idx", "kn_idx", "subject", "relation", "object", "source"])
        return df

    def _prepare_comparison_knowledge(self):
        """准备用于比较的知识分组"""
        cmp_knowledges = self.knowledges[self.knowledges['object'].apply(
            lambda x: (type(x) is float and not np.isnan(x)) or
                      type(x) is int or
                      str(x).isdigit())]

        return cmp_knowledges

    def _prepare_composition_knowledge(self):
        """准备用于组合的知识"""
        cmb_knowledges = self.knowledges.copy(deep=True)
        cmb_knowledges['subject'] = cmb_knowledges['subject'].astype(str)
        cmb_knowledges.sort_values(by=['subject'], inplace=True, ignore_index=True)

        subjects = cmb_knowledges['subject'].tolist()
        cmb_knowledges['successor'] = cmb_knowledges['object'].apply(lambda obj: find_all_indices_sorted(subjects, str(obj)))

        return cmb_knowledges


class TestCaseGenerator:
    """测试用例生成器基类"""

    def __init__(self, data_path: str, result_dir: str, middle_dir: str):
        self.data_path = data_path
        self.result_dir = result_dir
        self.middle_dir = middle_dir
        self.kg = KnowledgeGraph(data_path)
        self.test_cases = pd.DataFrame()

        # 确保目录存在
        os.makedirs(result_dir, exist_ok=True)
        os.makedirs(middle_dir, exist_ok=True)

    def generate(self, model=None, mode=None, **kwargs):
        """生成测试用例的模板方法"""
        raise NotImplementedError("子类必须实现此方法")

    def save_test_cases(self, filename):
        """保存测试用例到CSV"""
        path = os.path.join(self.result_dir, filename)
        write_file(path, self.test_cases)
        return path


class RecallTestCaseGenerator(TestCaseGenerator):
    """生成知识回忆测试用例"""

    def generate(self, few_shot=3, sample_num=10, model=None, mode=None, **kwargs):
        print(f'{GREEN}[Start] Generating {sample_num} knowledge-recall test cases...{RESET}')

        # 构造初始测试用例
        test_cases = []
        for d in self.kg.data:
            text_idx = d['idx']
            for k in d['knowledges']:
                subj, rel, obj = k['triplet']
                text = k['text']
                if obj in text:
                    expressions = text.strip().split(obj)
                    if len(expressions) == 2:
                        question = expressions[0].strip()
                        test_cases.append([text_idx, [[subj, rel, obj]], question, obj])

        # 创建DataFrame并采样
        self.test_cases = pd.DataFrame(
            test_cases,
            columns=['idx', 'knowledges', 'question', 'answer']
        )

        sample_num = len(self.test_cases) if sample_num == -1 else sample_num
        self.test_cases = self.test_cases.sample(min(sample_num, len(self.test_cases)))
        self.test_cases.reset_index(drop=True, inplace=True)

        # 添加上下文示例
        self._add_few_shot_context(few_shot)

        # 保存结果
        path = self.save_test_cases('kn_recall_test_cases.csv')
        print(f'{GREEN}[Done] Generated {len(self.test_cases)} knowledge-recall test cases{RESET}')
        return path

    def _add_few_shot_context(self, few_shot):
        """为每个测试用例添加上下文示例"""
        for ridx, row in self.test_cases.iterrows():
            relation = row['knowledges'][0][1]
            question = row['question']

            # 查找相同关系的其他问题
            same_relation = self.test_cases[
                self.test_cases.apply(
                    lambda r: r['knowledges'][0][1] == relation and r['question'] != question,
                    axis=1
                )
            ]

            # 创建上下文
            context = [f"{d['question']} {d['answer']}" for _, d in same_relation.iterrows()]
            context = random.sample(context, min(few_shot, len(context)))
            self.test_cases.loc[ridx, 'context'] = '\n'.join(context)


class ExtractionTestCaseGenerator(TestCaseGenerator):
    """生成知识提取测试用例"""

    def generate(self, few_shot=3, sample_num=-1, model=None, mode=None, **kwargs):
        print(f'{GREEN}[Start] Generating {sample_num} knowledge-extraction test cases...{RESET}')

        # 构造初始测试用例
        test_cases = []
        for d in self.kg.data:
            text_idx = d['idx']
            for k in d['knowledges']:
                subj, rel, obj = k['triplet']
                test_cases.append([text_idx, [[subj, rel, obj]]])

        self.test_cases = pd.DataFrame(test_cases, columns=['idx', 'knowledges'])
        self.test_cases['question'] = None
        self.test_cases['answer'] = None
        self.test_cases['context'] = None

        # 采样
        sample_num = len(self.test_cases) if sample_num == -1 else sample_num
        self.test_cases = self.test_cases.sample(min(sample_num, len(self.test_cases)))
        self.test_cases.reset_index(drop=True, inplace=True)

        # 生成问题
        self._generate_questions(model, mode, **kwargs)

        # 设置答案
        self.test_cases['answer'] = self.test_cases['knowledges'].apply(lambda x: x[0][2])

        # 添加上下文
        self._add_few_shot_context(few_shot)

        # 保存结果
        path = self.save_test_cases('kn_extraction_test_cases.csv')
        print(f'{GREEN}[Done] Generated {len(self.test_cases)} knowledge-extraction test cases{RESET}')
        return path

    def _generate_questions(self, model, mode, **kwargs):
        """使用LLM生成自然语言问题"""
        prompt_template = Template(
            "Given a factual triplet (subject, relation, object), generate a question that asks for the object "
            "based on the subject and relation.\n"
            "Triple: (Patrick,father,Matthew), Question: Who is Patrick's father?\n"
            "Triple: ($s, $r, $o), Question:"
        )

        prompts = [
            {
                'prompt': prompt_template.substitute({
                    's': row['knowledges'][0][0],
                    'r': row['knowledges'][0][1],
                    'o': row['knowledges'][0][2]
                }),
                'info': {'idx': idx}
            }
            for idx, row in self.test_cases.iterrows()
        ]

        middle_path = os.path.join(self.middle_dir, 'kn_extraction_questions.jsonl')
        interact_with_llm(
            prompts,
            model,
            mode,
            lambda x: {'text': x},
            middle_path,
            **kwargs
        )

        # 处理结果
        results = read_file(middle_path)
        for r in results:
            idx = r['info']['idx']
            question = self._clean_llm_output(r['text'])
            self.test_cases.loc[idx, 'question'] = question

    def _clean_llm_output(self, text):
        """清理LLM输出"""
        lines = text.split('\n')
        lines = [line.strip() for line in lines if line.strip() != '']
        text = '\n'.join(lines)

        # 移除冗余内容
        text = text.split('(Note')[0].split('Note')[0]
        text = text.split('\n')[0].strip()
        text = text.split('However')[0].strip()

        return text

    def _add_few_shot_context(self, few_shot):
        """为每个测试用例添加上下文示例"""
        for ridx, row in self.test_cases.iterrows():
            relation = row['knowledges'][0][1]
            question = row['question']

            # 查找相同关系的其他问题
            same_relation = self.test_cases[
                self.test_cases.apply(
                    lambda r: r['knowledges'][0][1] == relation and r['question'] != question,
                    axis=1
                )
            ]

            # 创建上下文
            context = [f"{d['question']} {d['answer']}" for _, d in same_relation.iterrows()]
            context = random.sample(context, min(few_shot, len(context)))
            self.test_cases.loc[ridx, 'context'] = '\n'.join(context)


class ReasoningTestCaseGenerator(TestCaseGenerator):
    """生成知识推理测试用例"""

    def __init__(self, data_path: str, result_dir: str, middle_dir: str):
        super().__init__(data_path, result_dir, middle_dir)
        self.reason_types = ["compare", "composition"]

        # 定义处理函数映射
        self._first_step_handlers = {
            "compare": self._handle_first_compare,
            "composition": self._handle_first_composition
        }
        self._next_step_handlers = {
            "compare": self._handle_next_compare,
            "composition": self._handle_next_composition
        }

    def generate(self, reason_steps=3, few_shot=3, sample_num=-1, multi_source=False,
                 model=None, mode=None, **kwargs):
        print(f'{GREEN}[Start] Generating reasoning test cases (steps 1-{reason_steps})...{RESET}')

        # 生成推理规则和知识链
        self._generate_reason_chains(reason_steps, sample_num, multi_source)

        # 生成自然语言问题
        self._generate_questions(model, mode, **kwargs)

        # 生成选择题选项
        self._generate_options()

        # 生成推理链（CoT）
        self._generate_cot(model, mode, **kwargs)

        # 添加上下文
        self._add_few_shot_context(few_shot)

        # 保存结果
        path = self.save_test_cases('kn_reasoning_test_cases.csv')
        print(f'{GREEN}[Done] Generated {len(self.test_cases)} reasoning test cases{RESET}')
        return path

    def _generate_reason_chains(self, max_steps, sample_num, multi_source):
        """生成推理规则和知识链"""
        test_cases = []
        for step in range(1, max_steps + 1):
            step_cases = 0
            while step_cases < sample_num:
                case = self._generate_single_chain(step, multi_source)
                if case:
                    test_cases.append(case)
                    step_cases += 1

        self.test_cases = pd.DataFrame(
            test_cases,
            columns=["idx", "reason_step", "reason_types", "knowledges", "rule", "answer"]
        )

    def _generate_single_chain(self, steps, multi_source):
        """生成单条推理规则链"""
        question, answer = None, None
        knowledges = []
        sources = []
        reason_types = [random.choice(self.reason_types) for _ in range(steps)]

        for i, reason_type in enumerate(reason_types):
            # 第一步推理
            if i == 0:
                handler = self._first_step_handlers[reason_type]
                result = handler(sources)
                if not result:
                    return None
                question, answer, new_knowledges = result
                knowledges.extend(new_knowledges)

            # 后续推理步骤
            else:
                handler = self._next_step_handlers[reason_type]
                result = handler(question, answer, sources)
                if not result:
                    return None
                question, answer, new_knowledges = result
                knowledges.extend(new_knowledges)

            # 检查多源要求
            if i == steps - 1 and multi_source and len(set(sources)) == 1:
                return None

        return [len(knowledges), steps, reason_types, knowledges, question, answer]

    def _handle_first_compare(self, sources):
        """处理第一步比较推理"""
        # 随机选择一组可比较的知识
        relations = self.kg.cmp_knowledges.groupby('relation')
        valid_relations = [name for name, group in relations if len(group) >= 2]

        if not valid_relations:
            return None

        relation = random.choice(valid_relations)
        group = relations.get_group(relation)
        kns = group.sample(2).reset_index(drop=True)

        # 确保值不相等
        if kns.loc[0, 'object'] == kns.loc[1, 'object']:
            return None

        sources.extend([kns.loc[0, 'source'], kns.loc[1, 'source']])

        # 构造比较表达式
        op = random.choice([">", "<"])
        question = [relation, kns.loc[0, 'subject'], kns.loc[1, 'subject'], op]

        # 确定正确答案
        try:
            val1 = float(kns.loc[0, 'object'])
            val2 = float(kns.loc[1, 'object'])
        except ValueError:
            val1 = kns.loc[0, 'object']
            val2 = kns.loc[1, 'object']

        if (val1 > val2 and op == '>') or (val1 < val2 and op == '<'):
            answer = kns.loc[0, 'subject']
        else:
            answer = kns.loc[1, 'subject']

        return question, answer, [
            kns.loc[0, ['subject', 'relation', 'object']].tolist(),
            kns.loc[1, ['subject', 'relation', 'object']].tolist()
        ]

    def _handle_first_composition(self, sources):
        """处理第一步组合推理"""
        # 随机选择起始知识
        valid_kn = self.kg.cmb_knowledges[self.kg.cmb_knowledges['successor'].apply(len) > 0]
        if valid_kn.empty:
            return None

        kn1 = valid_kn.sample(1).reset_index(drop=True).loc[0]
        sources.append(kn1['source'])

        # 查找后续知识
        successors = self.kg.cmb_knowledges.iloc[kn1['successor']]
        valid_successors = successors[successors['successor'].apply(len) > 0]

        if valid_successors.empty:
            if successors.empty:
                return None
            kn2 = successors.sample(1).reset_index(drop=True).loc[0]
        else:
            kn2 = valid_successors.sample(1).reset_index(drop=True).loc[0]

        # 构造组合表达式
        question = [[kn1['subject'], kn1['relation']], kn2['relation']]
        answer = kn2['object']

        return question, answer, [
            kn1[['subject', 'relation', 'object']].tolist(),
            kn2[['subject', 'relation', 'object']].tolist()
        ]

    def _handle_next_compare(self, current_question, current_answer, sources):
        """处理后续步骤的比较推理"""
        # 查找当前答案对应的知识
        k1_candidates = self.kg.cmp_knowledges[self.kg.cmp_knowledges['subject'] == current_answer]
        if k1_candidates.empty:
            return None

        k1 = k1_candidates.sample(1).reset_index(drop=True).loc[0]
        relation = k1['relation']

        # 查找可比较的其他知识
        k2_candidates = self.kg.cmp_knowledges[
            (self.kg.cmp_knowledges['subject'] != current_answer) &
            (self.kg.cmp_knowledges['relation'] == relation)
            ]

        if k2_candidates.empty:
            return None

        k2 = k2_candidates.sample(1).reset_index(drop=True).loc[0]

        # 确保值不相等
        if k1['object'] == k2['object']:
            return None

        sources.extend([k1['source'], k2['source']])

        # 构造比较表达式
        op = random.choice(['<', '>'])
        if random.random() < 0.5:
            question = [relation, current_question, k2['subject'], op]
        else:
            question = [relation, k2['subject'], current_question, op]

        # 确定正确答案
        try:
            val1 = float(k1['object'])
            val2 = float(k2['object'])
        except ValueError:
            val1 = k1['object']
            val2 = k2['object']

        if (val1 > val2 and op == '>') or (val1 < val2 and op == '<'):
            answer = current_answer
        else:
            answer = k2['subject']

        return question, answer, [
            k1[['subject', 'relation', 'object']].tolist(),
            k2[['subject', 'relation', 'object']].tolist()
        ]

    def _handle_next_composition(self, current_question, current_answer, sources):
        """处理后续步骤的组合推理"""
        # 查找当前答案对应的知识
        kn1_candidates = self.kg.cmb_knowledges[self.kg.cmb_knowledges['subject'] == current_answer]
        if kn1_candidates.empty:
            return None

        kn1 = kn1_candidates.sample(1).reset_index(drop=True).loc[0]
        sources.append(kn1['source'])

        # 构造组合表达式
        question = [current_question, kn1['relation']]
        answer = kn1['object']

        return question, answer, [kn1[['subject', 'relation', 'object']].tolist()]

    def _generate_questions(self, model, mode, **kwargs):
        """根据规则生成自然语言问题"""
        prompt_template = Template(
            "Define two basic operation rules:\n"
            "Comparison: (e1, a, v1) ∧ (e2, a, v2) ∧ v1 < v2 =⇒ e1=(a, e1, e2, <), e2=(a, e1, e2, >)\n"
            "Combination: (h, r1, b) ∧ (b, r2, t) =⇒ t=(h, r1, r2)\n\n"
            "Please complete the following task in the format given in the example.\n"
            "Task: Given an expression formed by basic operation rules, explain it step by step from the innermost rule to the outermost rule to create a complex reasoning question. The following requirements must be met: 1. The question must be purely a question and cannot include the reasoning results of the rules, nor can it change, add, or omit any of facts beyond the rule. 2. Only output a question in only one version without any additional explanations. 3. The question should be fluent, easy to read, and concise.\n\n"
            "Expression: ['Chase', 'spouse', 'father']\n"
            "Question: Who's Chase's spouse's father?\n\n"
            "Expression: [age, Jane, Bob, >]\n"
            "Question: What is the name of the older person between Jane and Bob?\n\n"
            "Expression: [[age, Jane, Bob, >], father, spouse]\n"
            "Question: Jane and Bob are comparing who is older. Who is the spouse of the older person's father?\n\n"
            "Expression: [[['bmi', 'Joseph', 'Wesley', '<'], 'spouse', 'father'], 'mother', 'spouse']\n"
            "Question: Joseph and Wesley are comparing BMI values. Who is the spouse of the father of the mother of the spouse of the person with the lower BMI?\n\n"
            "Expression: $expression\n"
            "Answer: $answer\n"
            "Question:"
        )

        prompts = [
            {
                'prompt': prompt_template.substitute({
                    'expression': row['rule'],
                    'answer': row['answer']
                }),
                'info': {'idx': idx}
            }
            for idx, row in self.test_cases.iterrows()
        ]

        middle_path = os.path.join(self.middle_dir, 'reasoning_questions.jsonl')
        interact_with_llm(
            prompts,
            model,
            mode,
            lambda x: {'text': x},
            middle_path,
            **kwargs
        )

        # 处理结果
        results = read_file(middle_path)
        for r in results:
            idx = r['info']['idx']
            question = self._clean_llm_output(r['text'])
            self.test_cases.loc[idx, 'question'] = question

    def _clean_llm_output(self, text):
        """清理LLM输出"""
        lines = text.split('\n')
        lines = [line.strip() for line in lines if line.strip() != '']
        text = '\n'.join(lines)

        # 移除冗余内容
        text = text.split('(Note')[0].split('Note')[0]
        text = text.split('\n')[0].strip()
        text = text.split('However')[0].strip()

        return text

    def _generate_options(self):
        """为问题生成选项"""
        for idx, row in self.test_cases.iterrows():
            answer = row['answer']
            knowledges = row['knowledges']

            # 收集候选实体
            candidates = set()
            kns = self.kg.knowledges

            # 通过关系查找候选
            rels = kns[kns['object'] == answer]['relation'].tolist()
            candidates.update(
                kns[(kns['relation'].isin(rels)) & (kns['object'] != answer)]['object'].tolist()
            )

            rels = kns[kns['subject'] == answer]['relation'].tolist()
            candidates.update(
                kns[(kns['relation'].isin(rels)) & (kns['subject'] != answer)]['subject'].tolist()
            )

            # 从当前知识中添加实体
            for kn in knowledges:
                if kn[0] != answer:
                    candidates.add(kn[0])
                if kn[2] != answer:
                    candidates.add(kn[2])

            # 确保有足够选项
            all_entities = kns['subject'].tolist() + kns['object'].tolist()
            while len(candidates) < 3:
                candidates.add(random.choice(all_entities))

            # 构建选项
            options = random.sample(list(candidates), 3) + [answer]
            random.shuffle(options)
            answer_idx = options.index(answer)

            # 格式化选项
            option_letters = ['A', 'B', 'C', 'D']
            formatted_options = '\n'.join([f"{letter}. {option}" for letter, option in zip(option_letters, options)])

            # 更新问题和答案
            self.test_cases.loc[idx, 'question'] = f"{row['question']}\n{formatted_options}"
            self.test_cases.loc[idx, 'answer'] = option_letters[answer_idx]

    def _generate_cot(self, model, mode, **kwargs):
        """生成推理链（CoT）"""
        prompt_template = Template(
            "Define two basic operation rules:\n"
            "Comparison: (e1, a, v1) ∧ (e2, a, v2) ∧ v1 < v2 =⇒ e1=(a, e1, e2, <), e2=(a, e1, e2, >)\n"
            "Combination: (h, r1) =⇒ t=(h, r1)\n\n"
            "Based on the given rule, knowledge and answer to the question, generate the thinking process to answer the question. The thinking process only includes factual deduction and does not include any additional information such as feasibility.\n\n"
            "Question: Which TV series has fewer episodes, Helen Chronicles or Dawn Blossoms?\n"
            "Rule: ['number of episodes', 'Helen Chronicles', 'Dawn Blossoms', '<']\n"
            "Knowledge: [['Helen Chronicles', 'number of episodes', '12'], ['Dawn Blossoms', 'number of episodes', '20']]\n"
            "Answer: Helen Chronicles\n"
            "Think: The number of episodes of Helen Chronicles is 12. The number of episodes of Dawn Blossoms is 20. Helen Chronicles has fewer episodes than Dawn Blossoms. The answer is Helen Chronicles.\n\n"
            "Question: Compare the number of participants in Julianne David Montalvo's spouse's country of citizenship with the number of population in Montebello. Which one has a larger number of population?\n"
            "Rule: ['number of participants', [['Julianne David Montalvo', 'spouse'], 'country of citizenship'], 'Montebello', '>']\n"
            "Knowledge: [['Julianne David Montalvo', 'spouse', 'Mayra Joanne Stevens'], ['Mayra Joanne Stevens', 'country of citizenship', 'Garyland'], ['Garyland', 'population', '12543201'], ['Montebello', 'population', '4567']]\n"
            "Answer: Montebello\n"
            "Think: Julianne David Montalvo's spouse is Mayra Joanne Stevens. Mayra Joanne Stevens's country of citizenship is Garyland. Garyland's population is 12543201. Montebello's population is 4567. Montebello has a larger number of population than Garyland. The answer is Montebello.\n\n"
            "Question: $question\n"
            "Rule: $rule\n"
            "Knowledge: $knowledge\n"
            "Answer: $answer\n"
            "Think:"
        )

        prompts = [
            {
                'prompt': prompt_template.substitute({
                    'question': row['question'],
                    'rule': row['rule'],
                    'knowledge': row['knowledges'],
                    'answer': row['answer']
                }),
                'info': {'idx': idx}
            }
            for idx, row in self.test_cases.iterrows()
        ]

        middle_path = os.path.join(self.middle_dir, 'reasoning_cot.jsonl')
        interact_with_llm(
            prompts,
            model,
            mode,
            lambda x: {'text': x},
            middle_path,
            **kwargs
        )

        # 处理结果
        results = read_file(middle_path)
        for r in results:
            idx = r['info']['idx']
            cot = self._clean_llm_output(r['text'])
            self.test_cases.loc[idx, 'cot'] = cot

    def _add_few_shot_context(self, few_shot):
        """添加上下文示例"""
        for ridx, row in self.test_cases.iterrows():
            # 清理问题
            question = row['question'].split('(Note')[0].split('Note')[0].strip()
            self.test_cases.loc[ridx, 'question'] = question

            # 查找相似类型的问题
            similar_cases = self.test_cases[
                (self.test_cases.index != ridx) &
                (self.test_cases['reason_types'] == row['reason_types'])
                ]

            # 添加上下文
            if not similar_cases.empty:
                context = [
                    f"{d['question']} {d['answer']}"
                    for _, d in similar_cases.iterrows()
                ]
                context = random.sample(context, min(few_shot, len(context)))
                self.test_cases.loc[ridx, 'context'] = '\n'.join(context)

                # 添加CoT上下文
                cot_context = [
                    f"{d['question']}\nThought: {d['cot']}\nAnswer: {d['answer']}"
                    for _, d in similar_cases.iterrows()
                ]
                cot_context = random.sample(cot_context, min(few_shot, len(cot_context)))
                self.test_cases.loc[ridx, 'cot_context'] = '\n\n'.join(cot_context)


def generate_test_cases(args):
    """主入口函数：生成所有测试用例"""
    config = args.test_sample_generation
    model = config.model.name
    mode = config.model.mode
    data_path = config.data.data_path
    result_dir = config.data.result_dir
    middle_dir = config.data.middle_result_dir

    # 创建目录
    os.makedirs(result_dir, exist_ok=True)
    os.makedirs(middle_dir, exist_ok=True)

    # 生成回忆测试用例
    recall_gen = RecallTestCaseGenerator(data_path, result_dir, middle_dir)
    recall_gen.generate(
        few_shot=config.recall.few_shot,
        sample_num=config.recall.sample_num,
        model=model,
        mode=mode,
        **config.inference.__dict__
    )

    # 生成提取测试用例
    extraction_gen = ExtractionTestCaseGenerator(data_path, result_dir, middle_dir)
    extraction_gen.generate(
        few_shot=config.extraction.few_shot,
        sample_num=config.extraction.sample_num,
        model=model,
        mode=mode,
        **config.inference.__dict__
    )

    # 生成推理测试用例
    reason_gen = ReasoningTestCaseGenerator(data_path, result_dir, middle_dir)
    reason_gen.generate(
        reason_steps=config.reason.reason_step,
        few_shot=config.reason.few_shot,
        sample_num=config.reason.sample_num,
        multi_source=config.reason.multi_source,
        model=model,
        mode=mode,
        **config.inference.__dict__
    )