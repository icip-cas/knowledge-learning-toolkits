prompt_template_role_inter_zh = """
Your goal is to extract structured information from the user's input that matches the form described below. When extracting information please make sure it matches the type information exactly. Do not add any attributes that do not appear in the schema shown below.

```TypeScript
script: Array<{{ // Adapted from the novel into script.
    role: string // The character involved in the relationship, specifically refer to the names of the individuals.
    relation: string // The relationship between the two roles, e.g. friendship, hostility, mediation, emotional conflict, and trust.
    role: string // The other character involved in the relationship, specifically refer to the names of the individuals.
}}>
```

Please output the extracted information in CSV format in Excel dialect. Please use a | as the delimiter. 
Do NOT add any clarifying information. Output MUST follow the schema above. Do NOT add any additional columns that do not appear in the schema.

Input: 在阴暗的森林中，哈利·波特和赫敏·格兰杰并肩走着，心中充满了紧张。突然，他们听到了远处传来的沉重脚步声。哈利转过头，看见了伏地魔的身影，心里一阵发慌。他紧握着魔杖，暗自下定决心要保护赫敏。
Output: role|relation|role
哈利·波特|亲密|赫敏·格兰杰
哈利·波特|保护|赫敏·格兰杰
伏地魔|对抗|哈利·波特

Input: 在一个阴雨连绵的下午，艾米莉坐在窗边，回忆起她与汤姆的友谊。自从他们初次在大学相遇后，两人便形影不离。然而，最近汤姆与他的室友杰克之间的冲突让艾米莉担心不已。杰克对汤姆的态度愈发敌对，甚至在一次聚会上当着所有人的面指责汤姆，弄得气氛非常尴尬。艾米莉试图调解两人之间的关系，但她发现杰克根本不愿意聆听。对此，汤姆则表现出对杰克的失望，但又不愿彻底断绝与他的友谊。
Output: role|relation|role
艾米莉|友谊|汤姆
杰克|敌对|汤姆
艾米莉|调解|汤姆
汤姆|情感冲突|杰克
汤姆|信任|艾米莉

Input: {text}
Output:
"""


prompt_template_role_inter_en = """
Your goal is to extract structured information from the user's input that matches the form described below. When extracting information please make sure it matches the type information exactly. Do not add any attributes that do not appear in the schema shown below.

```TypeScript
script: Array<{{ // Adapted from the novel into script.
    role: string // The character involved in the relationship, specifically refer to the names of the individuals.
    relation: string // The relationship between the two roles, e.g. friendship, hostility, mediation, emotional conflict, and trust.
    role: string // The other character involved in the relationship, specifically refer to the names of the individuals.
}}>
```

Please output the extracted information in CSV format in Excel dialect. Please use a | as the delimiter. 
Do NOT add any clarifying information. Output MUST follow the schema above. Do NOT add any additional columns that do not appear in the schema.

Input: In the dark forest, Harry Potter and Hermione Granger walked side by side, their hearts filled with tension. Suddenly, they heard the sound of heavy footsteps in the distance. Harry turned his head and saw the figure of Voldemort, a wave of panic washing over him. He tightened his grip on his wand, secretly resolving to protect Hermione.
Output: role|relation|role
Harry Potter|close relationship|Hermione Granger
Harry Potter|protects|Hermione Granger
Voldemort|opposes|Harry Potter

Input: On a rainy afternoon, Emily sat by the window, reminiscing about her friendship with Tom. Ever since they first met in college, the two had been inseparable. However, recently, the conflict between Tom and his roommate Jack had her worried. Jack's attitude towards Tom had become increasingly hostile, even accusing Tom in front of everyone at a party, creating an incredibly awkward atmosphere. Emily tried to mediate between the two, but she found that Jack was unwilling to listen at all. In response, Tom showed disappointment towards Jack, yet he was reluctant to completely sever their friendship.
Output: role|relation|role
Emily|friendship|Tom
Jack|hostility|Tom
Emily|mediation|Tom
Tom|emotional conflict|Jack
Tom|trust|Emily

Input: {text}
Output:
"""


prompt_template_role_inter_with_roles_en = """
Your goal is to extract structured information from the user's input that matches the form described below. When extracting information please make sure it matches the type information exactly. Do not add any attributes that do not appear in the schema shown below.

```TypeScript
script: Array<{{ // Adapted from the novel into script.
    role_1: string // The character involved in the relationship, specifically refer to the names of the individuals.
    relation: string // The relationship between the two roles, e.g. friendship, hostility, mediation, emotional conflict, and trust.
    role_2: string // The other character involved in the relationship, specifically refer to the names of the individuals.
}}>
```

Please output the extracted information in CSV format in Excel dialect. Please use a | as the delimiter. 
Do NOT add any clarifying information. Output MUST follow the schema above. Do NOT add any additional columns that do not appear in the schema.

Input: In the dark forest, Harry Potter and Hermione Granger walked side by side, their hearts filled with tension. Suddenly, they heard the sound of heavy footsteps in the distance. Harry turned his head and saw the figure of Voldemort, a wave of panic washing over him. He tightened his grip on his wand, secretly resolving to protect Hermione.
Roles: Harry Potter, Hermione Granger, Voldemort
Output: role_1|relation|role_2
Harry Potter|close relationship|Hermione Granger
Harry Potter|protects|Hermione Granger
Voldemort|opposes|Harry Potter

Input: On a rainy afternoon, Emily sat by the window, reminiscing about her friendship with Tom. Ever since they first met in college, the two had been inseparable. However, recently, the conflict between Tom and his roommate Jack had her worried. Jack's attitude towards Tom had become increasingly hostile, even accusing Tom in front of everyone at a party, creating an incredibly awkward atmosphere. Emily tried to mediate between the two, but she found that Jack was unwilling to listen at all. In response, Tom showed disappointment towards Jack, yet he was reluctant to completely sever their friendship.
Roles: Emily, Tom, Jack
Output: role_1|relation|role_2
Emily|friendship|Tom
Jack|hostility|Tom
Emily|mediation|Tom
Tom|emotional conflict|Jack
Tom|trust|Emily

Input: {text}
Roles: {roles}
Output:
"""


prompt_template_role_center_zh = """
Your goal is to extract structured information from the user's input that matches the form described below. When extracting information please make sure it matches the type information exactly. Do not add any attributes that do not appear in the schema shown below.

```TypeScript
script: Array<{{ // Adapted from the novel into script.
    role: string // The character involved in the relationship.
    relation: string // The relationship between a character and some object or attribute, e.g., possession, use, etc.
    object: string // The objects or attributes associated with the character.
}}>
```

Please output the extracted information in CSV format in Excel dialect. Please use a | as the delimiter. 
Do NOT add any clarifying information. Output MUST follow the schema above. Do NOT add any additional columns that do not appear in the schema.

Input: 在一个阴雨绵绵的下午，艾米莉坐在她的小书桌旁，神情专注。桌子上散落着几本厚厚的书，她正翻阅着一本关于古代魔法的书籍。艾米莉的背后，挂着她最喜欢的魔法披风，那是她父亲在一次探险中带回来的。她的眼睛闪烁着智慧的光芒，脸上总是挂着微笑，说明她对魔法学的热爱与执着。
Output: role|relation|object
艾米莉|拥有特征|智慧
艾米莉|拥有特征|热爱魔法
艾米莉|拥有|魔法披风
艾米莉|使用|古代魔法书

Input: 在阴暗的古堡中，哈利·波特独自站在石阶上，手中紧握着他的火弩箭。他的心跳加速，脑海中回想着在霍格沃茨学习的日子。在那里，他结识了聪明的赫敏·格兰杰和勇敢的罗恩·韦斯ley，三人共同克服了无数的困难。尽管面前充满了未知的危险，但哈利始终保持着对友谊的坚定信念。
Output: role|relation|object
哈利·波特|拥有特征|勇敢
赫敏·格兰杰|拥有特征|聪明
哈利·波特|拥有|火弩箭

Input: {text}
Output:
"""

prompt_template_role_center_en = """
Your goal is to extract structured information from the user's input that matches the form described below. When extracting information please make sure it matches the type information exactly. Do not add any attributes that do not appear in the schema shown below.

```TypeScript
script: Array<{{ // Adapted from the novel into script.
    role: string // The character involved in the relationship.
    relation: string // The relationship between a character and some object or attribute, e.g., possession, use, etc.
    object: string // The objects or attributes associated with the character.
}}>
```

Please output the extracted information in CSV format in Excel dialect. Please use a | as the delimiter. 
Do NOT add any clarifying information. Output MUST follow the schema above. Do NOT add any additional columns that do not appear in the schema.

Input: On a rainy afternoon, Emily sat at her small desk, her expression focused. Scattered on the table were several thick books, and she was flipping through one about ancient magic. Behind her hung her favorite magical cloak, which her father had brought back from an adventure. Her eyes sparkled with the light of wisdom, and a smile was always on her face, indicating her love and dedication to the study of magic.
Output: role|relation|object
Emily|has trait|wisdom
Emily|has trait|loves magic
Emily|has|magical cloak
Emily|uses|ancient magic book

Input: In the dimly lit castle, Harry Potter stood alone on the stone steps, gripping his Firebolt tightly. His heart raced as he recalled the days spent learning at Hogwarts. There, he had made friends with the clever Hermione Granger and the brave Ron Weasley, and together they had overcome countless challenges. Despite the unknown dangers ahead, Harry held firmly to his belief in the strength of friendship.
Output: role|relation|object
Harry Potter|has trait|brave
Hermione Granger|has trait|clever
Harry Potter|possesses|Firebolt

Input: {text}
Output:
"""


prompt_template_role_attribute_en = """
Your goal is to extract structured information from the user's input that matches the form described below. When extracting information please make sure it matches the type information exactly. Do not add any attributes that do not appear in the schema shown below.

```TypeScript
script: Array<{{ // Adapted from the novel into script.
    role: string // The target character.
    attribute: string // The name of certain attributes of the character itself or attributes closely related to the character, such as age, personality, hair color, skills, traits, residence, etc.
    value: string // The attribute value of the character.
}}>
```

Please output the extracted information in CSV format in Excel dialect. Please use a | as the delimiter. 
Do NOT add any clarifying information. Output MUST follow the schema above. Do NOT add any additional columns that do not appear in the schema.

Input: In a dimly lit forest, Priscilla sat quietly by the roots of a tree, sunlight filtering through the leaves onto her face. Her hair was a fiery mixture of red and black, reminiscent of burning flames. Priscilla's teeth were sharp and even, revealing her strength. Despite being only 18 years old, her eyes were full of wisdom and determination, as if she had experienced much of the world.
Output: role|attribute|value
Priscilla|age|18
Priscilla|hair color|red-black
Priscilla|tooth shape|sharp
Priscilla|eye characteristic|wisdom and determination
Priscilla|residence|forest

Input: In a distant kingdom, the young knight Arthur rode his steed through the dense forest. His friend Lily, a brave female archer, had long golden hair and shining blue eyes. Arthur was 20 years old, tall in stature, and skilled in swordsmanship, while Lily was only 19, direct in personality, often displaying remarkable precision in battle. During their journey, they encountered a powerful dragon; Arthur bravely confronted it, while Lily supported him from the sidelines.
Output: role|attribute|value
Arthur|age|20
Arthur|stature|tall
Arthur|skill|swordsmanship
Lily|age|19
Lily|hair color|golden
Lily|eye color|blue
Lily|personality|direct
Lily|skill|precision
Arthur|riding animal|steed

Input: {text}
Output:
"""


prompt_template_role_attribute_with_roles_en = """
Your goal is to extract structured information from the user's input that matches the form described below. When extracting information please make sure it matches the type information exactly. Do not add any attributes that do not appear in the schema shown below.

```TypeScript
script: Array<{{ // Adapted from the novel into script.
    role: string // The target character.
    attribute: string // The name of certain attributes of the character itself or attributes closely related to the character, such as age, personality, hair color, skills, traits, residence, etc.
    value: string // The attribute value of the character.
}}>
```

Please output the extracted information in CSV format in Excel dialect. Please use a | as the delimiter. 
Do NOT add any clarifying information. Output MUST follow the schema above. Do NOT add any additional columns that do not appear in the schema.

Input: In a dimly lit forest, Priscilla sat quietly by the roots of a tree, sunlight filtering through the leaves onto her face. Her hair was a fiery mixture of red and black, reminiscent of burning flames. Priscilla's teeth were sharp and even, revealing her strength. Despite being only 18 years old, her eyes were full of wisdom and determination, as if she had experienced much of the world.
Roles: Priscilla
Output: role|attribute|value
Priscilla|age|18
Priscilla|hair color|red-black
Priscilla|tooth shape|sharp
Priscilla|eye characteristic|wisdom and determination
Priscilla|residence|forest

Input: In a distant kingdom, the young knight Arthur rode his steed through the dense forest. His friend Lily, a brave female archer, had long golden hair and shining blue eyes. Arthur was 20 years old, tall in stature, and skilled in swordsmanship, while Lily was only 19, direct in personality, often displaying remarkable precision in battle. During their journey, they encountered a powerful dragon; Arthur bravely confronted it, while Lily supported him from the sidelines.
Roles: Arthur, Lily
Output: role|attribute|value
Arthur|age|20
Arthur|stature|tall
Arthur|skill|swordsmanship
Lily|age|19
Lily|hair color|golden
Lily|eye color|blue
Lily|personality|direct
Lily|skill|precision
Arthur|riding animal|steed

Input: {text}
Roles: {roles}
Output:
"""