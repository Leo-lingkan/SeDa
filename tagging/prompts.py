# -*- coding: utf-8 -*-

def get_tag_selection_prompt(dataset_name, dataset_desc, candidate_tags, top_k=2):
    prompt = f"""你是一位科学数据集标注专家，擅长为各类学术数据集分配准确的主题标签。

你的任务是：根据数据集的名称和描述，从候选标签列表中选择{top_k}个最能代表该数据集主题的标签。如果候选标签都不合适，你可以生成新的标签。

要求：
1. 标签应该简洁、准确、具有代表性
2. 优先从候选标签中选择
3. 新标签应该是通用的主题词或领域术语（如"计算机视觉"、"自然语言处理"、"生物信息学"等）
4. 避免过于宽泛（如"数据"）或过于具体的标签

输出格式：严格按照JSON格式输出
{{
    "tags": ["标签1", "标签2"],
    "new_tags": ["新标签1"]
}}
- tags: 最终选定的{top_k}个标签
- new_tags: 新生成的标签列表，如果没有则为空列表[]

示例：
数据集名称：ImageNet
数据集描述：ImageNet是一个大规模视觉识别数据集，包含超过100万张带标注的图像，涵盖1000个类别。广泛用于计算机视觉模型的训练和评估。
候选标签：computer vision, deep learning, object detection, neural networks

输出：
{{
    "tags": ["computer vision"],
    "new_tags": ["image classification"]
}}

现在请处理以下数据集：

数据集名称：{dataset_name}

数据集描述：{dataset_desc[:500]}

候选标签：{', '.join(candidate_tags)}

请直接输出JSON，不要包含任何解释。"""

    return prompt


def get_tag_generation_prompt(dataset_name, dataset_desc, num_tags=2):
    prompt = f"""你是一位科学数据集标注专家，擅长为各类学术数据集分配准确的主题标签。

你的任务是：根据数据集的名称和描述，生成{num_tags}个最能代表该数据集主题的标签。

要求：
1. 标签应该是通用的主题词或领域术语（如"计算机视觉"、"机器学习"、"生物信息学"等）
2. 标签要简洁、准确、具有代表性
3. 避免过于宽泛（如"数据"）或过于具体的标签

输出格式：严格按照JSON格式输出
{{
    "tags": ["标签1", "标签2"]
}}

示例：
数据集名称：COCO (Common Objects in Context)
数据集描述：COCO是一个大规模的物体检测、分割和字幕数据集，包含超过20万张图像和80个物体类别。

输出：
{{
    "tags": ["object detection", "computer vision"]
}}

现在请处理以下数据集：

数据集名称：{dataset_name}

数据集描述：{dataset_desc[:500]}

请直接输出JSON，不要包含任何解释。"""

    return prompt


def get_tag_validation_prompt(dataset_name, dataset_desc, tags):
    prompt = f"""你是一位科学数据集标注审核专家，擅长评估标签与数据集的匹配度。

你的任务是：验证给定的标签是否准确反映该数据集的主题，并给出验证结果。

要求：
1. 判断每个标签是否与数据集主题相关
2. 对于不准确的标签，提供更合适的替代标签
3. 替代标签应该是通用的主题词或领域术语

输出格式：严格按照JSON格式输出
{{
    "valid_tags": ["有效的标签1", "有效的标签2"],
    "invalid_tags": ["无效的标签"],
    "suggested_replacements": {{"无效的标签": "建议的替代标签"}}
}}

示例：
数据集名称：WikiText-103
数据集描述：WikiText-103是一个大规模的语言建模数据集，包含从维基百科提取的超过1亿个token，用于训练和评估语言模型。
待验证标签：natural language processing, computer vision, machine learning

输出：
{{
    "valid_tags": ["natural language processing", "machine learning"],
    "invalid_tags": ["computer vision"],
    "suggested_replacements": {{"computer vision": "language modeling"}}
}}

现在请处理以下数据集：

数据集名称：{dataset_name}

数据集描述：{dataset_desc[:500]}

待验证的标签：{', '.join(tags)}

请直接输出JSON，不要包含任何解释。"""

    return prompt

