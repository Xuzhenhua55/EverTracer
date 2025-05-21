import random
from LoggerUtils import LoggerUtils

class TextProcessingUtils:
    """
    文本统一处理工具类，支持文本扰动与LLM模板注入。
    """

    @staticmethod
    def perturb_text(text, mode="none", perturb_ratio=0.05, charset="abcdefghijklmnopqrstuvwxyz"):
        """
        对输入文本进行扰动处理：
        - mode="remove" 删除一定比例的字符
        - mode="add" 在文本首尾添加随机字符
        - mode="none" 不做处理（默认）

        :param text: 输入文本
        :param mode: 扰动模式，支持 "remove"、"add"、或 "none"
        :param perturb_ratio: 扰动比例（例如 0.05 表示 5%）
        :param charset: 添加时使用的字符集合（仅对 mode="add" 有效）
        :return: (扰动后的文本, 扰动字符数)
        """
        if not text or mode == "none" or perturb_ratio <= 0:
            return text, 0

        text_length = len(text)
        perturb_count = max(1, int(text_length * perturb_ratio))

        if mode == "remove":
            indices_to_remove = set(random.sample(range(text_length), min(perturb_count, text_length)))
            perturbed_text = ''.join(char for idx, char in enumerate(text) if idx not in indices_to_remove)

        elif mode == "add":
            prefix = ''.join(random.choice(charset) for _ in range(perturb_count))
            suffix = ''.join(random.choice(charset) for _ in range(perturb_count))
            perturbed_text = prefix + text + suffix

        else:
            perturbed_text = text

        return perturbed_text, perturb_count

    prompt_templates = {
        "llama2": lambda text: f"<s> [INST] {text} [/INST]",
        "llama3": lambda text: f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{text}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
        "mistral": lambda text: f"<s>[INST] {text} [/INST]",
        "vicuna": lambda text: f"A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: {text} ASSISTANT:",
        "qwen": lambda text: f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n",
        "default": lambda text: f"{text}"
    }
    
    @staticmethod
    def format_input_instruction(json_obj, format="alpaca") -> str:
        """
        根据 format 类型将 input 和 instruction 拼接成 prompt 文本。

        :param json_obj: 包含 input、instruction 的 dict 对象
        :param format: 格式类型（如 "alpaca"）
        :return: 拼接后的文本字符串
        """
        if not isinstance(json_obj, dict):
            raise ValueError("输入必须是一个 JSON 对象（字典）")

        instruction = json_obj.get("instruction", "").strip()
        input_text = json_obj.get("input", "").strip()

        format = format.lower()
        if format == "alpaca":
            if input_text:
                return f"{instruction}\n{input_text}"
            else:
                return instruction
        else:
            raise NotImplementedError(f"不支持的格式类型: {format}")
    
    @staticmethod
    def encode_with_template(text: str, template="llama2") -> str:
        """
        将纯文本包装成模板格式。

        :param text: 拼接后的纯文本
        :param template: 模板类型（如 "llama2"）
        :return: 添加模板后的 prompt 字符串
        """
        template_fn = TextProcessingUtils.prompt_templates.get(
            template.lower(), TextProcessingUtils.prompt_templates["default"]
        )
        return template_fn(text)