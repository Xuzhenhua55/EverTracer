import json
import os

class DatasetUtils:
    """
    数据集加载工具类（支持 JSON / JSONL 格式，并进行格式转换）。
    """

    @staticmethod
    def load_dataset(path, source_format="alpaca", target_format="alpaca", max_samples=None):
        """
        加载并转换数据集为目标格式。

        :param path: 数据集路径
        :param source_format: 原始数据集格式（如 'firefly', 'alpaca'）
        :param target_format: 目标数据集格式（如 'alpaca'）
        :param max_samples: 最大样本数
        :return: list of dict（统一为目标格式）
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"数据集文件不存在: {path}")

        # Step 1: 加载原始数据
        if source_format == "firefly":
            raw_data = DatasetUtils._load_jsonl(path, max_samples)
        elif source_format == "alpaca":
            raw_data = DatasetUtils._load_json(path, max_samples)
        else:
            raise ValueError(f"暂不支持的源数据集格式: {source_format}")

        # Step 2: 格式转换（如果需要）
        if source_format != target_format:
            if source_format == "firefly" and target_format == "alpaca":
                return DatasetUtils.convert_firefly_to_alpaca(raw_data)
            else:
                raise NotImplementedError(f"不支持从 {source_format} 转换为 {target_format}")
        else:
            return raw_data

    @staticmethod
    def _load_json(path, max_samples=None):
        """
        加载 JSON 文件（整体为 list）。

        :param path: 文件路径
        :param max_samples: 最大样本数
        :return: list of dict
        """
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if not isinstance(data, list):
            raise ValueError("JSON 文件内容必须为列表形式（list of dict）")

        return data if max_samples is None else data[:max_samples]

    @staticmethod
    def _load_jsonl(path, max_samples=None):
        """
        加载 JSONL 文件（逐行 JSON）。

        :param path: 文件路径
        :param max_samples: 最大样本数
        :return: list of dict
        """
        data = []
        with open(path, 'r', encoding='utf-8') as f:
            for line_idx, line in enumerate(f):
                if max_samples is not None and len(data) >= max_samples:
                    break
                try:
                    item = json.loads(line.strip())
                    data.append(item)
                except Exception as e:
                    print(f"[Line {line_idx}] JSONL 解析失败: {e}")
        return data

    @staticmethod
    def convert_firefly_to_alpaca(data: list) -> list:
        """
        将 Firefly 格式的数据转换为 Alpaca 格式。

        :param data: 原始 Firefly 格式数据（list of dict）
        :return: 转换后的 Alpaca 格式数据
        """
        alpaca_data = []
        for idx, item in enumerate(data):
            try:
                conv = item.get("conversation", [])
                if not conv or not isinstance(conv, list):
                    continue
                instruction = conv[0].get("human", "").strip()
                output = conv[0].get("assistant", "").strip()
                alpaca_data.append({
                    "instruction": instruction,
                    "input": "",
                    "output": output
                })
            except Exception as e:
                print(f"[Item {idx}] Firefly 转换失败: {e}")
        return alpaca_data
