# MatchUtils.py
from enum import Enum


class MatchType(Enum):
    EXACT = "exact"
    EDIT_DISTANCE = "edit_distance"


class MatchUtils:
    @staticmethod
    def is_match(pred: str, gold: str, match_config: dict) -> bool:
        pred = pred.strip()
        gold = gold.strip()

        match_type = match_config.get("match_type", MatchType.EXACT.value)
        distance_ratio = match_config.get("distance_ratio", 0.2)

        if match_type == MatchType.EXACT.value:
            return pred == gold
        elif match_type == MatchType.EDIT_DISTANCE.value:
            distance = MatchUtils.edit_distance(pred, gold)
            allowed_threshold = int(len(gold) * distance_ratio)
            return distance <= allowed_threshold
        else:
            raise ValueError(f"Unsupported match_type '{match_type}'")

    @staticmethod
    def edit_distance(s1: str, s2: str) -> int:
        dp = [[0] * (len(s2) + 1) for _ in range(len(s1) + 1)]

        for i in range(len(s1) + 1):
            dp[i][0] = i
        for j in range(len(s2) + 1):
            dp[0][j] = j

        for i in range(1, len(s1) + 1):
            for j in range(1, len(s2) + 1):
                cost = 0 if s1[i - 1] == s2[j - 1] else 1
                dp[i][j] = min(
                    dp[i - 1][j] + 1,      # 删除
                    dp[i][j - 1] + 1,      # 插入
                    dp[i - 1][j - 1] + cost  # 替换
                )
        return dp[len(s1)][len(s2)]