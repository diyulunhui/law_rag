import json
import os

from entity.legal_case import LegalCase


class FileLoader:
    case_list: list[LegalCase]

    def __init__(self, path: str):
        self.case_list = []
        self.load_cases(path)

    def load_cases(self, path: str):
        """加载文件夹下所有JSON文件并合并ctxs中的案例"""
        all_cases = []
        for filename in os.listdir(path):
            if filename.endswith('.json'):
                with open(os.path.join(path, filename), 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if "ctxs" in data:
                        all_cases.extend(data["ctxs"].values())
        self.case_list = [LegalCase.from_dict(case) for case in all_cases]
