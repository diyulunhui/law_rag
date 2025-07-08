from typing import List, Dict, Tuple
from dataclasses import dataclass


@dataclass
class LegalCase:
    case_id: str
    case_type: str
    case_proc: str
    case_record: str
    judge_accusation: str
    judge_reason: str
    judge_result: str
    keywords: List[str]
    parties: List[Dict]
    category: Dict[str, str]
    retrieval_score: float

    @classmethod
    def from_dict(cls, data: Dict):
        return cls(
            case_id=data.get("CaseId", ""),
            case_type=data.get("CaseType", ""),
            case_proc=data.get("CaseProc", ""),
            case_record=data.get("CaseRecord", ""),
            judge_accusation=data.get("JudgeAccusation", ""),
            judge_reason=data.get("JudgeReason", ""),
            judge_result=data.get("JudgeResult", ""),
            keywords=data.get("Keywords", []),
            parties=data.get("Parties", []),
            category=data.get("Category", {}),
            retrieval_score=0
        )
