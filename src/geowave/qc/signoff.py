from __future__ import annotations
from dataclasses import dataclass
from geowave.qc.pipeline import quality_grade

@dataclass(frozen=True)
class SignoffCheck:
    name: str
    passed: bool
    detail: str

@dataclass(frozen=True)
class SignoffResult:
    passed: bool
    checks: tuple[SignoffCheck, ...]

def signoff(dataset_qc_report: dict[str, object], audit_valid: bool, recipe_issues: list[str], minimum_grade: str='C') -> SignoffResult:
    grade, reasons = quality_grade(dataset_qc_report)
    rank = {'A': 4, 'B': 3, 'C': 2, 'D': 1, 'F': 0}
    checks = [SignoffCheck('dataset-quality', rank.get(grade, 0) >= rank.get(minimum_grade, 2), f"grade={grade}; {('; '.join(reasons) if reasons else 'no blocking reasons')}"), SignoffCheck('audit-chain', audit_valid, 'audit chain valid' if audit_valid else 'audit chain broken'), SignoffCheck('recipe', not recipe_issues, 'recipe valid' if not recipe_issues else '; '.join(recipe_issues))]
    return SignoffResult(all((c.passed for c in checks)), tuple(checks))
