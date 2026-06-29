"""Transformer Health Index rule features based on dissolved gas thresholds."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Sequence


@dataclass(frozen=True)
class GasRule:
    column: str
    label: str
    healthy_max: float
    moderate_max: float
    poor_max: float

    def score(self, value: float) -> float:
        if value <= self.healthy_max:
            return 1.0
        if value <= self.moderate_max:
            return 0.75
        if value <= self.poor_max:
            return 0.5
        return 0.0

    def condition(self, value: float) -> str:
        if value <= self.healthy_max:
            return "Healthy"
        if value <= self.moderate_max:
            return "Moderate"
        if value <= self.poor_max:
            return "Poor"
        return "Critical"


GAS_RULES = (
    GasRule("H2", "H2", 100, 700, 1800),
    GasRule("Methane", "CH4", 120, 400, 1000),
    GasRule("Ethane", "C2H6", 65, 100, 150),
    GasRule("Ethylene", "C2H4", 50, 100, 200),
    GasRule("Acetylene", "C2H2", 1, 9, 35),
    GasRule("CO", "CO", 350, 570, 1400),
    GasRule("CO2", "CO2", 2500, 4000, 10000),
)

GAS_REASONS = {
    "H2": "Hydrogen high: Partial discharge/corona or general electrical stress in oil.",
    "Methane": "Methane high: Low-temperature oil overheating / low-energy thermal fault.",
    "Ethane": "Ethane high: Oil decomposition due to low-to-medium temperature thermal fault.",
    "Ethylene": "Ethylene high: High-temperature oil overheating / severe thermal fault.",
    "Acetylene": "Acetylene high: Arcing or high-energy electrical discharge; needs urgent attention if rising.",
    "CO": "CO high: Cellulose/paper insulation overheating or degradation.",
    "CO2": "CO2 high: Paper insulation aging/thermal stress; compare with CO and trend.",
}

GAS_LABEL_TO_COLUMN = {
    "H2": "H2",
    "CH4": "Methane",
    "C2H6": "Ethane",
    "C2H4": "Ethylene",
    "C2H2": "Acetylene",
    "CO": "CO",
    "CO2": "CO2",
}


def set_gas_rules(rules: Sequence[GasRule]) -> None:
    global GAS_RULES
    GAS_RULES = tuple(rules)


def set_gas_reasons(reasons: Mapping[str, str]) -> None:
    global GAS_REASONS
    GAS_REASONS = dict(reasons)


def gas_reason(column: str) -> str:
    return GAS_REASONS.get(column, f"{column} is outside the normal DGA range.")


RAW_FEATURES = (
    "H2",
    "Methane",
    "Acetylene",
    "Ethylene",
    "Ethane",
    "CO",
    "CO2",
)


def gas_scores(row: Mapping[str, float]) -> Dict[str, float]:
    return {f"{rule.column}_rule_score": rule.score(float(row[rule.column])) for rule in GAS_RULES}


def gas_conditions(row: Mapping[str, float]) -> Dict[str, str]:
    return {rule.column: rule.condition(float(row[rule.column])) for rule in GAS_RULES}


def critical_violations(row: Mapping[str, float]) -> Dict[str, float]:
    return {
        rule.column: float(row[rule.column])
        for rule in GAS_RULES
        if float(row[rule.column]) > rule.poor_max
    }


def gas_threshold_features(row: Mapping[str, float]) -> Dict[str, float]:
    features: Dict[str, float] = {}
    for rule in GAS_RULES:
        value = float(row[rule.column])
        features[f"{rule.column}_healthy_max"] = rule.healthy_max
        features[f"{rule.column}_moderate_max"] = rule.moderate_max
        features[f"{rule.column}_poor_max"] = rule.poor_max
        features[f"{rule.column}_critical_over"] = rule.poor_max
        features[f"{rule.column}_critical_ratio"] = value / rule.poor_max if rule.poor_max else 0.0
        features[f"{rule.column}_critical_excess"] = max(0.0, value - rule.poor_max)
        features[f"{rule.column}_critical_flag"] = 1.0 if value > rule.poor_max else 0.0
    return features


def rule_summary(scores: Mapping[str, float]) -> Dict[str, float]:
    values = list(scores.values())
    critical = sum(1 for value in values if value == 0.0)
    poor_or_worse = sum(1 for value in values if value <= 0.5)
    return {
        "gas_rule_avg": sum(values) / len(values),
        "gas_rule_min": min(values),
        "gas_critical_count": float(critical),
        "gas_poor_or_worse_count": float(poor_or_worse),
    }


def label_rule_features(scores: Mapping[str, float]) -> Dict[str, float]:
    summary = rule_summary(scores)
    any_critical = summary["gas_critical_count"] > 0
    acceptable = (not any_critical) and summary["gas_rule_avg"] >= 0.65
    deteriorated = (not any_critical) and summary["gas_rule_avg"] < 0.65
    return {
        "label_rule_any_gas_critical": 1.0 if any_critical else 0.0,
        "label_rule_acceptable": 1.0 if acceptable else 0.0,
        "label_rule_deteriorated": 1.0 if deteriorated else 0.0,
        "binary_hi_rule": 1.0 if acceptable else 0.0,
    }


def label_rule_name(scores: Mapping[str, float]) -> str:
    features = label_rule_features(scores)
    if features["label_rule_any_gas_critical"]:
        return "Any gas critical => Binary HI 0"
    if features["label_rule_acceptable"]:
        return "Acceptable"
    return "Deteriorated"


def feature_names() -> list[str]:
    rule_names = [f"{rule.column}_rule_score" for rule in GAS_RULES]
    threshold_names = []
    for rule in GAS_RULES:
        threshold_names.extend(
            [
                f"{rule.column}_healthy_max",
                f"{rule.column}_moderate_max",
                f"{rule.column}_poor_max",
                f"{rule.column}_critical_over",
                f"{rule.column}_critical_ratio",
                f"{rule.column}_critical_excess",
                f"{rule.column}_critical_flag",
            ]
        )
    return [
        *RAW_FEATURES,
        *rule_names,
        "gas_rule_avg",
        "gas_rule_min",
        "gas_critical_count",
        "gas_poor_or_worse_count",
        "label_rule_any_gas_critical",
        "label_rule_acceptable",
        "label_rule_deteriorated",
        "binary_hi_rule",
        *threshold_names,
    ]


def build_feature_row(row: Mapping[str, float]) -> list[float]:
    raw = [float(row[name]) for name in RAW_FEATURES]
    scores = gas_scores(row)
    summary = rule_summary(scores)
    label_features = label_rule_features(scores)
    thresholds = gas_threshold_features(row)
    return [*raw, *scores.values(), *summary.values(), *label_features.values(), *thresholds.values()]


def validate_columns(columns: Iterable[str]) -> None:
    missing = [name for name in (*RAW_FEATURES, "HI") if name not in columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {', '.join(missing)}")
