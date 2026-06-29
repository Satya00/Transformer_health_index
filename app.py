"""Dynamic Transformer Health Monitoring System web app."""

from __future__ import annotations

import json
import html
import io
import os
import re
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, quote, urlparse

import numpy as np
import pandas as pd

import health_rules
from health_rules import RAW_FEATURES, build_feature_row, gas_conditions, gas_scores, label_rule_features, label_rule_name, rule_summary
from train_model import DEFAULT_DATASET, MODEL_DIR, configure_rules_from_workbook, predict


APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = MODEL_DIR / "health_index_model.npz"
METADATA_PATH = MODEL_DIR / "metadata.json"
CAPPING_REASONS_PATH = APP_DIR / "data" / "capping_with_reasons.xlsx"
configure_rules_from_workbook(DEFAULT_DATASET)


DEFAULT_INPUT = {
    "RatingMVA": 100,
    "H2": 80,
    "CO": 430,
    "CO2": 5000,
    "Methane": 85,
    "Acetylene": 0.5,
    "Ethylene": 18,
    "Ethane": 55,
    "BDV": 60,
    "OTI": 72,
    "WTI": 82,
}

TEXT_INPUT_DEFAULTS = {
    "SubstationName": "Substation-1",
    "VoltageRatio": "220/132 kV",
}


GAS_REASON_LABELS = {
    "H₂": "H2",
    "CH₄": "Methane",
    "C₂H₆": "Ethane",
    "C₂H₄": "Ethylene",
    "C₂H₂": "Acetylene",
    "CO": "CO",
    "CO₂": "CO2",
}


def parse_range(text: object) -> tuple[float | None, float | None]:
    value = str(text).strip().replace(",", "")
    numbers = [float(item) for item in re.findall(r"\d+(?:\.\d+)?", value)]
    if not numbers:
        return None, None
    if value.startswith(">"):
        return numbers[0], None
    if value.startswith("<"):
        return None, numbers[0]
    if len(numbers) == 1:
        return numbers[0], numbers[0]
    return min(numbers), max(numbers)


def value_in_range(value: float, range_text: object) -> bool:
    text = str(range_text).strip()
    low, high = parse_range(text)
    if low is not None and high is None:
        return value > low
    if low is None and high is not None:
        return value < high
    if low is not None and high is not None:
        return low <= value <= high
    return False


def load_capping_reason_rules() -> dict[str, dict]:
    gas_df = pd.read_excel(CAPPING_REASONS_PATH, sheet_name=0).dropna(subset=["Gas (ppm)"])
    temp_df = pd.read_excel(CAPPING_REASONS_PATH, sheet_name=1)
    bdv_df = pd.read_excel(CAPPING_REASONS_PATH, sheet_name=2)

    gas_rules = {}
    for record in gas_df.to_dict("records"):
        gas_label = str(record["Gas (ppm)"]).strip()
        column = GAS_REASON_LABELS.get(gas_label)
        if not column:
            continue
        gas_rules[column] = {
            "poor_range": record["value1 "],
            "poor_reason": str(record[" Reason"]).strip(),
            "critical_range": record["value 2"],
            "critical_reason": str(record["Reason"]).strip(),
        }

    temp_rules = {str(row["Parameter"]).split()[0]: row for row in temp_df.to_dict("records")}
    bdv_rules = bdv_df.to_dict("records")
    return {"gas": gas_rules, "temperature": temp_rules, "bdv": bdv_rules}


CAPPING_REASON_RULES = load_capping_reason_rules()


def parse_prediction_values(query: dict[str, list[str]]) -> dict[str, float]:
    return {key: float(query.get(key, [DEFAULT_INPUT[key]])[0]) for key in DEFAULT_INPUT}


def query_text(query: dict[str, list[str]], key: str, default: str = "") -> str:
    return str(query.get(key, [default])[0])


def safe_filename_part(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", value).strip("_")
    return cleaned or "substation"


def load_model() -> dict[str, np.ndarray]:
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Model not found. Run: python train_model.py")
    loaded = np.load(MODEL_PATH)
    return {name: loaded[name] for name in loaded.files}


def classify_hi(value: float) -> str:
    if value >= 85:
        return "Healthy"
    if value >= 65:
        return "Moderate"
    if value >= 40:
        return "Poor"
    return "Critical"


def gas_issue_reason(gas: str, value: float) -> dict[str, str] | None:
    rule = CAPPING_REASON_RULES["gas"].get(gas)
    if not rule:
        return None
    if value_in_range(value, rule["critical_range"]):
        return {"parameter": gas, "value": f"{value:g} ppm", "condition": "Critical", "reason": rule["critical_reason"]}
    if value_in_range(value, rule["poor_range"]):
        return {"parameter": gas, "value": f"{value:g} ppm", "condition": "Poor", "reason": rule["poor_reason"]}
    return None


def temperature_issue_reason(parameter: str, value: float) -> dict[str, str] | None:
    rule = CAPPING_REASON_RULES["temperature"].get(parameter)
    if not rule:
        return None
    if parameter == "OTI" and value > 90:
        return {"parameter": "OTI(MAX)", "value": f"{value:g} °C", "condition": str(rule["Condition"]), "reason": str(rule["Reason"])}
    if parameter == "WTI" and value > 95:
        return {"parameter": "WTI(MAX)", "value": f"{value:g} °C", "condition": str(rule["Condition"]), "reason": str(rule["Reason"])}
    return None


def bdv_issue_reason(value: float) -> dict[str, str] | None:
    for rule in CAPPING_REASON_RULES["bdv"]:
        if value_in_range(value, rule["BDV (kV)"]):
            condition = str(rule["Condition"]).strip()
            if condition == "Good":
                return None
            return {"parameter": "BDV", "value": f"{value:g} kV", "condition": condition, "reason": str(rule["Reason"])}
    return None


def issue_reasons(row: dict[str, float], values: dict[str, float], gas_status: dict[str, str], operating_violations: dict[str, float]) -> list[dict[str, str]]:
    reasons: list[dict[str, str]] = []
    for gas in RAW_FEATURES:
        reason = gas_issue_reason(gas, row[gas])
        if reason:
            reasons.append(reason)
    bdv_reason = bdv_issue_reason(float(values.get("BDV", DEFAULT_INPUT["BDV"])))
    if bdv_reason:
        reasons.append(bdv_reason)
    oti_reason = temperature_issue_reason("OTI", float(values.get("OTI", DEFAULT_INPUT["OTI"])))
    if oti_reason:
        reasons.append(oti_reason)
    wti_reason = temperature_issue_reason("WTI", float(values.get("WTI", DEFAULT_INPUT["WTI"])))
    if wti_reason:
        reasons.append(wti_reason)
    return reasons


def recommendation(condition: str, reasons: list[dict[str, str]]) -> str:
    if reasons:
        return "Review the issue points below."
    if condition == "Healthy":
        return "Transformer condition is healthy. Continue routine monitoring and periodic oil testing."
    if condition == "Moderate":
        return "Schedule closer monitoring. Review loading, cooling performance, and repeat DGA trend analysis."
    if condition == "Poor":
        return "Plan maintenance action. Verify oil quality, inspect cooling system, and increase testing frequency."
    return "Critical condition indicated. Carry out urgent diagnostic inspection and operational risk review."


def operating_limit_violations(values: dict[str, float]) -> dict[str, float]:
    violations: dict[str, float] = {}
    bdv = float(values.get("BDV", DEFAULT_INPUT["BDV"]))
    oti = float(values.get("OTI", DEFAULT_INPUT["OTI"]))
    wti = float(values.get("WTI", DEFAULT_INPUT["WTI"]))

    if bdv < 40:
        violations["BDV"] = bdv
    if oti > 90:
        violations["OTI"] = oti
    if wti > 95:
        violations["WTI"] = wti
    return violations


def predict_payload(values: dict[str, float]) -> dict:
    model = load_model()
    row = {name: float(values.get(name, DEFAULT_INPUT[name])) for name in RAW_FEATURES}
    features = np.array([build_feature_row(row)], dtype=np.float64)
    model_hi = float(predict(model, features)[0][0])
    scores = gas_scores(row)
    summary = rule_summary(scores)
    gas_status = gas_conditions(row)
    label_features = label_rule_features(scores)
    label_rule = label_rule_name(scores)
    critical_label_rule = label_features["label_rule_any_gas_critical"] == 1.0
    operating_violations = operating_limit_violations(values)
    operating_override = bool(operating_violations)
    hi = 0.0 if critical_label_rule or operating_override else model_hi
    condition = classify_hi(hi)
    reasons = issue_reasons(row, values, gas_status, operating_violations)

    return {
        "transformer_details": {
            "rating_mva": round(float(values.get("RatingMVA", DEFAULT_INPUT["RatingMVA"])), 2),
        },
        "health_index": round(hi, 2),
        "model_health_index": round(model_hi, 2),
        "label_rule_override": bool(critical_label_rule),
        "operating_limit_override": bool(operating_override),
        "operating_violations": operating_violations,
        "condition": condition,
        "recommendation": recommendation(condition, reasons),
        "reason_points": reasons,
        "gas_conditions": gas_status,
        "gas_scores": {key: round(value, 2) for key, value in scores.items()},
        "label_rule": label_rule,
        "label_rule_features": {key: round(value, 2) for key, value in label_features.items()},
        "rule_summary": {key: round(value, 2) for key, value in summary.items()},
    }


def json_response(handler: BaseHTTPRequestHandler, payload: dict, status: int = 200) -> None:
    body = json.dumps(payload).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def condition_class(condition: str) -> str:
    return condition if condition in {"Healthy", "Moderate", "Poor", "Critical"} else "Critical"


def bar_row(label: str, value: float, unit: str, scale: float, condition: str = "") -> str:
    width = max(2.0, min(100.0, (value / scale) * 100.0 if scale else 0.0))
    safe_label = html.escape(label)
    safe_condition = html.escape(condition)
    condition_badge = f"<span>{safe_condition}</span>" if safe_condition else ""
    return f"""
      <div class="bar-row">
        <div class="bar-head"><strong>{safe_label}</strong><em>{value:g} {html.escape(unit)}</em>{condition_badge}</div>
        <div class="bar-track"><i style="width:{width:.2f}%"></i></div>
      </div>
    """


def render_result_page(query: dict[str, list[str]], raw_query: str = "") -> str:
    values = parse_prediction_values(query)
    substation_name = html.escape(query_text(query, "SubstationName", "Substation-1"))
    voltage_ratio = html.escape(query_text(query, "VoltageRatio", "220/132 kV"))
    pdf_query = html.escape(raw_query)
    data = predict_payload(values)
    gas_status = data["gas_conditions"]
    gas_scale = {rule.column: rule.poor_max for rule in health_rules.GAS_RULES}

    gas_bars = "\n".join(
        bar_row(label, values[column], "ppm", gas_scale.get(column, max(values[column], 1)), gas_status[column])
        for label, column in (
            ("H2 Hydrogen", "H2"),
            ("CH4 Methane", "Methane"),
            ("C2H6 Ethane", "Ethane"),
            ("C2H4 Ethylene", "Ethylene"),
            ("C2H2 Acetylene", "Acetylene"),
            ("CO Carbon Monoxide", "CO"),
            ("CO2 Carbon Dioxide", "CO2"),
        )
    )
    operating_bars = "\n".join(
        [
            bar_row("Rating", values["RatingMVA"], "MVA", max(values["RatingMVA"], 1)),
            bar_row("BDV", values["BDV"], "kV", 80),
            bar_row("OTI(MAX)", values["OTI"], "°C", 120),
            bar_row("WTI(MAX)", values["WTI"], "°C", 130),
        ]
    )
    reason_cards = "\n".join(
        f"""
        <article class="reason-card">
          <strong>{html.escape(item["parameter"])}: {html.escape(item["value"])}</strong>
          <span>Condition: {html.escape(item["condition"])}</span>
          <p>{html.escape(item["reason"])}</p>
        </article>
        """
        for item in data.get("reason_points", [])
    )
    if not reason_cards:
        reason_cards = '<article class="reason-card ok"><strong>No abnormal parameter found</strong><p>All entered values are within the active rule limits.</p></article>'

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Transformer Health Result</title>
  <style>
    :root {{
      --bg: #03070c;
      --panel: rgba(9, 22, 34, .82);
      --line: rgba(119, 221, 255, .2);
      --text: #eefaff;
      --muted: #9bb2c0;
      --cyan: #2fe6ff;
      --green: #40ffa8;
      --amber: #ffd166;
      --red: #ff5d6c;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      min-height: 100vh;
      font-family: Inter, Segoe UI, Arial, sans-serif;
      color: var(--text);
      background:
        radial-gradient(circle at 20% 18%, rgba(47,230,255,.16), transparent 30%),
        radial-gradient(circle at 78% 72%, rgba(64,255,168,.12), transparent 32%),
        linear-gradient(135deg, #010306, #071421 52%, #020509);
      padding: clamp(16px, 3vw, 34px);
    }}
    .layout {{
      width: min(1180px, 100%);
      margin: 0 auto;
      display: grid;
      gap: 18px;
    }}
    .top {{
      display: flex;
      justify-content: space-between;
      gap: 16px;
      align-items: flex-start;
      border: 1px solid var(--line);
      border-radius: 10px;
      padding: 22px;
      background: var(--panel);
      box-shadow: 0 20px 70px rgba(0,0,0,.35);
    }}
    .top-title {{
      display: grid;
      gap: 14px;
    }}
    .top-actions {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
    }}
    .result-btn {{
      display: inline-flex;
      align-items: center;
      justify-content: center;
      min-height: 40px;
      border: 1px solid rgba(47, 230, 255, .42);
      border-radius: 8px;
      padding: 10px 14px;
      color: var(--text);
      text-decoration: none;
      font-weight: 800;
      background: rgba(255, 255, 255, .07);
    }}
    .result-btn.primary {{
      color: #021019;
      background: linear-gradient(135deg, var(--cyan), var(--green));
      box-shadow: 0 0 24px rgba(47, 230, 255, .18);
    }}
    h1 {{ margin: 0; font-size: clamp(26px, 4vw, 44px); }}
    .sub {{ color: var(--muted); margin-top: 8px; }}
    .score {{
      text-align: right;
      min-width: 180px;
    }}
    .score strong {{
      display: block;
      font-size: clamp(46px, 8vw, 76px);
      line-height: .9;
      color: var(--green);
      text-shadow: 0 0 26px rgba(64,255,168,.26);
    }}
    .badge {{
      display: inline-block;
      margin-top: 10px;
      padding: 8px 13px;
      border-radius: 999px;
      font-weight: 800;
      color: #041018;
      background: var(--green);
    }}
    .badge.Moderate {{ background: var(--amber); }}
    .badge.Poor, .badge.Critical {{ color: white; background: var(--red); }}
    .grid {{
      display: grid;
      grid-template-columns: 1.1fr .9fr;
      gap: 18px;
    }}
    .panel {{
      border: 1px solid var(--line);
      border-radius: 10px;
      padding: 18px;
      background: var(--panel);
    }}
    h2 {{ margin: 0 0 14px; font-size: 20px; }}
    .detail-row {{
      display: grid;
      grid-template-columns: repeat(4, 1fr);
      gap: 12px;
    }}
    .detail {{
      border: 1px solid rgba(145,205,224,.16);
      border-radius: 8px;
      padding: 12px;
      background: rgba(2,8,13,.45);
    }}
    .detail small {{ display: block; color: var(--muted); margin-bottom: 5px; }}
    .detail strong {{ font-size: 17px; }}
    .bar-row {{ margin-bottom: 13px; }}
    .bar-head {{
      display: grid;
      grid-template-columns: 1fr auto auto;
      gap: 10px;
      align-items: center;
      margin-bottom: 7px;
      color: #dffbff;
      font-size: 13px;
    }}
    .bar-head em {{ color: var(--muted); font-style: normal; }}
    .bar-head span {{ color: var(--amber); }}
    .bar-track {{
      height: 10px;
      border-radius: 999px;
      background: rgba(255,255,255,.08);
      overflow: hidden;
    }}
    .bar-track i {{
      display: block;
      height: 100%;
      border-radius: inherit;
      background: linear-gradient(90deg, var(--cyan), var(--green));
      box-shadow: 0 0 18px rgba(47,230,255,.25);
    }}
    .reason-card {{
      border: 1px solid rgba(255,93,108,.28);
      border-radius: 8px;
      padding: 13px;
      margin-bottom: 10px;
      background: rgba(255,93,108,.08);
    }}
    .reason-card.ok {{
      border-color: rgba(64,255,168,.25);
      background: rgba(64,255,168,.08);
    }}
    .reason-card strong {{ display: block; color: #fff; margin-bottom: 5px; }}
    .reason-card span {{ display: block; color: #ffdca0; font-size: 13px; margin-bottom: 7px; }}
    .reason-card p {{ margin: 0; color: #dcecf2; line-height: 1.45; }}
    .meta {{ color: var(--muted); line-height: 1.5; }}
    .result-footer {{
      text-align: center;
      color: #9fd4df;
      font-size: 13px;
      letter-spacing: .12em;
      text-transform: uppercase;
      padding: 8px 0 2px;
    }}
    @media (max-width: 860px) {{
      .top, .grid, .detail-row {{ grid-template-columns: 1fr; display: grid; }}
      .score {{ text-align: left; }}
    }}
  </style>
</head>
<body>
  <main class="layout">
    <section class="top">
      <div class="top-title">
        <h1>Transformer Health Result</h1>
        <div class="sub">AI-based Health Index Prediction for Transformers</div>
        <div class="top-actions">
          <a class="result-btn" href="/">Home Page</a>
          <a class="result-btn primary" href="/result.pdf?{pdf_query}">Download PDF</a>
        </div>
      </div>
      <div class="score">
        <strong>{data["health_index"]:.2f}</strong>
        <span class="badge {condition_class(data["condition"])}">{html.escape(data["condition"])}</span>
      </div>
    </section>

    <section class="panel">
      <h2>Transformer Details</h2>
      <div class="detail-row">
        <div class="detail"><small>Substation</small><strong>{substation_name}</strong></div>
        <div class="detail"><small>Rating</small><strong>{values["RatingMVA"]:g} MVA</strong></div>
        <div class="detail"><small>Voltage Ratio</small><strong>{voltage_ratio}</strong></div>
        <div class="detail"><small>Model Estimate</small><strong>{data["model_health_index"]:.2f}</strong></div>
      </div>
    </section>

    <section class="grid">
      <div class="panel">
        <h2>DGA Input Graph</h2>
        {gas_bars}
      </div>
      <div class="panel">
        <h2>Operating Input Graph</h2>
        {operating_bars}
      </div>
    </section>

    <section class="panel">
      <h2>Recommendation</h2>
      {reason_cards}
    </section>

    <footer class="result-footer">Created by Satya</footer>
  </main>
</body>
</html>"""


def render_result_pdf(query: dict[str, list[str]]) -> tuple[bytes, str]:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    values = parse_prediction_values(query)
    data = predict_payload(values)
    substation_name = query_text(query, "SubstationName", "Substation-1")
    voltage_ratio = query_text(query, "VoltageRatio", "220/132 kV")
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    filename = f"health_index_{safe_filename_part(substation_name)}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
    styles = getSampleStyleSheet()
    story = [
        Paragraph("Transformer Health Index Result", styles["Title"]),
        Spacer(1, 10),
        Paragraph(f"Generated: {html.escape(timestamp)}", styles["Normal"]),
        Paragraph(f"Substation Name: {html.escape(substation_name)}", styles["Normal"]),
        Paragraph(f"Rating: {values['RatingMVA']:g} MVA", styles["Normal"]),
        Paragraph(f"Voltage Ratio: {html.escape(voltage_ratio)}", styles["Normal"]),
        Spacer(1, 12),
    ]

    summary_table = Table(
        [
            ["Health Index", f"{data['health_index']:.2f}"],
            ["Condition", data["condition"]],
            ["Model Estimate", f"{data['model_health_index']:.2f}"],
            ["Label Rule", data["label_rule"]],
        ],
        colWidths=[150, 320],
    )
    summary_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0B2335")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#8FB8C8")),
                ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("PADDING", (0, 0), (-1, -1), 7),
            ]
        )
    )
    story.extend([summary_table, Spacer(1, 14), Paragraph("Input Parameters", styles["Heading2"])])

    input_rows = [["Parameter", "Value"]]
    input_rows.extend(
        [
            ["H2 Hydrogen", f"{values['H2']:g} ppm"],
            ["CH4 Methane", f"{values['Methane']:g} ppm"],
            ["C2H6 Ethane", f"{values['Ethane']:g} ppm"],
            ["C2H4 Ethylene", f"{values['Ethylene']:g} ppm"],
            ["C2H2 Acetylene", f"{values['Acetylene']:g} ppm"],
            ["CO Carbon Monoxide", f"{values['CO']:g} ppm"],
            ["CO2 Carbon Dioxide", f"{values['CO2']:g} ppm"],
            ["BDV", f"{values['BDV']:g} kV"],
            ["OTI(MAX)", f"{values['OTI']:g} deg C"],
            ["WTI(MAX)", f"{values['WTI']:g} deg C"],
        ]
    )
    input_table = Table(input_rows, colWidths=[220, 250])
    input_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0E4D64")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#9CB8C2")),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("PADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    story.extend([input_table, Spacer(1, 14), Paragraph("Recommendation", styles["Heading2"])])

    reason_points = data.get("reason_points", [])
    if reason_points:
        for item in reason_points:
            story.append(
                Paragraph(
                    f"<b>{html.escape(item['parameter'])}: {html.escape(item['value'])}</b><br/>"
                    f"Condition: {html.escape(item['condition'])}<br/>"
                    f"{html.escape(item['reason'])}",
                    styles["BodyText"],
                )
            )
            story.append(Spacer(1, 8))
    else:
        story.append(Paragraph("No abnormal parameter found. All entered values are within the active rule limits.", styles["BodyText"]))

    doc.build(story)
    return buffer.getvalue(), filename


def render_index() -> str:
    return """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Transformer Health Monitoring System</title>
  <style>
    :root {
      --bg: #03070c;
      --panel: rgba(9, 22, 34, .72);
      --panel-strong: rgba(11, 29, 44, .88);
      --line: rgba(119, 221, 255, .2);
      --text: #eefaff;
      --muted: #92a8b7;
      --cyan: #2fe6ff;
      --green: #40ffa8;
      --amber: #ffd166;
      --red: #ff5d6c;
      --shadow: rgba(47, 230, 255, .22);
    }

    * { box-sizing: border-box; }

    body {
      margin: 0;
      min-height: 100vh;
      font-family: Inter, "Segoe UI", Arial, sans-serif;
      color: var(--text);
      background:
        radial-gradient(circle at 18% 18%, rgba(47, 230, 255, .2), transparent 32%),
        radial-gradient(circle at 80% 70%, rgba(64, 255, 168, .16), transparent 30%),
        linear-gradient(135deg, #010306 0%, #06121e 48%, #020509 100%);
      overflow-x: hidden;
    }

    body::before {
      content: "";
      position: fixed;
      inset: 0;
      pointer-events: none;
      opacity: .55;
      background-image:
        linear-gradient(rgba(47, 230, 255, .12) 1px, transparent 1px),
        linear-gradient(90deg, rgba(47, 230, 255, .12) 1px, transparent 1px);
      background-size: 72px 72px;
      animation: gridMove 18s linear infinite;
      mask-image: linear-gradient(to bottom, rgba(0, 0, 0, .95), rgba(0, 0, 0, .08));
    }

    .pulse-line {
      position: fixed;
      left: -20%;
      width: 140%;
      height: 1px;
      pointer-events: none;
      background: linear-gradient(90deg, transparent, rgba(47, 230, 255, .84), rgba(64, 255, 168, .6), transparent);
      box-shadow: 0 0 22px var(--shadow);
      animation: pulseLine 6s ease-in-out infinite;
    }

    .pulse-line.one { top: 27%; }
    .pulse-line.two { top: 68%; animation-delay: -2.5s; opacity: .68; }

    .particles {
      position: fixed;
      inset: 0;
      overflow: hidden;
      pointer-events: none;
    }

    .particle {
      position: absolute;
      width: 4px;
      height: 4px;
      border-radius: 50%;
      background: var(--cyan);
      box-shadow: 0 0 14px var(--cyan);
      opacity: .45;
      animation: floatParticle 10s ease-in-out infinite;
    }

    .particle:nth-child(1) { left: 12%; top: 72%; animation-delay: -1s; }
    .particle:nth-child(2) { left: 26%; top: 22%; animation-delay: -4s; }
    .particle:nth-child(3) { left: 52%; top: 78%; animation-delay: -6s; }
    .particle:nth-child(4) { left: 74%; top: 28%; animation-delay: -2s; }
    .particle:nth-child(5) { left: 88%; top: 62%; animation-delay: -8s; }

    .shell {
      position: relative;
      z-index: 1;
      min-height: 100vh;
      display: grid;
      grid-template-rows: auto 1fr auto;
      padding: 28px clamp(18px, 4vw, 56px);
    }

    .topbar {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 18px;
    }

    .brand {
      display: flex;
      align-items: center;
      gap: 12px;
      color: var(--muted);
      font-size: 13px;
      letter-spacing: .12em;
      text-transform: uppercase;
    }

    .brand-mark {
      width: 34px;
      height: 34px;
      border: 1px solid rgba(47, 230, 255, .5);
      border-radius: 8px;
      display: grid;
      place-items: center;
      color: var(--green);
      box-shadow: 0 0 24px rgba(47, 230, 255, .18);
    }

    .system-chip {
      border: 1px solid rgba(64, 255, 168, .28);
      border-radius: 999px;
      padding: 9px 14px;
      color: #c9ffed;
      background: rgba(64, 255, 168, .08);
      font-size: 13px;
    }

    .hero {
      display: grid;
      place-items: center;
      text-align: center;
      padding: 58px 0 36px;
    }

    .hero-inner {
      width: min(980px, 100%);
    }

    h1 {
      margin: 0;
      font-size: clamp(38px, 7vw, 82px);
      line-height: 1.02;
      letter-spacing: 0;
      text-shadow: 0 0 34px rgba(47, 230, 255, .24);
    }

    .subtitle {
      margin: 22px auto 0;
      max-width: 780px;
      color: #bdd4df;
      font-size: clamp(16px, 2.2vw, 23px);
      line-height: 1.5;
    }

    .hero-credit {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      margin-top: 16px;
      border: 1px solid rgba(64, 255, 168, .3);
      border-radius: 999px;
      padding: 8px 16px;
      color: #d9fff2;
      background: rgba(64, 255, 168, .08);
      box-shadow: 0 0 24px rgba(47, 230, 255, .12);
      font-size: 13px;
      font-weight: 800;
      letter-spacing: .12em;
      text-transform: uppercase;
    }

    .status-row {
      display: grid;
      grid-template-columns: repeat(3, minmax(150px, 1fr));
      gap: 14px;
      margin: 42px auto 0;
      max-width: 760px;
    }

    .status-card {
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 16px;
      background: rgba(9, 22, 34, .55);
      backdrop-filter: blur(18px);
      box-shadow: 0 16px 44px rgba(0, 0, 0, .24);
    }

    .status-card strong {
      display: block;
      margin-bottom: 6px;
      color: var(--green);
      font-size: 22px;
    }

    .status-card span {
      color: var(--muted);
      font-size: 13px;
    }

    .cta-wrap {
      display: flex;
      justify-content: center;
      padding: 24px 0 18px;
    }

    .credit {
      text-align: center;
      color: #9fd4df;
      font-size: 14px;
      letter-spacing: .08em;
      text-transform: uppercase;
    }

    .insert-btn, .predict-btn, .home-btn {
      border: 1px solid rgba(47, 230, 255, .62);
      border-radius: 8px;
      padding: 15px 24px;
      color: #021019;
      background: linear-gradient(135deg, var(--cyan), var(--green));
      font: inherit;
      font-weight: 800;
      cursor: pointer;
      box-shadow: 0 0 28px rgba(47, 230, 255, .25), inset 0 1px 0 rgba(255, 255, 255, .6);
      transition: transform .2s ease, box-shadow .2s ease;
    }

    .home-btn {
      color: var(--text);
      background: rgba(255, 255, 255, .07);
      box-shadow: none;
    }

    .insert-btn:hover, .predict-btn:hover, .home-btn:hover {
      transform: translateY(-2px);
      box-shadow: 0 0 40px rgba(64, 255, 168, .28);
    }

    .modal {
      position: fixed;
      inset: 0;
      z-index: 5;
      display: none;
      align-items: center;
      justify-content: center;
      padding: 18px;
      background: rgba(0, 0, 0, .72);
      backdrop-filter: blur(14px);
    }

    .modal.open { display: flex; }

    .modal-card {
      width: min(1080px, 100%);
      max-height: 92vh;
      overflow: auto;
      border: 1px solid rgba(47, 230, 255, .28);
      border-radius: 10px;
      background: linear-gradient(145deg, rgba(9, 20, 32, .92), rgba(6, 14, 22, .96));
      box-shadow: 0 30px 90px rgba(0, 0, 0, .6), 0 0 40px rgba(47, 230, 255, .12);
    }

    .modal-head {
      position: sticky;
      top: 0;
      z-index: 2;
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 14px;
      padding: 20px 22px;
      background: rgba(6, 14, 22, .96);
      border-bottom: 1px solid rgba(47, 230, 255, .18);
    }

    .modal-head h2 { margin: 0; font-size: 22px; }

    .modal-actions {
      display: flex;
      align-items: center;
      gap: 10px;
    }

    .close-btn {
      width: 38px;
      height: 38px;
      border: 1px solid rgba(255, 255, 255, .14);
      border-radius: 8px;
      color: var(--text);
      background: rgba(255, 255, 255, .06);
      cursor: pointer;
      font-size: 22px;
    }

    form {
      display: grid;
      gap: 18px;
      padding: 22px;
    }

    .form-section, .output-card {
      border: 1px solid rgba(47, 230, 255, .18);
      border-radius: 8px;
      padding: 18px;
      background: var(--panel);
      backdrop-filter: blur(18px);
    }

    .form-section h3, .output-card h3 {
      margin: 0 0 14px;
      color: #dffbff;
      font-size: 16px;
    }

    .fields {
      display: grid;
      grid-template-columns: repeat(3, minmax(180px, 1fr));
      gap: 14px;
    }


    label span {
      display: block;
      margin-bottom: 7px;
      color: var(--muted);
      font-size: 13px;
    }

    input {
      width: 100%;
      border: 1px solid rgba(145, 205, 224, .22);
      border-radius: 7px;
      padding: 12px;
      color: var(--text);
      background: rgba(2, 8, 13, .62);
      font: inherit;
      outline: none;
    }

    input:focus {
      border-color: rgba(47, 230, 255, .75);
      box-shadow: 0 0 0 3px rgba(47, 230, 255, .12);
    }

    .actions {
      display: flex;
      gap: 12px;
      justify-content: center;
      flex-wrap: wrap;
    }

    .actions .predict-btn {
      min-width: min(360px, 100%);
      padding: 18px 34px;
      font-size: 17px;
      letter-spacing: .02em;
    }

    .modal-footer {
      border-top: 1px solid rgba(47, 230, 255, .16);
      padding: 15px 22px 18px;
      text-align: center;
      color: #9fd4df;
      font-size: 13px;
      letter-spacing: .12em;
      text-transform: uppercase;
      background: rgba(6, 14, 22, .72);
    }

    .output-card {
      display: none;
      grid-template-columns: .75fr 1.25fr;
      gap: 18px;
      align-items: center;
    }

    .output-card.show { display: grid; }

    .score {
      font-size: clamp(46px, 8vw, 76px);
      font-weight: 900;
      color: var(--green);
      text-shadow: 0 0 26px rgba(64, 255, 168, .28);
    }

    .condition {
      display: inline-block;
      margin-top: 8px;
      border-radius: 999px;
      padding: 8px 13px;
      color: #041018;
      background: var(--green);
      font-weight: 800;
    }

    .condition.Moderate { background: var(--amber); }
    .condition.Poor, .condition.Critical { color: white; background: var(--red); }

    .recommendation {
      color: #d6e8ef;
      line-height: 1.55;
    }

    .recommendation-list {
      display: grid;
      gap: 10px;
      margin-top: 12px;
    }

    .recommendation-item {
      border: 1px solid rgba(47, 230, 255, .18);
      border-radius: 8px;
      padding: 12px;
      background: rgba(2, 8, 13, .45);
    }

    .recommendation-item strong {
      display: block;
      color: var(--cyan);
      margin-bottom: 5px;
    }

    .recommendation-item span {
      display: block;
      color: #ffdca0;
      font-size: 13px;
      margin-bottom: 6px;
    }

    .recommendation-item p {
      margin: 0;
      color: #d6e8ef;
      line-height: 1.45;
    }

    .meta {
      margin-top: 12px;
      color: var(--muted);
      font-size: 13px;
      line-height: 1.5;
    }

    @keyframes gridMove {
      from { transform: translate3d(0, 0, 0); }
      to { transform: translate3d(72px, 72px, 0); }
    }

    @keyframes pulseLine {
      0%, 100% { transform: translateX(-18%) scaleX(.78); opacity: .15; }
      45% { transform: translateX(10%) scaleX(1); opacity: .9; }
    }

    @keyframes floatParticle {
      0%, 100% { transform: translateY(0) translateX(0); }
      50% { transform: translateY(-46px) translateX(18px); }
    }

    @media (max-width: 860px) {
      .shell { padding: 20px 14px; }
      .topbar { align-items: flex-start; flex-direction: column; }
      .status-row, .fields, .output-card { grid-template-columns: 1fr; }
      .modal-card { max-height: 94vh; }
      .actions { justify-content: stretch; }
      .predict-btn { width: 100%; }
      .actions .predict-btn { min-width: 100%; }
      .modal-actions .home-btn { padding: 11px 13px; }
    }
  </style>
</head>
<body>
  <div class="pulse-line one"></div>
  <div class="pulse-line two"></div>
  <div class="particles"><i class="particle"></i><i class="particle"></i><i class="particle"></i><i class="particle"></i><i class="particle"></i></div>

  <main class="shell">
    <header class="topbar">
      <div class="brand"><span class="brand-mark">HV</span><span>Power Asset Intelligence</span></div>
      <div class="system-chip">Live AI Health Index Console</div>
    </header>

    <section class="hero">
      <div class="hero-inner">
        <h1>Transformer Health Monitoring System</h1>
        <p class="subtitle">AI-based Health Index Prediction for Transformers</p>
        <div class="hero-credit">Created by Satya</div>
        <div class="status-row" aria-label="system highlights">
          <div class="status-card"><strong>DGA</strong><span>Seven gas input model</span></div>
          <div class="status-card"><strong>BDV</strong><span>Oil insulation strength</span></div>
          <div class="status-card"><strong>OTI / WTI</strong><span>Thermal condition monitoring</span></div>
        </div>
      </div>
    </section>

    <div class="cta-wrap">
      <button class="insert-btn" id="openModal" type="button">Insert Data</button>
    </div>
  </main>

  <section class="modal" id="dataModal" aria-hidden="true">
    <div class="modal-card" role="dialog" aria-modal="true" aria-labelledby="modalTitle">
      <div class="modal-head">
        <h2 id="modalTitle">Transformer Health Input</h2>
        <div class="modal-actions">
          <button class="home-btn" id="mainPageBtn" type="button">Main Page</button>
          <button class="close-btn" id="closeModal" type="button" aria-label="Close">x</button>
        </div>
      </div>
      <form id="healthForm">
        <section class="form-section">
          <h3>Transformer Details</h3>
          <div class="fields">
            <label><span>Substation Name</span><input name="SubstationName" type="text" value="Substation-1"></label>
            <label><span>Rating(MVA)</span><input name="RatingMVA" type="number" step="0.1" value="100"></label>
            <label><span>Voltage Ratio(kV)</span><input name="VoltageRatio" type="text" value="220/132 kV"></label>
          </div>
        </section>

        <section class="form-section">
          <h3>DGA Inputs</h3>
          <div class="fields">
            <label><span>H2 Hydrogen ppm</span><input name="H2" type="number" step="0.001" value="80"></label>
            <label><span>CH4 Methane ppm</span><input name="Methane" type="number" step="0.001" value="85"></label>
            <label><span>C2H6 Ethane ppm</span><input name="Ethane" type="number" step="0.001" value="55"></label>
            <label><span>C2H4 Ethylene ppm</span><input name="Ethylene" type="number" step="0.001" value="18"></label>
            <label><span>C2H2 Acetylene ppm</span><input name="Acetylene" type="number" step="0.001" value="0.5"></label>
            <label><span>CO Carbon Monoxide ppm</span><input name="CO" type="number" step="0.001" value="430"></label>
            <label><span>CO2 Carbon Dioxide ppm</span><input name="CO2" type="number" step="0.001" value="5000"></label>
          </div>
        </section>

        <section class="form-section">
          <h3>Oil &amp; Temperature Inputs</h3>
          <div class="fields">
            <label><span>BDV value in kV</span><input name="BDV" type="number" step="0.001" value="60"></label>
            <label><span>Maximum OTI in °C</span><input name="OTI" type="number" step="0.1" value="72"></label>
            <label><span>Maximum WTI in °C</span><input name="WTI" type="number" step="0.1" value="82"></label>
          </div>
        </section>

        <div class="actions">
          <button class="predict-btn" type="submit">Predict Health Index</button>
        </div>
      </form>
      <footer class="modal-footer">Created by Satya</footer>
    </div>
  </section>

  <script>
    const modal = document.getElementById('dataModal');
    const form = document.getElementById('healthForm');

    document.getElementById('openModal').addEventListener('click', () => {
      modal.classList.add('open');
      modal.setAttribute('aria-hidden', 'false');
    });

    document.getElementById('closeModal').addEventListener('click', closeModal);
    document.getElementById('mainPageBtn').addEventListener('click', closeModal);
    modal.addEventListener('click', event => {
      if (event.target === modal) closeModal();
    });

    function closeModal() {
      modal.classList.remove('open');
      modal.setAttribute('aria-hidden', 'true');
    }

    form.addEventListener('submit', event => {
      event.preventDefault();
      const params = new URLSearchParams(new FormData(form));
      window.open('/result?' + params.toString(), '_blank');
    });
  </script>
</body>
</html>"""


class AppHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            body = render_index().encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if parsed.path == "/result":
            query = parse_qs(parsed.query)
            body = render_result_page(query, parsed.query).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if parsed.path == "/result.pdf":
            query = parse_qs(parsed.query)
            body, filename = render_result_pdf(query)
            self.send_response(200)
            self.send_header("Content-Type", "application/pdf")
            self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if parsed.path == "/api/predict":
            query = parse_qs(parsed.query)
            values = {key: float(query.get(key, [DEFAULT_INPUT[key]])[0]) for key in DEFAULT_INPUT}
            json_response(self, predict_payload(values))
            return
        json_response(self, {"error": "Not found"}, status=404)

    def log_message(self, format: str, *args) -> None:
        return


def main() -> None:
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8000"))
    server = ThreadingHTTPServer((host, port), AppHandler)
    print(f"Transformer Health Monitoring System running at http://{host}:{port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
