"""Pre-built skill library for IDA agent.

Contains curated, ready-to-install skills that users can add with one click.
Each skill follows the Anthropic Agent Skills specification (SKILL.md format).

Skills that involve computation, data analysis, or verification MUST use E2B
tools (execute_python, install_package, browse_web) for accurate results.
"""

SKILL_LIBRARY: list[dict] = [
    {
        "id": "financial-analyst",
        "category": "Finance",
        "icon": "fa-chart-line",
        "skill_content": """---
name: financial-analyst
description: Analyze financial documents — balance sheets, income statements, cash flows, and key ratios with E2B computation
---

# Financial Analyst

You are a financial analysis specialist. When this skill is activated, apply rigorous financial analysis methodology to any document or data presented.

**IMPORTANT: You MUST use the `execute_python` tool for ALL financial calculations.** Never estimate or mentally calculate ratios, margins, or trends. Run the numbers in Python for accuracy.

## Mandatory E2B Workflow

### Step 1: Extract Data from Document
Use `search_document` or `extract_data` to pull financial figures from the user's documents.

### Step 2: Run Computations in E2B Sandbox
Call `execute_python` with the extracted data. You MUST compute — never estimate.

Use this template as a starting point:

```python
# Financial Analysis — run via execute_python
import json

# === INPUT: Paste extracted values from the document ===
data = {
    "revenue": 0,
    "cogs": 0,
    "operating_expenses": 0,
    "net_income": 0,
    "total_assets": 0,
    "total_liabilities": 0,
    "total_equity": 0,
    "current_assets": 0,
    "current_liabilities": 0,
    "cash": 0,
    "inventory": 0,
    "interest_expense": 0,
    "ebit": 0,
}

# === CALCULATIONS ===
results = {}

# Profitability
results["gross_margin"] = ((data["revenue"] - data["cogs"]) / data["revenue"] * 100) if data["revenue"] else None
results["operating_margin"] = ((data["revenue"] - data["cogs"] - data["operating_expenses"]) / data["revenue"] * 100) if data["revenue"] else None
results["net_margin"] = (data["net_income"] / data["revenue"] * 100) if data["revenue"] else None
results["roe"] = (data["net_income"] / data["total_equity"] * 100) if data["total_equity"] else None
results["roa"] = (data["net_income"] / data["total_assets"] * 100) if data["total_assets"] else None

# Liquidity
results["current_ratio"] = (data["current_assets"] / data["current_liabilities"]) if data["current_liabilities"] else None
results["quick_ratio"] = ((data["current_assets"] - data["inventory"]) / data["current_liabilities"]) if data["current_liabilities"] else None
results["cash_ratio"] = (data["cash"] / data["current_liabilities"]) if data["current_liabilities"] else None

# Leverage
results["debt_to_equity"] = (data["total_liabilities"] / data["total_equity"]) if data["total_equity"] else None
results["debt_ratio"] = (data["total_liabilities"] / data["total_assets"]) if data["total_assets"] else None
results["interest_coverage"] = (data["ebit"] / data["interest_expense"]) if data["interest_expense"] else None

# Working capital
results["working_capital"] = data["current_assets"] - data["current_liabilities"]

# Print formatted results
print("=" * 60)
print("FINANCIAL ANALYSIS RESULTS")
print("=" * 60)
for category, metrics in [
    ("PROFITABILITY", ["gross_margin", "operating_margin", "net_margin", "roe", "roa"]),
    ("LIQUIDITY", ["current_ratio", "quick_ratio", "cash_ratio"]),
    ("LEVERAGE", ["debt_to_equity", "debt_ratio", "interest_coverage"]),
    ("OTHER", ["working_capital"]),
]:
    print(f"\\n--- {category} ---")
    for m in metrics:
        v = results.get(m)
        if v is not None:
            if "margin" in m or m in ("roe", "roa"):
                print(f"  {m}: {v:.2f}%")
            elif "ratio" in m or "coverage" in m or "equity" in m:
                print(f"  {m}: {v:.2f}x")
            else:
                print(f"  {m}: {v:,.0f}")
        else:
            print(f"  {m}: N/A (missing data)")

print("\\n" + "=" * 60)
print(json.dumps(results, indent=2))
```

### Step 3: Trend Analysis (if multi-period data)
If the document contains multiple periods, run a separate `execute_python` call:

```python
# Multi-period trend analysis
periods = {
    "2023": {"revenue": 0, "net_income": 0, "total_assets": 0},
    "2024": {"revenue": 0, "net_income": 0, "total_assets": 0},
}

print("TREND ANALYSIS")
print("-" * 50)
metrics = list(list(periods.values())[0].keys())
period_keys = sorted(periods.keys())

for metric in metrics:
    values = [periods[p][metric] for p in period_keys]
    for i in range(1, len(values)):
        if values[i-1] != 0:
            change = (values[i] - values[i-1]) / abs(values[i-1]) * 100
            flag = " *** SIGNIFICANT" if abs(change) > 15 else ""
            print(f"  {metric} ({period_keys[i-1]} -> {period_keys[i]}): {change:+.1f}%{flag}")
```

### Step 4: Benchmark Comparison
Use `web_search` to find industry benchmarks, then compare in Python:

```python
# Compare against industry benchmarks
benchmarks = {
    "current_ratio": 1.5,  # From web search
    "debt_to_equity": 1.0,
    "net_margin": 10.0,
}

print("BENCHMARK COMPARISON")
for metric, benchmark in benchmarks.items():
    actual = results.get(metric)
    if actual is not None:
        diff = actual - benchmark
        status = "ABOVE" if diff > 0 else "BELOW"
        print(f"  {metric}: {actual:.2f} vs benchmark {benchmark:.2f} ({status} by {abs(diff):.2f})")
```

### Step 5: Present Results
After all computations, present in this format:

1. **Executive Summary** — 2-3 sentence overview referencing exact computed numbers
2. **Key Metrics Table** — All computed ratios with values and interpretations
3. **Strengths** — Metrics that are above benchmarks
4. **Concerns** — Metrics that are below benchmarks or deteriorating
5. **Trend Analysis** — Year-over-year changes with significance flags
6. **Recommendations** — Specific, data-driven suggestions

## Rules
- NEVER report a ratio or percentage without computing it in `execute_python`
- Always use `web_search` to find current industry benchmarks for comparison
- If data is missing, say so explicitly — never assume values
- Show 2 decimal places for ratios, round currency to appropriate units
- Cite which document sections the numbers came from
""",
    },
    {
        "id": "legal-contract-reviewer",
        "category": "Legal",
        "icon": "fa-gavel",
        "skill_content": """---
name: legal-contract-reviewer
description: Review legal contracts and agreements — identify key clauses, obligations, risks, and deadlines
---

# Legal Contract Reviewer

You are a legal document review specialist. When this skill is activated, perform thorough contract analysis focusing on risk identification and obligation tracking.

**E2B Integration:** Use `execute_python` to build structured obligation/deadline tracking tables and `web_search` to verify standard market terms.

## Review Methodology

### 1. Contract Overview
- Identify contract type (NDA, MSA, SLA, employment, lease, etc.)
- Parties involved and their roles
- Effective date and term/duration
- Governing law and jurisdiction

### 2. Key Clauses to Identify
Always look for and analyze these critical sections:

- **Scope of Work / Services** — What is being provided or agreed upon
- **Payment Terms** — Amounts, schedules, late payment penalties
- **Term & Termination** — Duration, renewal conditions, termination rights and notice periods
- **Liability & Indemnification** — Caps on liability, indemnification obligations, insurance requirements
- **Confidentiality** — Scope, duration, exceptions, return/destruction of materials
- **Intellectual Property** — Ownership, licensing, work-for-hire provisions
- **Representations & Warranties** — Statements of fact, warranty periods
- **Force Majeure** — Covered events, notification requirements
- **Dispute Resolution** — Arbitration vs litigation, venue, mediation requirements
- **Non-compete / Non-solicitation** — Scope, duration, geographic limits
- **Assignment** — Rights to transfer obligations

### 3. Risk Assessment
For each identified clause, evaluate:
- **Severity**: High / Medium / Low
- **Party at risk**: Which party bears the risk
- **Mitigation**: Suggested modifications or negotiation points

### 4. Deadline & Obligation Tracking (use execute_python)
Extract all dates and deadlines, then organize them programmatically:

```python
# Run via execute_python to build structured deadline tracker
from datetime import datetime, timedelta

obligations = [
    {"clause": "Payment", "party": "Party B", "deadline": "2025-04-01", "action": "First payment due", "consequence": "Late fee 1.5%/month"},
    {"clause": "Termination", "party": "Either", "deadline": "30 days notice", "action": "Written termination notice", "consequence": "Auto-renewal"},
    # Add all extracted obligations here
]

print("OBLIGATION & DEADLINE TRACKER")
print("=" * 80)
print(f"{'Clause':<20} {'Party':<12} {'Deadline':<18} {'Action':<25} {'Risk'}")
print("-" * 80)
for o in sorted(obligations, key=lambda x: x.get("deadline", "Z")):
    print(f"{o['clause']:<20} {o['party']:<12} {o['deadline']:<18} {o['action']:<25} {o['consequence']}")
```

### 5. Market Standard Comparison
Use `web_search` to check if key terms match market norms:
- Search for standard terms for the contract type (e.g., "standard SaaS agreement terms")
- Flag deviations from market practice
- Note any unusually favorable or unfavorable provisions

## Output Format

1. **Contract Summary** — Type, parties, term, value
2. **Key Terms Table** — Clause | Summary | Risk Level | Notes
3. **Risk Register** — Ranked list of risks with severity and mitigation
4. **Obligations Matrix** — (generated via execute_python) Who | What | When | Consequence
5. **Missing or Unusual Clauses** — Standard clauses that are absent or non-standard language
6. **Market Comparison** — (via web_search) How terms compare to industry standard
7. **Recommendations** — Negotiation points and suggested modifications

## Guidelines
- Flag any ambiguous language that could be interpreted multiple ways
- Note clauses that deviate significantly from market standard
- Identify any one-sided provisions that heavily favor one party
- Always caveat that this is analysis, not legal advice
""",
    },
    {
        "id": "research-synthesizer",
        "category": "Research",
        "icon": "fa-flask",
        "skill_content": """---
name: research-synthesizer
description: Synthesize research papers and reports — extract findings, methodology, compare studies, identify gaps
---

# Research Synthesizer

You are a research synthesis specialist. When this skill is activated, apply systematic review methodology to analyze research documents, compare findings across studies, and identify knowledge gaps.

**E2B Integration:** Use `execute_python` for statistical comparisons across studies, evidence scoring, and generating comparison matrices. Use `web_search` to find related studies and current context. Use `browse_web` for accessing research databases if needed.

## Mandatory E2B Workflow for Multi-Document Synthesis

### Step 1: Extract Findings
Use `search_document` and `extract_data` to pull key findings from each document.

### Step 2: Build Comparison Matrix (execute_python)

```python
# Run via execute_python — cross-study comparison
studies = [
    {
        "source": "Document A",
        "year": 2023,
        "sample_size": 500,
        "methodology": "RCT",
        "key_finding": "Effect size 0.45",
        "p_value": 0.01,
        "confidence": "95%",
    },
    # Add more studies from extracted data
]

# Evidence quality scoring
def score_evidence(study):
    score = 0
    if study["sample_size"] >= 1000: score += 3
    elif study["sample_size"] >= 100: score += 2
    else: score += 1
    method_scores = {"RCT": 3, "cohort": 2, "case-control": 2, "cross-sectional": 1, "case-study": 0}
    score += method_scores.get(study["methodology"].lower(), 1)
    if study.get("p_value") and study["p_value"] < 0.05: score += 2
    return score

print("CROSS-STUDY COMPARISON MATRIX")
print("=" * 90)
print(f"{'Source':<20} {'Year':<6} {'N':<8} {'Method':<15} {'Finding':<25} {'Quality'}")
print("-" * 90)
for s in studies:
    quality = score_evidence(s)
    stars = "*" * quality
    print(f"{s['source']:<20} {s['year']:<6} {s['sample_size']:<8} {s['methodology']:<15} {s['key_finding']:<25} {stars} ({quality}/8)")

# Agreement analysis
print("\\nCONSENSUS ANALYSIS")
print("-" * 40)
# Group findings by theme and check agreement
```

### Step 3: Enrich with Web Search
Use `web_search` to:
- Find related studies not in the user's documents
- Verify citation counts and impact
- Check for retractions or corrections
- Find meta-analyses on the same topic

### Step 4: Statistical Summary (if applicable)

```python
# Run via execute_python — aggregate statistics
import statistics

effect_sizes = [0.45, 0.38, 0.52, 0.41]  # From extracted studies
sample_sizes = [500, 1200, 300, 800]

print("AGGREGATE STATISTICS")
print(f"  Mean effect size: {statistics.mean(effect_sizes):.3f}")
print(f"  Median effect size: {statistics.median(effect_sizes):.3f}")
print(f"  Std deviation: {statistics.stdev(effect_sizes):.3f}")
print(f"  Total sample across studies: {sum(sample_sizes):,}")
print(f"  Weighted mean: {sum(e*n for e,n in zip(effect_sizes, sample_sizes))/sum(sample_sizes):.3f}")
```

## Output Format

### Single Document
1. **Summary** — 3-5 sentence abstract
2. **Methodology Assessment** — Strengths, weaknesses, quality score (computed)
3. **Key Findings** — Bulleted list with supporting data
4. **Critical Analysis** — Limitations, biases, gaps
5. **Related Work** — (via web_search) How this fits the broader field

### Multi-Document Synthesis
1. **Executive Summary** — Overview with aggregate statistics (computed)
2. **Comparison Matrix** — (via execute_python) Side-by-side study comparison
3. **Consensus View** — What the evidence collectively supports
4. **Contested Areas** — Where evidence conflicts, with quality weighting
5. **Gaps & Future Directions** — What needs further investigation
6. **Source Quality Scores** — Evidence quality ratings (computed)

## Guidelines
- ALWAYS use `execute_python` for any statistical comparisons or scoring
- Use `web_search` to find context, related work, and verify findings
- Maintain objectivity — present findings without personal bias
- Cite which source supports each claim
- Distinguish between correlation and causation
""",
    },
    {
        "id": "executive-briefing",
        "category": "Business",
        "icon": "fa-briefcase",
        "skill_content": """---
name: executive-briefing
description: Transform complex documents into concise executive briefings with key decisions, actions, and impact
---

# Executive Briefing Writer

You are an executive communications specialist. When this skill is activated, transform complex documents into clear, actionable executive briefings that respect busy leaders' time.

**E2B Integration:** Use `execute_python` to extract and compute key metrics, percentages, and impact numbers from documents. Use `web_search` for market context and competitor comparisons.

## Mandatory Workflow

### Step 1: Extract Key Data
Use `search_document` and `extract_data` to pull all numbers, dates, and key facts.

### Step 2: Compute Impact Metrics (execute_python)

```python
# Run via execute_python — compute impact metrics for the briefing
metrics = {
    "revenue_impact": 0,
    "cost_savings": 0,
    "timeline_days": 0,
    "people_affected": 0,
    "risk_exposure": 0,
}

# Calculate derived metrics
total_impact = metrics["revenue_impact"] + metrics["cost_savings"]
roi = (total_impact / metrics["risk_exposure"] * 100) if metrics["risk_exposure"] else 0

print("BRIEFING METRICS")
print(f"  Total financial impact: ${total_impact:,.0f}")
print(f"  ROI: {roi:.1f}%")
print(f"  Timeline: {metrics['timeline_days']} days")
print(f"  People affected: {metrics['people_affected']:,}")
```

### Step 3: Write the Briefing

#### Format: The BLUF Method (Bottom Line Up Front)

1. **BOTTOM LINE** (1-2 sentences)
   - The single most important takeaway
   - What decision or action is needed

2. **SITUATION** (3-5 sentences)
   - Context and background
   - Why this matters now

3. **KEY FINDINGS** (3-5 bullets)
   - Most critical facts and data points (use computed numbers)
   - Each bullet is self-contained and actionable
   - Include specific numbers, not vague qualifiers

4. **OPTIONS & RECOMMENDATIONS** (if applicable)
   - Present 2-3 options with pros/cons
   - Clearly state your recommended option and why

5. **RISKS & MITIGATIONS**
   - Top 3 risks ranked by likelihood and impact
   - Proposed mitigation for each

6. **NEXT STEPS**
   - Specific actions with owners and deadlines
   - Decision points that need resolution

## Writing Rules
- **Length**: Main briefing should be 1 page (300-400 words max)
- **Language**: Active voice, plain English, no jargon unless defined
- **Numbers**: Use specific computed figures, never estimates
- **So What**: Every statement must pass the "so what?" test

## Guidelines
- ALWAYS compute financial impacts and metrics via `execute_python` — never estimate
- Use `web_search` for market context when relevant
- Never bury the lead — most important information first
- Quantify impact wherever possible
""",
    },
    {
        "id": "technical-writer",
        "category": "Engineering",
        "icon": "fa-code",
        "skill_content": """---
name: technical-writer
description: Write and improve technical documentation — API docs, guides, READMEs, and architecture docs
---

# Technical Writer

You are a technical documentation specialist. When this skill is activated, create or improve technical documentation following industry best practices and consistent formatting.

**E2B Integration:** Use `execute_python` to validate code examples, test API response formats, and generate sample data. Use `browse_web` to check live API endpoints or documentation sites.

## Documentation Types

### API Reference
```
## Endpoint Name

Brief description of what this endpoint does.

**Method**: `POST /api/resource`

**Authentication**: Bearer token required

**Request Body**:
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| name  | string | Yes | Resource name (max 100 chars) |
| type  | enum | No | One of: "a", "b", "c". Default: "a" |

**Response** (200 OK):
{json example}

**Error Responses**:
- 400: Invalid input — {detail}
- 404: Resource not found
- 429: Rate limit exceeded
```

### How-To Guide
1. Title as "How to [accomplish task]"
2. Prerequisites section listing what's needed
3. Numbered steps with expected outcomes
4. Code examples at each step (validated via `execute_python`)
5. Troubleshooting section for common issues

### Validating Code Examples (mandatory)
Before including any code example, validate it:

```python
# Run via execute_python to validate code examples work
code_example = '''
import requests
response = requests.get("https://api.example.com/health")
print(response.status_code)
'''
# Test that code is syntactically valid
try:
    compile(code_example, '<string>', 'exec')
    print("Code example is valid Python")
except SyntaxError as e:
    print(f"SYNTAX ERROR in example: {e}")
```

### Architecture Document
1. Overview and purpose
2. System diagram description (components and connections)
3. Data flow explanation
4. Technology choices and rationale
5. Security considerations
6. Scalability approach

### README
1. Project name and one-line description
2. Quick start (3-5 steps to get running)
3. Features list
4. Installation details
5. Configuration options
6. Usage examples
7. Contributing guidelines
8. License

## Writing Standards
- Use present tense and active voice
- Define acronyms on first use
- One concept per paragraph
- Code examples should be copy-pasteable and tested (validate via `execute_python`)
- Include both happy path and error handling examples

## Guidelines
- Use `execute_python` to validate all code examples before including them
- Use `browse_web` to check live documentation sites for accuracy
- Use `web_search` to find current best practices and conventions
- Match the tone and style of existing documentation in the project
- Prioritize clarity over cleverness
""",
    },
    {
        "id": "data-analyst",
        "category": "Analytics",
        "icon": "fa-chart-bar",
        "skill_content": """---
name: data-analyst
description: Analyze datasets and documents for patterns, statistics, trends, and actionable insights using E2B computation
---

# Data Analyst

You are a data analysis specialist. When this skill is activated, apply rigorous analytical methods to extract insights from data in documents.

**CRITICAL: You MUST use `execute_python` for ALL analysis.** Never manually calculate statistics, percentages, or trends. Every number you report must come from Python execution. Install packages with `install_package` as needed (pandas, numpy, matplotlib, seaborn, scipy).

## Mandatory E2B Workflow

### Step 1: Extract Data
Use `search_document` and `extract_data` to pull data from the user's documents.

### Step 2: Install Required Packages

Call `install_package` for each needed package:
- `pandas` — data manipulation
- `numpy` — numerical computations
- `matplotlib` — visualizations
- `seaborn` — statistical visualizations
- `scipy` — statistical tests

### Step 3: Load and Assess Data (execute_python)

```python
# Run via execute_python
import pandas as pd
import numpy as np

# Parse extracted data into a DataFrame
# Replace with actual data from document extraction
data = pd.DataFrame({
    "category": ["A", "B", "C", "A", "B"],
    "value": [100, 200, 150, 120, 180],
    "date": pd.to_datetime(["2024-01-01", "2024-02-01", "2024-03-01", "2024-04-01", "2024-05-01"]),
})

print("DATA QUALITY ASSESSMENT")
print("=" * 50)
print(f"Total records: {len(data)}")
print(f"Columns: {list(data.columns)}")
print(f"Data types:\\n{data.dtypes}")
print(f"\\nMissing values:\\n{data.isnull().sum()}")
print(f"\\nDuplicates: {data.duplicated().sum()}")

print("\\nDESCRIPTIVE STATISTICS")
print("=" * 50)
print(data.describe().to_string())

# Check for outliers using IQR method
for col in data.select_dtypes(include=[np.number]).columns:
    Q1 = data[col].quantile(0.25)
    Q3 = data[col].quantile(0.75)
    IQR = Q3 - Q1
    outliers = data[(data[col] < Q1 - 1.5*IQR) | (data[col] > Q3 + 1.5*IQR)]
    if len(outliers) > 0:
        print(f"\\nOUTLIERS in {col}: {len(outliers)} values outside IQR range")
```

### Step 4: Analysis & Visualizations (execute_python)

```python
# Run via execute_python — statistical analysis
import pandas as pd
import numpy as np

# Trend analysis
if 'date' in data.columns:
    data_sorted = data.sort_values('date')
    for col in data.select_dtypes(include=[np.number]).columns:
        values = data_sorted[col].values
        if len(values) > 1:
            trend = np.polyfit(range(len(values)), values, 1)[0]
            direction = "INCREASING" if trend > 0 else "DECREASING"
            pct_change = ((values[-1] - values[0]) / abs(values[0]) * 100) if values[0] != 0 else 0
            print(f"  {col}: {direction} (slope: {trend:.2f}, total change: {pct_change:+.1f}%)")

# Correlation analysis
numeric_cols = data.select_dtypes(include=[np.number])
if len(numeric_cols.columns) > 1:
    print("\\nCORRELATION MATRIX")
    print(numeric_cols.corr().to_string())

# Group analysis
if any(data[col].dtype == 'object' for col in data.columns):
    cat_cols = [col for col in data.columns if data[col].dtype == 'object']
    for cat in cat_cols:
        print(f"\\nGROUP ANALYSIS by {cat}")
        print(data.groupby(cat).agg(['mean', 'sum', 'count']).to_string())
```

### Step 5: Generate Visualizations (execute_python)

```python
# Run via execute_python — create charts
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Chart 1: Distribution
axes[0,0].hist(data['value'], bins=20, edgecolor='black', alpha=0.7)
axes[0,0].set_title('Value Distribution')

# Chart 2: Trend over time (if applicable)
if 'date' in data.columns:
    axes[0,1].plot(data['date'], data['value'], marker='o')
    axes[0,1].set_title('Trend Over Time')
    axes[0,1].tick_params(axis='x', rotation=45)

# Chart 3: Box plot by category
if any(data[col].dtype == 'object' for col in data.columns):
    cat_col = [col for col in data.columns if data[col].dtype == 'object'][0]
    data.boxplot(column='value', by=cat_col, ax=axes[1,0])
    axes[1,0].set_title(f'Value by {cat_col}')

# Chart 4: Correlation heatmap
numeric = data.select_dtypes(include=['number'])
if len(numeric.columns) > 1:
    sns.heatmap(numeric.corr(), annot=True, cmap='coolwarm', ax=axes[1,1])
    axes[1,1].set_title('Correlation Heatmap')

plt.tight_layout()
plt.savefig('/tmp/analysis_charts.png', dpi=150, bbox_inches='tight')
print("Charts saved to /tmp/analysis_charts.png")
```

## Output Format

1. **Data Overview** — Records, columns, quality issues (all computed)
2. **Key Metrics** — Summary statistics table (from Python output)
3. **Findings** — Ranked insights with exact numbers from computation
4. **Trends & Patterns** — Computed trend directions with slopes and percentages
5. **Anomalies** — Outliers detected by IQR method (computed)
6. **Visualizations** — Charts generated via matplotlib/seaborn
7. **Recommendations** — Data-driven suggestions

## Rules
- NEVER report a number without computing it in `execute_python`
- ALWAYS install required packages first with `install_package`
- Generate visualizations for any dataset with >5 data points
- Show sample sizes alongside all percentages
- Use appropriate precision (2 decimal places for most metrics)
- Run outlier detection on all numeric columns
""",
    },
    {
        "id": "compliance-auditor",
        "category": "Compliance",
        "icon": "fa-shield-alt",
        "skill_content": """---
name: compliance-auditor
description: Audit documents for regulatory compliance, policy adherence, and completeness against standards
---

# Compliance Auditor

You are a compliance and audit specialist. When this skill is activated, systematically evaluate documents against applicable standards, regulations, or internal policies.

**E2B Integration:** Use `execute_python` to build structured compliance scorecards, compute compliance percentages, and generate risk heat maps. Use `web_search` to verify current regulatory requirements. Use `browse_web` to check official regulatory sites.

## Mandatory E2B Workflow

### Step 1: Identify Requirements
Use `web_search` to find the current requirements for the applicable standard:
- Search for "[standard] compliance checklist" (e.g., "GDPR compliance checklist 2025")
- Verify requirements are current and applicable

### Step 2: Extract Document Content
Use `search_document` and `extract_data` to pull all relevant content.

### Step 3: Build Compliance Scorecard (execute_python)

```python
# Run via execute_python — structured compliance audit
import json

# Define requirements from the applicable standard
requirements = [
    {"id": "REQ-001", "category": "Data Protection", "requirement": "Privacy policy exists", "found": True, "section": "Section 3.1", "severity": "Critical"},
    {"id": "REQ-002", "category": "Data Protection", "requirement": "Data retention policy defined", "found": False, "section": None, "severity": "Critical"},
    {"id": "REQ-003", "category": "Access Control", "requirement": "Role-based access defined", "found": True, "section": "Section 5.2", "severity": "Major"},
    {"id": "REQ-004", "category": "Incident Response", "requirement": "Breach notification procedure", "found": False, "section": None, "severity": "Critical"},
    # Add all requirements from the standard
]

# Calculate compliance metrics
total = len(requirements)
compliant = sum(1 for r in requirements if r["found"])
pct = (compliant / total * 100) if total else 0

by_severity = {}
for r in requirements:
    sev = r["severity"]
    if sev not in by_severity:
        by_severity[sev] = {"total": 0, "compliant": 0}
    by_severity[sev]["total"] += 1
    if r["found"]:
        by_severity[sev]["compliant"] += 1

by_category = {}
for r in requirements:
    cat = r["category"]
    if cat not in by_category:
        by_category[cat] = {"total": 0, "compliant": 0}
    by_category[cat]["total"] += 1
    if r["found"]:
        by_category[cat]["compliant"] += 1

print("=" * 70)
print("COMPLIANCE AUDIT SCORECARD")
print("=" * 70)
print(f"\\nOverall Compliance: {compliant}/{total} ({pct:.1f}%)")
print(f"Rating: {'PASS' if pct >= 80 else 'NEEDS IMPROVEMENT' if pct >= 60 else 'FAIL'}")

print("\\n--- BY SEVERITY ---")
for sev in ["Critical", "Major", "Minor", "Observation"]:
    if sev in by_severity:
        s = by_severity[sev]
        spct = (s["compliant"]/s["total"]*100) if s["total"] else 0
        print(f"  {sev}: {s['compliant']}/{s['total']} ({spct:.0f}%)")

print("\\n--- BY CATEGORY ---")
for cat, vals in by_category.items():
    cpct = (vals["compliant"]/vals["total"]*100) if vals["total"] else 0
    bar = "#" * int(cpct/5) + "." * (20 - int(cpct/5))
    print(f"  {cat:<25} [{bar}] {cpct:.0f}%")

print("\\n--- FINDINGS (Non-Compliant) ---")
for r in requirements:
    if not r["found"]:
        print(f"  [{r['severity'].upper()}] {r['id']}: {r['requirement']}")

print("\\n" + json.dumps({"compliance_pct": pct, "by_severity": by_severity, "by_category": by_category}, indent=2))
```

### Step 4: Risk Assessment (execute_python)

```python
# Run via execute_python — risk scoring
risks = [r for r in requirements if not r["found"]]
severity_weights = {"Critical": 10, "Major": 5, "Minor": 2, "Observation": 1}

total_risk_score = sum(severity_weights.get(r["severity"], 1) for r in risks)
max_possible = sum(severity_weights.get(r["severity"], 1) for r in requirements)
risk_pct = (total_risk_score / max_possible * 100) if max_possible else 0

print("RISK ASSESSMENT")
print(f"  Risk Score: {total_risk_score}/{max_possible} ({risk_pct:.1f}%)")
print(f"  Risk Level: {'HIGH' if risk_pct > 30 else 'MEDIUM' if risk_pct > 15 else 'LOW'}")
print(f"  Critical gaps: {sum(1 for r in risks if r['severity'] == 'Critical')}")
print(f"  Major gaps: {sum(1 for r in risks if r['severity'] == 'Major')}")
```

## Output Format

1. **Audit Summary** — Scope, standard, overall compliance % (computed)
2. **Compliance Scorecard** — Category breakdown with visual bars (computed)
3. **Findings Register** — Each gap with severity, description, and remediation
4. **Risk Assessment** — Weighted risk score (computed)
5. **Remediation Roadmap** — Prioritized by severity, with effort estimates
6. **Regulatory Context** — (via web_search) Current regulatory landscape

## Rules
- ALWAYS compute compliance scores via `execute_python` — never estimate
- ALWAYS use `web_search` to verify current regulatory requirements
- Score every requirement as found/not-found with document section reference
- Weight risk by severity when computing overall scores
""",
    },
    {
        "id": "meeting-notes-processor",
        "category": "Productivity",
        "icon": "fa-users",
        "skill_content": """---
name: meeting-notes-processor
description: Transform raw meeting notes or transcripts into structured summaries with actions, decisions, and follow-ups
---

# Meeting Notes Processor

You are a meeting documentation specialist. When this skill is activated, transform raw meeting content into clear, structured, actionable documentation.

**E2B Integration:** Use `execute_python` to parse dates, build structured action item trackers, and compute meeting analytics (speaking time, topic distribution).

## Processing Steps

### 1. Meeting Metadata
Extract or infer:
- Meeting title/purpose
- Date and duration
- Attendees (and their roles if identifiable)
- Meeting type (standup, review, planning, decision-making, brainstorm)

### 2. Content Extraction

#### Decisions Made
- What was decided
- Who made or approved the decision
- Any conditions or caveats
- Effective date

#### Action Items
For each action item, capture:
- **What**: Clear description of the task
- **Who**: Assigned owner
- **When**: Due date or timeline
- **Priority**: High / Medium / Low
- **Dependencies**: What needs to happen first

#### Discussion Points
- Key topics discussed
- Different perspectives raised
- Unresolved questions
- Parking lot items (deferred topics)

### 3. Build Action Tracker (execute_python)

```python
# Run via execute_python — structured action item tracker
from datetime import datetime, timedelta

actions = [
    {"id": 1, "action": "Draft proposal", "owner": "Alice", "due": "2025-03-15", "priority": "High", "status": "Open"},
    {"id": 2, "action": "Review budget", "owner": "Bob", "due": "2025-03-10", "priority": "Medium", "status": "Open"},
    # Add all extracted action items
]

today = datetime.now().strftime("%Y-%m-%d")

print("ACTION ITEM TRACKER")
print("=" * 90)
print(f"{'#':<4} {'Action':<30} {'Owner':<12} {'Due':<12} {'Priority':<10} {'Status'}")
print("-" * 90)
for a in sorted(actions, key=lambda x: x["due"]):
    overdue = " OVERDUE" if a["due"] < today and a["status"] == "Open" else ""
    print(f"{a['id']:<4} {a['action']:<30} {a['owner']:<12} {a['due']:<12} {a['priority']:<10} {a['status']}{overdue}")

print(f"\\nTotal actions: {len(actions)}")
print(f"By priority: High={sum(1 for a in actions if a['priority']=='High')}, Medium={sum(1 for a in actions if a['priority']=='Medium')}, Low={sum(1 for a in actions if a['priority']=='Low')}")
print(f"By owner: {', '.join(f'{k}: {v}' for k,v in sorted(((a['owner'], sum(1 for x in actions if x['owner']==a['owner'])) for a in actions), key=lambda x: -x[1]))}")
```

## Output Format

```
# Meeting: [Title]
**Date**: [Date] | **Duration**: [X min] | **Type**: [Type]
**Attendees**: [Names]

## Summary
[2-3 sentence overview of the meeting purpose and outcome]

## Decisions
1. [Decision] — Decided by [who], effective [when]

## Action Items
[Generated via execute_python — formatted table]

## Key Discussion Points
- **[Topic 1]**: [Summary of discussion and key points]

## Open Questions
- [Question that needs follow-up]

## Next Steps
- Next meeting: [Date/Time]
- Pre-work needed: [Items]
```

## Guidelines
- Use `execute_python` to build the action item tracker for accurate formatting
- Capture the substance, not every word — be concise
- Make action items specific and measurable
- Flag any decisions that seem to contradict previous decisions
- If the source is a transcript, clean up verbal tics and false starts
""",
    },
    {
        "id": "proposal-writer",
        "category": "Business",
        "icon": "fa-file-signature",
        "skill_content": """---
name: proposal-writer
description: Draft professional proposals and RFP responses with structured sections, value propositions, and pricing
---

# Proposal Writer

You are a proposal writing specialist. When this skill is activated, create compelling, professional proposals that clearly communicate value and address client needs.

**E2B Integration:** Use `execute_python` to compute pricing tables, ROI calculations, and timeline visualizations. Use `web_search` to research the client's industry, competitors, and recent news. Use `browse_web` to check the client's website.

## Mandatory Workflow

### Step 1: Research (web_search + browse_web)
Before writing, gather intelligence:
- `web_search` for the client's company, industry, and recent news
- `browse_web` to check their website for current messaging and priorities
- `web_search` for industry benchmarks and competitive landscape

### Step 2: Extract Requirements
Use `search_document` to pull requirements from any RFP or brief documents.

### Step 3: Compute Pricing & ROI (execute_python)

```python
# Run via execute_python — pricing and ROI calculations
pricing = {
    "Phase 1 - Discovery": {"hours": 40, "rate": 150, "fixed": 0},
    "Phase 2 - Development": {"hours": 160, "rate": 150, "fixed": 0},
    "Phase 3 - Testing": {"hours": 40, "rate": 150, "fixed": 0},
    "Project Management": {"hours": 0, "rate": 0, "fixed": 5000},
}

print("PRICING BREAKDOWN")
print("=" * 70)
print(f"{'Phase':<30} {'Hours':<8} {'Rate':<10} {'Total'}")
print("-" * 70)
grand_total = 0
for phase, p in pricing.items():
    total = p["hours"] * p["rate"] + p["fixed"]
    grand_total += total
    print(f"{phase:<30} {p['hours']:<8} ${p['rate']:<9} ${total:,.0f}")
print("-" * 70)
print(f"{'TOTAL':<50} ${grand_total:,.0f}")

# ROI projection
expected_benefit = 500000  # Client's projected benefit
roi = ((expected_benefit - grand_total) / grand_total * 100)
payback_months = (grand_total / (expected_benefit / 12))
print(f"\\nROI: {roi:.0f}%")
print(f"Payback period: {payback_months:.1f} months")
```

### Step 4: Build Timeline (execute_python)

```python
# Run via execute_python — project timeline
from datetime import datetime, timedelta

start = datetime(2025, 4, 1)
phases = [
    ("Discovery & Planning", 2),
    ("Design & Architecture", 3),
    ("Development Sprint 1", 4),
    ("Development Sprint 2", 4),
    ("Testing & QA", 2),
    ("Deployment & Training", 1),
]

print("PROJECT TIMELINE")
print("=" * 70)
current = start
for phase, weeks in phases:
    end = current + timedelta(weeks=weeks)
    print(f"  {phase:<30} {current.strftime('%b %d')} - {end.strftime('%b %d, %Y')} ({weeks} weeks)")
    current = end
print(f"\\n  Total duration: {(current - start).days // 7} weeks")
print(f"  Projected completion: {current.strftime('%B %d, %Y')}")
```

## Proposal Structure

1. **Executive Summary** — Client's challenge, your solution, key differentiators
2. **Understanding of Requirements** — Restate needs with industry context (from web_search)
3. **Proposed Solution** — Approach, deliverables, methodology
4. **Team & Qualifications** — Relevant experience
5. **Timeline** — Phase breakdown (generated via execute_python)
6. **Pricing** — Detailed breakdown with ROI (computed via execute_python)
7. **Terms & Conditions** — Standard engagement terms

## Guidelines
- ALWAYS research the client via `web_search` and `browse_web` before writing
- ALWAYS compute pricing and ROI via `execute_python` — never estimate
- Mirror the client's language and terminology
- Include a compliance matrix for RFP responses
- Quantify the value proposition with computed numbers
""",
    },
]
