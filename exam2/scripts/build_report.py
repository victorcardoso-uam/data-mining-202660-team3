"""Build reports/Exam2_Team3_Report.pdf (black & white, Times 12, 1.5 spacing, justified).

Re-run at the end to include manual screenshots:
  python scripts/build_report.py
Manual screenshots in reports/screenshots/ must start with "manual_". Put "task1".."task4"
in the name to place them in that section (e.g. manual_task4_pr_review.png); others go to Task 4.
The PR link is read from reports/pr_url.txt if present.
"""
import json
from pathlib import Path

import pandas as pd
from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (Image, KeepTogether, Paragraph, SimpleDocTemplate, Spacer, Table,
                                TableStyle)

ROOT = Path(__file__).resolve().parents[1]
REPORTS, FIGS, SHOTS = ROOT / "reports", ROOT / "reports" / "figures", ROOT / "reports" / "screenshots"
OUT = REPORTS / "Exam2_Team3_Report.pdf"

# Fonts: Times New Roman if available, otherwise built-in Times-Roman
FONT, BOLD = "Times-Roman", "Times-Bold"
win = Path("C:/Windows/Fonts")
if (win / "times.ttf").exists() and (win / "timesbd.ttf").exists():
    pdfmetrics.registerFont(TTFont("TNR", str(win / "times.ttf")))
    pdfmetrics.registerFont(TTFont("TNR-Bold", str(win / "timesbd.ttf")))
    FONT, BOLD = "TNR", "TNR-Bold"

body = ParagraphStyle("body", fontName=FONT, fontSize=12, leading=18, alignment=TA_JUSTIFY, spaceAfter=6)
h1 = ParagraphStyle("h1", parent=body, fontName=BOLD, fontSize=14, leading=21, spaceBefore=10,
                    keepWithNext=1)
title = ParagraphStyle("title", parent=body, fontName=BOLD, fontSize=14, alignment=TA_CENTER)
center = ParagraphStyle("center", parent=body, alignment=TA_CENTER, fontSize=10, leading=15)
cell = ParagraphStyle("cell", fontName=FONT, fontSize=10, leading=13)
cellb = ParagraphStyle("cellb", fontName=BOLD, fontSize=10, leading=13)

MAX_W = 6.5 * inch


def P(t, s=body):
    return Paragraph(t, s)


def table(df, index=True, fmt="{:.4f}", col_widths=None):
    d = df.reset_index() if index else df
    data = [[Paragraph(str(c), cellb) for c in d.columns]]
    for _, r in d.iterrows():
        data.append([Paragraph(fmt.format(v) if isinstance(v, float) else str(v), cell) for v in r])
    t = Table(data, hAlign="CENTER", repeatRows=1, colWidths=col_widths)
    t.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.5, colors.black),
                           ("VALIGN", (0, 0), (-1, -1), "MIDDLE")]))
    return t


FIG = [0]


def img(path, caption, max_h=7.5 * inch):
    """Image + auto-numbered caption (Figure 1, 2, 3 ... in order of appearance)."""
    FIG[0] += 1
    w, h = PILImage.open(path).size
    s = min(MAX_W / w, max_h / h)
    im = Image(str(path), width=w * s, height=h * s)
    return KeepTogether([im, P(f"Figure {FIG[0]}. {caption}.", center)])


def shot(name, caption, max_h=7.5 * inch):
    f = SHOTS / name
    return [img(f, caption, max_h)] if f.exists() else []


def manual(task):
    out = []
    for f in sorted(SHOTS.glob("manual_*")):
        name = f.stem.lower()
        tagged = next((n for n in range(1, 5) if f"task{n}" in name), 4)
        if tagged == task and f.suffix.lower() in (".png", ".jpg", ".jpeg"):
            nice = {"merge": "PR #23 merged into master and approved by the reviewer",
                    "commits": "Commits included in PR #23"}
            cap = next((v for k, v in nice.items() if k in name), f.stem.replace("_", " "))
            out.append(img(f, cap))
    return out


meta = json.loads((ROOT / "models" / "metadata.json").read_text())
t1 = pd.read_csv(REPORTS / "task1_metrics.csv", index_col=0)
comp = pd.read_csv(REPORTS / "model_comparison.csv", index_col=0)
diag = pd.read_csv(REPORTS / "task1_diagnostics.csv", index_col=0).iloc[:, 0]
presets = pd.read_csv(REPORTS / "task3_presets.csv")
for df_ in (t1, comp):
    df_["p"] = df_["p"].astype(int).astype(str)
pr_file = REPORTS / "pr_url.txt"
pr_url = pr_file.read_text().strip() if pr_file.exists() else "[PR link pending]"
win_name = meta["winning_model"]
wm = comp.loc[win_name]
mlp = comp.loc["MLP (64, 32)"]
y = pd.read_csv(ROOT / "data" / "exam2_team3_ev_battery.csv")["battery_cell_temp_c"]
p75, p90 = y.quantile(.75), y.quantile(.9)
pv = presets.set_index("Scenario")["Predicted_C"]

s = []
s += [P("Examen Parcial 2, Part 2 - Team 3", title),
      P("Data Mining &amp; Modern AI Systems (202660)", center),
      P("<b>Dataset:</b> exam2_team3_ev_battery.csv &nbsp;|&nbsp; <b>Target:</b> battery_cell_temp_c", center),
      P("<b>Team:</b> Eduardo Gallegos, Carlos Mejia, Jimena Escalante, Victoria Morales, "
        "Jose Carlos Vera, Rodrigo Meouchi", center), Spacer(1, 8)]

# Task 1
s += [P("Task 1: Regression and Residual Diagnostics", h1),
      P("We checked the data first (600 rows, 6 numeric columns, no missing values). We used an 80/20 split "
        "with random_state=42 and fit three models: a base OLS, an OLS on log(y) that we transform back with "
        "exp before computing metrics, and a degree 2 polynomial regression (20 generated features). "
        "Adjusted R2 uses n = 120 test rows and p = number of features each model really uses."),
      table(t1), Spacer(1, 8)]
s += shot("terminal_task1.png", "Console output of scripts/task1_regression.py")
s += [img(FIGS / "task1_residual_dashboard.png", "Residual diagnostics of the base OLS model"),
      P(f"<b>Team answer:</b> The base OLS wins with the lowest RMSE ({t1.loc['OLS Linear', 'RMSE']:.3f}) and "
        f"the highest adjusted R2 ({t1.loc['OLS Linear', 'Adj_R2']:.3f}). The polynomial model is almost the same "
        f"on R2 but its adjusted R2 drops because it uses 20 terms, and the log model is worse, so the relation "
        f"is basically linear. The residual plots agree: the Residuals vs Fitted line is flat (linearity OK), the "
        f"Q-Q plot follows the line except a few tail points (Shapiro-Wilk p = {float(diag['shapiro_p']):.3f}), "
        f"the Scale-Location line is flat (Breusch-Pagan p = {float(diag['breusch_pagan_p']):.3f}, so constant "
        f"variance), and no point has a large Cook's distance (max = {float(diag['max_cooks']):.3f}, far below "
        f"0.5).")]
s += manual(1)

# Task 2
s += [P("Task 2: Artificial Neural Network", h1),
      P(f"We scaled the features with a StandardScaler fit only on the training set and trained an "
        f"MLPRegressor with two hidden layers (64, 32), ReLU, early stopping and max_iter=1000. It stopped after "
        f"{meta['mlp']['n_iter']} epochs. Then we compared it with the three Task 1 models on the same test set "
        f"and saved the model with the lowest RMSE."),
      table(comp), Spacer(1, 8)]
s += shot("terminal_task2.png", "Console output of scripts/task2_ann.py")
s += [img(FIGS / "task2_loss_curve.png", "MLP training loss per epoch", max_h=4 * inch)]
s += shot("terminal_models.png", "Saved artifacts in the models/ folder", max_h=2.5 * inch)
s += [P(f"<b>Team answer:</b> The loss drops fast in the first epochs and then flattens, so the network "
        f"converged and early stopping ended training before overfitting. Even so, the MLP (RMSE = "
        f"{mlp['RMSE']:.3f}) does not beat the {win_name} (RMSE = {wm['RMSE']:.3f}). Since the data is close "
        f"to linear, a simple linear model is enough and the ANN only adds complexity. The winning model "
        f"({win_name}) is saved as models/best_model.joblib, the MLP as models/mlp_model.joblib and the scaler "
        f"as models/scaler.joblib; metadata.json says if the scaler or log transform is needed.")]
s += manual(2)

# Task 3
scen = presets.rename(columns={"discharge_rate_c": "Discharge", "ambient_temp_c": "Ambient",
                               "coolant_flow_l_min": "Coolant", "internal_resistance_mOhm": "Resistance",
                               "Predicted_C": "Pred. C"})
scen["Scenario"] = scen["Scenario"].str.replace(" scenario", "", regex=False)
scen["Alert"] = scen["Alert"].str.title()
scen = scen[["Scenario", "Discharge", "Ambient", "Coolant", "Resistance", "Pred. C", "Alert"]]
widths = [w * inch for w in (1.15, 0.8, 0.75, 0.75, 0.9, 0.7, 1.15)]
s += [KeepTogether([P("Task 3: Streamlit App", h1)]
                   + shot("terminal_streamlit.png", "Launching the app with streamlit run app.py",
                          max_h=1.5 * inch))]
s += [P(f"We built app.py with four tabs that load the .joblib files with st.cache_resource and show an error "
        f"if a file is missing. A status line under the title shows which artifacts were loaded. Tab 4 has "
        f"sliders for the five features (dataset min/max, default at the median) and predicts the temperature "
        f"with {win_name}. Active cooling is triggered at {round(p75)} C (75th percentile, {p75:.2f} C) and the "
        f"critical alert at {round(p90)} C (90th percentile, {p90:.2f} C). We checked the three presets with "
        f"the real model:"),
      table(scen, index=False, fmt="{:.2f}", col_widths=widths), Spacer(1, 8)]
for f, c in [("tab1.png", "Tab 1 - Data Overview"), ("tab2.png", "Tab 2 - Model Comparison"),
             ("tab3.png", "Tab 3 - ANN Training"),
             ("tab4_normal.png", "Tab 4 - Normal scenario (success)"),
             ("tab4_active_cooling.png", "Tab 4 - Active cooling scenario (warning)"),
             ("tab4_high_risk.png", "Tab 4 - High-risk scenario (critical error)")]:
    s += shot(f, c)
s += [P(f"<b>Team answer:</b> Discharge rate and ambient temperature push the cell temperature up the most, "
        f"and coolant flow pulls it down. With low discharge, cool ambient and high coolant the prediction "
        f"stays near {pv['Normal scenario']:.0f} C (normal). With a high discharge, median ambient, low coolant "
        f"and high resistance it goes to {pv['Active cooling scenario']:.2f} C, which is between {round(p75)} "
        f"and {round(p90)} C, so active cooling turns on. High discharge, hot ambient, low coolant and high "
        f"resistance give about {pv['High-risk scenario']:.0f} C, above the critical threshold. The green status "
        f"line at the top confirms that the four .joblib/metadata files loaded correctly, and it is visible in "
        f"all four tabs. Using percentiles keeps the alerts tied to what the batteries actually experienced in "
        f"the data.")]
s += manual(3)

# Task 4
s += [P("Task 4: Git and Pull Request", h1),
      P("We made small commits per task on the branch <b>exam/partial-exam-2</b>, pushed it and opened a "
        "Pull Request to master for code review by a teammate before merging."),
      P(f"<b>Branch:</b> exam/partial-exam-2<br/><b>Pull Request:</b> {pr_url}"),
      P(f"The PR was reviewed and approved by a teammate and merged into master. Merge commit: "
        f"{(REPORTS / 'merge_commit.txt').read_text().strip() if (REPORTS / 'merge_commit.txt').exists() else '-'}.")]
m4 = manual(4)
s += m4 if m4 else [P("<b>[Insert screenshot: PR review and merge]</b>"), Spacer(1, 60)]

SimpleDocTemplate(str(OUT), pagesize=LETTER, leftMargin=inch, rightMargin=inch,
                  topMargin=inch, bottomMargin=inch, title="Exam2 Team3 Report").build(s)
print("Report written to", OUT)
