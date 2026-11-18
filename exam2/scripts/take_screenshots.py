"""Headless Playwright screenshots of the 4 Streamlit tabs + Tab 4 presets."""
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

URL = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8501"
OUT = Path(__file__).resolve().parents[1] / "reports" / "screenshots"
OUT.mkdir(parents=True, exist_ok=True)
TABS = ["1. Data Overview", "2. Model Comparison", "3. ANN Training",
        "4. Real-Time What-If Scenario Inference"]

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page(viewport={"width": 1400, "height": 1000})
    page.goto(URL, wait_until="networkidle")
    page.wait_for_selector("[role=tab]", timeout=60000)
    page.wait_for_timeout(2500)
    for i, name in enumerate(TABS, 1):
        page.get_by_role("tab", name=name).click()
        page.wait_for_timeout(2500)
        page.screenshot(path=str(OUT / f"tab{i}.png"), full_page=True)
        print("saved", f"tab{i}.png")
    for label, fname, alert in [("Normal scenario", "tab4_normal.png", "Normal operation"),
                                ("High-risk scenario", "tab4_high_risk.png", "CRITICAL")]:
        page.get_by_role("button", name=label).click()
        page.get_by_text(alert, exact=False).first.wait_for(timeout=20000)
        page.wait_for_timeout(2000)
        page.screenshot(path=str(OUT / fname), full_page=True)
        print("saved", fname)
    browser.close()
