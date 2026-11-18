"""Headless Playwright screenshots of the 4 Streamlit tabs + 3 verified Tab 4 scenarios."""
import re
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

URL = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8501"
OUT = Path(__file__).resolve().parents[1] / "reports" / "screenshots"
OUT.mkdir(parents=True, exist_ok=True)
TABS = ["1. Data Overview", "2. Model Comparison", "3. ANN Training",
        "4. Real-Time What-If Scenario Inference"]
# (button, file, expected prediction, alert text, Streamlit alert test id)
SCENARIOS = [("Normal scenario", "tab4_normal.png", 25.72, "Normal operation", "stAlertContentSuccess"),
             ("Active cooling scenario", "tab4_active_cooling.png", 56.65, "ACTIVE COOLING TRIGGERED",
              "stAlertContentWarning"),
             ("High-risk scenario", "tab4_high_risk.png", 71.34, "CRITICAL THERMAL ALERT", "stAlertContentError")]


def wait_idle(page):
    """Wait until Streamlit finished rerunning (no 'Running' status widget), then 2 s more."""
    page.wait_for_timeout(500)
    page.wait_for_function(
        """() => { const w = document.querySelector('[data-testid="stStatusWidget"]');
                   return !w || !/Running/i.test(w.innerText); }""", timeout=60000)
    page.wait_for_function(
        """() => !document.querySelector('[data-stale="true"]')""", timeout=60000)
    page.wait_for_timeout(2000)


def full_shot(page, path):
    """Streamlit scrolls inside its own container, so grow the viewport to the content height first."""
    h = page.evaluate("""() => { const m = document.querySelector('[data-testid="stMain"]')
                                   || document.querySelector('[data-testid="stAppViewContainer"]');
                                 const c = m ? m.scrollHeight : 0;
                                 return Math.max(c, document.body.scrollHeight, 1000); }""")
    page.set_viewport_size({"width": 1400, "height": int(h) + 40})
    page.wait_for_timeout(2500)
    page.screenshot(path=str(path), full_page=True)
    page.set_viewport_size({"width": 1400, "height": 1000})
    page.wait_for_timeout(500)


with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page(viewport={"width": 1400, "height": 1000})
    page.goto(URL, wait_until="networkidle")
    page.wait_for_selector("[role=tab]", timeout=60000)
    wait_idle(page)
    for i, name in enumerate(TABS, 1):
        page.get_by_role("tab", name=name).click()
        wait_idle(page)
        # scroll through the page so lazy charts render, then back to top
        page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
        page.wait_for_timeout(1500)
        page.evaluate("window.scrollTo(0, 0)")
        if i < 4:
            full_shot(page, OUT / f"tab{i}.png")
            print("saved", f"tab{i}.png")

    for label, fname, expected, alert_text, testid in SCENARIOS:
        for attempt in range(1, 4):
            page.get_by_role("button", name=label).click()
            wait_idle(page)
            metric_box = page.locator('[data-testid="stMetric"]').filter(
                has_text="Predicted battery cell temperature").first
            metric = metric_box.locator('[data-testid="stMetricValue"]').inner_text()
            value = float(re.search(r"[-\d.]+", metric).group())
            alert = page.locator(f'[data-testid="{testid}"]').filter(has_text=alert_text)
            visible = alert.count() > 0 and alert.first.is_visible()
            # effective opacity of the alert and the metric (stale Streamlit elements are greyed out)
            eff = """el => { let o = 1; for (let e = el; e; e = e.parentElement)
                               o *= parseFloat(getComputedStyle(e).opacity); return o; }"""
            opacity = min(alert.first.evaluate(eff) if visible else 0, metric_box.evaluate(eff))
            ok = abs(value - expected) < 0.01 and visible and opacity > 0.9
            print(f"{label}: shown {value:.2f} C (expected {expected:.2f}), alert visible={visible}, "
                  f"min opacity={opacity:.2f} -> {'OK' if ok else 'RETRY'} (attempt {attempt})")
            if ok:
                full_shot(page, OUT / fname)
                print("saved", fname)
                break
        else:
            sys.exit(f"Could not get a clean screenshot for {label}")
    browser.close()
