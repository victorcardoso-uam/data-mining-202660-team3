"""Run the exam scripts and render their console output as terminal-style PNGs."""
import os
import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "screenshots"
PROMPT = "PS ...\Data_Mining\exam2> "


def font(size=16):
    for f in ("C:/Windows/Fonts/consola.ttf", "C:/Windows/Fonts/cour.ttf"):
        if Path(f).exists():
            return ImageFont.truetype(f, size)
    return ImageFont.load_default()


def render(command, text, out_name):
    fnt = font()
    lines = [PROMPT + command] + text.rstrip().splitlines()
    lines = [l.replace("\t", "    ") for l in lines]
    cw = fnt.getbbox("M")[2]
    lh = 22
    width = max(900, min(1600, 30 + cw * max(len(l) for l in lines)))
    img = Image.new("RGB", (width, 30 + lh * len(lines)), (12, 12, 12))
    d = ImageDraw.Draw(img)
    for i, line in enumerate(lines):
        color = (120, 220, 120) if i == 0 else (230, 230, 230)
        d.text((15, 15 + i * lh), line, font=fnt, fill=color)
    img.save(OUT / out_name)
    print("saved", out_name)


def run(cmd):
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    r = subprocess.run([sys.executable] + cmd, cwd=ROOT, capture_output=True, text=True,
                       encoding="utf-8", env=env)
    return r.stdout + r.stderr


if __name__ == "__main__":
    render("python scripts/task1_regression.py", run(["scripts/task1_regression.py"]), "terminal_task1.png")
    render("python scripts/task2_ann.py", run(["scripts/task2_ann.py"]), "terminal_task2.png")
    listing = "\n".join(f"{p.stat().st_size:>10,} bytes   {p.name}"
                        for p in sorted((ROOT / "models").iterdir()))
    render("ls models/", "\n    Directory: exam2\models\n\n" + listing, "terminal_models.png")
    # Streamlit startup text: pass the saved server log as argv[1]
    if len(sys.argv) > 1:
        log = Path(sys.argv[1]).read_text(encoding="utf-8", errors="ignore").splitlines()
        keep = [l for l in log if "Local URL" in l or "You can now view" in l]
        render("streamlit run app.py", "\n" + "\n".join(keep) + "\n", "terminal_streamlit.png")
