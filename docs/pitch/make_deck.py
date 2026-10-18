"""Builds docs/pitch/RoboTokens_pitch.pptx (3-minute pitch, 8 slides, speaker notes = the script).

Needs python-pptx (`pip install python-pptx`), kept out of the project environment on purpose.
Assets: docs/figures/* and docs/pitch/assets/*: robot_demo_720p.mp4 (H.264 copy of docs/figures/RoboTokens_Can.MOV),
stills from it, dashboard screenshot.  `--preview` writes a copy with the video replaced by its poster frame.
"""

from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt

ROOT = Path(__file__).resolve().parents[2]
FIG, ASSETS, OUT = ROOT / "docs/figures", ROOT / "docs/pitch/assets", ROOT / "docs/pitch/RoboTokens_pitch.pptx"
PREVIEW = "--preview" in sys.argv
if PREVIEW:
    OUT = Path("/tmp/RoboTokens_pitch_preview.pptx")
VIDEO, POSTER = ASSETS / "robot_demo_720p.mp4", ASSETS / "still_36.jpg"

NAVY, TEAL, ORANGE, GREY, BG = "16294A", "3D8F8C", "E2A03D", "5B6578", "F7F9FB"
GREEN, BLUE, AMBER, WHITE, LINE = "2A9D5B", "2F6FDB", "D98A1A", "FFFFFF", "E3E8EF"
FONT = "Arial"
# Team slide: (name, role). Names from the git history where known; fill in the placeholders.
TEAM = [("Pietro Dardano", "[role]"), ("Ziad", "[role]"), ("Jiwon You", "[role]"), ("[Name]", "[role]")]


def rgb(h: str) -> RGBColor:
    return RGBColor.from_string(h)


prs = Presentation()
prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
BLANK = prs.slide_layouts[6]


def new_slide(notes: str, bg: str = BG):
    s = prs.slides.add_slide(BLANK)
    s.background.fill.solid()
    s.background.fill.fore_color.rgb = rgb(bg)
    s.notes_slide.notes_text_frame.text = notes
    return s


def text(s, x, y, w, h, runs, size=20, color=NAVY, bold=False, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP,
         spacing=1.1):
    """runs: str, or list of paragraphs; a paragraph is a str or a list of (text, {overrides}) runs."""
    tb = s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = Inches(0.05)
    paras = [runs] if isinstance(runs, str) else runs
    for i, para in enumerate(paras):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment, p.line_spacing = align, spacing
        for t, o in ([(para, {})] if isinstance(para, str) else para):
            r = p.add_run()
            r.text = t
            f = r.font
            f.name, f.size, f.bold = FONT, Pt(o.get("size", size)), o.get("bold", bold)
            f.color.rgb = rgb(o.get("color", color))
            f.italic = o.get("italic", False)
        if isinstance(para, list) and para and para[0][1].get("space_after") is not None:
            p.space_after = Pt(para[0][1]["space_after"])
    return tb


def box(s, x, y, w, h, fill=WHITE, line=LINE, shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.08):
    sh = s.shapes.add_shape(shape, Inches(x), Inches(y), Inches(w), Inches(h))
    sh.fill.solid()
    sh.fill.fore_color.rgb = rgb(fill)
    if line:
        sh.line.color.rgb = rgb(line)
        sh.line.width = Pt(1)
    else:
        sh.line.fill.background()
    sh.shadow.inherit = False
    if shape == MSO_SHAPE.ROUNDED_RECTANGLE:
        sh.adjustments[0] = radius
    return sh


def picture(s, path: Path, x, y, w=None, h=None, crop=None):
    """Fits the image inside (w, h) keeping its aspect ratio, centered in that box. crop = (l, t, r, b) in px."""
    im = Image.open(path)
    iw, ih = im.size
    if crop:
        l, t, r, b = crop
        iw, ih = r - l, b - t
    ar = iw / ih
    if w and h:
        bw, bh = (w, w / ar) if w / ar <= h else (h * ar, h)
        x, y = x + (w - bw) / 2, y + (h - bh) / 2
    elif w:
        bw, bh = w, w / ar
    else:
        bw, bh = h * ar, h
    pic = s.shapes.add_picture(str(path), Inches(x), Inches(y), Inches(bw), Inches(bh))
    if crop:
        W0, H0 = im.size
        pic.crop_left, pic.crop_top = crop[0] / W0, crop[1] / H0
        pic.crop_right, pic.crop_bottom = 1 - crop[2] / W0, 1 - crop[3] / H0
    return pic


def title(s, t, kicker=None):
    s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.6), Inches(0.55), Inches(0.09), Inches(0.62)).fill.solid()
    bar = s.shapes[-1]
    bar.fill.fore_color.rgb = rgb(TEAL)
    bar.line.fill.background()
    if kicker:
        text(s, 0.85, 0.25, 11, 0.35, kicker.upper(), size=12, color=TEAL, bold=True)
    text(s, 0.85, 0.47, 11.8, 0.8, t, size=34, bold=True)


def footer(s, n):
    picture(s, FIG / "logo.png", 11.75, 6.92, h=0.42)
    text(s, 0.6, 7.0, 6, 0.3, "RoboTokens  ·  Vision is not enough", size=11, color=GREY)
    text(s, 10.9, 7.0, 0.7, 0.3, str(n), size=11, color=GREY, align=PP_ALIGN.RIGHT)


def bullets(s, x, y, w, h, items, size=20, color=NAVY, gap=10):
    paras = [[("•  ", {"color": TEAL, "bold": True, "size": size, "space_after": gap}), (t, {"size": size})]
             for t in items]
    return text(s, x, y, w, h, paras, size=size, color=color, spacing=1.15)


# 1 ---------------------------------------------------------------------------------------------------------
s = new_slide(
    "0:00-0:25 HOOK\n"
    "Two cans. They look identical. One is full, one is empty. Any camera sees the same image twice. "
    "Vision is not enough. And that is exactly what robots rely on today.")
picture(s, FIG / "banner.png", 0.6, 0.55, w=12.13, h=5.6)
text(s, 0.6, 6.45, 12.13, 0.5, "Real OpenArm v1  ·  MuJoCo digital twin  ·  3-minute pitch",
     size=16, color=GREY, align=PP_ALIGN.CENTER)

# 2 ---------------------------------------------------------------------------------------------------------
s = new_slide(
    "0:25-0:55 PROBLEM\n"
    "Physical AI has to deal with physics: mass, stiffness, fill level. You cannot see those. You only learn them "
    "by interacting with the object. A robot that sorts parcels, hands a tool to a person, or loads a lab rack makes "
    "wrong decisions the moment two objects look the same.")
title(s, "Two cans. Same image.", "The problem")
picture(s, ASSETS / "still_12.jpg", 0.6, 1.55, w=5.4, h=5.0, crop=(760, 60, 1800, 1080))
text(s, 0.6, 6.62, 5.4, 0.3, "Our OpenArm in front of two look-alike cans. One is full, one is empty.",
     size=11, color=GREY, align=PP_ALIGN.CENTER)
text(s, 6.6, 1.65, 6.2, 0.8, "Which one is full?", size=34, bold=True, color=ORANGE)
bullets(s, 6.6, 2.65, 6.2, 2.6, [
    "A camera sees shape, colour and position.",
    "Mass, fill level and stiffness are invisible.",
    "Physical AI needs physics: these properties are revealed only by interaction.",
], size=21)
b = box(s, 6.6, 5.35, 6.2, 1.05, fill="E9F4F3", line=None)
text(s, 6.85, 5.35, 5.8, 1.05, [[("Vision-only robots guess. ", {"bold": True}),
                                 ("Sorting, handover and lab handling fail when objects look alike.", {})]],
     size=18, color=NAVY, anchor=MSO_ANCHOR.MIDDLE)
footer(s, 2)

# 3 ---------------------------------------------------------------------------------------------------------
s = new_slide(
    "0:55-1:30 OUR IDEA\n"
    "RoboTokens adds a physical sensing layer. The arm already carries the sensor we need: every joint reports its "
    "torque. We lift the object, read the joints, and estimate its mass. No load cell, no extra hardware, just "
    "software. The result is a mass with an uncertainty, not a guess.")
title(s, "The arm is the sensor.", "Our idea")
picture(s, ASSETS / "still_36.jpg", 0.6, 1.55, w=3.6, h=5.25, crop=(880, 0, 1720, 1080))
steps = [("Lift", "the object", TEAL), ("Read", "the joint torques", BLUE), ("Estimate", "mass ± uncertainty", ORANGE)]
for i, (a, b_, c) in enumerate(steps):
    x = 4.65 + i * 2.75
    sh = box(s, x, 1.75, 2.85, 1.45, fill=c, line=None, shape=MSO_SHAPE.CHEVRON if i else MSO_SHAPE.PENTAGON)
    text(s, x + (0.78 if i else 0.25), 1.75, 1.85 if i else 2.2, 1.45, [[(a, {"bold": True, "size": 24, "color": WHITE})],
                                                         [(b_, {"size": 15, "color": WHITE})]],
         anchor=MSO_ANCHOR.MIDDLE, spacing=1.0)
text(s, 4.65, 3.5, 8.2, 1.3, [
    [("How: ", {"bold": True}), ("a mass m at the hand adds  J", {}), ("z", {"size": 13}),
     ("(q) · m · g  to the joint torques. A 1° dither cancels friction; least squares gives m and its standard error.",
      {})]], size=17, color=NAVY, spacing=1.2)
for i, (k, v) in enumerate([("No load cell", "uses the torques the motors already report"),
                            ("No wrist sensor", "no added hardware on the arm"),
                            ("Software only", "a layer on top of the vision pipeline")]):
    x = 4.65 + i * 2.78
    box(s, x, 4.95, 2.6, 1.65)
    text(s, x + 0.15, 5.05, 2.3, 1.5, [[(k, {"bold": True, "size": 18, "color": TEAL})], [(v, {"size": 14})]],
         spacing=1.15)
footer(s, 3)

# 4 ---------------------------------------------------------------------------------------------------------
s = new_slide(
    "1:30-2:10 LIVE DEMO  (run: robotokens run, default task 'Give me the heavy can.')\n"
    "I say: give me the heavy can. Vision finds the two cans. The robot lifts each one, weighs it, and compares them "
    "statistically. Only when the difference is clear does it act. Then it carries the heavy can to me and holds it. "
    "It opens the gripper only when it feels me take it. If the evidence is weak, it re-measures, and if it is still "
    "unclear, it refuses to guess.\n\n"
    "If the live run fails: say 'the robot refused to guess' and play the video on this slide.")
title(s, "Demo: “Give me the heavy can.”", "Live")
flow = [("See", "GroundingDINO + SAM2 find the two cans from the sentence"),
        ("Weigh", "lift each can, mass from the joint torques"),
        ("Decide", "act only if  z = Δm / SE ≥ 3;  else re-measure, else refuse to guess"),
        ("Hand over", "carry the heavy can to the person"),
        ("Release", "open the gripper only when contact is felt")]
for i, (k, v) in enumerate(flow):
    y = 1.6 + i * 1.03
    c = [TEAL, BLUE, ORANGE, TEAL, GREEN][i]
    box(s, 0.6, y + 0.06, 0.7, 0.7, fill=c, line=None, shape=MSO_SHAPE.OVAL)
    text(s, 0.6, y + 0.06, 0.7, 0.7, str(i + 1), size=20, bold=True, color=WHITE, align=PP_ALIGN.CENTER,
         anchor=MSO_ANCHOR.MIDDLE)
    text(s, 1.45, y - 0.02, 4.75, 0.9, [[(k, {"bold": True, "size": 19})], [(v, {"size": 13.5, "color": GREY})]],
         anchor=MSO_ANCHOR.MIDDLE, spacing=1.05)
vx, vy, vw = 6.45, 1.75, 6.3
vh = vw * 9 / 16
if PREVIEW:
    picture(s, POSTER, vx, vy, w=vw, h=vh)
else:
    s.shapes.add_movie(str(VIDEO), Inches(vx), Inches(vy), Inches(vw), Inches(vh),
                       poster_frame_image=str(POSTER), mime_type="video/mp4")
text(s, vx, vy + vh + 0.12, vw, 0.6, [[("Real OpenArm v1: ", {"bold": True}),
                                       ("detecting, lifting and weighing two look-alike cans, with the MuJoCo twin on "
                                        "the laptop. Backup if the live run fails: click to play.", {})]],
     size=12.5, color=GREY, align=PP_ALIGN.CENTER)
footer(s, 4)

# 5 ---------------------------------------------------------------------------------------------------------
s = new_slide(
    "2:10-2:20 EVIDENCE (part 1)\n"
    "We also simulated the same recipe on three different arms: OpenArm, UR5e and Franka. "
    "(Let the animation run for a few seconds.)")
title(s, "Same recipe, three different arms.", "Cross-embodiment simulation")
picture(s, FIG / "exploration.gif", 0.6, 1.6, w=12.13, h=4.4)
text(s, 0.6, 6.15, 12.13, 0.6,
     [[("MuJoCo, 2x speed: ", {"bold": True}),
       ("each arm lifts two look-alike cans, wiggles its joints to weigh them, and puts them back. "
        "Simulation, not hardware data.", {})]], size=15, color=GREY, align=PP_ALIGN.CENTER)
footer(s, 5)

# 6 ---------------------------------------------------------------------------------------------------------
s = new_slide(
    "2:20-2:35 EVIDENCE (part 2), honestly\n"
    "On the real OpenArm, the full pipeline runs end to end. In simulation we added realistic error sources: drift, "
    "friction, ripple, grasp position, model error. Full versus empty is resolved on all three arms every time; the "
    "OpenArm resolves about 20 to 30 grams. And the simulation showed us something to fix: today the robot is "
    "overconfident on tiny differences. With a floor from repeated weighings it stops acting wrongly and abstains "
    "instead. Real benchmarks on the robot are our first next step.")
title(s, "Evidence, stated honestly.", "Results")
picture(s, FIG / "cross_embodiment_results.png", 0.6, 1.45, w=12.13, h=3.85)
cards = [("HARDWARE", GREEN, "Full pipeline end to end on the real OpenArm: see, weigh, decide, hand over."),
         ("SIMULATION", BLUE, "Realistic error budget. Full vs. empty: 300 / 300 on all arms. OpenArm resolves ~20–30 g."),
         ("TO FIX / NOT YET", AMBER, "Uncertainty is overconfident below ~20 g: add a repeatability floor. "
                                     "Real benchmarks next.")]
for i, (k, c, v) in enumerate(cards):
    x = 0.6 + i * 4.1
    box(s, x, 5.45, 3.93, 1.35)
    s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(x), Inches(5.6), Inches(0.08), Inches(1.05))
    bar = s.shapes[-1]
    bar.fill.solid()
    bar.fill.fore_color.rgb = rgb(c)
    bar.line.fill.background()
    text(s, x + 0.22, 5.52, 3.6, 1.25, [[(k, {"bold": True, "size": 12, "color": c})], [(v, {"size": 13.5})]],
         spacing=1.1)
footer(s, 6)

# 7 ---------------------------------------------------------------------------------------------------------
s = new_slide(
    "2:35-2:55 WHY IT MATTERS\n"
    "Warehouses already check weight to catch mis-picks, but with extra scales. Labs verify fills by weight. "
    "Assistive robots must hand things over safely. We bring that check into the grasp itself. Every run produces an "
    "auditable report, and our dashboard shows the vision, the digital twin and the live torques.")
title(s, "Why it matters.", "Use cases")
uses = [("Warehouse picking", "Weight checks catch mis-picks, today with extra scales. We check inside the grasp."),
        ("Lab & pharma", "Fills are verified by weight. A screening check in flexible cells."),
        ("Assistive robots", "Hand-over must release at the right moment: on felt contact."),
        ("Robot learning", "Measured physical labels instead of appearance-based guesses.")]
for i, (k, v) in enumerate(uses):
    x, y = 0.6 + (i % 2) * 3.2, 1.6 + (i // 2) * 2.45
    box(s, x, y, 3.0, 2.25)
    text(s, x + 0.18, y + 0.15, 2.65, 2.0, [[(k, {"bold": True, "size": 18, "color": TEAL})], [(v, {"size": 15})]],
         spacing=1.15)
picture(s, ASSETS / "dashboard.png", 7.2, 1.6, w=5.55, h=3.6)
text(s, 7.2, 5.3, 5.55, 1.3, [[("Every run is auditable. ", {"bold": True}),
                               ("Dashboard: camera detections, MuJoCo twin rebuilt from vision, mass ± error, "
                                "decision, live joint torques. (Shown with demo data.)", {})]],
     size=14, color=GREY)
footer(s, 7)

# team ------------------------------------------------------------------------------------------------------
s = new_slide("TEAM\n(Optional, a few seconds) Introduce the four of us, or let this slide stay up during questions.")
title(s, "The team.", "Who we are")
for i, (name, role) in enumerate(TEAM):
    x = 0.6 + i * 3.08
    box(s, x, 1.85, 2.88, 4.1)
    circ = box(s, x + 0.79, 2.25, 1.3, 1.3, fill=[TEAL, BLUE, ORANGE, GREEN][i], line=None, shape=MSO_SHAPE.OVAL)
    initials = "".join(w[0] for w in name.strip("[]").split()[:2]).upper() if not name.startswith("[") else "?"
    text(s, x + 0.79, 2.25, 1.3, 1.3, initials, size=30, bold=True, color=WHITE, align=PP_ALIGN.CENTER,
         anchor=MSO_ANCHOR.MIDDLE)
    text(s, x + 0.15, 3.85, 2.58, 0.9, name, size=20, bold=True, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
    text(s, x + 0.15, 4.75, 2.58, 1.0, role, size=14, color=GREY, align=PP_ALIGN.CENTER)
footer(s, 8)

# 8 ---------------------------------------------------------------------------------------------------------
s = new_slide("2:55-3:00 CLOSE\nVision tells the robot what to pick. Interaction tells it what it picked. "
              "RoboTokens: vision is not enough.")
text(s, 0.6, 1.5, 12.13, 1.0, "Vision tells the robot what to pick.", size=40, bold=True, align=PP_ALIGN.CENTER)
text(s, 0.6, 2.5, 12.13, 1.0, "Interaction tells it what it picked.", size=40, bold=True, color=TEAL,
     align=PP_ALIGN.CENTER)
picture(s, FIG / "logo.png", 4.55, 3.85, w=4.2, h=1.8)
text(s, 0.6, 5.85, 12.13, 0.6, "Vision is not enough.", size=24, bold=True, color=ORANGE, align=PP_ALIGN.CENTER)

prs.save(OUT)
print("wrote", OUT, f"({OUT.stat().st_size / 1e6:.1f} MB, {len(prs.slides)} slides)")
