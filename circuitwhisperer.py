"""
BUILDCORED ORCAS — Day 22: CircuitWhisperer
Photograph a circuit schematic. Local vision model
identifies components and describes the circuit.

Hardware concept: Schematic Reading
Every hardware engineer reads schematics. This builds
your visual vocabulary for hardware documentation.

CONTROLS:
- SPACE → capture from webcam
- 'f'   → load from file (place circuit.jpg/png in folder)
- 't'   → use generated test schematic
- 'q'   → quit

PREREQS:
- ollama running: ollama serve
- Vision model: ollama pull moondream
- pip install pillow opencv-python
"""

import cv2
import subprocess
import sys
import os
import time
import tempfile
import numpy as np

try:
    from PIL import Image, ImageDraw, ImageFont
    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    print("pip install Pillow for best results")


# ──────────────────────────────────────────────
#  SETUP CHECK
# ──────────────────────────────────────────────

def check_setup():
    """Verify ollama is running and moondream is available."""
    try:
        r = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if r.returncode != 0:
            print("ERROR: ollama not running. Run: ollama serve")
            sys.exit(1)
        if "moondream" not in r.stdout.lower():
            print("ERROR: moondream not found.")
            print("Fix: ollama pull moondream")
            sys.exit(1)
        print("✓ moondream ready")
    except FileNotFoundError:
        print("ERROR: ollama not installed. See https://ollama.ai")
        sys.exit(1)


check_setup()

MODEL = "moondream"
MAX_SIZE = 512   # resize before sending to keep inference fast


# ──────────────────────────────────────────────
#  TEST SCHEMATIC GENERATOR
# ──────────────────────────────────────────────

def generate_test_circuit(output_path="test_circuit.png"):
    """
    Draw a simple RC low-pass filter schematic so the project
    works even without a webcam or hand-drawn image.
    """
    if not HAS_PIL:
        # OpenCV fallback
        img = np.ones((300, 500, 3), dtype=np.uint8) * 255
        cv2.rectangle(img, (50, 100), (200, 200), (0, 0, 0), 2)
        cv2.rectangle(img, (250, 100), (400, 200), (0, 0, 0), 2)
        cv2.line(img, (200, 150), (250, 150), (0, 0, 0), 2)
        cv2.putText(img, "R1", (100, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        cv2.putText(img, "C1", (300, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        cv2.putText(img, "RC Low-pass Filter", (100, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        cv2.imwrite(output_path, img)
        return output_path

    W, H = 620, 360
    img = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(img)
    lw = 3

    # Title
    draw.text((W // 2 - 110, 10), "RC Low-Pass Filter Circuit", fill="black")

    # VIN label + wire in
    draw.text((10, 142), "VIN", fill="black")
    draw.line([(70, 150), (120, 150)], fill="black", width=lw)

    # Resistor (zigzag) R1
    draw.text((145, 112), "R1 = 10kΩ", fill="black")
    rx, ry = 120, 150
    zag_pts = [(rx, ry)]
    for i in range(8):
        offset = 8 if i % 2 == 0 else -8
        zag_pts.append((rx + 20 + i * 20, ry + offset))
    zag_pts.append((rx + 200, ry))
    draw.line(zag_pts, fill="black", width=lw)

    # Wire to node
    nx, ny = 320, 150
    draw.line([(nx, ny), (nx + 60, ny)], fill="black", width=lw)
    draw.ellipse([(nx - 5, ny - 5), (nx + 5, ny + 5)], fill="black")   # junction dot

    # Capacitor C1 (two horizontal plates)
    cx, cy = 380, 150
    draw.text((388, 112), "C1 = 100nF", fill="black")
    draw.line([(cx, cy), (cx, cy - 40)], fill="black", width=lw)
    draw.line([(cx - 30, cy - 40), (cx + 30, cy - 40)], fill="black", width=lw)
    draw.line([(cx - 30, cy - 52), (cx + 30, cy - 52)], fill="black", width=lw)
    draw.line([(cx, cy - 52), (cx, cy - 80)], fill="black", width=lw)

    # Capacitor bottom → GND
    draw.line([(cx, cy), (cx, cy + 55)], fill="black", width=lw)
    draw.line([(cx - 28, cy + 55), (cx + 28, cy + 55)], fill="black", width=lw)
    draw.line([(cx - 18, cy + 67), (cx + 18, cy + 67)], fill="black", width=lw)
    draw.line([(cx - 8,  cy + 79), (cx + 8,  cy + 79)], fill="black", width=lw)
    draw.text((cx + 6, cy + 55), "GND", fill="black")

    # VIN bottom → GND
    draw.line([(70, 150), (70, 200)], fill="black", width=lw)
    draw.line([(70, 200), (70, 255)], fill="black", width=lw)
    draw.line([(42, 255), (98, 255)], fill="black", width=lw)
    draw.line([(52, 267), (88, 267)], fill="black", width=lw)
    draw.line([(62, 279), (78, 279)], fill="black", width=lw)
    draw.text((102, 253), "GND", fill="black")

    # VOUT wire + label
    draw.line([(nx + 60, ny), (nx + 110, ny)], fill="black", width=lw)
    draw.text((nx + 112, 138), "VOUT", fill="black")

    img.save(output_path)
    print(f"✓ Generated test schematic: {output_path}")
    return output_path


# ──────────────────────────────────────────────
#  IMAGE PRE-PROCESSING
# ──────────────────────────────────────────────

def preprocess_image(img_path, max_size=MAX_SIZE):
    """
    Resize, convert to high-contrast B&W, and save to a temp file.
    Returns path to processed image, or None on failure.
    """
    img = cv2.imread(img_path)
    if img is None:
        return None

    h, w = img.shape[:2]
    scale = max_size / max(h, w)
    if scale < 1.0:
        img = cv2.resize(img, (int(w * scale), int(h * scale)))

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Adaptive threshold works better than simple threshold for hand-drawn photos
    thresh = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=15,
        C=8
    )

    enhanced = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)

    temp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
    cv2.imwrite(temp.name, enhanced, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return temp.name


# ──────────────────────────────────────────────
#  PROMPTS  (TODO #1 — tuned for structured output)
# ──────────────────────────────────────────────

# TODO #1: Tuned component detection prompt.
# Key changes vs starter:
#   - Forces a strict numbered-list format the code can parse
#   - Asks for confidence so we can flag hallucinations
#   - Caps at 10 items to prevent runaway lists
#   - Explicitly names every common schematic symbol moondream knows
COMPONENT_PROMPT = (
    "You are a hardware engineer reading an electronic circuit schematic. "
    "List every distinct component symbol you can see in the image. "
    "Use this exact format for each line — nothing else:\n"
    "  N. <component type> | <label or value if visible> | confidence: <high/medium/low>\n\n"
    "Component types to look for: resistor, capacitor, inductor, diode, LED, "
    "Zener diode, transistor (NPN/PNP), MOSFET, op-amp, voltage source, "
    "current source, ground symbol, power rail, wire junction, switch, "
    "input terminal, output terminal.\n\n"
    "Rules:\n"
    "- List at most 10 items.\n"
    "- If a component has no visible label, write 'unlabeled'.\n"
    "- Only list things you can actually see — do not guess.\n"
    "- If nothing is visible, write: 1. none | none | confidence: high\n\n"
    "Start your answer with '1.' — no preamble."
)

FUNCTION_PROMPT = (
    "You are a hardware engineer. Look at this circuit schematic. "
    "Answer these two questions in plain language:\n"
    "FUNCTION: What does this circuit do? (1–2 sentences, mention signal flow)\n"
    "TYPE: What category of circuit is this? "
    "(e.g. filter, amplifier, oscillator, power supply, logic gate, sensor interface)\n\n"
    "Be specific. Do not say 'I cannot determine' — make your best technical guess."
)

# TODO #2: Wiring error detection query.
# Checks for the five most common schematic mistakes:
#   1. Floating inputs / unconnected pins
#   2. Missing or misplaced ground
#   3. Short circuits (two incompatible nodes joined)
#   4. Reversed polarity (electrolytic cap, diode, transistor pinout)
#   5. Disconnected or dangling wires
WIRING_ERROR_PROMPT = (
    "You are a hardware engineer doing a design review. "
    "Carefully inspect this circuit schematic for wiring errors.\n\n"
    "Check specifically for:\n"
    "1. Floating inputs — pins not connected to anything\n"
    "2. Missing ground — no GND symbol or return path\n"
    "3. Short circuits — two nodes that should be separate are joined\n"
    "4. Reversed polarity — diode, electrolytic cap, or transistor connected backwards\n"
    "5. Dangling wires — a wire that ends without connecting to a component\n\n"
    "Format your answer exactly like this:\n"
    "ERRORS FOUND: <number, or 0 if none>\n"
    "1. <error type>: <short description of the specific problem>\n"
    "2. <error type>: <short description>\n"
    "...\n\n"
    "If you see no errors write:\n"
    "ERRORS FOUND: 0\n"
    "No wiring errors detected.\n\n"
    "List at most 5 errors. Be concise and specific."
)


# ──────────────────────────────────────────────
#  VLM QUERY
# ──────────────────────────────────────────────

def query_vlm(image_path, prompt):
    """Send an image + prompt to the local moondream model via ollama."""
    try:
        result = subprocess.run(
            ["ollama", "run", MODEL,
             f"{prompt}\n\nImage: {image_path}"],
            capture_output=True,
            text=True,
            timeout=90
        )
        out = result.stdout.strip()
        return out if out else "[No response from model]"
    except subprocess.TimeoutExpired:
        return "[Model timed out — try a smaller image or simpler schematic]"
    except Exception as e:
        return f"[Error: {e}]"


# ──────────────────────────────────────────────
#  RESULT PARSER
# ──────────────────────────────────────────────

def parse_components(raw_text):
    """
    Parse the structured component list.
    Returns (list_of_dicts, any_high_confidence_found).
    """
    lines = [l.strip() for l in raw_text.splitlines() if l.strip()]
    components = []
    for line in lines:
        if not line[0].isdigit():
            continue
        parts = line.split("|")
        # Strip leading "N. " from first part
        ctype = parts[0].split(".", 1)[-1].strip() if parts else "unknown"
        label = parts[1].strip() if len(parts) > 1 else "unlabeled"
        conf_raw = parts[2].strip().lower() if len(parts) > 2 else "low"
        confidence = "high" if "high" in conf_raw else ("medium" if "medium" in conf_raw else "low")
        components.append({"type": ctype, "label": label, "confidence": confidence})

    any_correct = any(c["confidence"] in ("high", "medium") for c in components)
    return components, any_correct


def parse_errors(raw_text):
    """Return the number of errors found from the wiring error response."""
    for line in raw_text.splitlines():
        if line.upper().startswith("ERRORS FOUND"):
            try:
                return int(line.split(":")[-1].strip())
            except ValueError:
                pass
    return -1   # unknown


# ──────────────────────────────────────────────
#  MAIN ANALYSIS PIPELINE
# ──────────────────────────────────────────────

def analyze_circuit(image_path):
    print("\n" + "─" * 55)
    print("🔍  Analyzing circuit schematic...")
    print("─" * 55)

    processed = preprocess_image(image_path)
    if processed is None:
        print("ERROR: Could not load image. Check the file path.")
        return

    try:
        # ── Step 1: Component Detection ──────────────────────
        print("\n[1/3] Detecting components...")
        t0 = time.time()
        raw_components = query_vlm(processed, COMPONENT_PROMPT)
        elapsed = time.time() - t0

        components, any_correct = parse_components(raw_components)
        conf_icon = "✅" if any_correct else "⚠️ "

        print(f"      ⚡ {elapsed:.1f}s\n")
        print("  COMPONENTS IDENTIFIED")
        print("  " + "─" * 40)
        if components:
            for i, c in enumerate(components, 1):
                conf_marker = {"high": "●", "medium": "◑", "low": "○"}.get(c["confidence"], "?")
                print(f"  {i:>2}. {conf_marker} {c['type']:<22} {c['label']}")
        else:
            print("  (raw model output below)")
            print(raw_components)
        print()
        print(f"  {conf_icon} At least one high/medium-confidence component: {any_correct}")

        # ── Step 2: Circuit Function ──────────────────────────
        print("\n[2/3] Identifying circuit function...")
        t0 = time.time()
        function = query_vlm(processed, FUNCTION_PROMPT)
        elapsed = time.time() - t0
        print(f"      ⚡ {elapsed:.1f}s\n")
        print("  CIRCUIT FUNCTION")
        print("  " + "─" * 40)
        for line in function.splitlines():
            print(f"  {line}")

        # ── Step 3: Wiring Error Detection ───────────────────
        print("\n[3/3] Checking for wiring errors...")
        t0 = time.time()
        raw_errors = query_vlm(processed, WIRING_ERROR_PROMPT)
        elapsed = time.time() - t0
        n_errors = parse_errors(raw_errors)
        print(f"      ⚡ {elapsed:.1f}s\n")

        error_icon = "🔴" if n_errors > 0 else ("🟢" if n_errors == 0 else "🟡")
        print("  WIRING ERROR CHECK")
        print("  " + "─" * 40)
        for line in raw_errors.splitlines():
            print(f"  {line}")
        print(f"\n  {error_icon} Errors flagged: {'unknown' if n_errors < 0 else n_errors}")

    finally:
        try:
            os.unlink(processed)
        except Exception:
            pass

    print("\n" + "─" * 55)
    print("✓  Analysis complete.")
    print("─" * 55 + "\n")


# ──────────────────────────────────────────────
#  ENTRY POINT
# ──────────────────────────────────────────────

def main():
    print("\n" + "=" * 58)
    print("  ⚡  CircuitWhisperer — Day 22")
    print("  Local vision AI for schematic analysis")
    print("=" * 58)
    print()
    print("  SPACE = capture from webcam")
    print("  'f'   = load circuit.jpg / circuit.png from folder")
    print("  't'   = use generated RC low-pass filter schematic")
    print("  'q'   = quit")
    print()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        cap = cv2.VideoCapture(1)
    has_webcam = cap.isOpened()

    if not has_webcam:
        print("⚠️  No webcam found — running in keyboard mode.\n")

    frame = None

    while True:
        if has_webcam:
            ret, frame = cap.read()
            if ret:
                display = frame.copy()
                cv2.putText(
                    display,
                    "SPACE=capture  f=file  t=test  q=quit",
                    (10, frame.shape[0] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 0), 1
                )
                cv2.imshow("CircuitWhisperer - Day 22", display)
            key = cv2.waitKey(1) & 0xFF
        else:
            command = input("Command (t / f / q): ").strip().lower()
            key = ord(command[0]) if command else 0

        # ── Quit ─────────────────────────────────────────────
        if key == ord("q"):
            break

        # ── Webcam capture ────────────────────────────────────
        elif key == ord(" ") and has_webcam and frame is not None:
            temp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
            cv2.imwrite(temp.name, frame)
            print(f"\n📸 Captured frame → {temp.name}")
            analyze_circuit(temp.name)
            try:
                os.unlink(temp.name)
            except Exception:
                pass

        # ── Load from file ────────────────────────────────────
        elif key == ord("f"):
            found = False
            for fname in ["circuit.jpg", "circuit.png", "circuit.jpeg",
                          "schematic.jpg", "schematic.png"]:
                if os.path.exists(fname):
                    print(f"\n📂 Loading: {fname}")
                    analyze_circuit(fname)
                    found = True
                    break
            if not found:
                print("\n⚠️  No circuit image found in the current folder.")
                print("   Name your file 'circuit.jpg' or 'circuit.png' and place it here.")

        # ── Generated test schematic ──────────────────────────
        elif key == ord("t"):
            print("\n🔧 Generating RC low-pass filter test schematic...")
            test_path = generate_test_circuit()
            analyze_circuit(test_path)

    if has_webcam:
        cap.release()
    cv2.destroyAllWindows()
    print("\nCircuitWhisperer ended. See you tomorrow for Day 23! ⚡")


if __name__ == "__main__":
    main()