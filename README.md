# ⚡ CircuitWhisperer — Day 22

> Local vision AI that reads circuit schematics, identifies components, explains circuit function, and flags wiring errors.

---

## What it does

| Step | Query | Output |
|------|-------|--------|
| 1 | Component detection | Numbered list with labels and confidence levels |
| 2 | Circuit function | FUNCTION + TYPE description |
| 3 | Wiring error check | Flagged errors or "No errors detected" |

---

## Prerequisites

```bash
# 1. Install ollama
# https://ollama.ai

# 2. Start the ollama server
ollama serve

# 3. Pull the vision model
ollama pull moondream

# 4. Install Python dependencies
pip install pillow opencv-python
```

---

## Running

```bash
python day22_circuitwhisperer.py
```

### Controls

| Key | Action |
|-----|--------|
| `SPACE` | Capture from webcam |
| `f` | Load `circuit.jpg` / `circuit.png` from current folder |
| `t` | Use built-in RC low-pass filter test schematic |
| `q` | Quit |

---

## Follow-up queries

The three analysis prompts are constants at the top of the file — edit them to try new queries:

```python
COMPONENT_PROMPT    # what components are in the schematic
FUNCTION_PROMPT     # what does the circuit do
WIRING_ERROR_PROMPT # are there any mistakes
```

To add a **fourth query** (e.g. "estimate signal gain"):

```python
GAIN_PROMPT = (
    "Based on this schematic, estimate the voltage gain or attenuation "
    "at low and high frequencies. Show the calculation if possible."
)

# Then inside analyze_circuit():
print("[4/4] Estimating gain...")
gain = query_vlm(processed, GAIN_PROMPT)
print(gain)
```

---

## Common fixes

| Problem | Fix |
|---------|-----|
| Model hallucinates components | Check `confidence: low` items — ignore them; tighten the prompt |
| Image too dark | Use white paper + dark pen in bright light |
| Slow on large images | Starter auto-resizes to 512 px before sending |
| No webcam | Press `t` to use the built-in generated test schematic |
| moondream not found | `ollama pull moondream` then `ollama serve` |

---

## Hardware concept: Schematic Reading

Schematics are the universal language of electronics. Learning to read them means understanding:

- **Component symbols** — each has a standardized shape (zigzag = resistor, parallel lines = capacitor)
- **Net connections** — wires that connect define voltage nodes
- **Signal flow** — typically left (input) → right (output)
- **Ground reference** — all voltages are measured relative to GND

This project shows where AI vision models succeed (identifying obvious symbols) and where they still struggle (counting exact component values, detecting subtle wiring errors on hand-drawn diagrams).

---

## Shipped checklist

- [x] Image fed to local vision model
- [x] Model returns structured component list
- [x] At least one component correctly identified (high/medium confidence flagged)
- [x] Follow-up wiring error query works
- [x] README filled in
