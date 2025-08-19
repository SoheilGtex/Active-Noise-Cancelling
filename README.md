# Active Noise Cancelling (Real-Time Mic Noise Suppression)

> Low-latency, real-time microphone noise suppression in Python using spectral subtraction + adaptive noise tracking and 50% overlap-add (OLA). Works cross-platform with PortAudio via `sounddevice`.

https://github.com/SoheilGtex

---

## ✨ Features

- **Real-time** mic noise suppression (20 ms frames, 50% overlap).
- **Adaptive noise estimate** using exponential moving average (EMA).
- **Spectral subtraction + Wiener-style gain**, smoothed per band.
- **High-pass filter** (optional) to reduce rumble / hum.
- **Calibration step** (1–2 s) to capture baseline ambient noise.
- **Low latency** (frame = 20 ms, hop = 10 ms).
- **Pure Python + NumPy/SciPy**. No heavyweight ML runtime required.

> This is **noise suppression** (post-filtering of mic input), not feedforward/feedback “anti-noise” (phase-inversion) for headphones.

---

## 📦 Install

```bash
git clone https://github.com/SoheilGtex/active-noise-cancelling.git
cd active-noise-cancelling
python -m venv .venv && source .venv/bin/activate  # (Linux/macOS)
# .venv\Scripts\activate                            # (Windows)
pip install -r requirements.txt
