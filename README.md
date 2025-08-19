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
```
## Dependencies

sounddevice (PortAudio binding)

numpy

scipy

pyyaml (for reading config.yaml, optional)

Windows: if PortAudio devices are not listed, update your audio drivers or install WASAPI loopback support (typically not needed).
Linux: ensure your user is in audio group and PulseAudio/PipeWire is running.
macOS: grant microphone permission to the terminal/IDE.

##  ▶️ Run

Quick start (defaults are fine):

```python main.py```


Useful options

python main.py \
  --samplerate 16000 \
  --frame_ms 20 \
  --calib_sec 1.0 \
  --device_in default \
  --device_out default \
  --highpass 80


--samplerate : 16000 or 48000 are common.

--frame_ms : 20 ms frame (10 ms hop with 50% overlap).

--calib_sec : initial ambient-noise calibration duration (seconds).

--highpass : cut below N Hz (0 to disable).

Press Ctrl+C to stop.

## ⚙️ How it works (DSP)

Framing & Windowing
Signal is split into frames (e.g., 20 ms) with 50% overlap and Hann analysis/synthesis windows (perfect reconstruction with OLA).

Noise Spectrum Estimation (EMA)
Magnitude spectrum of noise is tracked with an exponential moving average. During low-energy segments it adapts faster.

Spectral Subtraction + Gain Smoothing
We estimate clean magnitude per bin: |X|_clean = max(|X| - β·|N|, floor·|N|).
Then we compute a Wiener-like gain and smooth it over time to avoid musical noise.

High-Pass
Optional IIR HPF removes rumble.

OLA Synthesis
Inverse FFT → window → overlap-add. The output chunk each hop is sent to the soundcard with minimal latency.

## 📁 Config

config.yaml (optional; args override file):

samplerate: 16000
frame_ms: 20
calib_sec: 1.0
highpass_hz: 80
noise_beta: 1.0
noise_floor: 0.02
ema_alpha: 0.96
gain_smooth: 0.8
device_in: default
device_out: default

## 🧪 Benchmark notes

Latency ≈ frame_ms (20 ms) because hop is 10 ms and OLA buffering is 1 frame.

CPU < 10% on mid-range laptops at 16 kHz mono.

## 📜 License

MIT — see LICENSE.

## 🙌 Acknowledgements

sounddevice (PortAudio) for reliable cross-platform I/O.

Classic spectral subtraction literature and Wiener filtering techniques for single-channel speech enhancement.


---

## 2) `requirements.txt`

```txt
sounddevice>=0.4.6
numpy>=1.24
scipy>=1.10
pyyaml>=6.0
