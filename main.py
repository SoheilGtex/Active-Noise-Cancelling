import argparse
import queue
import sys
import time
import yaml

import numpy as np
import sounddevice as sd

from dsp import NoiseSuppressor

def list_devices():
    print(sd.query_devices())

def parse_args():
    p = argparse.ArgumentParser(
        description="Real-time Active Noise Cancelling (mic noise suppression)"
    )
    p.add_argument("--samplerate", type=int, default=None, help="Sample rate (default: device default)")
    p.add_argument("--frame_ms", type=int, default=None, help="Frame length in ms (default from config)")
    p.add_argument("--calib_sec", type=float, default=None, help="Calibration seconds (default from config)")
    p.add_argument("--device_in", type=str, default=None, help="Input device name or 'default'")
    p.add_argument("--device_out", type=str, default=None, help="Output device name or 'default'")
    p.add_argument("--highpass", type=float, default=None, help="High-pass cutoff Hz (0 to disable)")
    p.add_argument("--config", type=str, default="config.yaml")
    p.add_argument("--list-devices", action="store_true", help="Print available devices and exit")
    return p.parse_args()

def load_config(path: str):
    cfg = {
        "samplerate": 16000,
        "frame_ms": 20,
        "calib_sec": 1.0,
        "highpass_hz": 80,
        "noise_beta": 1.0,
        "noise_floor": 0.02,
        "ema_alpha": 0.96,
        "gain_smooth": 0.8,
        "device_in": "default",
        "device_out": "default",
    }
    try:
        with open(path, "r", encoding="utf-8") as f:
            loaded = yaml.safe_load(f) or {}
            cfg.update(loaded)
    except FileNotFoundError:
        pass
    return cfg

def main():
    args = parse_args()
    if args.list_devices:
        list_devices()
        return

    cfg = load_config(args.config)

    # CLI overrides
    if args.samplerate is not None: cfg["samplerate"] = args.samplerate
    if args.frame_ms is not None:   cfg["frame_ms"] = args.frame_ms
    if args.calib_sec is not None:  cfg["calib_sec"] = args.calib_sec
    if args.highpass is not None:   cfg["highpass_hz"] = args.highpass
    if args.device_in is not None:  cfg["device_in"] = args.device_in
    if args.device_out is not None: cfg["device_out"] = args.device_out

    sr = int(cfg["samplerate"])
    frame_ms = int(cfg["frame_ms"])

    ns = NoiseSuppressor(
        sr=sr,
        frame_ms=frame_ms,
        beta=cfg["noise_beta"],
        noise_floor=cfg["noise_floor"],
        ema_alpha=cfg["ema_alpha"],
        gain_smooth=cfg["gain_smooth"],
        highpass_hz=cfg["highpass_hz"],
    )

    hop = ns.hop
    q_out = queue.Queue(maxsize=8)

    def callback(indata, outdata, frames, time_info, status):
        if status:
            print(status, file=sys.stderr)
        # Make mono
        x = indata[:, 0].copy()
        # Ensure hop size
        if len(x) != hop:
            # Resample or pad (should not happen with blocksize=hop)
            x = x[:hop] if len(x) > hop else np.pad(x, (0, hop-len(x)))
        y = ns.process(x)
        outdata[:, 0] = y
        if outdata.shape[1] > 1:
            outdata[:, 1] = y
        # Non-critical monitoring
        try:
            q_out.put_nowait(float(np.sqrt(np.mean(y**2))))
        except queue.Full:
            pass

    print("• Devices:")
    try:
        default_sr = sd.query_devices(kind="input")["default_samplerate"]
        print("  Default input SR:", default_sr)
    except Exception:
        pass

    print(f"• Using SR={sr} Hz, frame={frame_ms} ms, hop={ns.hop} samples")
    print("• Calibrating noise… Speak as little as possible.")

    # Calibration stream
    with sd.Stream(
        device=(cfg["device_in"], cfg["device_out"]),
        samplerate=sr,
        blocksize=hop,
        dtype="float32",
        channels=1,
        callback=None,
    ):
        t_end = time.time() + float(cfg["calib_sec"])
        while time.time() < t_end:
            # read one hop from default input
            inbuf = sd.rec(hop, samplerate=sr, channels=1, dtype="float32")
            sd.wait()
            ns.calibrate_noise(inbuf[:, 0])

    print("• Calibration done. Starting real-time suppression. Ctrl+C to stop.")

    with sd.Stream(
        device=(cfg["device_in"], cfg["device_out"]),
        samplerate=sr,
        blocksize=hop,
        dtype="float32",
        channels=1,
        callback=callback,
    ):
        try:
            while True:
                time.sleep(0.5)
        except KeyboardInterrupt:
            print("\nStopping…")

if __name__ == "__main__":
    main()
