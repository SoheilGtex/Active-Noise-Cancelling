import numpy as np
from scipy.signal import butter, lfilter, lfilter_zi
from numpy.fft import rfft, irfft

_EPS = 1e-8

def hann_sqrt(N: int) -> np.ndarray:
    """Sqrt-Hann window for perfect 50% OLA reconstruction."""
    w = np.hanning(N)
    return np.sqrt(w + _EPS)

def design_highpass(sr: int, cutoff_hz: float, order: int = 2):
    if cutoff_hz <= 0:
        return None
    b, a = butter(order, cutoff_hz / (0.5 * sr), btype="highpass")
    zi = lfilter_zi(b, a)
    return (b, a, zi)

class NoiseSuppressor:
    """
    Real-time spectral-subtraction noise suppressor with 50% OLA.
    """
    def __init__(
        self,
        sr: int,
        frame_ms: int = 20,
        beta: float = 1.0,
        noise_floor: float = 0.02,
        ema_alpha: float = 0.96,
        gain_smooth: float = 0.8,
        highpass_hz: float = 0.0,
    ):
        self.sr = sr
        self.frame_len = int(sr * frame_ms / 1000)
        self.hop = self.frame_len // 2  # 50% overlap
        self.win = hann_sqrt(self.frame_len)

        self.beta = float(beta)
        self.noise_floor = float(noise_floor)
        self.ema_alpha = float(ema_alpha)
        self.gain_smooth = float(gain_smooth)

        # buffers
        self._prev_input_tail = np.zeros(self.frame_len - self.hop, dtype=np.float32)
        self._ola = np.zeros(self.frame_len, dtype=np.float32)
        self._prev_gain = None

        # spectrum template
        self._noise_mag = np.ones(self.frame_len // 2 + 1, dtype=np.float32) * 1e-3

        # high-pass
        hp = design_highpass(sr, highpass_hz, order=2)
        self.hp = hp  # (b,a,zi) or None

    def _apply_highpass(self, x: np.ndarray):
        if self.hp is None:
            return x
        b, a, zi = self.hp
        y, zf = lfilter(b, a, x, zi=zi * x[0])
        self.hp = (b, a, zf)
        return y

    def calibrate_noise(self, chunk: np.ndarray):
        """
        Update noise magnitude estimate using EMA during calibration period.
        """
        # Build full frame using previous tail + current hop
        frame = np.concatenate([self._prev_input_tail, chunk]).astype(np.float32)
        self._prev_input_tail = chunk[-(self.frame_len - self.hop):].copy()

        frame = self._apply_highpass(frame)
        X = rfft(frame * self.win)
        mag = np.abs(X)
        self._noise_mag = self.ema_alpha * self._noise_mag + (1 - self.ema_alpha) * mag

    def process(self, chunk: np.ndarray) -> np.ndarray:
        """
        Process one hop-sized chunk and return hop-sized denoised audio.
        """
        # Construct frame
        frame = np.concatenate([self._prev_input_tail, chunk]).astype(np.float32)
        self._prev_input_tail = chunk[-(self.frame_len - self.hop):].copy()

        frame = self._apply_highpass(frame)

        # Analysis
        X = rfft(frame * self.win)
        mag = np.abs(X)
        phase = np.angle(X)

        # Update noise (slower update during speech; here basic EMA)
        self._noise_mag = self.ema_alpha * self._noise_mag + (1 - self.ema_alpha) * mag

        # Spectral subtraction with flooring
        clean_mag = np.maximum(mag - self.beta * self._noise_mag,
                               self.noise_floor * self._noise_mag)

        # Wiener-like gain
        gain = clean_mag / (mag + _EPS)

        # Smooth gain over time
        if self._prev_gain is None:
            self._prev_gain = gain
        gain = self.gain_smooth * self._prev_gain + (1 - self.gain_smooth) * gain
        self._prev_gain = gain

        # Synthesis
        Y = gain * mag * np.exp(1j * phase)
        y_frame = irfft(Y).astype(np.float32)

        # OLA with sqrt-hann synthesis
        y_frame *= self.win

        # Output hop
        out = self._ola[:self.hop].copy()
        # Shift ola buffer
        self._ola[:-self.hop] = self._ola[self.hop:]
        self._ola[-self.hop:] = 0.0
        # Add current frame at buffer start
        self._ola += y_frame

        return out
