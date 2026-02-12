from scipy.signal import butter, filtfilt, firwin, hilbert, sosfiltfilt
from scipy.ndimage import median_filter 
import numpy as np

"""
Functions to filter analog signals
"""

def butter_lowpass(cutoff, fs, order=4):
    """
    Apply generic butter filter

    Args:
        cutff (int): cutoff frequency for lowpass filter
        fs (int; default is intan sample rate 20e3): sampling rate of signal
    Returns: 
        b, a (ndarray, ndarray): numerator (b) and denominator (a) polynomials of the IIR filter.
    """
    # calculate nyquist with cutoff
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff=500, fs=20e3, order=3):
    """
    Apply both butter and filtfilt filters

    Args:
        data (array-like): 
        cutoff (int): cutoff frequency for lowpass filter
        fs (int; default is intan sample rate 20e3): sampling rate of signal
        order (int; default is 3): steepness of signal above cutoff frequency
    """
    b, a = butter_lowpass(cutoff=cutoff, fs=fs, order=order)
    y = filtfilt(b, a, data)
    return y

def sinc_lowpass_filter(signal, cutoff, fs, numtaps=101):
    """
    Apply a sinc lowpass filter to signal
    
    Args:
        signal (np.ndarray): signal to lowpass filter
        cutoff (int): cutoff frequency for filter
        fs (int or float): sampling rate of signal
        numtaps (int): length of filter/num coefficients
    """
    # calc nyquist sampling freq to avoid aliasing 
    nyquist_freq = fs / 2
    #normalize cutoff from 0 to 1, use to get coefficients for finite impulse response
    # apply forward and backward linear filters to sharpen signal and cancel phase shifts
    # convolves input signal with coefficients from FIR
    fir = firwin(numtaps, cutoff / nyquist_freq)
    return filtfilt(fir, [1.0], signal, axis=-1)

def despike_cliffs(signal, k=9, z=6.0, only_drops=False):
    """
    Remove large step changes by flagging outliers in the first difference
    using rolling median/MAD, then linearly interpolating flagged samples.
    """
    signal = np.asarray(signal, dtype=float).copy()
    if signal.ndim != 1 or signal.size < 3:
        return signal
    dx = np.diff(signal, prepend=signal[0])
    win = int(2 * k + 1)
    med = median_filter(dx, size=win, mode='nearest')
    mad = median_filter(np.abs(dx - med), size=win, mode='nearest') + 1e-8
    zscore = (dx - med) / (1.4826 * mad)
    flags = (zscore < -z) if only_drops else (np.abs(zscore) > z)
    # include “landing” sample after a jump
    flags[:-1] |= flags[1:]
    if not np.any(flags):
        return signal
    good = ~flags
    # extend ends to allow interpolation
    if not good[0]:
        first = np.argmax(good)
        signal[:first] = signal[first]; good[:first] = True
    if not good[-1]:
        last = len(good) - 1 - np.argmax(good[::-1])
        signal[last:] = signal[last]; good[last:] = True
    xi = np.interp(np.arange(signal.size), np.flatnonzero(good), signal[good])
    signal[flags] = xi[flags]
    return signal

def bandpass_filter(data, fs, lowcut=40, highcut=300, order=4):
    nyq = fs / 2.0
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype="band")
    return filtfilt(b, a, data)

def bandpass_filter_sos(data, fs, lowcut=0.5, highcut=4.0, order=4):
    nyq = fs / 2.0
    low = lowcut / nyq
    high = highcut / nyq
    sos = butter(order, [low, high], btype="band", output='sos')
    return sosfiltfilt(sos, data)

def band_envelope_slow(signal, fs_hz, band_hz, lp_envelope_hz=5.0):
    """Return a slow-varying amplitude envelope for an LFP band."""
    lo, hi = band_hz
    # Band-pass
    b, a = butter(4, [lo/(fs_hz/2), hi/(fs_hz/2)], btype='band')
    band_sig = filtfilt(b, a, signal)

    # Hilbert envelope
    env = np.abs(hilbert(band_sig))

    # Low-pass the envelope to focus on slow modulations
    if lp_envelope_hz is not None:
        b2, a2 = butter(2, lp_envelope_hz/(fs_hz/2), btype='low')
        env = filtfilt(b2, a2, env)
    return env

def band_hilbert(x, fs, f_lo, f_hi, order=801):
    '''
    Typical bands: δ(1–4), θ(6–10), β(15–30), low-γ(30–60), high-γ(60–120).

    Band-pass per channel → Hilbert:

    Return amplitude envelope (power proxy) and instantaneous phase (for spike–phase locking).
    '''
    taps = firwin(order, [f_lo, f_hi], pass_zero=False, fs=fs)
    xb = filtfilt(taps, [1.0], x)
    h  = hilbert(xb)
    amp = np.abs(h)          # envelope
    phs = np.angle(h)        # phase
    return amp, phs


# '''Apply a sinc lowpass filter to signal'''
# def sinc_lowpass_filter_GPU(signal, cutoff, fs, numtaps=101, gpu_available=False):
#     # calc nyquist sampling freq to avoid aliasing 
#     nyquist_freq = fs / 2
#     # convert to cupy array for faster processing if gpu available else use numpy
#     if GPU_AVAILABLE and cp is not None:
#         signal = cp.asarray(signal)
#         #normalize cutoff from 0 to 1, use to get coefficients for finite impulse response
#         fir = firwin_gpu(numtaps, cutoff / nyquist_freq)
#         # apply forward and backward linear filters to sharpen signal and cancel phase shifts
#         # convolves input signal with coefficients from FIR
#         return filtfilt_gpu(fir, cp.asarray([1.0]), signal, axis=-1)
#     else:
#         fir = firwin(numtaps, cutoff / nyquist_freq)
#         return filtfilt(fir, [1.0], signal, axis=-1)