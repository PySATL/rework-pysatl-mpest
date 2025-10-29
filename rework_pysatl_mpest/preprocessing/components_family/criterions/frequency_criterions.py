"""Module which contains frequency features (from sound recognition) for a mixture classifier"""

__author__ = "Mark Dubrovchenko"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

import numpy as np
from pywt import wavedec
from rework_pysatl_mpest.preprocessing.components_family.criterions.abstract_criterion import (
    AHistRecognitionCriterion,
)
from scipy.fft import rfft
from scipy.fftpack import dct
from scipy.signal import periodogram


class CDct(AHistRecognitionCriterion):
    def __init__(self, dct_type: int) -> None:
        self.dct_type = dct_type

    @property
    def name(self) -> str:
        return f"DCT C{self.dct_type} Criterion"

    def score(self, hist: np.ndarray) -> float:
        hist = hist / np.sum(hist)

        d = dct(hist, norm="ortho")
        return d[self.dct_type] if len(d) > self.dct_type else 0


class CDctEnergy(AHistRecognitionCriterion):
    @property
    def name(self) -> str:
        return "DCT Energy Criterion"

    def score(self, hist: np.ndarray) -> float:
        k = 4
        hist = hist / np.sum(hist)

        d = dct(hist, norm="ortho")
        return np.sum(d[k:] ** 2) if len(d) > k else 0


class CSpecBandwidth(AHistRecognitionCriterion):
    def __init__(self, noise: float = 10**-12) -> None:
        self.noise = noise

    @property
    def name(self) -> str:
        return "Spectral Bandwidth Criterion"

    def score(self, hist: np.ndarray) -> float:
        hist = hist / np.sum(hist)

        spec = np.abs(rfft(hist))
        freq = np.arange(len(spec))
        centroid = np.sum(spec * freq) / (np.sum(spec) + self.noise)
        return np.sqrt(np.sum((freq - centroid) ** 2 * spec) / (np.sum(spec) + self.noise))


class CSpecCentroid(AHistRecognitionCriterion):
    def __init__(self, noise: float = 10**-12) -> None:
        self.noise = noise

    @property
    def name(self) -> str:
        return "Spectral Centroid Criterion"

    def score(self, hist: np.ndarray) -> float:
        hist = hist / np.sum(hist)

        spec = np.abs(rfft(hist))
        freq = np.arange(len(spec))
        return np.sum(spec * freq) / (np.sum(spec) + self.noise)


class CSpecDecrease(AHistRecognitionCriterion):
    def __init__(self, noise: float = 10**-12) -> None:
        self.noise = noise

    @property
    def name(self) -> str:
        return "Spectral Decrease Criterion"

    def score(self, hist: np.ndarray) -> float:
        hist = hist / np.sum(hist)

        spec = np.abs(rfft(hist))
        if len(spec) <= 1:
            return 0

        m1 = spec[1:]
        return np.sum((m1[1:] - m1[:-1]) / np.arange(1, len(m1))) / (np.sum(m1) + self.noise)


class CSpecEnergy(AHistRecognitionCriterion):
    @property
    def name(self) -> str:
        return "Spectral Energy Criterion"

    def score(self, hist: np.ndarray) -> float:
        hist = hist / np.sum(hist)

        spec = rfft(hist)
        return np.sum(np.abs(spec**2))


class CSpecEntropy(AHistRecognitionCriterion):
    def __init__(self, noise: float = 10**-12) -> None:
        self.noise = noise

    @property
    def name(self) -> str:
        return "Spectral Entropy Criterion"

    def score(self, hist: np.ndarray) -> float:
        hist = hist / np.sum(hist)

        freq, psd = periodogram(hist)
        psd = psd / (np.sum(psd) + self.noise)
        return -np.sum(psd * np.log2(psd + self.noise))


class CSpecFlatness(AHistRecognitionCriterion):
    def __init__(self, noise: float = 10**-12) -> None:
        self.noise = noise

    @property
    def name(self) -> str:
        return "Spectral Flatness Criterion"

    def score(self, hist: np.ndarray) -> float:
        hist = hist / np.sum(hist)

        spec = np.abs(rfft(hist)) + self.noise
        gmean = np.exp(np.mean(np.log(spec)))
        return gmean / np.mean(spec)


class CSpecRolloff(AHistRecognitionCriterion):
    def __init__(self, roll: float = 0.85) -> None:
        self.roll = roll

    @property
    def name(self) -> str:
        return "Spectral Rolloff Criterion"

    def score(self, hist: np.ndarray) -> float:
        hist = hist / np.sum(hist)

        spec = np.abs(rfft(hist))
        cumsum = np.cumsum(spec)
        return np.where(cumsum >= self.roll * cumsum[-1])[0][0]


class CSpecSlope(AHistRecognitionCriterion):
    def __init__(self, noise: float = 10**-12) -> None:
        self.noise = noise

    @property
    def name(self) -> str:
        return "Spectral Slope Criterion"

    def score(self, hist: np.ndarray) -> float:
        hist = hist / np.sum(hist)

        spec = np.abs(rfft(hist))
        freq = np.arange(len(spec))
        fm, sm = freq.mean(), spec.mean()
        return np.sum((freq - fm) * (spec - sm)) / (np.sum((freq - fm) ** 2) + self.noise)


class CWaveletEnergy(AHistRecognitionCriterion):
    def __init__(self, level: int = 1, level_max: int = 3, wavelet: str = "haar") -> None:
        self.level = level
        self.level_max = level_max
        self.wavelet = wavelet

    @property
    def name(self) -> str:
        return f"Wavelet {self.level} Energy Criterion"

    def score(self, hist: np.ndarray) -> float:
        hist = hist / np.sum(hist)

        coeffs = wavedec(hist, self.wavelet, level=self.level_max)
        total = sum(np.sum(c**2) for c in coeffs)

        return np.sum(coeffs[self.level - 1] ** 2) / total


class CWaveletEntropy(AHistRecognitionCriterion):
    def __init__(
        self,
        level: int = 1,
        level_max: int = 3,
        wavelet: str = "haar",
        noise: float = 10**-12,
    ) -> None:
        self.level = level
        self.level_max = level_max
        self.wavelet = wavelet
        self.noise = noise

    @property
    def name(self) -> str:
        return f"Wavelet {self.level} Entropy Criterion"

    def score(self, hist: np.ndarray) -> float:
        hist = hist / np.sum(hist)

        coeffs = wavedec(hist, self.wavelet, level=self.level_max)
        c_abs = np.abs(coeffs[self.level - 1])
        c_norm = c_abs / (c_abs.sum() + self.noise)

        return -np.sum(c_norm * np.log(c_norm + self.noise))


class CWaveletLarge(AHistRecognitionCriterion):
    def __init__(
        self,
        level: int = 1,
        threshold: float = 0.1,
        level_max: int = 3,
        wavelet: str = "haar",
    ) -> None:
        self.level = level
        self.level_max = level_max
        self.threshold = threshold
        self.wavelet = wavelet

    @property
    def name(self) -> str:
        return f"Wavelet {self.level} Large {self.threshold} Criterion"

    def score(self, hist: np.ndarray) -> float:
        hist = hist / np.sum(hist)

        coeffs = wavedec(hist, self.wavelet, level=self.level_max)
        c_abs = np.abs(coeffs[self.level - 1])

        return np.mean(c_abs > np.max(c_abs) * self.threshold)


class CWaveletMean(AHistRecognitionCriterion):
    def __init__(self, level: int = 1, level_max: int = 3, wavelet: str = "haar") -> None:
        self.level = level
        self.level_max = level_max
        self.wavelet = wavelet

    @property
    def name(self) -> str:
        return f"Wavelet {self.level} Mean Criterion"

    def score(self, hist: np.ndarray) -> float:
        hist = hist / np.sum(hist)

        coeffs = wavedec(hist, self.wavelet, level=self.level_max)

        return np.mean(coeffs[self.level - 1])


class CWaveletStd(AHistRecognitionCriterion):
    def __init__(self, level: int = 1, level_max: int = 3, wavelet: str = "haar") -> None:
        self.level = level
        self.level_max = level_max
        self.wavelet = wavelet

    @property
    def name(self) -> str:
        return f"Wavelet {self.level} Std Criterion"

    def score(self, hist: np.ndarray) -> float:
        hist = hist / np.sum(hist)

        coeffs = wavedec(hist, self.wavelet, level=self.level_max)

        return np.std(coeffs[self.level - 1])
