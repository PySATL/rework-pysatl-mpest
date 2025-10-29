"""Module which contains peaks features for a mixture classifier"""

__author__ = "Mark Dubrovchenko"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

import numpy as np
from rework_pysatl_mpest.preprocessing.components_family.criterions.abstract_criterion import (
    APeaksRecognitionCriterion,
)
from scipy.signal import peak_widths


class CPeaksCount(APeaksRecognitionCriterion):
    @property
    def name(self) -> str:
        return "Peaks Count Criterion"

    def score(self, hist: np.ndarray) -> float:
        peaks = self._get_peaks(hist)[1]
        return len(peaks)


class CPeaksDistMax(APeaksRecognitionCriterion):
    @property
    def name(self) -> str:
        return "Peaks Distance Max Criterion"

    def score(self, hist: np.ndarray) -> float:
        peaks = self._get_peaks(hist)[1]
        return np.max(np.abs(np.diff(peaks)) - 1) / len(hist) if len(peaks) > 1 else 0


class CPeaksDistMean(APeaksRecognitionCriterion):
    @property
    def name(self) -> str:
        return "Peaks Distance Mean Criterion"

    def score(self, hist: np.ndarray) -> float:
        peaks = self._get_peaks(hist)[1]
        return np.mean(np.abs(np.diff(peaks)) - 1) / len(hist) if len(peaks) > 1 else 0


class CPeaksDistMin(APeaksRecognitionCriterion):
    @property
    def name(self) -> str:
        return "Peaks Distance Min Criterion"

    def score(self, hist: np.ndarray) -> float:
        peaks = self._get_peaks(hist)[1]
        return np.min(np.abs(np.diff(peaks)) - 1) / len(hist) if len(peaks) > 1 else 0


class CPeaksFirst(APeaksRecognitionCriterion):
    @property
    def name(self) -> str:
        return "Peaks First Criterion"

    def score(self, hist: np.ndarray) -> float:
        peaks = self._get_peaks(hist)[1]
        return float(1 in peaks)


class CPeaksLast(APeaksRecognitionCriterion):
    @property
    def name(self) -> str:
        return "Peaks Last Criterion"

    def score(self, hist: np.ndarray) -> float:
        peaks = self._get_peaks(hist)[1]
        return float(len(hist) in peaks)


class CPeaksMax(APeaksRecognitionCriterion):
    @property
    def name(self) -> str:
        return "Peaks Max Criterion"

    def score(self, hist: np.ndarray) -> float:
        hist, peaks = self._get_peaks(hist)
        hist /= np.sum(hist)
        return hist[peaks].max()


class CPeaksMean(APeaksRecognitionCriterion):
    @property
    def name(self) -> str:
        return "Peaks Mean Criterion"

    def score(self, hist: np.ndarray) -> float:
        hist, peaks = self._get_peaks(hist)
        hist /= np.sum(hist)
        return hist[peaks].mean()


class CPeaksMin(APeaksRecognitionCriterion):
    @property
    def name(self) -> str:
        return "Peaks Min Criterion"

    def score(self, hist: np.ndarray) -> float:
        hist, peaks = self._get_peaks(hist)
        hist /= np.sum(hist)
        return hist[peaks].min()


class CPeaksWidthMax(APeaksRecognitionCriterion):
    @property
    def name(self) -> str:
        return "Peaks Width Max Criterion"

    def score(self, hist: np.ndarray) -> float:
        hist, peaks = self._get_peaks(hist)
        return np.max(peak_widths(hist, peaks)[0]) / (len(hist) - 2)


class CPeaksWidthMean(APeaksRecognitionCriterion):
    @property
    def name(self) -> str:
        return "Peaks Width Mean Criterion"

    def score(self, hist: np.ndarray) -> float:
        hist, peaks = self._get_peaks(hist)
        return np.mean(peak_widths(hist, peaks)[0]) / (len(hist) - 2)


class CPeaksWidthMin(APeaksRecognitionCriterion):
    @property
    def name(self) -> str:
        return "Peaks Width Min Criterion"

    def score(self, hist: np.ndarray) -> float:
        hist, peaks = self._get_peaks(hist)
        return np.min(peak_widths(hist, peaks)[0]) / (len(hist) - 2)


class CValleysDistMax(APeaksRecognitionCriterion):
    @property
    def name(self) -> str:
        return "Valleys Distance Max Criterion"

    def score(self, hist: np.ndarray) -> float:
        valleys = self._get_peaks(hist, True)[1]
        return np.max(np.abs(np.diff(valleys)) - 1) / len(hist) if len(valleys) > 1 else 0


class CValleysDistMean(APeaksRecognitionCriterion):
    @property
    def name(self) -> str:
        return "Valleys Distance Mean Criterion"

    def score(self, hist: np.ndarray) -> float:
        valleys = self._get_peaks(hist, True)[1]
        return np.mean(np.abs(np.diff(valleys)) - 1) / len(hist) if len(valleys) > 1 else 0


class CValleysDistMin(APeaksRecognitionCriterion):
    @property
    def name(self) -> str:
        return "Valleys Distance Min Criterion"

    def score(self, hist: np.ndarray) -> float:
        valleys = self._get_peaks(hist, True)[1]
        return np.min(np.abs(np.diff(valleys)) - 1) / len(hist) if len(valleys) > 1 else 0


class CValleysMax(APeaksRecognitionCriterion):
    @property
    def name(self) -> str:
        return "Valleys Max Criterion"

    def score(self, hist: np.ndarray) -> float:
        hist, valleys = self._get_peaks(hist, True)
        hist /= np.sum(hist)
        return hist[valleys].max() if len(valleys) != 0 else 0


class CValleysMean(APeaksRecognitionCriterion):
    @property
    def name(self) -> str:
        return "Valleys Mean Criterion"

    def score(self, hist: np.ndarray) -> float:
        hist, valleys = self._get_peaks(hist, True)
        hist /= np.sum(hist)
        return hist[valleys].mean() if len(valleys) != 0 else 0


class CValleysMin(APeaksRecognitionCriterion):
    @property
    def name(self) -> str:
        return "Valleys Min Criterion"

    def score(self, hist: np.ndarray) -> float:
        hist, valleys = self._get_peaks(hist, True)
        hist /= np.sum(hist)
        return hist[valleys].min() if len(valleys) != 0 else 0


class CValleysWidthMax(APeaksRecognitionCriterion):
    @property
    def name(self) -> str:
        return "Valleys Width Max Criterion"

    def score(self, hist: np.ndarray) -> float:
        hist, valleys = self._get_peaks(hist, True)
        return np.max(peak_widths(-hist, valleys)[0]) / (len(hist) - 2) if len(valleys) != 0 else 0


class CValleysWidthMean(APeaksRecognitionCriterion):
    @property
    def name(self) -> str:
        return "Valleys Width Mean Criterion"

    def score(self, hist: np.ndarray) -> float:
        hist, valleys = self._get_peaks(hist, True)
        return np.mean(peak_widths(-hist, valleys)[0]) / (len(hist) - 2) if len(valleys) != 0 else 0


class CValleysWidthMin(APeaksRecognitionCriterion):
    @property
    def name(self) -> str:
        return "Valleys Width Min Criterion"

    def score(self, hist: np.ndarray) -> float:
        hist, valleys = self._get_peaks(hist, True)
        return np.min(peak_widths(-hist, valleys)[0]) / (len(hist) - 2) if len(valleys) != 0 else 0
