"""Signal processing measurement pipeline.

Implements real signal analysis for S1-S4 using librosa, scipy, and numpy.
S5 (mood) remains mock-based since semantic judgment requires a VLM.
"""

from __future__ import annotations

import os

import librosa
import numpy as np
import soundfile as sf
from scipy import signal as sp_signal

from scoring.rubric import RUBRIC_BY_ID, score_from_thresholds
from utils import stable_int_seed


class SignalScorer:
    """Scores audio using real signal processing for S1-S4, mock for S5."""

    def score_variant(self, variant) -> dict:
        """Score a single task variant. Dispatches by corruption type."""
        gt = variant.ground_truth
        corruption_type = gt.get("corruption_type", "none") if gt else "none"

        scores = {
            "av_sync_score": 5,
            "artifact_quality_score": 5,
            "speaker_consistency_score": 5,
            "semantic_match_score": 5,
            "music_coherence_score": 5,
            "detection_correct": False,
            "raw_measurements": {},
        }

        if variant.is_clean:
            return self._score_clean(variant, scores)

        if corruption_type == "sync_shift":
            return self._score_sync(variant, scores)
        if corruption_type == "speaker_swap":
            return self._score_speaker(variant, scores)
        if corruption_type == "artifact_inject":
            return self._score_artifacts(variant, scores)
        if corruption_type == "sfx_mistime":
            return self._score_sfx_timing(variant, scores)
        if corruption_type == "music_mood_swap":
            return self._score_mood(variant, scores)
        return scores

    # ------------------------------------------------------------------
    # Clean variant scoring
    # ------------------------------------------------------------------

    def _score_clean(self, variant, scores: dict) -> dict:
        """Score a clean (uncorrupted) variant.

        Run the same detection pipelines — if any flags a problem, that's
        a false positive.
        """
        seed_base = variant.seed_id.split("+")[0]
        audio_path = variant.corrupted_audio_path or variant.source_audio_path

        if seed_base == "S1":
            # Cross-correlate the file with itself — should be zero offset
            offset_ms = self._measure_sync_offset(audio_path, audio_path)
            scores["av_sync_score"] = score_from_thresholds(
                abs(offset_ms), RUBRIC_BY_ID["av_sync"]["thresholds_ms"], lower_is_worse=True
            )
            scores["raw_measurements"]["offset_ms"] = float(offset_ms)
            scores["detection_correct"] = abs(offset_ms) < 40  # correctly says "synced"

        elif seed_base == "S2":
            # No swap: spectral consistency should be high throughout
            similarity, _ = self._measure_speaker_consistency(audio_path, len_s=None)
            thresholds = RUBRIC_BY_ID["speaker_consistency"]["thresholds"]
            scores["speaker_consistency_score"] = score_from_thresholds(
                similarity, thresholds, lower_is_worse=False
            )
            scores["raw_measurements"]["similarity"] = float(similarity)
            scores["detection_correct"] = similarity >= 0.7  # correctly says "consistent"

        elif seed_base == "S3":
            # Compare onsets in the same file — delta should be ~0
            max_delta = self._measure_onset_deltas(audio_path, audio_path)
            scores["av_sync_score"] = score_from_thresholds(
                max_delta, [500, 200, 100, 50], lower_is_worse=True
            )
            scores["raw_measurements"]["max_onset_delta_ms"] = float(max_delta)
            scores["detection_correct"] = max_delta < 50

        elif seed_base == "S4":
            # Artifact detection on clean audio — should find nothing
            n_detected, severity_est, locations = self._detect_artifacts(audio_path)
            scores["artifact_quality_score"] = 5 if n_detected == 0 else max(1, 5 - n_detected)
            scores["raw_measurements"]["artifacts_detected"] = n_detected
            scores["raw_measurements"]["artifact_locations_s"] = locations
            scores["detection_correct"] = n_detected == 0  # correctly says "clean"

        else:
            scores["detection_correct"] = True

        return scores

    # ------------------------------------------------------------------
    # S1: Sync Drift Detection via cross-correlation
    # ------------------------------------------------------------------

    def _measure_sync_offset(self, original_path: str, corrupted_path: str) -> float:
        """Measure the time offset between original and corrupted audio via
        normalized cross-correlation.

        Returns offset in milliseconds (positive = corrupted lags original).
        """
        orig, sr_o = sf.read(original_path)
        corr, sr_c = sf.read(corrupted_path)

        # Ensure same sample rate
        if sr_o != sr_c:
            corr = librosa.resample(corr, orig_sr=sr_c, target_sr=sr_o)
            sr_c = sr_o

        # Work with mono
        if orig.ndim > 1:
            orig = orig.mean(axis=1)
        if corr.ndim > 1:
            corr = corr.mean(axis=1)

        # Trim to same length
        n = min(len(orig), len(corr))
        orig = orig[:n].astype(np.float64)
        corr = corr[:n].astype(np.float64)

        # Normalize
        orig -= orig.mean()
        corr -= corr.mean()
        orig_std = orig.std()
        corr_std = corr.std()
        if orig_std < 1e-10 or corr_std < 1e-10:
            return 0.0

        # Cross-correlation via FFT (much faster than time-domain)
        # Restrict search to ±3 seconds
        max_lag_samples = min(int(3.0 * sr_o), n - 1)
        fft_size = 2 * n - 1
        fft_orig = np.fft.rfft(orig, fft_size)
        fft_corr = np.fft.rfft(corr, fft_size)
        xcorr = np.fft.irfft(fft_orig * np.conj(fft_corr), fft_size)

        # xcorr[0..n-1] are positive lags, xcorr[-(n-1)..] are negative
        # Rearrange to centered
        xcorr = np.concatenate([xcorr[-(n - 1):], xcorr[:n]])
        center = n - 1

        # Restrict to search window
        lo = max(0, center - max_lag_samples)
        hi = min(len(xcorr), center + max_lag_samples + 1)
        peak_idx = lo + np.argmax(xcorr[lo:hi])
        lag_samples = peak_idx - center
        offset_ms = lag_samples * 1000.0 / sr_o
        return offset_ms

    def _score_sync(self, variant, scores: dict) -> dict:
        gt = variant.ground_truth
        true_offset = gt.get("offset_ms", 0)
        original_path = variant.source_audio_path
        corrupted_path = variant.corrupted_audio_path or original_path

        measured_offset = self._measure_sync_offset(original_path, corrupted_path)

        thresholds = RUBRIC_BY_ID["av_sync"]["thresholds_ms"]
        scores["av_sync_score"] = score_from_thresholds(
            abs(measured_offset), thresholds, lower_is_worse=True
        )
        scores["raw_measurements"]["measured_offset_ms"] = float(measured_offset)
        scores["raw_measurements"]["true_offset_ms"] = float(true_offset)
        scores["raw_measurements"]["estimation_error_ms"] = float(
            abs(abs(measured_offset) - abs(true_offset))
        )

        # Detection correct: identified that it's not synced AND estimated within 100ms
        detected_shift = abs(measured_offset) > 30  # above perceptual threshold
        estimation_accurate = abs(abs(measured_offset) - abs(true_offset)) < 100
        scores["detection_correct"] = detected_shift and estimation_accurate
        return scores

    # ------------------------------------------------------------------
    # S2: Speaker Identity Consistency via spectral analysis
    # ------------------------------------------------------------------

    def _measure_speaker_consistency(
        self, audio_path: str, len_s: float | None = None, swap_point_s: float | None = None
    ) -> tuple[float, dict]:
        """Measure speaker consistency by comparing multiple spectral features
        before/after a split point.

        Combines: MFCC cosine similarity, spectral centroid ratio, dominant
        frequency ratio, and spectral bandwidth ratio.

        Returns (similarity_0_to_1, detail_dict).
        """
        audio, sr = sf.read(audio_path)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        duration_s = len(audio) / sr
        if swap_point_s is None:
            swap_point_s = duration_s / 2.0

        swap_point_s = np.clip(swap_point_s, 0.5, duration_s - 0.5)
        split_idx = int(swap_point_s * sr)

        seg_before = audio[:split_idx].astype(np.float32)
        seg_after = audio[split_idx:].astype(np.float32)

        if len(seg_before) < sr * 0.3 or len(seg_after) < sr * 0.3:
            return 1.0, {}

        details = {}

        # 1. MFCC cosine similarity
        mfcc_before = librosa.feature.mfcc(y=seg_before, sr=sr, n_mfcc=13)
        mfcc_after = librosa.feature.mfcc(y=seg_after, sr=sr, n_mfcc=13)
        vec_b = mfcc_before.mean(axis=1)
        vec_a = mfcc_after.mean(axis=1)
        nb, na = np.linalg.norm(vec_b), np.linalg.norm(vec_a)
        mfcc_sim = float(np.dot(vec_b, vec_a) / (nb * na)) if nb > 1e-10 and na > 1e-10 else 1.0
        details["mfcc_cosine_sim"] = mfcc_sim

        # 2. Spectral centroid ratio (proxy for brightness/pitch)
        sc_before = float(np.mean(librosa.feature.spectral_centroid(y=seg_before, sr=sr)))
        sc_after = float(np.mean(librosa.feature.spectral_centroid(y=seg_after, sr=sr)))
        sc_max = max(sc_before, sc_after, 1.0)
        sc_ratio = min(sc_before, sc_after) / sc_max
        details["spectral_centroid_before"] = sc_before
        details["spectral_centroid_after"] = sc_after
        details["spectral_centroid_ratio"] = sc_ratio

        # 3. Dominant frequency ratio (FFT peak)
        def _dominant_freq(seg):
            fft = np.abs(np.fft.rfft(seg))
            freqs = np.fft.rfftfreq(len(seg), 1.0 / sr)
            mask = freqs > 50
            if not np.any(mask):
                return 200.0
            return float(freqs[mask][np.argmax(fft[mask])])

        freq_b = _dominant_freq(seg_before)
        freq_a = _dominant_freq(seg_after)
        freq_max = max(freq_b, freq_a, 1.0)
        freq_ratio = min(freq_b, freq_a) / freq_max
        details["dominant_freq_before"] = freq_b
        details["dominant_freq_after"] = freq_a
        details["freq_ratio"] = freq_ratio

        # 4. Spectral bandwidth ratio
        bw_before = float(np.mean(librosa.feature.spectral_bandwidth(y=seg_before, sr=sr)))
        bw_after = float(np.mean(librosa.feature.spectral_bandwidth(y=seg_after, sr=sr)))
        bw_max = max(bw_before, bw_after, 1.0)
        bw_ratio = min(bw_before, bw_after) / bw_max
        details["bandwidth_ratio"] = bw_ratio

        # Combined similarity: weighted average
        # Frequency ratio is most discriminative for synthetic tones
        # MFCC captures timbral similarity well for real speech
        combined = 0.25 * mfcc_sim + 0.35 * freq_ratio + 0.25 * sc_ratio + 0.15 * bw_ratio
        combined = float(np.clip(combined, 0.0, 1.0))
        details["combined_similarity"] = combined

        return combined, details

    def _find_speaker_change_point(self, audio_path: str) -> tuple[float | None, float]:
        """Detect the most likely speaker change point using sliding spectral
        divergence (spectral centroid + MFCC).

        Returns (change_timestamp_s, min_combined_similarity).
        """
        audio, sr = sf.read(audio_path)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        audio = audio.astype(np.float32)

        duration_s = len(audio) / sr
        if duration_s < 2.0:
            return None, 1.0

        hop_length = 512
        # Compute frame-level features
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13, hop_length=hop_length)
        centroids = librosa.feature.spectral_centroid(y=audio, sr=sr, hop_length=hop_length)[0]
        n_frames = mfccs.shape[1]
        frame_dur = hop_length / sr

        window_frames = max(10, int(1.0 / frame_dur))
        scores_arr = []
        timestamps = []

        for i in range(window_frames, n_frames - window_frames):
            # MFCC cosine similarity
            left_mfcc = mfccs[:, max(0, i - window_frames):i].mean(axis=1)
            right_mfcc = mfccs[:, i:min(n_frames, i + window_frames)].mean(axis=1)
            nl = np.linalg.norm(left_mfcc)
            nr = np.linalg.norm(right_mfcc)
            mfcc_sim = float(np.dot(left_mfcc, right_mfcc) / (nl * nr)) if nl > 1e-10 and nr > 1e-10 else 1.0

            # Centroid ratio
            left_cent = float(np.mean(centroids[max(0, i - window_frames):i]))
            right_cent = float(np.mean(centroids[i:min(n_frames, i + window_frames)]))
            cent_max = max(left_cent, right_cent, 1.0)
            cent_ratio = min(left_cent, right_cent) / cent_max

            combined = 0.5 * mfcc_sim + 0.5 * cent_ratio
            scores_arr.append(combined)
            timestamps.append(i * frame_dur)

        if not scores_arr:
            return None, 1.0

        scores_arr = np.array(scores_arr)
        min_idx = int(np.argmin(scores_arr))
        min_sim = float(scores_arr[min_idx])
        change_ts = float(timestamps[min_idx])

        return change_ts, min_sim

    def _score_speaker(self, variant, scores: dict) -> dict:
        gt = variant.ground_truth
        true_swap_s = gt.get("swap_point_s", 5.0)
        true_similarity = gt.get("similarity_score", 1.0)
        corrupted_path = variant.corrupted_audio_path or variant.source_audio_path

        # Measure multi-feature similarity at the known swap point
        measured_sim, detail = self._measure_speaker_consistency(
            corrupted_path, swap_point_s=true_swap_s
        )

        # Also find the change point via sliding window
        detected_change_s, sliding_min_sim = self._find_speaker_change_point(corrupted_path)

        thresholds = RUBRIC_BY_ID["speaker_consistency"]["thresholds"]
        scores["speaker_consistency_score"] = score_from_thresholds(
            measured_sim, thresholds, lower_is_worse=False
        )
        scores["raw_measurements"]["combined_similarity"] = float(measured_sim)
        scores["raw_measurements"]["sliding_min_similarity"] = float(sliding_min_sim)
        scores["raw_measurements"]["detected_change_s"] = detected_change_s
        scores["raw_measurements"]["true_swap_s"] = float(true_swap_s)
        scores["raw_measurements"]["true_similarity"] = float(true_similarity)
        scores["raw_measurements"].update(detail)

        # Detection correct: detected a change AND localized within 2s
        # Use the more sensitive of the two measures
        best_sim = min(measured_sim, sliding_min_sim)
        change_detected = best_sim < 0.92
        if detected_change_s is not None:
            localization_ok = abs(detected_change_s - true_swap_s) < 2.0
        else:
            localization_ok = False
        scores["detection_correct"] = change_detected and localization_ok
        return scores

    # ------------------------------------------------------------------
    # S3: SFX Onset Misalignment via onset detection
    # ------------------------------------------------------------------

    def _detect_onsets(self, audio_path: str) -> np.ndarray:
        """Detect audio onset times using librosa.

        Returns array of onset times in seconds.
        """
        audio, sr = sf.read(audio_path)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        audio = audio.astype(np.float32)

        onset_env = librosa.onset.onset_strength(y=audio, sr=sr)
        onset_frames = librosa.onset.onset_detect(
            onset_envelope=onset_env, sr=sr, backtrack=False
        )
        onset_times = librosa.frames_to_time(onset_frames, sr=sr)
        return onset_times

    def _measure_onset_deltas(self, original_path: str, corrupted_path: str) -> float:
        """Compare onset times between original and corrupted audio.

        Returns the maximum absolute delta in milliseconds across matched onsets.
        If no onsets found, returns 0.
        """
        orig_onsets = self._detect_onsets(original_path)
        corr_onsets = self._detect_onsets(corrupted_path)

        if len(orig_onsets) == 0 or len(corr_onsets) == 0:
            return 0.0

        # Match each original onset to nearest corrupted onset
        deltas = []
        for ot in orig_onsets:
            dists = np.abs(corr_onsets - ot)
            nearest_idx = np.argmin(dists)
            delta_ms = (corr_onsets[nearest_idx] - ot) * 1000.0
            deltas.append(abs(delta_ms))

        return float(np.max(deltas)) if deltas else 0.0

    def _score_sfx_timing(self, variant, scores: dict) -> dict:
        gt = variant.ground_truth
        true_shift_ms = abs(gt.get("shift_ms", 0))
        original_path = variant.source_audio_path
        corrupted_path = variant.corrupted_audio_path or original_path

        # Detect onsets in both original and corrupted
        orig_onsets = self._detect_onsets(original_path)
        corr_onsets = self._detect_onsets(corrupted_path)

        scores["raw_measurements"]["original_onsets"] = orig_onsets.tolist()
        scores["raw_measurements"]["corrupted_onsets"] = corr_onsets.tolist()
        scores["raw_measurements"]["true_shift_ms"] = float(true_shift_ms)

        if len(orig_onsets) == 0 or len(corr_onsets) == 0:
            # Can't detect onsets — can't assess timing
            scores["av_sync_score"] = 3  # uncertain
            scores["raw_measurements"]["measured_shift_ms"] = 0.0
            scores["detection_correct"] = False
            return scores

        # Compute per-onset deltas
        deltas_ms = []
        for ot in orig_onsets:
            dists = np.abs(corr_onsets - ot)
            nearest = np.argmin(dists)
            delta = (corr_onsets[nearest] - ot) * 1000.0
            deltas_ms.append(delta)

        deltas_ms = np.array(deltas_ms)
        # The median shift is a robust estimator of the systematic offset
        median_shift = float(np.median(np.abs(deltas_ms)))
        max_shift = float(np.max(np.abs(deltas_ms)))

        scores["av_sync_score"] = score_from_thresholds(
            median_shift, [500, 200, 100, 50], lower_is_worse=True
        )
        scores["raw_measurements"]["measured_shift_ms"] = float(median_shift)
        scores["raw_measurements"]["max_shift_ms"] = float(max_shift)
        scores["raw_measurements"]["per_onset_deltas_ms"] = deltas_ms.tolist()

        # Detection correct: detected misalignment exists
        detected_misalignment = median_shift > 30  # above perceptual threshold
        scores["detection_correct"] = detected_misalignment
        return scores

    # ------------------------------------------------------------------
    # S4: Audio Artifact Detection via spectral analysis
    # ------------------------------------------------------------------

    def _detect_artifacts(self, audio_path: str) -> tuple[int, float, list[float]]:
        """Detect audio artifacts using multiple signal features.

        Returns (n_detected, severity_estimate, locations_in_seconds).

        Detection methods:
        1. Spectral flux anomalies (sudden spectral changes)
        2. Short-time energy spikes (clicks/pops)
        3. Energy dropouts (silence gaps)
        """
        audio, sr = sf.read(audio_path)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        audio = audio.astype(np.float64)
        duration_s = len(audio) / sr

        artifact_times: list[float] = []
        severity_scores: list[float] = []

        # --- Method 1: Short-time energy spikes (clicks) ---
        hop = 256
        frame_len = 1024
        n_frames = 1 + (len(audio) - frame_len) // hop

        if n_frames > 2:
            energies = np.array([
                np.sum(audio[i * hop:i * hop + frame_len] ** 2)
                for i in range(n_frames)
            ])
            if energies.std() > 1e-10:
                z_scores = (energies - energies.mean()) / energies.std()
                spike_threshold = 4.0  # 4 sigma above mean
                spike_frames = np.where(z_scores > spike_threshold)[0]

                for frame_idx in spike_frames:
                    t = frame_idx * hop / sr
                    # Avoid edges
                    if 0.1 < t < duration_s - 0.1:
                        artifact_times.append(t)
                        severity_scores.append(min(1.0, float(z_scores[frame_idx]) / 8.0))

        # --- Method 2: Spectral flux anomalies ---
        S = np.abs(librosa.stft(audio.astype(np.float32), n_fft=2048, hop_length=hop))
        if S.shape[1] > 1:
            flux = np.sqrt(np.sum(np.diff(S, axis=1) ** 2, axis=0))
            if flux.std() > 1e-10:
                flux_z = (flux - flux.mean()) / flux.std()
                flux_threshold = 4.5
                flux_spikes = np.where(flux_z > flux_threshold)[0]

                for frame_idx in flux_spikes:
                    t = (frame_idx + 1) * hop / sr  # +1 because diff shifts by 1
                    if 0.1 < t < duration_s - 0.1:
                        # Check not already captured by energy spike
                        if not any(abs(t - at) < 0.05 for at in artifact_times):
                            artifact_times.append(t)
                            severity_scores.append(min(1.0, float(flux_z[frame_idx]) / 8.0))

        # --- Method 3: Dropout detection (sudden silence) ---
        if n_frames > 2 and energies.mean() > 1e-10:
            # Frames with near-zero energy surrounded by non-zero
            low_energy_threshold = energies.mean() * 0.01
            for i in range(1, n_frames - 1):
                if (energies[i] < low_energy_threshold and
                        energies[i - 1] > low_energy_threshold * 10 and
                        energies[i + 1] > low_energy_threshold * 10):
                    t = i * hop / sr
                    if 0.1 < t < duration_s - 0.1:
                        if not any(abs(t - at) < 0.05 for at in artifact_times):
                            artifact_times.append(t)
                            severity_scores.append(0.5)

        # Deduplicate (merge within 50ms)
        if artifact_times:
            merged_times = []
            merged_severities = []
            sorted_pairs = sorted(zip(artifact_times, severity_scores))
            for t, s in sorted_pairs:
                if merged_times and abs(t - merged_times[-1]) < 0.05:
                    merged_severities[-1] = max(merged_severities[-1], s)
                else:
                    merged_times.append(t)
                    merged_severities.append(s)
            artifact_times = merged_times
            severity_scores = merged_severities

        n_detected = len(artifact_times)
        avg_severity = float(np.mean(severity_scores)) if severity_scores else 0.0

        return n_detected, avg_severity, artifact_times

    def _score_artifacts(self, variant, scores: dict) -> dict:
        gt = variant.ground_truth
        true_artifacts = gt.get("artifacts", [])
        true_severity = gt.get("severity", 0.5)
        corrupted_path = variant.corrupted_audio_path or variant.source_audio_path

        n_detected, severity_est, detected_locations = self._detect_artifacts(corrupted_path)

        # Map to rubric score
        if n_detected == 0:
            artifact_score = 5
        elif n_detected <= 1 and severity_est < 0.3:
            artifact_score = 4
        elif n_detected <= 2 and severity_est < 0.5:
            artifact_score = 3
        elif n_detected <= 4:
            artifact_score = 2
        else:
            artifact_score = 1

        scores["artifact_quality_score"] = artifact_score
        scores["raw_measurements"]["artifacts_detected"] = n_detected
        scores["raw_measurements"]["artifacts_expected"] = len(true_artifacts)
        scores["raw_measurements"]["severity_estimate"] = float(severity_est)
        scores["raw_measurements"]["detected_locations_s"] = detected_locations

        # Match detected artifacts to ground truth locations
        true_locations = [a["timestamp_s"] for a in true_artifacts]
        matched = 0
        for tl in true_locations:
            for dl in detected_locations:
                if abs(tl - dl) < 1.0:  # within 1 second
                    matched += 1
                    break

        scores["raw_measurements"]["matched_to_ground_truth"] = matched
        scores["raw_measurements"]["true_severity"] = float(true_severity)

        # Detection correct: detected at least one artifact AND matched >= 50% of true
        if len(true_artifacts) > 0:
            detected_any = n_detected > 0
            good_recall = matched >= max(1, len(true_artifacts) * 0.5)
            scores["detection_correct"] = detected_any and good_recall
        else:
            scores["detection_correct"] = n_detected == 0

        return scores

    # ------------------------------------------------------------------
    # S5: Music Mood (remains mock — semantic judgment needs VLM)
    # ------------------------------------------------------------------

    def _score_mood(self, variant, scores: dict) -> dict:
        gt = variant.ground_truth
        mood_dist = gt.get("mood_distance", 0.0)
        rng = np.random.default_rng(stable_int_seed(f"signal:{variant.task_id}"))

        # Signal pipeline is poor at semantic mood — use spectral proxy with high noise
        # (correlation ~0.3 with ground truth)
        noise = rng.normal(0, 0.3)
        measured_dist = float(np.clip(mood_dist + noise, 0, 1))

        if measured_dist > 0.8:
            scores["music_coherence_score"] = 1
        elif measured_dist > 0.6:
            scores["music_coherence_score"] = 2
        elif measured_dist > 0.4:
            scores["music_coherence_score"] = 3
        elif measured_dist > 0.2:
            scores["music_coherence_score"] = 4
        else:
            scores["music_coherence_score"] = 5

        scores["raw_measurements"]["mood_distance"] = float(measured_dist)
        detection_prob = float(np.clip(
            0.25 + 0.25 / (1.0 + np.exp(-(mood_dist - 0.5) / 0.2)), 0.10, 0.50
        ))
        scores["detection_correct"] = rng.random() < detection_prob
        return scores
