"""Model comparison analysis and divergence detection."""

from __future__ import annotations

import json
import os
from collections import defaultdict

import numpy as np

from config import OUTPUT_DIR, PLOTS_DIR


class ModelComparator:
    """Analyzes results from both models to surface differential failure modes."""

    def __init__(
        self,
        variants: list,
        signal_results: dict,
        gemini_results: dict,
        report_path: str | None = None,
        plots_dir: str | None = None,
        plot_prefix: str = "",
    ):
        self.variants = variants
        self.signal = signal_results
        self.gemini = gemini_results
        self.report_path = report_path or os.path.join(OUTPUT_DIR, "comparison_report.json")
        self.plots_dir = plots_dir or PLOTS_DIR
        self.plot_prefix = plot_prefix

    def run_all(self) -> dict:
        """Run all analyses and return combined report."""
        report = {
            "per_seed_accuracy": self.per_seed_accuracy(),
            "tier_performance": self.tier_performance(),
            "divergence_matrix": self.divergence_matrix(),
            "difficulty_calibration": self.difficulty_calibration(),
        }

        report_dir = os.path.dirname(self.report_path) or "."
        os.makedirs(report_dir, exist_ok=True)
        with open(self.report_path, "w") as f:
            json.dump(report, f, indent=2, default=str)
        print(f"Comparison report saved to {self.report_path}")

        self._print_summary(report)
        self._generate_plots()
        return report

    def per_seed_accuracy(self) -> dict:
        """Per-seed accuracy table grouped by severity."""
        results = {}
        by_seed: dict[str, list] = defaultdict(list)
        for v in self.variants:
            by_seed[v.seed_id].append(v)

        for seed_id, variants in sorted(by_seed.items()):
            seed_data = {"signal": {}, "gemini": {}, "total": len(variants)}
            sig_correct = sum(1 for v in variants
                             if self.signal.get(v.task_id, {}).get("detection_correct", False))
            gem_correct = sum(1 for v in variants
                             if self.gemini.get(v.task_id, {}).get("detection_correct", False))
            seed_data["signal"]["accuracy"] = sig_correct / max(1, len(variants))
            seed_data["gemini"]["accuracy"] = gem_correct / max(1, len(variants))

            # Group by difficulty
            for diff in ["easy", "medium", "hard"]:
                diff_vars = [v for v in variants if v.difficulty_estimate == diff]
                if diff_vars:
                    sc = sum(1 for v in diff_vars
                             if self.signal.get(v.task_id, {}).get("detection_correct"))
                    gc = sum(1 for v in diff_vars
                             if self.gemini.get(v.task_id, {}).get("detection_correct"))
                    seed_data["signal"][diff] = sc / len(diff_vars)
                    seed_data["gemini"][diff] = gc / len(diff_vars)

            results[seed_id] = seed_data
        return results

    def tier_performance(self) -> dict:
        """Average accuracy per tier per model."""
        by_tier: dict[int, list] = defaultdict(list)
        for v in self.variants:
            by_tier[v.tier].append(v)

        results = {}
        for tier, variants in sorted(by_tier.items()):
            sc = sum(1 for v in variants
                     if self.signal.get(v.task_id, {}).get("detection_correct"))
            gc = sum(1 for v in variants
                     if self.gemini.get(v.task_id, {}).get("detection_correct"))
            n = len(variants)
            results[f"tier_{tier}"] = {
                "signal_accuracy": sc / max(1, n),
                "gemini_accuracy": gc / max(1, n),
                "n_variants": n,
            }
        return results

    def divergence_matrix(self) -> dict:
        """Classify each variant into 4 quadrants of agreement/disagreement."""
        both_correct = []
        both_wrong = []
        signal_advantage = []  # signal correct, gemini wrong
        gemini_advantage = []  # gemini correct, signal wrong

        for v in self.variants:
            sc = self.signal.get(v.task_id, {}).get("detection_correct", False)
            gc = self.gemini.get(v.task_id, {}).get("detection_correct", False)
            entry = {"task_id": v.task_id, "seed_id": v.seed_id,
                     "tier": v.tier, "difficulty": v.difficulty_estimate}
            if sc and gc:
                both_correct.append(entry)
            elif not sc and not gc:
                both_wrong.append(entry)
            elif sc and not gc:
                signal_advantage.append(entry)
            else:
                gemini_advantage.append(entry)

        return {
            "both_correct": {"count": len(both_correct), "examples": both_correct[:5]},
            "both_wrong": {"count": len(both_wrong), "examples": both_wrong[:5]},
            "signal_advantage": {"count": len(signal_advantage), "examples": signal_advantage[:5]},
            "gemini_advantage": {"count": len(gemini_advantage), "examples": gemini_advantage[:5]},
        }

    def difficulty_calibration(self) -> dict:
        """Report what severity ranges produce 35-75% pass rate per model per seed."""
        by_seed: dict[str, list] = defaultdict(list)
        for v in self.variants:
            if not v.is_clean:
                by_seed[v.seed_id].append(v)

        results = {}
        for seed_id, variants in by_seed.items():
            # Group by difficulty
            for model_name, model_results in [("signal", self.signal), ("gemini", self.gemini)]:
                for diff in ["easy", "medium", "hard"]:
                    diff_vars = [v for v in variants if v.difficulty_estimate == diff]
                    if diff_vars:
                        rate = sum(1 for v in diff_vars
                                   if model_results.get(v.task_id, {}).get("detection_correct")) / len(diff_vars)
                        key = f"{seed_id}_{model_name}_{diff}"
                        results[key] = {
                            "pass_rate": rate,
                            "n": len(diff_vars),
                            "in_discriminating_band": 0.35 <= rate <= 0.75,
                        }
        return results

    def _print_summary(self, report: dict) -> None:
        print("\n" + "=" * 60)
        print("MODEL COMPARISON SUMMARY")
        print("=" * 60)

        # Tier performance
        print("\nTier-Level Performance:")
        print(f"  {'Tier':<8} {'Signal':>12} {'Gemini':>12} {'N':>6}")
        print(f"  {'-'*8} {'-'*12} {'-'*12} {'-'*6}")
        for tier_key, data in sorted(report["tier_performance"].items()):
            print(f"  {tier_key:<8} {data['signal_accuracy']:>11.1%} "
                  f"{data['gemini_accuracy']:>11.1%} {data['n_variants']:>6}")

        # Divergence matrix
        div = report["divergence_matrix"]
        total = sum(div[k]["count"] for k in div)
        print(f"\nDivergence Matrix (n={total}):")
        print(f"  Both correct:      {div['both_correct']['count']:>4} "
              f"({div['both_correct']['count']/max(1,total):.1%})")
        print(f"  Both wrong:        {div['both_wrong']['count']:>4} "
              f"({div['both_wrong']['count']/max(1,total):.1%})")
        print(f"  Signal advantage:  {div['signal_advantage']['count']:>4} "
              f"({div['signal_advantage']['count']/max(1,total):.1%})")
        print(f"  Gemini advantage:  {div['gemini_advantage']['count']:>4} "
              f"({div['gemini_advantage']['count']/max(1,total):.1%})")

    def _generate_plots(self) -> None:
        """Generate severity-accuracy curve plots."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import seaborn as sns
        except ImportError:
            print("  [matplotlib not available, skipping plots]")
            return

        os.makedirs(self.plots_dir, exist_ok=True)
        sns.set_theme(style="whitegrid")

        # Per-seed severity curves
        by_seed: dict[str, list] = defaultdict(list)
        for v in self.variants:
            if not v.is_clean:
                by_seed[v.seed_id.split("+")[0]].append(v)

        for seed_id, variants in sorted(by_seed.items()):
            if "+" in seed_id:
                continue
            # Extract numeric severity
            sev_signal = defaultdict(list)
            sev_gemini = defaultdict(list)
            for v in variants:
                # Get first numeric param value as severity proxy
                sev = None
                for val in v.corruption_params.values():
                    if isinstance(val, (int, float)):
                        sev = val
                        break
                if sev is None:
                    continue
                sc = self.signal.get(v.task_id, {}).get("detection_correct", False)
                gc = self.gemini.get(v.task_id, {}).get("detection_correct", False)
                sev_signal[sev].append(int(sc))
                sev_gemini[sev].append(int(gc))

            if not sev_signal:
                continue

            sevs = sorted(sev_signal.keys())
            sig_acc = [np.mean(sev_signal[s]) for s in sevs]
            gem_acc = [np.mean(sev_gemini[s]) for s in sevs]

            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(sevs, sig_acc, "o-", label="Signal Pipeline", color="#2196F3")
            ax.plot(sevs, gem_acc, "s--", label="Gemini 2.5 Flash", color="#FF5722")
            ax.set_xlabel("Corruption Severity")
            ax.set_ylabel("Detection Accuracy")
            ax.set_title(f"Seed {seed_id}: Detection Accuracy vs Severity")
            ax.set_ylim(0, 1.05)
            ax.legend()
            ax.axhspan(0.35, 0.75, alpha=0.1, color="green", label="Discriminating band")
            fig.tight_layout()
            fig.savefig(
                os.path.join(self.plots_dir, f"{self.plot_prefix}severity_curve_{seed_id}.png"),
                dpi=150,
            )
            plt.close(fig)

        # Tier comparison bar chart
        tier_data = {}
        for v in self.variants:
            tier_data.setdefault(v.tier, {"signal": [], "gemini": []})
            tier_data[v.tier]["signal"].append(
                int(self.signal.get(v.task_id, {}).get("detection_correct", False)))
            tier_data[v.tier]["gemini"].append(
                int(self.gemini.get(v.task_id, {}).get("detection_correct", False)))

        tiers = sorted(tier_data.keys())
        sig_means = [np.mean(tier_data[t]["signal"]) for t in tiers]
        gem_means = [np.mean(tier_data[t]["gemini"]) for t in tiers]

        fig, ax = plt.subplots(figsize=(7, 5))
        x = np.arange(len(tiers))
        w = 0.35
        ax.bar(x - w / 2, sig_means, w, label="Signal Pipeline", color="#2196F3")
        ax.bar(x + w / 2, gem_means, w, label="Gemini 2.5 Flash", color="#FF5722")
        ax.set_xticks(x)
        ax.set_xticklabels([f"Tier {t}" for t in tiers])
        ax.set_ylabel("Detection Accuracy")
        ax.set_title("Detection Accuracy by Task Tier")
        ax.set_ylim(0, 1.05)
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(self.plots_dir, f"{self.plot_prefix}tier_comparison.png"), dpi=150)
        plt.close(fig)

        print(f"  Plots saved to {self.plots_dir}/")
