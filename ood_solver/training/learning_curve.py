from __future__ import annotations

import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from ood_solver.utils import move_episode_to_device


def _mean_metrics(sum_metrics: dict[str, float], count: int) -> dict[str, float]:
    if count <= 0:
        return {}
    return {k: v / float(count) for k, v in sum_metrics.items()}


class LearningCurveMonitor:
    def __init__(
        self,
        metric: str = "seq_acc",
        mode: str = "max",
        min_delta: float = 1e-4,
        patience: int = 10,
        warmup_epochs: int = 0,
        loss_key: str = "loss",
        log_jsonl_path: str | None = None,
        instability_window: int = 5,
        loss_spike_ratio: float = 0.35,
        acc_drop_tol: float = 0.08,
        metric_ema_beta: float = 0.0,
    ):
        if mode not in {"max", "min"}:
            raise ValueError(f"Unsupported mode={mode!r}; expected 'max' or 'min'.")
        self.metric = metric
        self.mode = mode
        self.min_delta = float(min_delta)
        self.patience = max(1, int(patience))
        self.warmup_epochs = max(0, int(warmup_epochs))
        self.loss_key = loss_key
        self.best_value: float | None = None
        self.best_epoch: int = -1
        self.bad_epochs = 0
        self.metric_history: list[float] = []
        self.loss_history: list[float] = []
        self.instability_window = max(3, int(instability_window))
        self.loss_spike_ratio = float(loss_spike_ratio)
        self.acc_drop_tol = float(acc_drop_tol)
        self.metric_ema_beta = float(metric_ema_beta)
        self.metric_ema_value: float | None = None
        self.log_jsonl_path = Path(log_jsonl_path) if log_jsonl_path else None
        if self.log_jsonl_path is not None:
            self.log_jsonl_path.parent.mkdir(parents=True, exist_ok=True)

    def _is_improvement(self, value: float) -> bool:
        if self.best_value is None:
            return True
        if self.mode == "max":
            return value > (self.best_value + self.min_delta)
        return value < (self.best_value - self.min_delta)

    def evaluate_epoch(self, epoch: int, metrics: dict[str, float]) -> dict:
        warnings: list[str] = []
        suggestions: list[str] = []
        should_stop = False
        stop_reason = None

        value = metrics.get(self.metric, None)
        loss = metrics.get(self.loss_key, None)

        if value is None:
            warnings.append(f"metric_missing:{self.metric}")
        else:
            raw_value = float(value)
            self.metric_history.append(raw_value)
            if raw_value != raw_value:  # NaN
                should_stop = True
                stop_reason = f"nan_metric:{self.metric}"
            if self.metric_ema_beta > 0.0:
                if self.metric_ema_value is None:
                    self.metric_ema_value = raw_value
                else:
                    self.metric_ema_value = (
                        self.metric_ema_beta * self.metric_ema_value
                        + (1.0 - self.metric_ema_beta) * raw_value
                    )
                monitor_value = self.metric_ema_value
            else:
                monitor_value = raw_value
            if (not should_stop) and self._is_improvement(float(monitor_value)):
                self.best_value = float(monitor_value)
                self.best_epoch = int(epoch)
                self.bad_epochs = 0
            elif (not should_stop) and epoch >= self.warmup_epochs:
                self.bad_epochs += 1

        if loss is not None:
            self.loss_history.append(float(loss))
            if loss != loss:  # NaN
                should_stop = True
                stop_reason = "nan_loss"

        if len(self.loss_history) >= 2:
            prev_loss = self.loss_history[-2]
            cur_loss = self.loss_history[-1]
            if prev_loss > 0.0 and cur_loss > prev_loss * (1.0 + self.loss_spike_ratio):
                warnings.append("loss_spike")
                suggestions.append("Loss spiked; consider lowering lr or increasing regularization.")

        metric_is_acc = self.metric.endswith("seq_acc") or self.metric.endswith("seq_acc_lcb")
        if metric_is_acc and len(self.metric_history) >= 2:
            prev_acc = self.metric_history[-2]
            cur_acc = self.metric_history[-1]
            if (prev_acc - cur_acc) > self.acc_drop_tol:
                warnings.append("accuracy_drop")
                suggestions.append("Accuracy dropped sharply; inspect batch difficulty and lr stability.")

        if len(self.metric_history) >= self.instability_window:
            window = self.metric_history[-self.instability_window :]
            mean = sum(window) / len(window)
            var = sum((x - mean) ** 2 for x in window) / len(window)
            std = var ** 0.5
            if metric_is_acc and std > 0.06:
                warnings.append("high_metric_variance")
                suggestions.append("Learning curve is noisy; increase steps_per_epoch or reduce lr.")

        if (not should_stop) and (epoch >= self.warmup_epochs) and (self.bad_epochs >= self.patience):
            should_stop = True
            stop_reason = "early_stop_plateau"
            suggestions.append(
                "Metric plateaued; try stronger probe/rule supervision, rebalanced aux weights, or lr schedule."
            )

        event = {
            "epoch": int(epoch),
            "metrics": metrics,
            "monitor": {
                "metric": self.metric,
                "mode": self.mode,
                "monitor_value": monitor_value if value is not None else None,
                "best_value": self.best_value,
                "best_epoch": self.best_epoch,
                "bad_epochs": self.bad_epochs,
                "metric_ema_beta": self.metric_ema_beta,
                "metric_ema_value": self.metric_ema_value,
                "warnings": warnings,
                "suggestions": suggestions,
                "should_stop": should_stop,
                "stop_reason": stop_reason,
            },
        }
        if self.log_jsonl_path is not None:
            with self.log_jsonl_path.open("a") as f:
                f.write(json.dumps(event) + "\n")
        return event


def update_running_sums(sum_metrics: dict[str, float], step_metrics: dict[str, float]) -> None:
    for key, value in step_metrics.items():
        if isinstance(value, (int, float)):
            sum_metrics[key] = sum_metrics.get(key, 0.0) + float(value)


def format_metrics(metrics: dict[str, float], keys: list[str]) -> str:
    chunks: list[str] = []
    for key in keys:
        if key in metrics:
            val = metrics[key]
            if isinstance(val, float):
                chunks.append(f"{key}={val:.4f}")
            else:
                chunks.append(f"{key}={val}")
    return " ".join(chunks)


def mean_metrics(sum_metrics: dict[str, float], count: int) -> dict[str, float]:
    return _mean_metrics(sum_metrics, count)


@torch.no_grad()
def evaluate_fixed_batches(
    model,
    batches,
    probe_executor,
    device: str,
    structured: bool,
) -> dict[str, float]:
    model.eval()
    losses: list[float] = []
    seq_accs: list[float] = []
    count = 0

    for batch in batches:
        batch = move_episode_to_device(batch, device)
        if structured:
            logits, _, _ = model(batch, probe_executor)
        else:
            logits, _ = model(batch, probe_executor)

        b, l, v = logits.shape
        loss = F.cross_entropy(logits.reshape(b * l, v), batch.final_target.reshape(b * l))
        pred = logits.argmax(dim=-1)
        seq_acc = (pred == batch.final_target).float().mean()

        losses.append(float(loss.item()))
        seq_accs.append(float(seq_acc.item()))
        count += 1

    if count <= 0:
        return {
            "eval_loss": float("nan"),
            "eval_seq_acc": float("nan"),
            "eval_seq_acc_std": float("nan"),
            "eval_seq_acc_ci95": float("nan"),
            "eval_seq_acc_lcb": float("nan"),
            "eval_seq_acc_ucb": float("nan"),
        }
    mean_loss = sum(losses) / float(count)
    mean_acc = sum(seq_accs) / float(count)
    if count > 1:
        var = sum((x - mean_acc) ** 2 for x in seq_accs) / float(count - 1)
        std = math.sqrt(max(var, 0.0))
    else:
        std = 0.0
    ci95 = 1.96 * std / math.sqrt(float(count))
    lcb = mean_acc - ci95
    ucb = mean_acc + ci95
    return {
        "eval_loss": mean_loss,
        "eval_seq_acc": mean_acc,
        "eval_seq_acc_std": std,
        "eval_seq_acc_ci95": ci95,
        "eval_seq_acc_lcb": lcb,
        "eval_seq_acc_ucb": ucb,
    }
