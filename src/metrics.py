import json
from pathlib import Path
import math
import matplotlib.pyplot as plt

# =========================================================
# CONFIG
# =========================================================
TRAIN_DIR = Path("/home/ap2194.UNT/micro/data/flan_t5_micro_grouped/checkpoint-42")
STATE_PATH = TRAIN_DIR / "trainer_state.json"
FIG_DIR = TRAIN_DIR / "figures"

# =========================================================
# HELPERS
# =========================================================
def safe_float(x):
    if x is None:
        return None
    try:
        x = float(x)
        if math.isnan(x) or math.isinf(x):
            return None
        return x
    except (TypeError, ValueError):
        return None


def filter_xy(x_vals, y_vals):
    pairs = [(x, y) for x, y in zip(x_vals, y_vals) if x is not None and y is not None]
    if not pairs:
        return [], []
    xs, ys = zip(*pairs)
    return list(xs), list(ys)


def save_single_line_plot(x, y, title, xlabel, ylabel, out_path):
    x, y = filter_xy(x, y)
    if not x:
        print(f"Skipped {out_path.name}: no valid data.")
        return

    plt.figure(figsize=(8, 5))
    plt.plot(x, y, marker="o")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def save_multi_line_plot(x, series, title, xlabel, ylabel, out_path):
    valid_any = False

    plt.figure(figsize=(8, 5))
    for label, y in series.items():
        fx, fy = filter_xy(x, y)
        if fx:
            plt.plot(fx, fy, marker="o", label=label)
            valid_any = True

    if not valid_any:
        plt.close()
        print(f"Skipped {out_path.name}: no valid data.")
        return

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


# =========================================================
# MAIN
# =========================================================
def main():
    if not STATE_PATH.exists():
        raise FileNotFoundError(f"trainer_state.json not found at: {STATE_PATH}")

    FIG_DIR.mkdir(parents=True, exist_ok=True)

    with open(STATE_PATH, "r", encoding="utf-8") as f:
        state = json.load(f)

    log_history = state.get("log_history", [])
    if not log_history:
        raise ValueError("No log_history found in trainer_state.json")

    train_steps = []
    train_losses = []

    eval_epochs = []
    eval_losses = []
    eval_rouge1 = []
    eval_rouge2 = []
    eval_rougeL = []
    eval_sacrebleu = []
    eval_count_match = []
    eval_within_range = []
    eval_format_rate = []
    eval_exact_match = []
    eval_avg_generated_count = []
    eval_avg_reference_count = []

    for row in log_history:
        # training logs
        if "loss" in row and "eval_loss" not in row:
            train_steps.append(safe_float(row.get("step")))
            train_losses.append(safe_float(row.get("loss")))

        # evaluation logs
        if "eval_loss" in row:
            eval_epochs.append(safe_float(row.get("epoch")))
            eval_losses.append(safe_float(row.get("eval_loss")))
            eval_rouge1.append(safe_float(row.get("eval_rouge1")))
            eval_rouge2.append(safe_float(row.get("eval_rouge2")))
            eval_rougeL.append(safe_float(row.get("eval_rougeL")))

            # support either name if you changed it mid-run
            eval_sacrebleu.append(
                safe_float(
                    row.get("eval_sacrebleu", row.get("eval_bleu"))
                )
            )

            eval_count_match.append(safe_float(row.get("eval_count_match_rate")))
            eval_within_range.append(safe_float(row.get("eval_within_range_rate")))
            eval_format_rate.append(safe_float(row.get("eval_format_rate")))
            eval_exact_match.append(safe_float(row.get("eval_exact_match_rate")))
            eval_avg_generated_count.append(safe_float(row.get("eval_avg_generated_count")))
            eval_avg_reference_count.append(safe_float(row.get("eval_avg_reference_count")))

    # -----------------------------------------------------
    # Plot 1: Training loss by step
    # -----------------------------------------------------
    save_single_line_plot(
        train_steps,
        train_losses,
        title="Training Loss",
        xlabel="Step",
        ylabel="Loss",
        out_path=FIG_DIR / "train_loss.png",
    )

    # -----------------------------------------------------
    # Plot 2: Evaluation loss by epoch
    # -----------------------------------------------------
    save_single_line_plot(
        eval_epochs,
        eval_losses,
        title="Evaluation Loss by Epoch",
        xlabel="Epoch",
        ylabel="Loss",
        out_path=FIG_DIR / "eval_loss.png",
    )

    # -----------------------------------------------------
    # Plot 3: ROUGE scores by epoch
    # -----------------------------------------------------
    save_multi_line_plot(
        eval_epochs,
        {
            "ROUGE-1": eval_rouge1,
            "ROUGE-2": eval_rouge2,
            "ROUGE-L": eval_rougeL,
        },
        title="ROUGE Scores by Epoch",
        xlabel="Epoch",
        ylabel="Score",
        out_path=FIG_DIR / "rouge_scores.png",
    )

    # -----------------------------------------------------
    # Plot 4: Structure metrics by epoch
    # -----------------------------------------------------
    save_multi_line_plot(
        eval_epochs,
        {
            "Count Match Rate": eval_count_match,
            "Within Range Rate": eval_within_range,
            "Format Rate": eval_format_rate,
            "Exact Match Rate": eval_exact_match,
        },
        title="Structure Metrics by Epoch",
        xlabel="Epoch",
        ylabel="Score",
        out_path=FIG_DIR / "structure_metrics.png",
    )

    # -----------------------------------------------------
    # Plot 5: Generated vs reference count by epoch
    # -----------------------------------------------------
    save_multi_line_plot(
        eval_epochs,
        {
            "Avg Generated Count": eval_avg_generated_count,
            "Avg Reference Count": eval_avg_reference_count,
        },
        title="Average Number of Sub-Questions by Epoch",
        xlabel="Epoch",
        ylabel="Count",
        out_path=FIG_DIR / "average_counts.png",
    )

    # -----------------------------------------------------
    # Plot 6: SacreBLEU by epoch, if present
    # -----------------------------------------------------
    if any(v is not None for v in eval_sacrebleu):
        save_single_line_plot(
            eval_epochs,
            eval_sacrebleu,
            title="SacreBLEU by Epoch",
            xlabel="Epoch",
            ylabel="Score",
            out_path=FIG_DIR / "sacrebleu.png",
        )
    else:
        print("Skipped sacrebleu.png: metric not found in trainer_state.json")

    print(f"\nAll done. Figures saved in: {FIG_DIR}")


if __name__ == "__main__":
    main()