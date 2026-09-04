import os
import json

import numpy as np
import pandas as pd

DATA_PATH_PARQUET = "returns_dataset.parquet"
DATA_PATH_CSV = "returns_dataset.csv"
CALIB_PREDICTIONS_PATH = "return_risk_calib_predictions.csv"
MODEL_RESULTS_PATH = "return_risk_results.json"

THRESHOLDS_OUT = "return_financial_thresholds.json"
LANDSCAPE_OUT = "return_cost_landscape.csv"
SENSITIVITY_OUT = "return_threshold_sensitivity.csv"

COST_PHOTO_REVIEW = 2.00
FRICTION_PENALTY = 10.00

TRAIN_FRACTION = 0.70
CALIB_FRACTION = 0.15

N_THRESHOLDS = 99
SENSITIVITY_RADIUS = 2


def load_calibration_set():
    if os.path.exists(DATA_PATH_PARQUET):
        df = pd.read_parquet(DATA_PATH_PARQUET)
    elif os.path.exists(DATA_PATH_CSV):
        df = pd.read_csv(DATA_PATH_CSV)
    else:
        raise FileNotFoundError(f"Neither {DATA_PATH_PARQUET} nor {DATA_PATH_CSV} found.")

    df = df.sort_values('ReturnDT').reset_index(drop=True)
    n = len(df)
    train_end = int(n * TRAIN_FRACTION)
    calib_end = int(n * (TRAIN_FRACTION + CALIB_FRACTION))

    calib_df = df.iloc[train_end:calib_end].copy().reset_index(drop=True)
    preds = pd.read_csv(CALIB_PREDICTIONS_PATH)

    if len(preds) != len(calib_df):
        raise ValueError(
            f"Row count mismatch: recreated calibration split has {len(calib_df)} rows, "
            f"{CALIB_PREDICTIONS_PATH} has {len(preds)}. The dataset or split fractions "
            f"used here don't match what train_return_scorer produced."
        )

    if not np.array_equal(preds['actual_fraud'].to_numpy(), calib_df['Is_Fraud'].to_numpy()):
        raise ValueError(
            f"Label mismatch between the recreated calibration split and "
            f"{CALIB_PREDICTIONS_PATH} -- they are not row-aligned. Re-run "
            f"train_return_scorer against the current dataset before calibrating."
        )

    calib_df['fraud_probability'] = preds['fraud_probability'].to_numpy()
    return calib_df


def compute_baselines(actuals, margins, ltvs):
    refund_all = float(np.where(actuals == 1, margins, 0.0).sum())
    reject_all = float(np.where(actuals == 0, ltvs, 0.0).sum())
    return refund_all, reject_all


def single_threshold_cost(probs, actuals, margins, ltvs, threshold):
    reject = probs >= threshold
    refund = ~reject
    cost = (
        np.where(refund & (actuals == 1), margins, 0.0).sum()
        + np.where(reject & (actuals == 0), ltvs, 0.0).sum()
    )
    return float(cost)


def build_cost_grid(probs, actuals, margins, ltvs, thresholds):
    order = np.argsort(probs, kind='mergesort')
    sorted_probs = probs[order]

    fraud_cost_if_refunded = np.where(actuals == 1, margins, 0.0)[order]
    review_cost_component = (COST_PHOTO_REVIEW + np.where(actuals == 0, FRICTION_PENALTY, 0.0))[order]
    reject_cost_if_rejected = np.where(actuals == 0, ltvs, 0.0)[order]

    cum_refund = np.concatenate([[0.0], np.cumsum(fraud_cost_if_refunded)])
    cum_review = np.concatenate([[0.0], np.cumsum(review_cost_component)])
    cum_reject = np.concatenate([[0.0], np.cumsum(reject_cost_if_rejected)])
    total_reject_pool = cum_reject[-1]

    idx_low = np.searchsorted(sorted_probs, thresholds, side='left')
    idx_high = np.searchsorted(sorted_probs, thresholds, side='right')

    idx_low_grid = idx_low[:, None]
    idx_high_grid = idx_high[None, :]

    cost_refund = cum_refund[idx_low_grid]
    cost_review = cum_review[idx_high_grid] - cum_review[idx_low_grid]
    cost_reject = total_reject_pool - cum_reject[idx_high_grid]

    cost_grid = cost_refund + cost_review + cost_reject

    valid_mask = thresholds[:, None] < thresholds[None, :]
    cost_grid = np.where(valid_mask, cost_grid, np.inf)

    return cost_grid, idx_low, idx_high


def routing_breakdown(probs, actuals, margins, ltvs, t_low, t_high):
    auto_refund = probs < t_low
    require_photo = (probs >= t_low) & (probs <= t_high)
    auto_reject = probs > t_high

    fraud_mask = actuals == 1
    legit_mask = actuals == 0

    return {
        'auto_refund_count': int(auto_refund.sum()),
        'require_photo_count': int(require_photo.sum()),
        'auto_reject_count': int(auto_reject.sum()),
        'fraud_missed_via_refund': int((auto_refund & fraud_mask).sum()),
        'fraud_caught_via_review_friction': int((require_photo & fraud_mask).sum()),
        'fraud_blocked_via_reject': int((auto_reject & fraud_mask).sum()),
        'legit_frictioned_via_review': int((require_photo & legit_mask).sum()),
        'legit_churned_via_reject': int((auto_reject & legit_mask).sum()),
    }


def run_sensitivity(cost_grid, thresholds, best_low_i, best_high_i, radius):
    rows = []
    lo_range = range(max(0, best_low_i - radius), min(len(thresholds) - 1, best_low_i + radius) + 1)
    hi_range = range(max(0, best_high_i - radius), min(len(thresholds) - 1, best_high_i + radius) + 1)

    best_cost = cost_grid[best_low_i, best_high_i]

    for i in lo_range:
        for j in hi_range:
            cost = cost_grid[i, j]
            if not np.isfinite(cost):
                continue
            rows.append({
                't_low': float(thresholds[i]),
                't_high': float(thresholds[j]),
                'total_cost': float(cost),
                'cost_delta_from_optimum': float(cost - best_cost)
            })

    return pd.DataFrame(rows).sort_values('total_cost').reset_index(drop=True)


def calibrate_financial_thresholds():
    print("Loading data and predictions...")
    calib_df = load_calibration_set()

    probs = calib_df['fraud_probability'].to_numpy()
    margins = calib_df['Item_Margin_USD'].to_numpy()
    ltvs = calib_df['Customer_LTV'].to_numpy()
    actuals = calib_df['Is_Fraud'].to_numpy()

    refund_all_cost, reject_all_cost = compute_baselines(actuals, margins, ltvs)

    f1_threshold = None
    f1_policy_cost = None
    if os.path.exists(MODEL_RESULTS_PATH):
        with open(MODEL_RESULTS_PATH) as f:
            model_results = json.load(f)
        f1_threshold = model_results.get('decision_threshold')
        if f1_threshold is not None:
            f1_policy_cost = single_threshold_cost(probs, actuals, margins, ltvs, f1_threshold)

    print(f"Baseline (Auto-Refund Everyone): ${refund_all_cost:,.2f}")
    print(f"Baseline (Auto-Reject Everyone): ${reject_all_cost:,.2f}")
    if f1_policy_cost is not None:
        print(f"F1-Optimal Threshold as a Single Cutoff ({f1_threshold:.4f}): ${f1_policy_cost:,.2f}")

    print("Running vectorized financial grid search...")
    thresholds = np.round(np.linspace(0.01, 0.99, N_THRESHOLDS), 4)
    cost_grid, idx_low, idx_high = build_cost_grid(probs, actuals, margins, ltvs, thresholds)

    best_flat_idx = int(np.argmin(cost_grid))
    best_low_i, best_high_i = np.unravel_index(best_flat_idx, cost_grid.shape)
    best_t_low = float(thresholds[best_low_i])
    best_t_high = float(thresholds[best_high_i])
    best_cost = float(cost_grid[best_low_i, best_high_i])

    savings_vs_refund_all = refund_all_cost - best_cost
    savings_vs_best_static = min(refund_all_cost, reject_all_cost) - best_cost
    reduction_pct = (savings_vs_refund_all / refund_all_cost * 100) if refund_all_cost > 0 else 0.0

    breakdown = routing_breakdown(probs, actuals, margins, ltvs, best_t_low, best_t_high)
    pct_refund = breakdown['auto_refund_count'] / len(probs) * 100
    pct_photo = breakdown['require_photo_count'] / len(probs) * 100
    pct_reject = breakdown['auto_reject_count'] / len(probs) * 100

    print("\n" + "=" * 50)
    print("OPTIMIZED FINANCIAL THRESHOLDS")
    print("=" * 50)
    print(f"Optimal T_low  (Auto-Refund cutoff):  {best_t_low:.4f}")
    print(f"Optimal T_high (Auto-Reject cutoff):  {best_t_high:.4f}")
    print("\n--- Business Impact ---")
    print(f"Optimized System Loss:     ${best_cost:,.2f}")
    print(f"Savings vs Refund-All:     ${savings_vs_refund_all:,.2f} ({reduction_pct:.2f}%)")
    print(f"Savings vs Best Static:    ${savings_vs_best_static:,.2f}")
    print("\n--- Traffic Routing ---")
    print(f"Auto-Refund:   {pct_refund:.1f}% ({breakdown['auto_refund_count']:,}) | "
          f"fraud missed: {breakdown['fraud_missed_via_refund']:,}")
    print(f"Require Photo: {pct_photo:.1f}% ({breakdown['require_photo_count']:,}) | "
          f"fraud in lane: {breakdown['fraud_caught_via_review_friction']:,} | "
          f"legit frictioned: {breakdown['legit_frictioned_via_review']:,}")
    print(f"Auto-Reject:   {pct_reject:.1f}% ({breakdown['auto_reject_count']:,}) | "
          f"fraud blocked: {breakdown['fraud_blocked_via_reject']:,} | "
          f"legit churned: {breakdown['legit_churned_via_reject']:,}")
    print("=" * 50)

    finite_grid = cost_grid[np.isfinite(cost_grid)]
    landscape_df = pd.DataFrame(
        [(thresholds[i], thresholds[j], cost_grid[i, j])
         for i in range(len(thresholds)) for j in range(len(thresholds))
         if np.isfinite(cost_grid[i, j])],
        columns=['t_low', 't_high', 'total_cost']
    )
    landscape_df.to_csv(LANDSCAPE_OUT, index=False)

    sensitivity_df = run_sensitivity(cost_grid, thresholds, best_low_i, best_high_i, SENSITIVITY_RADIUS)
    sensitivity_df.to_csv(SENSITIVITY_OUT, index=False)

    thresholds_out = {
        "t_low": best_t_low,
        "t_high": best_t_high,
        "baseline_loss_refund_all": refund_all_cost,
        "baseline_loss_reject_all": reject_all_cost,
        "f1_threshold": f1_threshold,
        "f1_single_cutoff_loss": f1_policy_cost,
        "optimized_loss": best_cost,
        "dollars_saved_vs_refund_all": savings_vs_refund_all,
        "dollars_saved_vs_best_static": savings_vs_best_static,
        "loss_reduction_vs_refund_all_pct": reduction_pct,
        "routing": {
            "auto_refund_pct": pct_refund,
            "require_photo_pct": pct_photo,
            "auto_reject_pct": pct_reject,
            **breakdown
        },
        "financial_assumptions": {
            "review_cost_usd": COST_PHOTO_REVIEW,
            "friction_penalty_usd": FRICTION_PENALTY
        },
        "calibration_rows": int(len(calib_df)),
        "threshold_combinations_evaluated": int(np.isfinite(cost_grid).sum())
    }

    with open(THRESHOLDS_OUT, 'w') as f:
        json.dump(thresholds_out, f, indent=4)

    print(f"\nFinancial thresholds saved to '{THRESHOLDS_OUT}'")
    print(f"Full cost landscape saved to '{LANDSCAPE_OUT}'")
    print(f"Threshold sensitivity saved to '{SENSITIVITY_OUT}'")


if __name__ == "__main__":
    calibrate_financial_thresholds()
