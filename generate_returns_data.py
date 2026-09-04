import os
import numpy as np
import pandas as pd

SEED = 42
np.random.seed(SEED)
RNG = np.random.default_rng(SEED)

PERSONA_NAMES = np.array(["normal", "wardrobe", "empty_box", "policy_abuse"])
PERSONA_PROBS = np.array([0.920, 0.040, 0.015, 0.025])

PERSONA_EVENT_LAMBDA = {
    "normal": 0.6,
    "wardrobe": 2.0,
    "empty_box": 1.0,
    "policy_abuse": 3.5,
}

PERSONA_FRAUD_PROB = {
    "normal": 0.015,
    "wardrobe": 0.65,
    "empty_box": 0.75,
    "policy_abuse": 0.70,
}

INCIDENTAL_FRAUD_TYPE_WEIGHTS = {"Wardrobing": 0.50, "Empty_Box": 0.20, "Policy_Abuse": 0.30}

ROLLING_WINDOW_DAYS = 90
SIMULATION_WINDOW_DAYS = 270

ITEM_CATEGORIES = ["Apparel", "Electronics", "Home", "Consumables"]
CLAIM_TYPES = ["Wrong Size", "Did Not Like", "Item Defective", "Arrived Late",
                "Empty Box", "Item Missing"]

OUTPUT_PATH = "returns_dataset.parquet"

def generate_users(n_samples):
    expected_events_per_user = sum(
        PERSONA_PROBS[i] * (1.0 + PERSONA_EVENT_LAMBDA[name])
        for i, name in enumerate(PERSONA_NAMES)
    )
    n_users = max(1, int(round(n_samples / expected_events_per_user)))

    user_id = np.arange(100000, 100000 + n_users)
    persona = RNG.choice(PERSONA_NAMES, size=n_users, p=PERSONA_PROBS)

    event_counts = np.empty(n_users, dtype=np.int64)
    for name in PERSONA_NAMES:
        mask = persona == name
        n_in_group = int(mask.sum())
        if n_in_group == 0:
            continue
        event_counts[mask] = 1 + RNG.poisson(PERSONA_EVENT_LAMBDA[name], size=n_in_group)

    ltv_baseline = np.empty(n_users, dtype=np.float64)
    ltv_params = {
        "normal": (2.0, 300.0),
        "wardrobe": (1.8, 250.0),
        "empty_box": (1.0, 40.0),
        "policy_abuse": (1.5, 150.0),
    }
    
    for name in PERSONA_NAMES:
        mask = persona == name
        n_in_group = int(mask.sum())
        if n_in_group == 0:
            continue
        shape, scale = ltv_params[name]
        ltv_baseline[mask] = RNG.gamma(shape, scale, size=n_in_group)

    users = pd.DataFrame({
        "User_ID": user_id,
        "persona": persona,
        "event_count": event_counts,
        "ltv_baseline": ltv_baseline,
    })

    return users

def expand_to_events(users):
    n_events = int(users["event_count"].sum())

    events = users.loc[users.index.repeat(users["event_count"])].reset_index(drop=True)
    events = events.drop(columns=["event_count"])

    max_seconds = SIMULATION_WINDOW_DAYS * 86400
    events["ReturnDT"] = RNG.integers(0, max_seconds, size=n_events)

    ltv_jitter = RNG.lognormal(mean=0.0, sigma=0.05, size=n_events)
    events["Customer_LTV"] = np.round(events["ltv_baseline"] * ltv_jitter, 2)
    events = events.drop(columns=["ltv_baseline"])

    return events

def assign_true_outcomes(events):
    n = len(events)
    fraud_roll = RNG.random(n)

    persona = events["persona"].to_numpy()
    fraud_prob = np.vectorize(PERSONA_FRAUD_PROB.get)(persona)
    is_fraud = fraud_roll < fraud_prob

    fraud_type = np.full(n, "Legitimate", dtype=object)

    fraud_type[is_fraud & (persona == "wardrobe")] = "Wardrobing"
    fraud_type[is_fraud & (persona == "empty_box")] = "Empty_Box"
    fraud_type[is_fraud & (persona == "policy_abuse")] = "Policy_Abuse"

    incidental_mask = is_fraud & (persona == "normal")
    n_incidental = int(incidental_mask.sum())
    if n_incidental > 0:
        types, weights = zip(*INCIDENTAL_FRAUD_TYPE_WEIGHTS.items())
        weights = np.array(weights) / np.sum(weights)
        fraud_type[incidental_mask] = RNG.choice(types, size=n_incidental, p=weights)

    events["Is_Fraud"] = is_fraud.astype(np.int8)
    events["Fraud_Type"] = fraud_type

    return events

_TIME_TO_RETURN_PARAMS = {
    "Legitimate": dict(kind="normal", loc=12.0, scale=6.0),
    "Wardrobing": dict(kind="beta", a=6.0, b=2.0),      
    "Empty_Box": dict(kind="beta", a=1.5, b=6.0),       
    "Policy_Abuse": dict(kind="normal", loc=11.0, scale=7.0),  
}

_MARGIN_PARAMS = {
    "Legitimate": dict(kind="exponential", scale=40.0, shift=10.0),
    "Wardrobing": dict(kind="exponential", scale=140.0, shift=40.0),
    "Empty_Box": dict(kind="lognormal", mean=6.5, sigma=0.5),
    "Policy_Abuse": dict(kind="exponential", scale=55.0, shift=10.0),
}

_CATEGORY_PROBS = {
    "Legitimate": [0.40, 0.20, 0.30, 0.10],
    "Wardrobing": [0.82, 0.05, 0.10, 0.03],
    "Empty_Box": [0.05, 0.78, 0.15, 0.02],
    "Policy_Abuse": [0.40, 0.20, 0.30, 0.10], 
}

_CLAIM_PROBS = {
    "Legitimate":    [0.30, 0.30, 0.25, 0.10, 0.02, 0.03],
    "Wardrobing":    [0.45, 0.45, 0.05, 0.03, 0.00, 0.02],
    "Empty_Box":     [0.03, 0.02, 0.10, 0.00, 0.65, 0.20],
    "Policy_Abuse":  [0.15, 0.15, 0.55, 0.10, 0.02, 0.03],
}

def _draw_time_to_return(outcome, n):
    p = _TIME_TO_RETURN_PARAMS[outcome]
    if p["kind"] == "normal":
        raw = RNG.normal(p["loc"], p["scale"], size=n)
    else:  
        raw = 1.0 + RNG.beta(p["a"], p["b"], size=n) * 29.0
    return np.clip(raw, 1, 30)

def _draw_margin(outcome, n):
    p = _MARGIN_PARAMS[outcome]
    if p["kind"] == "exponential":
        return RNG.exponential(p["scale"], size=n) + p["shift"]
    else:  
        return RNG.lognormal(p["mean"], p["sigma"], size=n)

def generate_observable_features(events):
    n = len(events)
    outcome = events["Fraud_Type"].to_numpy()

    time_to_return = np.empty(n, dtype=np.float64)
    margin = np.empty(n, dtype=np.float64)
    category = np.empty(n, dtype=object)
    claim = np.empty(n, dtype=object)

    for outcome_name in _TIME_TO_RETURN_PARAMS:
        mask = outcome == outcome_name
        n_in_group = int(mask.sum())
        if n_in_group == 0:
            continue

        time_to_return[mask] = _draw_time_to_return(outcome_name, n_in_group)
        margin[mask] = _draw_margin(outcome_name, n_in_group)
        category[mask] = RNG.choice(ITEM_CATEGORIES, size=n_in_group, p=_CATEGORY_PROBS[outcome_name])
        claim[mask] = RNG.choice(CLAIM_TYPES, size=n_in_group, p=_CLAIM_PROBS[outcome_name])

    events["Time_to_Return_Days"] = np.clip(time_to_return, 1, 30).astype(np.int64)
    events["Item_Margin_USD"] = np.round(margin, 2)
    events["Item_Category"] = category
    events["Claim_Type"] = claim

    return events

def add_causal_history_features(events):
    events = events.sort_values(["User_ID", "ReturnDT"]).reset_index(drop=True)
    events["_return_dt_datetime"] = pd.to_datetime(events["ReturnDT"], unit="s")

    def _rolling_prior_count(group):
        s = pd.Series(1, index=group)
        return s.rolling(f"{ROLLING_WINDOW_DAYS}D", closed="left").count()

    events["Returns_Count_Last_90D"] = (
        events
        .groupby("User_ID")["_return_dt_datetime"]
        .apply(_rolling_prior_count)
        .reset_index(level=0, drop=True)
        .fillna(0)
        .astype(np.int32)
        .to_numpy()
    )

    events["Prior_Confirmed_Fraud_Count"] = (
        events
        .groupby("User_ID")["Is_Fraud"]
        .apply(lambda s: s.shift(1).expanding().sum())
        .reset_index(level=0, drop=True)
        .fillna(0)
        .astype(np.int32)
        .to_numpy()
    )

    events = events.drop(columns=["_return_dt_datetime"])

    return events

def generate_synthetic_returns(n_samples=15000):
    print(f"Generating a synthetic return-fraud dataset targeting ~{n_samples:,} events...")

    users = generate_users(n_samples)
    print(f"  Users generated: {len(users):,} "
          f"(expected total events: {int(users['event_count'].sum()):,})")

    events = expand_to_events(users)
    events = assign_true_outcomes(events)
    events = generate_observable_features(events)
    events = add_causal_history_features(events)

    events = events.drop(columns=["persona"])

    column_order = [
        "User_ID", "ReturnDT", "Time_to_Return_Days", "Item_Category",
        "Item_Margin_USD", "Claim_Type", "Customer_LTV",
        "Returns_Count_Last_90D", "Prior_Confirmed_Fraud_Count",
        "Is_Fraud", "Fraud_Type",
    ]
    events = events[column_order].sort_values("ReturnDT").reset_index(drop=True)

    return events

def print_dataset_diagnostics(df):
    print("\n--- Dataset Summary ---")
    print(f"Total Records: {len(df):,}")
    print(f"Unique Users:  {df['User_ID'].nunique():,}")
    print(f"Overall Fraud Rate: {(df['Is_Fraud'].mean() * 100):.2f}%")

    print("\nBreakdown by Fraud Type:")
    print(df["Fraud_Type"].value_counts())

    print("\n--- Overlap Diagnostics (should NOT be 0% or 100%) ---")

    empty_box_claim_but_legit = df.loc[
        (df["Claim_Type"] == "Empty Box") & (df["Is_Fraud"] == 0)
    ]
    empty_box_fraud_total = df.loc[df["Fraud_Type"] == "Empty_Box"]
    print(
        f"Legit returns that used the 'Empty Box' claim type: "
        f"{len(empty_box_claim_but_legit)} of "
        f"{(df['Claim_Type'] == 'Empty Box').sum()} 'Empty Box' claims "
        f"({len(empty_box_claim_but_legit) / max((df['Claim_Type'] == 'Empty Box').sum(), 1) * 100:.1f}%)"
    )
    if len(empty_box_fraud_total) > 0:
        non_empty_box_claim = (empty_box_fraud_total["Claim_Type"] != "Empty Box").mean() * 100
        print(f"Confirmed Empty_Box fraud that used a DIFFERENT claim type: {non_empty_box_claim:.1f}%")

    apparel_fraud = df.loc[df["Fraud_Type"] == "Wardrobing"]
    if len(apparel_fraud) > 0:
        non_apparel = (apparel_fraud["Item_Category"] != "Apparel").mean() * 100
        print(f"Confirmed Wardrobing fraud on a non-Apparel item: {non_apparel:.1f}%")

    print(
        f"Returns with 0 prior history "
        f"(Returns_Count_Last_90D == 0 and Prior_Confirmed_Fraud_Count == 0): "
        f"{((df['Returns_Count_Last_90D'] == 0) & (df['Prior_Confirmed_Fraud_Count'] == 0)).mean() * 100:.1f}% "
        f"of all rows -- expected to be a large chunk, since most users only return once "
        f"or twice in this window and have no fraud history yet."
    )

if __name__ == "__main__":
    df_returns = generate_synthetic_returns(n_samples=15000)

    try:
        df_returns.to_parquet(OUTPUT_PATH, index=False)
        saved_path = OUTPUT_PATH
    except (ImportError, ValueError) as exc:
        fallback_path = OUTPUT_PATH.replace(".parquet", ".csv")
        print(f"Could not write parquet ({exc}); falling back to CSV.")
        df_returns.to_csv(fallback_path, index=False)
        saved_path = fallback_path

    print_dataset_diagnostics(df_returns)
    print(f"\nDataset successfully saved to '{saved_path}'")
    print(
        "\nNOTE: ReturnDT is a real (if synthetic) timestamp -- when you build the "
        "return-risk model on this file, split train/calib/test chronologically by "
        "ReturnDT, the same way the transaction-fraud track splits by TransactionDT. "
        "A random split would let a user's future returns leak into training data "
        "used to predict their past ones."
    )