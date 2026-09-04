import pandas as pd
import numpy as np
import os


def run_validation():
    print("PIPELINE AND DATA VALIDATION REPORT")

    files = {
        "Train (M1-M4)": "train_m1_m4.parquet",
        "Calibration (M5)": "calib_m5.parquet",
        "Real Test (M6)": "test_m6_real.parquet",
        "Stress Test (M6)": "test_m6_stress.parquet"
    }

    dfs = {}

    for name, path in files.items():
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Required file {path} missing! Run dataB(generated).py first."
            )

        dfs[name] = pd.read_parquet(path)

        print(
            f"Loaded {name}: "
            f"{len(dfs[name]):,} rows, "
            f"{dfs[name].shape[1]} columns"
        )

    # CLASS PREVALENCE CHECK
    print("\nClass Prevalence Check")

    for name, df in dfs.items():

        if 'isFraud' not in df.columns:
            print(f"{name} -> ERROR: isFraud column missing")
            continue

        fraud_counts = (
            df['isFraud']
            .value_counts(normalize=True)
            * 100
        )

        print(
            f"{name} -> Fraud Rate: "
            f"{fraud_counts.get(1, 0):.2f}% "
            f"(Legit: {fraud_counts.get(0, 0):.2f}%)"
        )

    # TEMPORAL CHRONOLOGICAL BOUNDARY CHECK
    print("\nTemporal Chronological Boundary Check")

    train_max = dfs["Train (M1-M4)"]['TransactionDT'].max()

    calib_min = dfs["Calibration (M5)"]['TransactionDT'].min()
    calib_max = dfs["Calibration (M5)"]['TransactionDT'].max()

    test_min = dfs["Real Test (M6)"]['TransactionDT'].min()
    test_max = dfs["Real Test (M6)"]['TransactionDT'].max()

    print(f"Train Max Time: {train_max}")
    print(f"Calib Min Time: {calib_min} | Max Time: {calib_max}")
    print(f"Test Min Time:  {test_min} | Max Time: {test_max}")

    if train_max < calib_min and calib_max < test_min:
        print(
            "Temporal Integrity Confirmed: "
            "No time overlap between splits."
        )
    else:
        print(
            "WARNING: Temporal overlap detected across splits!"
        )

    # C. FEATURE SANITY CHECK
    print("\nFeature Sanity Check (Engineered Features)")

    engineered_keywords = [
        'transactions_per',
        'unique',
        'txn_last',
        'avg_amount',
        'amount_vs',
        'amount_deviation'
    ]

    engineered_cols = sorted([
        c
        for c in dfs["Train (M1-M4)"].columns
        if any(keyword in c for keyword in engineered_keywords)
    ])

    if engineered_cols:

        print(
            f"Checking {len(engineered_cols)} engineered features"
        )

        for name, df in dfs.items():

            print(f"\n{name}")

            stats = []

            for col in engineered_cols:

                if col not in df.columns:
                    stats.append({
                        'Feature': col,
                        'Missing %': 'MISSING',
                        'Min': '-',
                        'Median': '-',
                        'Max': '-'
                    })
                    continue

                missing_pct = (
                    df[col].isnull().mean() * 100
                )

                c_min = df[col].min()
                c_med = df[col].median()
                c_max = df[col].max()

                stats.append({
                    'Feature': col,
                    'Missing %': f"{missing_pct:.2f}%",
                    'Min': c_min,
                    'Median': c_med,
                    'Max': c_max
                })

            stats_df = pd.DataFrame(stats)

            pd.set_option(
                'display.max_rows',
                None
            )

            pd.set_option(
                'display.max_columns',
                None
            )

            pd.set_option(
                'display.width',
                1000
            )

            print(
                stats_df.to_string(
                    index=False
                )
            )

    else:
        print("No engineered features found.")

    # CAUSAL LEAKAGE SPOT-CHECK
    print("\nCausal Leakage Spot-Check")

    train_df = dfs["Train (M1-M4)"]

    if (
        'transactions_per_device' in train_df.columns
        and 'device_id' in train_df.columns
    ):

        first_indices = (
            train_df
            .groupby('device_id')
            .head(1)
        )

        non_zero_first = (
            first_indices[
                'transactions_per_device'
            ] > 0
        ).sum()

        print(
            f"First-seen training transactions with history > 0: "
            f"{non_zero_first} "
            f"(Expected: 0 for strict causality)"
        )

        if non_zero_first == 0:
            print(
                "Device causality check passed."
            )
        else:
            print(
                "WARNING: Possible future-information leakage "
                "in device history features!"
            )

    # VELOCITY FEATURE CONSISTENCY CHECK
    print("\nVelocity Feature Consistency Check")

    velocity_cols = [
        'device_txn_last_5min',
        'device_txn_last_1hr',
        'device_txn_last_24hr',
        'ip_txn_last_5min',
        'ip_txn_last_1hr',
        'ip_txn_last_24hr',
        'card_txn_last_5min',
        'card_txn_last_1hr',
        'card_txn_last_24hr'
    ]

    found_velocity = False

    for col in velocity_cols:

        if col not in train_df.columns:
            continue

        found_velocity = True

        min_value = train_df[col].min()

        negative_count = (
            train_df[col] < 0
        ).sum()

        print(
            f"{col} -> "
            f"Min: {min_value}, "
            f"Negative values: {negative_count}"
        )

        if negative_count > 0:
            print(
                f"WARNING: Negative values detected in {col}"
            )

    if not found_velocity:
        print("No velocity features found.")

    # BASIC FEATURE RELATIONSHIP CHECK
    print("\nFeature Relationship Check")

    if (
        'device_txn_last_5min' in train_df.columns
        and 'device_txn_last_1hr' in train_df.columns
        and 'device_txn_last_24hr' in train_df.columns
    ):

        invalid_device_velocity = (
            (train_df['device_txn_last_5min'] >
             train_df['device_txn_last_1hr'])
            |
            (train_df['device_txn_last_1hr'] >
             train_df['device_txn_last_24hr'])
        ).sum()

        print(
            "Device velocity ordering violations: "
            f"{invalid_device_velocity}"
        )

        if invalid_device_velocity == 0:
            print(
                "Device velocity ordering check passed."
            )
        else:
            print(
                "WARNING: Device velocity features "
                "are internally inconsistent."
            )

    if (
        'ip_txn_last_5min' in train_df.columns
        and 'ip_txn_last_1hr' in train_df.columns
        and 'ip_txn_last_24hr' in train_df.columns
    ):

        invalid_ip_velocity = (
            (train_df['ip_txn_last_5min'] >
             train_df['ip_txn_last_1hr'])
            |
            (train_df['ip_txn_last_1hr'] >
             train_df['ip_txn_last_24hr'])
        ).sum()

        print(
            "IP velocity ordering violations: "
            f"{invalid_ip_velocity}"
        )

        if invalid_ip_velocity == 0:
            print(
                "IP velocity ordering check passed."
            )
        else:
            print(
                "WARNING: IP velocity features "
                "are internally inconsistent."
            )

    # REAL TEST VS STRESS TEST CHECK
    print("\nReal Test vs Stress Test Check")

    real_test = dfs["Real Test (M6)"]
    stress_test = dfs["Stress Test (M6)"]

    if len(real_test) != len(stress_test):
        print(
            "WARNING: Real and Stress test row counts differ!"
        )
    else:
        print(
            f"Row count preserved: {len(real_test):,}"
        )

    common_columns = sorted(
        set(real_test.columns)
        & set(stress_test.columns)
    )

    differences = []

    for col in common_columns:

        if real_test[col].dtype == 'category':
            real_values = real_test[col].astype(str)
            stress_values = stress_test[col].astype(str)
        else:
            real_values = real_test[col]
            stress_values = stress_test[col]

        different = (
            real_values
            .fillna('__NULL__')
            .ne(
                stress_values
                .fillna('__NULL__')
            )
        )

        if different.any():
            differences.append({
                'Column': col,
                'Changed Rows': int(different.sum())
            })

    if differences:

        differences_df = pd.DataFrame(
            differences
        )

        print(
            differences_df.to_string(
                index=False
            )
        )

    else:
        print(
            "WARNING: No differences found between "
            "real and stress test."
        )

    # TARGET INTEGRITY CHECK
    print("\nTarget Integrity Check")

    real_target = real_test['isFraud']
    stress_target = stress_test['isFraud']

    target_changed = (
        real_target
        .fillna(-1)
        .ne(
            stress_target
            .fillna(-1)
        )
    ).sum()

    print(
        f"Target label differences between Real and Stress: "
        f"{target_changed}"
    )

    if target_changed == 0:
        print(
            "Target integrity confirmed."
        )
    else:
        print(
            "WARNING: Stress-test target labels were modified!"
        )

    # VALIDATION STATUS
    print("\nValidation Summary")

    print("Pipeline checks completed.")
    print("Review any WARNING messages above before model training.")

    print("\nVALIDATION COMPLETE. READY FOR MODEL EXPERIMENTATION.")


if __name__ == "__main__":
    run_validation()