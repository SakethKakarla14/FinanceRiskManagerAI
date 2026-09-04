import sys
import subprocess
import os

def install_requirements():
    required_packages = ['pandas', 'numpy', 'pyarrow', 'fastparquet']
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            print(f"Package '{package}' not found. Installing")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    print("All dependencies installed.\n")

install_requirements()

import pandas as pd
import numpy as np


# ENTERPRISE PREPROCESSING
def optimize_and_preprocess(df, reference_df=None, preprocessing_info=None):
    print("Executing Preprocessing & Memory Optimization")

    df = df.copy()

    # 1. Drop high-null columns (Noise reduction)
    if preprocessing_info is None:
        if reference_df is None:
            reference_df = df

        null_percentages = reference_df.isnull().mean()
        cols_to_drop = null_percentages[null_percentages > 0.90].index.tolist()

        preprocessing_info = {
            'cols_to_drop': cols_to_drop
        }
    else:
        cols_to_drop = preprocessing_info['cols_to_drop']

    df.drop(columns=cols_to_drop, inplace=True, errors='ignore')

    print(f"Dropped {len(cols_to_drop)} columns with >90% missing values.")

    # 2. Downcast numerical types (Memory compression)
    start_mem = df.memory_usage().sum() / 1024**2

    if preprocessing_info.get('numeric_dtypes') is None:
        numeric_dtypes = {}

        if reference_df is None:
            reference_df = df

        for col in reference_df.columns:
            if col not in df.columns:
                continue

            if pd.api.types.is_integer_dtype(reference_df[col]):
                c_min = reference_df[col].min()
                c_max = reference_df[col].max()

                if (
                    pd.notna(c_min)
                    and pd.notna(c_max)
                    and c_min >= np.iinfo(np.int32).min
                    and c_max <= np.iinfo(np.int32).max
                ):
                    numeric_dtypes[col] = np.int32

            elif pd.api.types.is_float_dtype(reference_df[col]):
                c_min = reference_df[col].min()
                c_max = reference_df[col].max()

                if (
                    pd.notna(c_min)
                    and pd.notna(c_max)
                    and c_min >= np.finfo(np.float32).min
                    and c_max <= np.finfo(np.float32).max
                ):
                    numeric_dtypes[col] = np.float32

        preprocessing_info['numeric_dtypes'] = numeric_dtypes

    for col, dtype in preprocessing_info['numeric_dtypes'].items():
        if col in df.columns:
            try:
                df[col] = df[col].astype(dtype)
            except (TypeError, ValueError, OverflowError):
                pass

    # 3. Cast strings to Categorical (Required for LightGBM)
    object_cols = df.select_dtypes(include=['object', 'string']).columns

    for col in object_cols:
        df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024**2
    print(f"Memory compressed from {start_mem:.2f} MB to {end_mem:.2f} MB.")

    return df, preprocessing_info


# RELATIONAL GRAPH & VELOCITY FEATURE ENGINEERING
def add_causal_rolling_count(df, group_col, time_col, window_seconds):
    """
    Counts only transactions that happened before the current transaction
    and within the specified time window.
    """

    result = np.zeros(len(df), dtype=np.float32)

    if group_col not in df.columns or time_col not in df.columns:
        return result

    grouped = df.groupby(group_col, sort=False).groups

    for _, positions in grouped.items():

        positions = np.asarray(positions)

        if len(positions) == 0:
            continue

        times = df.loc[positions, time_col].to_numpy()

        order = np.argsort(times, kind='stable')
        sorted_positions = positions[order]
        sorted_times = times[order]

        # Strictly earlier timestamps only.
        right = np.searchsorted(
            sorted_times,
            sorted_times,
            side='left'
        )

        left = np.searchsorted(
            sorted_times,
            sorted_times - window_seconds,
            side='left'
        )

        counts = right - left

        result[sorted_positions] = counts.astype(np.float32)

    return result


def add_previous_unique_count(df, group_col, value_col):
    """
    Counts unique values observed previously within each group.
    The current transaction is not included.
    """

    if group_col not in df.columns or value_col not in df.columns:
        return np.zeros(len(df), dtype=np.float32)

    new_value = ~df.duplicated(
        subset=[group_col, value_col],
        keep='first'
    )

    cumulative = (
        new_value.astype(np.int32)
        .groupby(df[group_col], sort=False)
        .cumsum()
    )

    previous_unique = cumulative - new_value.astype(np.int32)

    return previous_unique.astype(np.float32).to_numpy()


def engineer_graph_features(df):
    print("Engineering Relational Graph & Velocity Features...")

    df = df.copy()
    df = df.sort_values('TransactionDT').reset_index(drop=True)

    # Previous transactions associated with each device
    if 'device_id' in df.columns and 'TransactionID' in df.columns:

        df['transactions_per_device'] = (
            df.groupby('device_id', sort=False)
              .cumcount()
              .astype(np.float32)
        )

        # Unique transaction amounts observed BEFORE current transaction
        if 'TransactionAmt' in df.columns:
            df['device_unique_transaction_amounts'] = (
                add_previous_unique_count(
                    df,
                    'device_id',
                    'TransactionAmt'
                )
            )

        # Unique cards observed BEFORE current transaction
        if 'card1' in df.columns:
            df['device_unique_cards'] = (
                add_previous_unique_count(
                    df,
                    'device_id',
                    'card1'
                )
            )

        # Device velocity
        df['device_txn_last_5min'] = add_causal_rolling_count(
            df,
            'device_id',
            'TransactionDT',
            5 * 60
        )

        df['device_txn_last_1hr'] = add_causal_rolling_count(
            df,
            'device_id',
            'TransactionDT',
            60 * 60
        )

        df['device_txn_last_24hr'] = add_causal_rolling_count(
            df,
            'device_id',
            'TransactionDT',
            24 * 60 * 60
        )

    # Previous transactions associated with each IP
    if 'ip_address' in df.columns and 'TransactionID' in df.columns:

        df['transactions_per_ip'] = (
            df.groupby('ip_address', sort=False)
              .cumcount()
              .astype(np.float32)
        )

        if 'card1' in df.columns:
            df['ip_unique_cards'] = (
                add_previous_unique_count(
                    df,
                    'ip_address',
                    'card1'
                )
            )

        # IP velocity
        df['ip_txn_last_5min'] = add_causal_rolling_count(
            df,
            'ip_address',
            'TransactionDT',
            5 * 60
        )

        df['ip_txn_last_1hr'] = add_causal_rolling_count(
            df,
            'ip_address',
            'TransactionDT',
            60 * 60
        )

        df['ip_txn_last_24hr'] = add_causal_rolling_count(
            df,
            'ip_address',
            'TransactionDT',
            24 * 60 * 60
        )

    # Card velocity and historical spending behavior
    if 'card1' in df.columns and 'TransactionID' in df.columns:

        df['card_txn_last_5min'] = add_causal_rolling_count(
            df,
            'card1',
            'TransactionDT',
            5 * 60
        )

        df['card_txn_last_1hr'] = add_causal_rolling_count(
            df,
            'card1',
            'TransactionDT',
            60 * 60
        )

        df['card_txn_last_24hr'] = add_causal_rolling_count(
            df,
            'card1',
            'TransactionDT',
            24 * 60 * 60
        )

        if 'TransactionAmt' in df.columns:

            # Historical amount sum excluding current transaction
            previous_amount_sum = (
                df.groupby('card1', sort=False)['TransactionAmt']
                  .cumsum()
                - df['TransactionAmt']
            )

            previous_amount_count = (
                df.groupby('card1', sort=False)
                  .cumcount()
            )

            previous_amount_avg = (
                previous_amount_sum /
                previous_amount_count.replace(0, np.nan)
            )

            previous_amount_avg = previous_amount_avg.fillna(
                df['TransactionAmt'].median()
            )

            df['card_avg_amount_before'] = (
                previous_amount_avg.astype(np.float32)
            )

            df['amount_vs_card_avg'] = (
                df['TransactionAmt'] /
                df['card_avg_amount_before'].replace(0, np.nan)
            ).replace(
                [np.inf, -np.inf],
                np.nan
            ).fillna(1.0).astype(np.float32)

            df['amount_deviation_from_card_avg'] = (
                (
                    df['TransactionAmt'] -
                    df['card_avg_amount_before']
                ).abs()
                .astype(np.float32)
            )

    return df


# DATA MUTATION & GENERATION
def generate_data_b():
    print("Starting Data B Mutator (Temporal Split & Graph Injection)")

    # Bugfix: seed the RNG once, unconditionally, at the very top of
    # this function. Previously np.random.seed(42) was only called
    # inside the `if 'ip_address' not in df.columns` branch, so
    # reproducibility of every downstream random draw in this function
    # (fraud-ring device/IP selection, stress-test row selection) was
    # accidental rather than guaranteed -- it happened to work only
    # because that branch always fired for this dataset.
    np.random.seed(42)

    raw_path = 'dataA.csv'

    if not os.path.exists(raw_path):
        raise FileNotFoundError(
            f"'{raw_path}' not found. Run dataA(opensource).py first!"
        )

    df = pd.read_csv(raw_path, low_memory=False)

    df = df.copy()

    print("\nStandardizing Graph Nodes (Devices & IPs)")

    if 'device_id' not in df.columns:
        # NOTE: DeviceInfo is ~80% missing in the raw data. For those
        # rows device_id below becomes a synthetic value unique to
        # that single row (so device velocity = 0 always for them).
        # For the ~20% where it IS populated, values are OS/browser
        # strings like "Windows" or "iOS Device" shared by thousands
        # of unrelated users -- not a real per-device fingerprint. The
        # "device velocity" features downstream are really "OS-family
        # transaction count," a much weaker signal than a true device
        # ID would give. Flag this in any writeup.
        df['device_id'] = (
            df.get('DeviceInfo', 'UNKNOWN')
            .fillna('UNKNOWN')
        )

        unknowns = df['device_id'] == 'UNKNOWN'

        df.loc[unknowns, 'device_id'] = [
            f"DEV_SYNTH_{i}"
            for i in range(unknowns.sum())
        ]

    if 'ip_address' not in df.columns:
        # NOTE: dataA.csv (the IEEE-CIS mirror) has no real IP field,
        # so a synthetic IP is fabricated here -- one fresh random
        # value per row. With ~59K rows over ~64.5K possible values,
        # collisions happen mostly by chance (birthday paradox), so
        # every ip_* velocity/graph feature derived from this column
        # downstream is illustrative/synthetic, not a measurement of
        # real shared-network behavior. Flag this in any writeup.
        ip_blocks = np.random.randint(
            1,
            255,
            size=(len(df), 2)
        )

        df['ip_address'] = [
            f"192.168.{x[0]}.{x[1]}"
            for x in ip_blocks
        ]

    print("\nSorting chronologically to simulate streaming data")

    df = df.sort_values(
        'TransactionDT'
    ).reset_index(drop=True)

    n = len(df)

    train_idx = int(n * 0.70)
    calib_idx = int(n * 0.85)

    train_df = df.iloc[:train_idx].copy()
    calib_df = df.iloc[train_idx:calib_idx].copy()
    test_real_df = df.iloc[calib_idx:].copy()  # Pristine real test set

    print(
        f"Original Split -> Train: {len(train_df)}, "
        f"Calib: {len(calib_df)}, "
        f"Real Test (M6): {len(test_real_df)}"
    )

    # Learn preprocessing decisions from training data only
    train_df, preprocessing_info = optimize_and_preprocess(
        train_df,
        reference_df=train_df
    )

    # Apply the exact same preprocessing decisions to later periods
    calib_df, _ = optimize_and_preprocess(
        calib_df,
        preprocessing_info=preprocessing_info
    )

    test_real_df, _ = optimize_and_preprocess(
        test_real_df,
        preprocessing_info=preprocessing_info
    )

    # Restore chronological ordering after preprocessing
    train_df = train_df.sort_values(
        'TransactionDT'
    ).reset_index(drop=True)

    calib_df = calib_df.sort_values(
        'TransactionDT'
    ).reset_index(drop=True)

    test_real_df = test_real_df.sort_values(
        'TransactionDT'
    ).reset_index(drop=True)

    # Create a combined chronological feature-engineering frame so
    # calibration and test transactions can use historical information
    # from earlier periods without using future transactions.
    combined_df = pd.concat(
        [
            train_df,
            calib_df,
            test_real_df
        ],
        axis=0,
        ignore_index=True
    )

    combined_df = combined_df.sort_values(
        'TransactionDT'
    ).reset_index(drop=True)

    # Bugfix: pd.concat silently reverts categorical columns to
    # plain object dtype whenever the category sets differ slightly
    # across train/calib/test_real (which they will, since each was
    # categorized independently in optimize_and_preprocess), undoing
    # the memory-compression step. Re-cast them here. device_id and
    # ip_address are deliberately excluded -- the fraud-ring injection
    # right below needs to assign brand-new string labels
    # (FRAUD_RING_DEV_1/2, 10.0.0.99) into those two columns, which a
    # fixed-category dtype would reject.
    recompress_cols = [
        col
        for col in combined_df.select_dtypes(include=['object']).columns
        if col not in ('device_id', 'ip_address')
    ]

    for col in recompress_cols:
        combined_df[col] = combined_df[col].astype('category')

    # Inject synthetic fraud ring before feature engineering so that
    # structural features actually reflect the injected relationships.
    print("\nInjecting multi-hop Fraud Rings into Train/Calib sets")

    fraud_indices_train = combined_df[
        (combined_df['isFraud'] == 1) &
        (combined_df.index < len(train_df))
    ].index

    if len(fraud_indices_train) > 100:

        selected_fraud = np.random.choice(
            fraud_indices_train,
            100,
            replace=False
        )

        combined_df['device_id'] = (
            combined_df['device_id']
            .astype(str)
        )

        combined_df['ip_address'] = (
            combined_df['ip_address']
            .astype(str)
        )

        combined_df.loc[
            selected_fraud,
            'device_id'
        ] = np.random.choice(
            [
                "FRAUD_RING_DEV_1",
                "FRAUD_RING_DEV_2"
            ],
            100
        )

        combined_df.loc[
            selected_fraud,
            'ip_address'
        ] = "10.0.0.99"

    # Engineer causal relational features after the structural
    # relationships have been created.
    combined_df = engineer_graph_features(
        combined_df
    )

    # Recover the original temporal boundaries
    train_end_time = train_df['TransactionDT'].max()
    calib_end_time = calib_df['TransactionDT'].max()

    train_df = combined_df[
        combined_df['TransactionDT'] <= train_end_time
    ].copy()

    calib_df = combined_df[
        (combined_df['TransactionDT'] > train_end_time) &
        (combined_df['TransactionDT'] <= calib_end_time)
    ].copy()

    test_real_df = combined_df[
        combined_df['TransactionDT'] > calib_end_time
    ].copy()

    train_df = train_df.reset_index(drop=True)
    calib_df = calib_df.reset_index(drop=True)
    test_real_df = test_real_df.reset_index(drop=True)

    print(
        f"Original Split -> Train: {len(train_df)}, "
        f"Calib: {len(calib_df)}, "
        f"Real Test (M6): {len(test_real_df)}"
    )

    # Create Adversarial Stress-Test Copy separated from the pristine M6 real test set
    print("Injecting Concept Drift (Stealth Fraud) into M6 Stress-Test copy")

    test_stress_df = test_real_df.copy()

    fraud_indices_test = test_stress_df[
        test_stress_df['isFraud'] == 1
    ].index

    if len(fraud_indices_test) > 50:

        stealth_fraud = np.random.choice(
            fraud_indices_test,
            50,
            replace=False
        )

        test_stress_df.loc[
            stealth_fraud,
            'TransactionAmt'
        ] = np.random.uniform(
            5,
            15,
            50
        ).astype(np.float32)

        test_stress_df.loc[
            stealth_fraud,
            'transactions_per_device'
        ] = 500.0

    print("\nSaving compressed Parquet files")

    train_df.to_parquet(
        'train_m1_m4.parquet',
        index=False
    )

    calib_df.to_parquet(
        'calib_m5.parquet',
        index=False
    )

    test_real_df.to_parquet(
        'test_m6_real.parquet',
        index=False
    )

    test_stress_df.to_parquet(
        'test_m6_stress.parquet',
        index=False
    )

    print("Files saved to root directory:")
    print(" - train_m1_m4.parquet")
    print(" - calib_m5.parquet")
    print(" - test_m6_real.parquet (Pristine)")
    print(" - test_m6_stress.parquet (Adversarial)")


if __name__ == "__main__":
    generate_data_b()
    print("\nData generation & preprocessing completed.")