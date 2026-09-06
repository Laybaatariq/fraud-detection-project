import pandas as pd
import numpy as np

NUMERIC_COLS_TO_SCALE = [
    "step", "amount_log", "oldbalanceOrg_log", "newbalanceOrig_log",
    "oldbalanceDest_log", "newbalanceDest_log", "hour_of_day"
]


def preprocess_transaction(txn, scaler, feature_columns) -> pd.DataFrame:
    row = {
        "step": txn.step,
        "hour_of_day": txn.hour_of_day,
        "orig_balance_was_zero": 1 if txn.oldbalanceOrg == 0 else 0,
        "dest_balance_was_zero": 1 if txn.oldbalanceDest == 0 else 0,
        "amount_log": np.log1p(txn.amount),
        "oldbalanceOrg_log": np.log1p(txn.oldbalanceOrg),
        "newbalanceOrig_log": np.log1p(txn.newbalanceOrig),
        "oldbalanceDest_log": np.log1p(txn.oldbalanceDest),
        "newbalanceDest_log": np.log1p(txn.newbalanceDest),
        "type_CASH_OUT": 1 if txn.transaction_type == "CASH_OUT" else 0,
        "type_DEBIT": 1 if txn.transaction_type == "DEBIT" else 0,
        "type_PAYMENT": 1 if txn.transaction_type == "PAYMENT" else 0,
        "type_TRANSFER": 1 if txn.transaction_type == "TRANSFER" else 0,
    }

    input_df = pd.DataFrame([row])
    input_df = input_df.reindex(columns=feature_columns, fill_value=0)

    cols_present = [c for c in NUMERIC_COLS_TO_SCALE if c in input_df.columns]
    input_df[cols_present] = scaler.transform(input_df[cols_present])

    return input_df
