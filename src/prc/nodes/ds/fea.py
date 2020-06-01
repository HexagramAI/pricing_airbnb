"""Feature engineering pipe."""
import numpy as np


def process_log_cols(df, log_cols):
    """Use np.log to scale columns.

    Arguments:
        df {[type]} -- [description]
        columns {[type]} -- [description]
    Returns:
        df: with log processed columns
    """
    # log_cols = ["price", "minimum_nights", "lag"]
    df[log_cols] = np.log(df[log_cols])
    return df


def drop_useless_cols(df, fea_drop_columns):
    """Drop useless columns."""
    df = df.drop(columns=fea_drop_columns)
    return df
