"""Primary pipeline data nodes."""
import pandas as pd


def process_date_cols(df):
    """Process date related columns.

    1: filter date.
    2: add lag (future date - cur date)
    3: add month: add month info

    Args:
        df: main df
    Returns:
        df: with date related col processed
    """
    # filter days with in 60 days
    min_date = pd.to_datetime(min(df.date))

    df = df[df.date <= (min_date + pd.Timedelta(days=60)).strftime("%Y-%m-%d")]

    df["date"] = pd.to_datetime(df["date"])
    # add lag
    df["lag"] = (df["date"] - min_date).dt.days

    # add month
    df["month"] = df.date.dt.month
    return df


def process_price_cols(df):
    """Process price columns."""
    df["price"] = df["price"].map(lambda x: float(str(x).strip("$").replace(",", "")))
    return df
