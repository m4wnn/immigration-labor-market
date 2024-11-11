import pandas as pd
import numpy as np

# Functions for weighted aggregation


def WtSum(
    df: pd.core.frame.DataFrame,
    cols: list,
    weight_col: str,
    by_cols: list,
    outw=False,
    mask=None,
):
    """
    Calculates the weighted sum of specified columns in a DataFrame.

    This function computes the weighted sum of the columns specified in `cols` using the
    values in `weight_col`. The result is grouped by the columns specified in `by_cols`.
    If `outw=True`, the function also returns the sum of the weight column. Optionally,
    the calculation can be applied to a subset of the DataFrame using the `mask` parameter.

    Args:
        df (pd.DataFrame): The input DataFrame containing the data to be aggregated.
        cols (list): A list of column names to be summed with weights applied.
        weight_col (str): The name of the column containing weights.
        by_cols (list): A list of columns to group by when aggregating the weighted sums.
        outw (bool, optional): If True, the weight column is also included in the output.
                               Defaults to False.
        mask (pd.Series, optional): A boolean mask to filter the DataFrame before applying
                                    the weighted sum. Defaults to None.

    Returns:
        pd.DataFrame: A DataFrame containing the weighted sums of the specified columns,
                      grouped by `by_cols`. If `outw=True`, the weight column sum is also
                      included in the output.
    """

    out = df[[*cols, weight_col, *by_cols]].copy()
    out[[*cols, weight_col]] = out[[*cols, weight_col]].astype(
        np.float64
    )  # for sum precision

    if mask is not None:
        out = out[mask]

    for c in cols:
        out[c] = out[c] * out[weight_col]

    if outw:
        return out.groupby(by_cols)[[*cols, weight_col]].sum()
    else:
        return out.groupby(by_cols)[cols].sum()


def WtMean(
    df: pd.core.frame.DataFrame, cols: list, weight_col: str, by_cols: list, mask=None
):
    """
    Calculates the weighted mean of specified columns in a DataFrame.

    This function computes the weighted mean for each column in `cols`, using the weights provided
    in `weight_col`, and groups the results by the columns specified in `by_cols`. Optionally, the
    calculation can be filtered using the `mask` parameter. Missing values are removed before
    performing the weighted mean calculation.

    Args:
        df (pd.DataFrame): The input DataFrame containing the data to be aggregated.
        cols (list): A list of column names for which to calculate the weighted mean.
        weight_col (str): The name of the column containing weights.
        by_cols (list): A list of columns to group by when calculating the weighted means.
        mask (pd.Series, optional): A boolean mask to filter the DataFrame before calculating
                                    the weighted mean. Defaults to None.

    Returns:
        pd.DataFrame: A DataFrame containing the weighted means of the specified columns, grouped
                        by the columns in `by_cols`.
    """
    out_list = []
    for c in cols:
        out = df[[c, weight_col, *by_cols]].copy()
        out[[c, weight_col]] = out[[c, weight_col]].astype(
            np.float64
        )  # for sum precision

        if mask is not None:
            out = out[mask]

        out = out[~np.isnan(out[c])]  # remove missings
        out.loc[:, c] = out.loc[:, c] * out.loc[:, weight_col]  # multiply by weights
        out = out.groupby(by_cols)[[c, weight_col]].sum()  # sum
        out.loc[:, c] = (
            out.loc[:, c] / out.loc[:, weight_col]
        )  # divide by total weights

        out_list.append(out[c])

    return pd.concat(out_list, axis=1)
