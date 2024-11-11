"""
Creating the dataset:

Loading and merging data from 1990, 2000 and 2008 Census and ACS.

"""

# %%
import os
import sqlite3
import pandas as pd
import numpy as np

from toolz import pipe
from econtools import group_id

from src.write_variable_labels import write_variable_labels
from src.agg import WtSum, WtMean


def create_base_df(mainp):
    """
    Creates a base DataFrame from census data and processes it into grouped categories.

    This function loads census data, processes demographic and employment-related variables,
    creates grouped categories based on gender, birthplace, age, education, and race,
    and merges the data with corresponding geographic information. It then creates employment
    indicators and calculates additional variables such as log weekly wages and total hours worked.

    Args:
        mainp (str): The main path where data is stored. This path is adjusted to point to the
                     'data' directory of the project.

    Returns:
        tuple:
            pd.DataFrame: The processed DataFrame with demographic, employment, and wage variables.
            list: A list of group column names used for further analysis.

    Raises:
        FileNotFoundError: If the required data files are not found in the specified path.
    """

    mainp = os.path.join("data")
    # This path will work relatively to the root of the project.
    mainp = os.path.join("data")
    print(f"--> Path to data: {mainp}")

    group_cols = ["male", "native", "agebin", "educbin", "white"]
    print("--> Group columns defined.")
    print(group_cols)

    write_variable_labels()

    # Keep those aged 16-64 and not in group quarters:
    df = pipe(
        os.path.join(mainp, "usa_00137.dta"),
        lambda x: pd.read_stata(x, convert_categoricals=False),
        lambda x: x[(x.age >= 16) & (x.age <= 64) & (x.gq <= 2)],
    )

    # Define (128) groups over which we CA:
    #    - gender (2)
    #    - US born (2)
    #    - age bin (4)
    #    - education bin (4)
    #    - race bin (2)

    df["male"] = np.where(df.sex == 1, 1, 0)
    df["native"] = np.where(df.bpl <= 99, 1, 0)
    df["agebin"] = pd.cut(df.age, bins=[15, 27, 39, 51, 64], labels=False)
    df["educbin"] = pd.cut(df.educ, bins=[-1, 5, 6, 9, 11], labels=False)
    df["white"] = np.where(df.race == 1, 1, 0)
    df["college"] = np.where((df.educ > 9) & (df.educ <= 11), 1, 0)

    df.drop(columns=["age", "educ", "race", "bpl", "sex"], inplace=True)
    print("--> Groups created.")

    df = group_id(df, cols=group_cols, merge=True, name="groups")
    print("--> Groups id created.")

    # Get geography to cz level
    # Katrina data issue
    df.loc[(df.statefip == 22) & (df.puma == 77777), "puma"] = 1801

    df["PUMA"] = df["statefip"].astype(str).str.zfill(2) + df["puma"].astype(
        str
    ).str.zfill(4)

    df["PUMA"] = df["PUMA"].astype("int")

    df = df.rename(columns={"puma": "puma_original"})

    df1990 = df[df.year == 1990].merge(
        pd.read_stata(os.path.join(mainp, "cw_puma1990_czone.dta")),
        left_on="PUMA",
        right_on="puma1990",
    )

    df2000 = df[df.year != 1990].merge(
        pd.read_stata(os.path.join(mainp, "cw_puma2000_czone.dta")),
        left_on="PUMA",
        right_on="puma2000",
    )

    df = pd.concat([df1990, df2000])
    df["perwt"] = df["perwt"] * df["afactor"]

    del df1990
    del df2000
    print("--> 1990 and 2000 census information merged.")

    # #### Aggregate to cz x group x year level
    # Employment status:
    df["emp"] = np.where(df.empstat == 1, 1, 0)
    df["unemp"] = np.where(df.empstat == 2, 1, 0)
    df["nilf"] = np.where(df.empstat == 3, 1, 0)
    print("--> Employment indicator columns created.")

    # Manufacturing employment:
    df["manuf"] = np.where(
        (df.emp == 1) & (df.ind1990 >= 100) & (df.ind1990 < 400), 1, 0
    )
    df["nonmanuf"] = np.where(
        (df.emp == 1) & ((df.ind1990 < 100) | (df.ind1990 >= 400)), 1, 0
    )
    print("--> Manufacturer and non-manufacturer indicator columns created.")

    # Filling in weeks worked for 2008 ACS (using midpoint):
    df.loc[df.wkswork2 == 1, "wkswork1"] = 7
    df.loc[df.wkswork2 == 2, "wkswork1"] = 20
    df.loc[df.wkswork2 == 3, "wkswork1"] = 33
    df.loc[df.wkswork2 == 4, "wkswork1"] = 43.5
    df.loc[df.wkswork2 == 5, "wkswork1"] = 48.5
    df.loc[df.wkswork2 == 6, "wkswork1"] = 51
    print("--> Weeks worked for 2008 ACS filled with midpoints.")

    # Log weekly wage:
    df["lnwkwage"] = np.log(df.incwage / df.wkswork1)
    df.loc[df["lnwkwage"] == -np.inf, "lnwkwage"] = np.nan
    print("--> Log of weekly wages created.")

    # Hours:
    df["hours"] = df["uhrswork"] * df["wkswork1"]
    print("--> Hours worked created.")

    df.drop(columns=["empstat", "wkswork2", "incwage"], inplace=True)

    return df, group_cols


def create_sql(mainp):
    """
    Creates an SQLite database from the processed census DataFrame.

    This function processes census data using `create_base_df`, then stores the resulting
    DataFrame in an SQLite database. The database is created or overwritten at the specified
    path, and the DataFrame is saved as a table named 'census'.

    Args:
        mainp (str): The main path where the SQLite database will be saved. The function will
                     create a database named 'dataset.db' in this path.

    Raises:
        Exception: If there is an issue writing the DataFrame to the SQLite database,
                   the error is printed.
    """

    df, _ = create_base_df()

    # Creating the sql database.
    conn = sqlite3.connect(os.path.join(mainp, "dataset.db"))

    try:
        df.to_sql("census", conn, if_exists="replace", index=False)
        print("DataFrame successfully written to the database.")
    except Exception as e:
        print(f"An error occurred: {e}")


def create_base_igt(mainp):
    """
    Creates a grouped DataFrame with weighted means and sums for various employment and wage variables.

    This function processes a DataFrame to calculate the weighted mean of log wages (`lnwkwage`)
    and the weighted sums for various employment-related indicators, such as manufacturing and
    non-manufacturing employment, unemployment, and hours worked. The data is grouped by geographic
    zones, year, demographic groups, and other relevant factors, producing a DataFrame for further
    analysis.

    Args:
        mainp (str): The main path where the data is stored. The function assumes the existence
                     of the necessary data files and directories.

    Returns:
        pd.DataFrame: A DataFrame grouped by zone, year, and demographic group, containing weighted
                      means and sums of key variables, as well as calculated population shares and
                      their log transformations.

    Calculations:
        - Weighted mean of log wages (`lnwkwage`) using the `WtMean` function.
        - Weighted sum of employment status indicators (`manuf`, `nonmanuf`, `emp`, `unemp`, `nilf`,
          `hours`) using the `WtSum` function.
        - Population shares for manufacturing, non-manufacturing, unemployment, and NILF statuses.
        - Log transformations of summed columns (`manuf`, `nonmanuf`, `emp`, `unemp`, `nilf`, `hours`,
          `pop`).

    Example:
        >>> df_cgy = create_base_igt("/path/to/project")
        >>> print(df_cgy.head())
    """
    df, group_cols = create_base_df(mainp)
    """
    Recall the definition of average log wages.  

    Defining $\mathcal{I}_{gy}$ as the set of people in group $g$ in year $y$, the average log wage is defined as:

    $$
    W_{gy} = \frac{1}{P_{gy}} \sum_{i \in \mathcal{I}_{gy}} p_{i} w_{i}
    $$

    Where:

    - $w_{i}$ is the log wage of individual $i \in \mathcal{I}_{gy}$.
    - $p_{i}$ is the weight of individual $i \in \mathcal{I}_{gy}$.
    - $P_{gy} = \sum_{i \in \mathcal{I}_{gy}} p_{i}$ is the total population of group $g$ in year $y$.  


    This is the operation that the `WtMean` function performs, using `by_cols` to define the different $\mathcal{I}_{gy}$ groups and `perwt` as the weights $p_{i}$.
    
    For the Weighted Sum operation, we have:

    $$
    W^*_{gy} = \sum_{i \in \mathcal{I}_{gy}} p_{i} w_{i}
    $$

    This is the operation that the `WtSum` function performs.
    """

    # columns to take weighted mean
    wmean_cols = ["lnwkwage"]

    # columns to sum
    sum_cols = ["manuf", "nonmanuf", "emp", "unemp", "nilf", "hours"]

    # columns to group by (equivalent to the `gy` index in the equations.)
    by_cols = ["czone", "year", "groups", *group_cols, "college"]

    df_cgy = pd.concat(
        [
            WtMean(df, cols=wmean_cols, weight_col="perwt", by_cols=by_cols),
            WtSum(df, cols=sum_cols, weight_col="perwt", by_cols=by_cols, outw=True),
        ],
        axis=1,
    )
    df_cgy.rename(columns={"perwt": "pop"}, inplace=True)

    print(f"--> Dataset grouped by: {by_cols}")
    print(f"--> {wmean_cols} aggregated using weighted mean.")
    print(f"--> {sum_cols} aggregated using weighted sum.")

    for c in ["manuf", "nonmanuf", "unemp", "nilf"]:
        df_cgy["{}_share".format(c)] = df_cgy[c] / df_cgy["pop"]

    for c in [*sum_cols, "pop"]:
        df_cgy["ln{}".format(c)] = np.log(df_cgy[c])
        df_cgy.loc[df_cgy["ln{}".format(c)] == -np.inf, "ln{}".format(c)] = np.nan

    df_cgy = df_cgy.reset_index().set_index(["czone", "year", "groups"])

    print(f"--> Dataset shape: {df_cgy.shape}")
    print(f"--> Dataset columns: {df_cgy.columns}")

    return df_cgy, group_cols


# %%
if __name__ == "__main__":
    mainp = os.path.join("data")
    create_sql(mainp)
