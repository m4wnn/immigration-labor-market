import os
import pandas as pd
import numpy as np

from toolz import pipe
from econtools import group_id

# from src.write_variable_labels import write_variable_labels
# from src.agg import WtSum, WtMean


def data_path():
    file = os.path.dirname(os.path.abspath(__file__))
    data = os.path.join(file, "..", "data")
    return data


def stata_paths():
    paths = {
        "census": os.path.join(data_path(), "usa_00137.dta"),
        "cz_pre": os.path.join(data_path(), "cw_puma1990_czone.dta"),
        "cz_post": os.path.join(data_path(), "cw_puma2000_czone.dta"),
        "controls": os.path.join(data_path(), "workfile_china.dta"),
    }
    return paths


def load_census_data():
    df = pipe(
        stata_paths()["census"],
        lambda x: pd.read_stata(x, convert_categoricals=False),
        lambda x: x[(x.age >= 16) & (x.age <= 64) & (x.gq <= 2)],
    )
    return df


def load_cz_pre_data():
    df = pd.read_stata(stata_paths()["cz_pre"])
    return df


def load_cz_post_data():
    df = pd.read_stata(stata_paths()["cz_post"])
    return df


def census_groups():
    df = load_census_data()
    groups_cols = ["male", "native", "agebin", "educbin", "white"]

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

    df.drop(columns=["age", "educ", "race", "sex"], inplace=True)

    df = group_id(df, cols=groups_cols, merge=True, name="groups")
    return df


def census_katrina_correction():
    df = census_groups()

    df.loc[(df.statefip == 22) & (df.puma == 77777), "puma"] = 1801

    df["PUMA"] = df["statefip"].astype(str).str.zfill(2) + df["puma"].astype(
        str
    ).str.zfill(4)

    df["PUMA"] = df["PUMA"].astype("int")

    df = df.rename(columns={"puma": "puma_original"})

    return df


def census_cz_merge():
    df = census_katrina_correction()

    df1990 = df[df.year == 1990].merge(
        load_cz_pre_data(),
        left_on="PUMA",
        right_on="puma1990",
    )

    df2000 = df[df.year != 1990].merge(
        load_cz_post_data(),
        left_on="PUMA",
        right_on="puma2000",
    )

    df = pd.concat([df1990, df2000])

    df["perwt"] = df["perwt"] * df["afactor"]

    return df


def main():
    tmp = census_cz_merge()
    print(tmp)
    print(tmp.dtypes)
    print(tmp.describe().round(0))


if __name__ == "__main__":
    main()
