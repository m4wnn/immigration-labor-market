import os
import pandas as pd
import numpy as np

from toolz import pipe
from econtools import group_id

# from src.write_variable_labels import write_variable_labels
# from src.agg import WtSum, WtMean


class CensusDataPipeline:
    DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")

    def __init__(self):
        pass

    def load_data(self, path_key, convert_categoricals=True):
        paths = {
            "census": os.path.join(self.DATA_DIR, "usa_00137.dta"),
            "cz_pre": os.path.join(self.DATA_DIR, "cw_puma1990_czone.dta"),
            "cz_post": os.path.join(self.DATA_DIR, "cw_puma2000_czone.dta"),
            "controls": os.path.join(self.DATA_DIR, "workfile_china.dta"),
        }
        return pd.read_stata(paths[path_key], convert_categoricals=convert_categoricals)

    def filter_working_age_population(self):
        df = pipe(
            self.load_data("census", convert_categoricals=False),
            lambda x: x[(x.age >= 16) & (x.age <= 64) & (x.gq <= 2)],
        )
        return df

    def group_by_demographics(self):
        df = self.filter_working_age_population()
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

    def katrina_correction(self):
        df = self.group_by_demographics()

        df.loc[(df.statefip == 22) & (df.puma == 77777), "puma"] = 1801

        df["PUMA"] = df["statefip"].astype(str).str.zfill(2) + df["puma"].astype(
            str
        ).str.zfill(4)

        df["PUMA"] = df["PUMA"].astype("int")

        df = df.rename(columns={"puma": "puma_original"})

        return df

    def cz_merge(self):
        df = self.katrina_correction()

        df1990 = df[df.year == 1990].merge(
            self.load_data("cz_pre"),
            left_on="PUMA",
            right_on="puma1990",
        )

        df2000 = df[df.year != 1990].merge(
            self.load_data("cz_post"),
            left_on="PUMA",
            right_on="puma2000",
        )

        df = pd.concat([df1990, df2000])

        df["perwt"] = df["perwt"] * df["afactor"]

        return df

    def run(self):
        return self.cz_merge()


def main():
    tmp = CensusDataPipeline().run()
    print(tmp)
    print(tmp.dtypes)
    print(tmp.describe().round(0))


if __name__ == "__main__":
    main()
