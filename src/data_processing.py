# %%
import pandas as pd
from econtools import group_id
import numpy as np
import os

from functools import cached_property


# %%
class Paths:
    def __init__(self, data_path):
        self.FILE_PATH = os.path.dirname(os.path.abspath(__file__))
        self.ROOT_PATH = os.path.join(self.FILE_PATH, "..")
        self.DATA_PATH = os.path.join(self.ROOT_PATH, data_path)


# %%
class CensusDataPipeline:

    def __init__(self, data_path):
        self.PATHS = Paths(data_path)

    @cached_property
    def _census_data_path(self):
        return os.path.join(self.PATHS.DATA_PATH, "usa_00121.dta")

    @cached_property
    def _census_data(self):
        df = pd.read_stata(self._census_data_path, convert_categoricals=False)
        # Keep those aged 20-60 and not in group quarters:
        df = df[(df.age >= 20) & (df.age <= 60) & (df.gq <= 2)]
        # Katrina data issue:
        df.loc[(df.statefip == 22) & (df.puma == 77777), "puma"] = 1801
        return df

    @cached_property
    def _census_data_info(self):
        return pd.read_stata(self._census_data_path, iterator=True).variable_labels()

    def cz_census_data_2008(self):

        df2008 = self._census_data[self._census_data.year == 2008].copy()

        # 2008 PUMAs:
        df2008["PUMA"] = df2008["statefip"].astype(str).str.zfill(2) + df2008[
            "puma"
        ].astype("int").astype(str).str.zfill(4)

        df2008["PUMA"] = df2008["PUMA"].astype("int")

        # Merge to CZs:
        df2008 = pd.merge(
            df2008,
            pd.read_stata(
                os.path.join(self.PATHS.DATA_PATH, "cw_puma2000_czone.dta"),
            ),
            left_on="PUMA",
            right_on="puma2000",
        )
        return df2008

    def cz_census_data_1980(self):

        df1980 = self._census_data[self._census_data.year == 1980].copy()
        # 1980 county groups:
        df1980["ctygrp1980"] = df1980["statefip"].astype(str).str.zfill(2) + df1980[
            "cntygp98"
        ].astype(int).astype(str).str.zfill(3)
        df1980["ctygrp1980"] = pd.to_numeric(df1980["ctygrp1980"])

        # Merge:
        df1980 = pd.merge(
            df1980,
            pd.read_stata(
                os.path.join(self.PATHS.DATA_PATH, "cw_ctygrp1980_czone_corr.dta"),
            ),
            on="ctygrp1980",
        )
        return df1980

    @cached_property
    def _cz_census_data(self):
        df = pd.concat([self.cz_census_data_1980(), self.cz_census_data_2008()])
        # Create new individual weights at the CZ level:
        df["weight"] = df["perwt"] * df["afactor"]
        df.drop(columns=["perwt", "afactor"])

        return df

    def get_data(self):
        return self._cz_census_data


# %%
class DataPipeline:
    def __init__(self, data_path):
        self.PATHS = Paths(data_path)
        self.CENSUS = CensusDataPipeline(data_path)

    def _key_shock_numerator_elements(self):
        """
        Construct elements of the numerator of the key shock and instrument

        I.e. $I_{c,2007}$, $I_{c,1980}$, $f_{cs,1980}$, $I_{s,2007}$, and $I_{s,1980}$
        """
        imm = self.CENSUS.get_data().copy()
        imm.loc[(imm.bpl >= 450) & (imm.bpl <= 459), "nativity"] = 450
        imm.loc[(imm.bpl >= 460) & (imm.bpl <= 465), "nativity"] = 460
        imm.loc[imm.bpl == 499, "nativity"] = 499
        imm.loc[imm.bpl == 500, "nativity"] = 500
        imm.loc[imm.bpl == 501, "nativity"] = 501
        imm.loc[imm.bpl == 502, "nativity"] = 502
        imm.loc[imm.bpl == 509, "nativity"] = 509
        imm.loc[(imm.bpl >= 510) & (imm.bpl <= 519), "nativity"] = 510
        imm.loc[(imm.bpl >= 520) & (imm.bpl <= 529), "nativity"] = 520
        imm.loc[(imm.bpl >= 530) & (imm.bpl <= 550), "nativity"] = 530
        imm.loc[imm.bpl == 599, "nativity"] = 599
        imm.loc[imm.bpl == 600, "nativity"] = 600
        imm.loc[(imm.bpl >= 700) & (imm.bpl <= 710), "nativity"] = 700

        imm.dropna(subset=["nativity"], inplace=True)
        imm = group_id(imm, cols=["nativity"], merge=True, name="source")

        imm = imm[["czone", "source", "year", "weight"]]
        return imm

    def _aggregated_immigrant_counts(self):
        """
        Compute immigrant counts at different levels of aggregation:
        """
        imm = self._key_shock_numerator_elements()
        imm["weight"] = imm["weight"].astype(float)  # to ensure precision in sum
        imm["I_csy"] = imm.groupby(["czone", "source", "year"])[["weight"]].transform(
            "sum"
        )
        imm["I_cy"] = imm.groupby(["czone", "year"])[["weight"]].transform("sum")
        imm["I_sy"] = imm.groupby(["source", "year"])[["weight"]].transform("sum")
        imm = imm[["czone", "source", "year", "I_cy", "I_csy", "I_sy"]].copy()
        imm.drop_duplicates(inplace=True)
        return imm

    @cached_property
    def _source_immigrant_fraction(self):
        """
        Construct the fraction of immigrants from a source who are in a CZ in 1980:
        """
        imm = self._aggregated_immigrant_counts()
        imm1980 = imm[imm.year == 1980].copy()
        imm1980["share_cs80"] = imm1980["I_csy"] / imm1980["I_sy"]
        imm1980 = imm1980.groupby(["czone", "source"])[["share_cs80"]].sum()
        imm = pd.merge(imm, imm1980, on=["czone", "source"], how="left")

        imm.drop(columns=["I_csy"], inplace=True)
        imm.rename(columns={"I_cy": "I_c", "I_sy": "I_s"}, inplace=True)

        # Reshape to wide format:
        imm = imm.pivot_table(index=["czone", "source"], columns="year")

        # Fill in missing values:
        for y in [1980, 2008]:
            imm["I_s", y] = imm.groupby(level=["source"]).transform(np.nanmax)["I_s", y]
            imm["I_c", y] = imm.groupby(level=["czone"]).transform(np.nanmax)["I_c", y]
            imm.loc[imm["share_cs80", y].isnull(), ("share_cs80", y)] = 0.0

        # Compute the time differences:
        for c in ["I_s", "I_c"]:
            imm["D{}".format(c)] = imm[c, 2008] - imm[c, 1980]
            imm["{}_80".format(c)] = imm[c, 1980]

        imm["fDI_s"] = imm["DI_s"] * imm["share_cs80", 1980]
        return imm

    @cached_property
    def _numerator_shock_and_instrument(self):

        imm = self._source_immigrant_fraction.copy()
        # Construct numerator of shock and instrument:
        imm["fDI_s"] = imm["DI_s"] * imm["share_cs80", 1980]

        num_c = pd.concat(
            [
                imm.groupby(level=["czone"])["DI_c"].max(),
                imm.groupby(level=["czone"])["I_c_80"].max(),
                imm.groupby(level=["czone"])["fDI_s"].sum(),
            ],
            axis=1,
        )

        num_c["one_over_I_c_80"] = 1 / num_c["I_c_80"]
        return num_c

    @cached_property
    def _controls(self):
        # #### Construct denominator of key shock and instrument + controls + outcomes
        df = self.CENSUS.get_data().copy()

        def MySum(mask, newname, col="weight"):
            return df[mask].groupby("czone")[[col]].sum().rename(columns={col: newname})

        # Controls:
        is_1980 = df.year == 1980
        is_manuf = (df.ind1990 >= 100) & (df.ind1990 < 400)
        is_emp = df.empstat == 1
        is_fem = df.sex == 2
        is_col = df.educ >= 10
        is_fborn = (df.bpl > 120) & (df.bpl < 900)

        df_c = pd.concat(
            [
                MySum(is_1980, "pop_80"),
                MySum(is_1980 & is_manuf & is_emp, "manuf_80"),
                MySum(is_1980 & is_fem & is_emp, "female_80"),
                MySum(is_1980 & is_emp, "emp_80"),
                MySum(is_1980 & is_col, "col_80"),
                MySum(is_1980 & is_fborn, "fborn_80"),
                MySum(is_1980 & (df.bpl < 900), "fborn_denom_80"),
            ],
            axis=1,
        )

        df_c["manuf_share_80"] = (
            df_c.manuf_80 / df_c.emp_80
        )  # manufacturing share of employed
        df_c["female_share_80"] = (
            df_c.female_80 / df_c.emp_80
        )  # female share of employed
        df_c["col_share_80"] = df_c.col_80 / df_c.pop_80  # college share of population
        df_c["lnpop_80"] = np.log(df_c.pop_80)  # log of population (in age range)
        df_c["fborn_share_80"] = (
            df_c.fborn_80 / df_c.fborn_denom_80
        )  # foreign-born share of employed
        return df_c

    @cached_property
    def _outcomes(self):
        df = self.CENSUS.get_data().copy()

        def MySum(mask, newname, col="weight"):
            return df[mask].groupby("czone")[[col]].sum().rename(columns={col: newname})

        # Filling in weeks worked for 2008 ACS (using midpoint):
        df.loc[(df.year == 2008) & (df.wkswork2 == 1), "wkswork1"] = 7
        df.loc[(df.year == 2008) & (df.wkswork2 == 2), "wkswork1"] = 20
        df.loc[(df.year == 2008) & (df.wkswork2 == 3), "wkswork1"] = 33
        df.loc[(df.year == 2008) & (df.wkswork2 == 4), "wkswork1"] = 43.5
        df.loc[(df.year == 2008) & (df.wkswork2 == 5), "wkswork1"] = 48.5
        df.loc[(df.year == 2008) & (df.wkswork2 == 6), "wkswork1"] = 51
        df["hours"] = df["uhrswork"] * df["wkswork1"]

        df = df[df.bpl < 100].copy()  # excluding US OUTLYING AREAS/TERRITORIES

        df["incwage"] = df["incwage"] * df["weight"]
        df["hours"] = df["hours"] * df["weight"]

        is_emp = df.empstat == 1
        is_unemp = df.empstat == 2
        is_nilf = df.empstat == 3

        for y in [1980, 2008]:

            def concat_df(y):
                df_c = pd.concat(
                    [
                        MySum((df.year == y) & is_nilf, "nilf_num_{}".format(y)),
                        MySum(
                            (df.year == y) & (is_emp | is_unemp | is_nilf),
                            "nilf_denom_{}".format(y),
                        ),
                        MySum((df.year == y) & is_unemp, "unemp_num_{}".format(y)),
                        MySum(
                            (df.year == y) & (is_emp | is_unemp),
                            "unemp_denom_{}".format(y),
                        ),
                        MySum(df.year == y, "inc_{}".format(y), "incwage"),
                        MySum(df.year == y, "hours_{}".format(y), "hours"),
                    ],
                    axis=1,
                )
                return df_c

            if y == 1980:
                df_c = concat_df(y)
            else:
                df_c = pd.concat([df_c, concat_df(y)], axis=1)

            df_c["nilf_rate_{}".format(y)] = (
                df_c["nilf_num_{}".format(y)] / df_c["nilf_denom_{}".format(y)]
            )
            df_c["unemp_rate_{}".format(y)] = (
                df_c["unemp_num_{}".format(y)] / df_c["unemp_denom_{}".format(y)]
            )
            df_c["ln_wage_{}".format(y)] = np.log(
                df_c["inc_{}".format(y)] / df_c["hours_{}".format(y)]
            )

            df_c.drop(
                columns=[
                    "nilf_num_{}".format(y),
                    "nilf_denom_{}".format(y),
                    "unemp_num_{}".format(y),
                    "unemp_denom_{}".format(y),
                    "inc_{}".format(y),
                    "hours_{}".format(y),
                ],
                inplace=True,
            )

        for v in ["ln_wage", "unemp_rate", "nilf_rate"]:
            df_c["D{}".format(v)] = (
                df_c["{}_2008".format(v)] - df_c["{}_1980".format(v)]
            )
        return df_c

    @cached_property
    def _shock(self):
        ## Construct the shock and the instrument
        df_c = pd.concat(
            [self._numerator_shock_and_instrument, self._outcomes, self._controls],
            axis=1,
        )

        df_c["x"] = df_c["DI_c"] / df_c["pop_80"]

        return df_c["x"]

    @cached_property
    def _instruments(self):
        ## Construct the shock and the instrument
        df_c = pd.concat(
            [self._numerator_shock_and_instrument, self._outcomes, self._controls],
            axis=1,
        )

        df_c["z_1"] = df_c["fDI_s"] / df_c["pop_80"]
        df_c["z_2"] = df_c["fDI_s"] / df_c["I_c_80"]

        return df_c[["z_1", "z_2"]]

    @cached_property
    def _cz_clustering(self):
        return pd.read_stata(os.path.join(self.PATHS.DATA_PATH, "cz_state.dta"))

    def load_data(self):
        df_c = pd.concat(
            [
                self._shock,
                self._outcomes[["Dln_wage", "Dunemp_rate", "Dnilf_rate"]],
                self._controls,
                self._instruments,
            ],
            axis=1,
        )

        # Merge in state associated with each czone for clustering
        df_c = pd.merge(
            df_c,
            self._cz_clustering,
            on="czone",
            how="left",
        )
        # Assign Alaska and Hawaii the same cluster:
        df_c.loc[df_c.statefip.isnull(), "statefip"] = 99
        return df_c


# %%


def main():
    import warnings

    warnings.simplefilter(action="ignore", category=FutureWarning)

    tmp = DataPipeline(data_path="data")
    tmp_df = tmp.load_data()
    tmp_df.to_csv(os.path.join(tmp.PATHS.DATA_PATH, "data.csv"))

    print(f"-- DATA SAVED IN: {tmp.PATHS.DATA_PATH}")


if __name__ == "__main__":
    main()
