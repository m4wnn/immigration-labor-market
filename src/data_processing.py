# %%
import os
import pandas as pd
import numpy as np

from toolz import pipe
from econtools import group_id
from functools import lru_cache
import logging
from agg import WtSum, WtMean


# Configuración del logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# %%


class CensusDataPipeline:
    # Directorio de datos
    DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
    # Columnas para la creación de grupos demográficos
    GROUPS_COLS = ["male", "native", "agebin", "educbin", "white"]
    BY_COLS = ["czone", "year", "groups", *GROUPS_COLS, "college"]
    # Configuraciones de columnas y binning
    AGE_BINS = [15, 27, 39, 51, 64]
    EDUC_BINS = [-1, 5, 6, 9, 11]

    def __init__(self):
        logger.info("CensusDataPipeline instance created")

    def load_data(self, path_key, convert_categoricals=True):
        """
        Load a dataset from the specified path key.

        Parameters:
        path_key (str): Key indicating which dataset to load.
        convert_categoricals (bool): Whether to convert categorical variables.

        Returns:
        DataFrame: Loaded dataset.
        """
        paths = {
            "census": os.path.join(self.DATA_DIR, "usa_00137.dta"),
            "cz_pre": os.path.join(self.DATA_DIR, "cw_puma1990_czone.dta"),
            "cz_post": os.path.join(self.DATA_DIR, "cw_puma2000_czone.dta"),
            "controls": os.path.join(self.DATA_DIR, "workfile_china.dta"),
        }
        logger.info(f"Loading data from {paths[path_key]}")
        return pd.read_stata(paths[path_key], convert_categoricals=convert_categoricals)

    def filter_working_age_population(self):
        """
        Filter dataset for individuals of working age (16-64) and in households.

        Returns:
        DataFrame: Filtered dataset.
        """
        df = self.load_data("census", convert_categoricals=False)
        df = df[(df.age >= 16) & (df.age <= 64) & (df.gq <= 2)]
        logger.info("Filtered working age population")
        return df

    def group_by_demographics(self):
        """
        Group population by demographic bins.

        Returns:
        DataFrame: Grouped dataset with additional demographic columns.
        """
        df = self.filter_working_age_population()

        df["male"] = np.where(df.sex == 1, 1, 0)
        df["native"] = np.where(df.bpl <= 99, 1, 0)
        df["agebin"] = pd.cut(df.age, bins=self.AGE_BINS, labels=False)
        df["educbin"] = pd.cut(df.educ, bins=self.EDUC_BINS, labels=False)
        df["white"] = np.where(df.race == 1, 1, 0)
        df["college"] = np.where((df.educ > 9) & (df.educ <= 11), 1, 0)

        df.drop(columns=["age", "educ", "race", "sex"], inplace=True)

        df = group_id(df, cols=self.GROUPS_COLS, merge=True, name="groups")
        logger.info("Grouped population by demographics")
        return df

    def katrina_correction(self):
        """
        Apply Katrina correction to PUMA codes for affected areas.

        Returns:
        DataFrame: Dataset with corrected PUMA codes.
        """
        df = self.group_by_demographics()
        df.loc[(df.statefip == 22) & (df.puma == 77777), "puma"] = 1801
        df["PUMA"] = df["statefip"].astype(str).str.zfill(2) + df["puma"].astype(
            str
        ).str.zfill(4)
        df["PUMA"] = df["PUMA"].astype("int")
        df = df.rename(columns={"puma": "puma_original"})
        logger.info("Applied Katrina correction")
        return df

    def cz_merge(self):
        """
        Merge Census data with commuting zone data.

        Returns:
        DataFrame: Dataset with commuting zone merged.
        """
        df = self.katrina_correction()
        df1990 = df[df.year == 1990].merge(
            self.load_data("cz_pre"), left_on="PUMA", right_on="puma1990"
        )
        df2000 = df[df.year != 1990].merge(
            self.load_data("cz_post"), left_on="PUMA", right_on="puma2000"
        )
        df = pd.concat([df1990, df2000])
        df["perwt"] = df["perwt"] * df["afactor"]
        logger.info("Merged with commuting zone data")
        return df

    def employment_status(self):
        """
        Determine employment status: employed, unemployed, and not in labor force.

        Returns:
        DataFrame: Dataset with employment status columns.
        """
        df = self.cz_merge()
        df["emp"] = np.where(df.empstat == 1, 1, 0)
        df["unemp"] = np.where(df.empstat == 2, 1, 0)
        df["nilf"] = np.where(df.empstat == 3, 1, 0)
        logger.info("Assigned employment status")
        return df

    def labor_force_groups(self):
        """
        Categorize labor force into manufacturing and non-manufacturing.

        Returns:
        DataFrame: Dataset with manufacturing and non-manufacturing columns.
        """
        df = self.employment_status()
        df["manuf"] = np.where(
            (df.emp == 1) & (df.ind1990 >= 100) & (df.ind1990 < 400), 1, 0
        )
        df["nonmanuf"] = np.where(
            (df.emp == 1) & ((df.ind1990 < 100) | (df.ind1990 >= 400)), 1, 0
        )
        logger.info("Categorized labor force groups")
        return df

    def log_wages(self):
        """
        Calculate log weekly wages based on income and weeks worked.

        Returns:
        DataFrame: Dataset with log weekly wage column.
        """
        df = self.labor_force_groups()
        df.loc[df.wkswork2 == 1, "wkswork1"] = 7
        df.loc[df.wkswork2 == 2, "wkswork1"] = 20
        df.loc[df.wkswork2 == 3, "wkswork1"] = 33
        df.loc[df.wkswork2 == 4, "wkswork1"] = 43.5
        df.loc[df.wkswork2 == 5, "wkswork1"] = 48.5
        df.loc[df.wkswork2 == 6, "wkswork1"] = 51
        df["lnwkwage"] = np.log(df.incwage / df.wkswork1)
        df.loc[df["lnwkwage"] == -np.inf, "lnwkwage"] = np.nan
        logger.info("Calculated log wages")
        return df

    @lru_cache(maxsize=None)
    def hours_worked(self):
        """
        Calculate total hours worked based on weekly hours and weeks worked.

        Returns:
        DataFrame: Dataset with hours worked column.
        """
        df = self.log_wages()
        df["hours"] = df["uhrswork"] * df["wkswork1"]
        df.drop(columns=["empstat", "wkswork2", "incwage"], inplace=True)
        logger.info("Calculated hours worked")
        return df

    def agg_WMean(self):
        """
        Calculate weighted mean for specific columns.

        Returns:
        DataFrame: Dataset with weighted mean calculations.
        """
        self.wmean_cols = ["lnwkwage"]
        df = self.hours_worked()
        df_cgy = WtMean(
            df, cols=self.wmean_cols, weight_col="perwt", by_cols=self.BY_COLS
        )
        logger.info("Aggregated weighted mean")
        return df_cgy

    def agg_WSum(self):
        """
        Calculate weighted sum for specified columns.

        Returns:
        DataFrame: Dataset with weighted sum calculations.
        """
        self.sum_cols = ["manuf", "nonmanuf", "emp", "unemp", "nilf", "hours"]
        df = self.hours_worked()
        df_cgy = WtSum(
            df, cols=self.sum_cols, weight_col="perwt", by_cols=self.BY_COLS, outw=True
        )
        logger.info("Aggregated weighted sum")
        return df_cgy

    def concat_agg(self):
        """
        Concatenate weighted mean and weighted sum aggregations.

        Returns:
        DataFrame: Concatenated aggregation dataset.
        """
        df_cgy = pd.concat([self.agg_WMean(), self.agg_WSum()], axis=1)
        df_cgy.rename(columns={"perwt": "pop"}, inplace=True)
        logger.info("Concatenated aggregated data")
        return df_cgy

    def labor_participation_shares(self):
        """
        Calculate labor participation shares and logarithmic values for labor data.

        Returns:
        DataFrame: Final dataset with labor shares and log-transformed columns.
        """
        df_cgy = self.concat_agg()
        for c in ["manuf", "nonmanuf", "unemp", "nilf"]:
            df_cgy[f"{c}_share"] = df_cgy[c] / df_cgy["pop"]
        for c in [*self.sum_cols, "pop"]:
            df_cgy[f"ln{c}"] = np.log(df_cgy[c])
            df_cgy.loc[df_cgy[f"ln{c}"] == -np.inf, f"ln{c}"] = np.nan
        df_cgy = df_cgy.reset_index().set_index(["czone", "year", "groups"])
        logger.info("Calculated labor participation shares")
        return df_cgy

    def run(self):
        """
        Run the entire pipeline and return the final aggregated dataset.

        Returns:
        DataFrame: Final dataset.
        """
        logger.info("Running CensusDataPipeline")
        return self.labor_participation_shares()


# %%


class OutcomesDataPipeline:

    # Initialize CensusDataPipeline and group columns as class-level constants
    CENSUS_PIPELINE = CensusDataPipeline()
    GROUPS_COLS = CENSUS_PIPELINE.GROUPS_COLS

    def __init__(self, CA=False):
        self.CA = CA
        self._weights = (
            None  # Cache for the result of weights() to avoid recalculations
        )

    def weights(self):
        # Calculate weights only once and store in self._weights for efficiency
        if self._weights is None:
            df_cgy = self.CENSUS_PIPELINE.run()
            df_w = df_cgy.reset_index()[["czone", "year", "groups", "hours"]].copy()
            df_w = df_w.set_index(["czone", "year", "groups"])
            df_w = df_w.groupby(
                level=["czone", "year", "groups"]
            ).sum()  # Sum hours for duplicate indices
            df_w = df_w.unstack(level=[1, 2], fill_value=0.0)
            df_w = df_w.stack(level=[1, 2])

            # Assign weights
            df_w["weight_cgt"] = df_w["hours"] / df_w.groupby(["czone", "year"])[
                "hours"
            ].transform("sum")
            df_w["weight_cg"] = df_w.groupby(["czone", "groups"])[
                "weight_cgt"
            ].transform("mean")

            df_cgy = pd.concat(
                [
                    df_cgy,
                    df_w[["weight_cgt", "weight_cg"]].rename(
                        columns={
                            "weight_cg": "weight",
                            "weight_cgt": "weight_non_adjusted",
                        }
                    ),
                ],
                axis=1,
            )
            self._weights = df_cgy  # Store result in class context to reuse

        return self._weights

    def _get_masks(self, df_cgy):
        """Generates and returns commonly used boolean masks."""
        df_reset = df_cgy.reset_index()
        col_mask = df_reset.college == 1
        ncol_mask = df_reset.college == 0
        male_mask = df_reset.male == 1
        female_mask = df_reset.male == 0
        native_mask = df_reset.native == 1
        return col_mask, ncol_mask, male_mask, female_mask, native_mask

    def avg_log_wages(self):
        # Access cached weights data
        df_cgy = self.weights()
        # Generate masks only once
        col_mask, ncol_mask, male_mask, female_mask, native_mask = self._get_masks(
            df_cgy
        )

        def fun(m):
            return WtMean(
                df_cgy.reset_index(),
                cols=["lnwkwage"],
                weight_col="weight" if self.CA else "weight_non_adjusted",
                by_cols=["czone", "year"],
                mask=m,
            )

        # Concatenate results of various aggregations using masks
        df_cy = pd.concat(
            [
                fun(None),
                fun(col_mask).rename(columns={"lnwkwage": "lnwkwage_col"}),
                fun(ncol_mask).rename(columns={"lnwkwage": "lnwkwage_ncol"}),
                fun(male_mask).rename(columns={"lnwkwage": "lnwkwage_male"}),
                fun(female_mask).rename(columns={"lnwkwage": "lnwkwage_female"}),
                fun(col_mask & male_mask).rename(
                    columns={"lnwkwage": "lnwkwage_col_male"}
                ),
                fun(col_mask & female_mask).rename(
                    columns={"lnwkwage": "lnwkwage_col_female"}
                ),
                fun(ncol_mask & male_mask).rename(
                    columns={"lnwkwage": "lnwkwage_ncol_male"}
                ),
                fun(ncol_mask & female_mask).rename(
                    columns={"lnwkwage": "lnwkwage_ncol_female"}
                ),
                fun(native_mask).rename(columns={"lnwkwage": "lnwkwage_native"}),
                fun(col_mask & native_mask).rename(
                    columns={"lnwkwage": "lnwkwage_col_native"}
                ),
                fun(ncol_mask & native_mask).rename(
                    columns={"lnwkwage": "lnwkwage_ncol_native"}
                ),
            ],
            axis=1,
        )
        return df_cy

    def avg_labor_shares(self):
        # Access cached weights data
        df_cgy = self.weights()
        # Generate masks only once
        col_mask, ncol_mask, _, _, native_mask = self._get_masks(df_cgy)

        self.share_cols = ["manuf_share", "nonmanuf_share", "unemp_share", "nilf_share"]

        def fun(m):
            return WtMean(
                df_cgy.reset_index(),
                cols=self.share_cols,
                weight_col="weight" if self.CA else "weight_non_adjusted",
                by_cols=["czone", "year"],
                mask=m,
            )

        # Concatenate results of various aggregations using masks
        df_cy = pd.concat(
            [
                fun(None),
                fun(col_mask).add_suffix("_col"),
                fun(ncol_mask).add_suffix("_ncol"),
                fun(native_mask).add_suffix("_native"),
                fun(col_mask & native_mask).add_suffix("_col_native"),
                fun(ncol_mask & native_mask).add_suffix("_ncol_native"),
            ],
            axis=1,
        )
        return df_cy

    def avg_labor_counts(self):
        # Access cached weights data
        df_cgy = self.weights()
        self.count_cols = [
            "lnmanuf",
            "lnnonmanuf",
            "lnemp",
            "lnunemp",
            "lnnilf",
            "lnpop",
        ]
        df_cy = WtMean(
            df_cgy.reset_index(),
            cols=self.count_cols,
            weight_col="weight" if self.CA else "weight_non_adjusted",
            by_cols=["czone", "year"],
        )
        return df_cy

    def concat_avg(self):
        # Concatenate results from avg_log_wages, avg_labor_shares, and avg_labor_counts
        df_cy = pd.concat(
            [self.avg_log_wages(), self.avg_labor_shares(), self.avg_labor_counts()],
            axis=1,
        )
        return df_cy

    def change_10_years(self):
        df_cy = self.concat_avg()
        cols = df_cy.columns.to_list()

        # Reshape to wide format:
        df_cy = df_cy.reset_index().pivot_table(index="czone", columns="year")

        # Compute decadal differences:
        for c in cols:
            df_cy["D{}".format(c), 1990] = df_cy[c, 2000] - df_cy[c, 1990]
            df_cy["D{}".format(c), 2000] = (df_cy[c, 2008] - df_cy[c, 2000]) * (10 / 7)
        # Reshape back to long format:
        df_cy = df_cy.stack().drop(columns=cols)
        return df_cy

    def rename_change_10_years(self):
        df_cy = self.change_10_years()

        for c in self.share_cols:
            df_cy["D{}".format(c)] = df_cy["D{}".format(c)] * 100.0
            df_cy["D{}_col".format(c)] = df_cy["D{}_col".format(c)] * 100.0
            df_cy["D{}_ncol".format(c)] = df_cy["D{}_ncol".format(c)] * 100.0
            df_cy["D{}_native".format(c)] = df_cy["D{}_native".format(c)] * 100.0
            df_cy["D{}_col_native".format(c)] = (
                df_cy["D{}_col_native".format(c)] * 100.0
            )
            df_cy["D{}_ncol_native".format(c)] = (
                df_cy["D{}_ncol_native".format(c)] * 100.0
            )

        # Multiply by 100 b/c reports log points:
        cols_mask = df_cy.columns.str.contains("Dln")
        for c in df_cy.columns[cols_mask]:
            df_cy[c] = df_cy[c] * 100.0

        self.ADHnames = {
            # outcome for Table 3
            "Dmanuf_share": "d_sh_empl_mfg",
            # outcomes for Table 5
            # panel A
            "Dlnmanuf": "lnchg_no_empl_mfg",
            "Dlnnonmanuf": "lnchg_no_empl_nmfg",
            "Dlnunemp": "lnchg_no_unempl",
            "Dlnnilf": "lnchg_no_nilf",
            # panel B
            "Dmanuf_share": "d_sh_empl_mfg",
            "Dnonmanuf_share": "d_sh_empl_nmfg",
            "Dunemp_share": "d_sh_unempl",
            "Dnilf_share": "d_sh_nilf",
            # panel C
            "Dmanuf_share_col": "d_sh_empl_mfg_edu_c",
            "Dnonmanuf_share_col": "d_sh_empl_nmfg_edu_c",
            "Dunemp_share_col": "d_sh_unempl_edu_c",
            "Dnilf_share_col": "d_sh_nilf_edu_c",
            # panel D
            "Dmanuf_share_ncol": "d_sh_empl_mfg_edu_nc",
            "Dnonmanuf_share_ncol": "d_sh_empl_nmfg_edu_nc",
            "Dunemp_share_ncol": "d_sh_unempl_edu_nc",
            "Dnilf_share_ncol": "d_sh_nilf_edu_nc",
            # panel (Native)
            "Dmanuf_share_native": "d_sh_empl_mfg_native",
            "Dnonmanuf_share_native": "d_sh_empl_nmfg_native",
            "Dunemp_share_native": "d_sh_unempl_native",
            "Dnilf_share_native": "d_sh_nilf_native",
            # panel (Native)
            "Dmanuf_share_col_native": "d_sh_empl_mfg_edu_c_native",
            "Dnonmanuf_share_col_native": "d_sh_empl_nmfg_edu_c_native",
            "Dunemp_share_col_native": "d_sh_unempl_edu_c_native",
            "Dnilf_share_col_native": "d_sh_nilf_edu_c_native",
            # panel (Native)
            "Dmanuf_share_ncol_native": "d_sh_empl_mfg_edu_nc_native",
            "Dnonmanuf_share_ncol_native": "d_sh_empl_nmfg_edu_nc_native",
            "Dunemp_share_ncol_native": "d_sh_unempl_edu_nc_native",
            "Dnilf_share_ncol_native": "d_sh_nilf_edu_nc_native",
            # outcomes for Table 6
            "Dlnwkwage": "d_avg_lnwkwage",
            "Dlnwkwage_col": "d_avg_lnwkwage_c",
            "Dlnwkwage_ncol": "d_avg_lnwkwage_nc",
            "Dlnwkwage_male": "d_avg_lnwkwage_m",
            "Dlnwkwage_female": "d_avg_lnwkwage_f",
            "Dlnwkwage_col_male": "d_avg_lnwkwage_c_m",
            "Dlnwkwage_col_female": "d_avg_lnwkwage_c_f",
            "Dlnwkwage_ncol_male": "d_avg_lnwkwage_nc_m",
            "Dlnwkwage_ncol_female": "d_avg_lnwkwage_nc_f",
            "Dlnwkwage_native": "d_avg_lnwkwage_native",
            "Dlnwkwage_col_native": "d_avg_lnwkwage_c_native",
            "Dlnwkwage_ncol_native": "d_avg_lnwkwage_nc_native",
        }

        df_cy.rename(columns=self.ADHnames, inplace=True)
        return df_cy

    def merge_with_dorn_data(self):
        df_cy = self.rename_change_10_years()

        df_NCA = self.CENSUS_PIPELINE.load_data("controls")

        # CA data:
        CA_cols = [v for k, v in self.ADHnames.items()]

        other_cols = df_NCA.columns.difference(CA_cols)

        df_CA = pd.merge(
            df_cy,
            df_NCA[other_cols],
            left_on=["czone", "year"],
            right_on=["czone", "yr"],
            how="inner",
        )

        return df_CA

    def run(self):
        # Run the full pipeline
        return self.merge_with_dorn_data()


# %%


def main():
    DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")

    OutcomesDataPipeline(CA=True).run().hto_csv(
        os.path.join(DATA_DIR, "outcomes_CA.csv")
    )

    OutcomesDataPipeline(CA=False).run().to_csv(
        os.path.join(DATA_DIR, "outcomes_CA.csv")
    )


if __name__ == "__main__":
    main()
