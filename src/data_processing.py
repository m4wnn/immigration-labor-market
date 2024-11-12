import os
import pandas as pd
import numpy as np

from toolz import pipe
from econtools import group_id
from functools import lru_cache
import logging
from src.agg import WtSum, WtMean


# Configuración del logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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


class OutcomesDataPipeline:
    """
    A pipeline to process and analyze census and outcomes data with various filtering
    and masking techniques, computing log wages, labor shares, and labor counts.
    """

    # Initialize CensusDataPipeline and group columns as class-level constants
    CENSUS_PIPELINE = CensusDataPipeline()
    GROUPS_COLS = CENSUS_PIPELINE.GROUPS_COLS
    AVG_MASKS = [
        ["native"],
        ["native", "col"],
        ["native", "ncol"],
        ["native", "male"],
        ["native", "female"],
    ]

    # Set up logger for the class
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)

    def __init__(self, CA=False):
        """
        Initialize the OutcomesDataPipeline.

        Parameters:
        CA (bool): Determines whether to use adjusted weights or non-adjusted weights.
        """
        self.CA = CA
        self.logger.info("OutcomesDataPipeline initialized with CA=%s", CA)

    @lru_cache(maxsize=None)
    def weights(self):
        """
        Calculate and cache the weights for the dataset.

        Returns:
        DataFrame: Dataset with calculated weights columns.
        """
        self.logger.info("Calculating weights")
        df_cgy = self.CENSUS_PIPELINE.run()
        df_w = df_cgy.reset_index()[["czone", "year", "groups", "hours"]].copy()
        df_w = (
            df_w.set_index(["czone", "year", "groups"])
            .groupby(level=["czone", "year", "groups"])
            .sum()
        )
        df_w = df_w.unstack(level=[1, 2], fill_value=0.0).stack(level=[1, 2])

        # Assign weights
        df_w["weight_cgt"] = df_w["hours"] / df_w.groupby(["czone", "year"])[
            "hours"
        ].transform("sum")
        df_w["weight_cg"] = df_w.groupby(["czone", "groups"])["weight_cgt"].transform(
            "mean"
        )

        df_cgy = pd.concat(
            [
                df_cgy,
                df_w[["weight_cgt", "weight_cg"]].rename(
                    columns={"weight_cg": "weight", "weight_cgt": "weight_non_adjusted"}
                ),
            ],
            axis=1,
        )
        self.logger.info("Weights calculation completed")
        return df_cgy

    @lru_cache(maxsize=None)
    def _get_masks(self):
        """
        Generates and returns commonly used boolean masks.

        Returns:
        dict: Dictionary of masks based on specific conditions.
        """
        self.logger.info("Generating masks")
        df_reset = self.weights().reset_index()
        masks = {
            "col": df_reset.college == 1,
            "ncol": df_reset.college == 0,
            "male": df_reset.male == 1,
            "female": df_reset.male == 0,
            "native": df_reset.native == 1,
        }
        return masks

    def avg_log_wages(self):
        """
        Calculate the weighted mean of log wages with various masks and return a DataFrame.

        Returns:
        DataFrame: Aggregated log wage data with columns named according to applied masks.
        """
        self.logger.info("Calculating average log wages")
        df_cgy = self.weights()
        masks = self._get_masks()

        def apply_wtmean(mask):
            return WtMean(
                df_cgy.reset_index(),
                cols=["lnwkwage"],
                weight_col="weight" if self.CA else "weight_non_adjusted",
                by_cols=["czone", "year"],
                mask=mask,
            )

        results = [apply_wtmean(None)]
        for mask_list in self.AVG_MASKS:
            combined_mask = np.logical_and.reduce([masks[var] for var in mask_list])
            result_df = apply_wtmean(combined_mask)
            mask_name = "_".join(["lnwkwage"] + mask_list)
            result_df = result_df.rename(columns={"lnwkwage": mask_name})
            results.append(result_df)

        df_cy = pd.concat(results, axis=1)
        self.logger.info("Average log wages calculation completed")
        return df_cy

    def avg_labor_shares(self):
        """
        Calculate the weighted mean of labor shares with various masks and return a DataFrame.

        Returns:
        DataFrame: Aggregated labor share data with columns named according to applied masks.
        """
        self.logger.info("Calculating average labor shares")
        df_cgy = self.weights()
        masks = self._get_masks()
        self.share_cols = ["manuf_share", "nonmanuf_share", "unemp_share", "nilf_share"]

        def apply_wtmean(mask):
            return WtMean(
                df_cgy.reset_index(),
                cols=self.share_cols,
                weight_col="weight" if self.CA else "weight_non_adjusted",
                by_cols=["czone", "year"],
                mask=mask,
            )

        results = [apply_wtmean(None)]
        for mask_list in self.AVG_MASKS:
            combined_mask = np.logical_and.reduce([masks[var] for var in mask_list])
            result_df = apply_wtmean(combined_mask)
            mask_name = "_".join(mask_list)
            result_df = result_df.add_suffix(f"_{mask_name}")
            results.append(result_df)

        df_cy = pd.concat(results, axis=1)
        self.logger.info("Average labor shares calculation completed")
        return df_cy

    def avg_labor_counts(self):
        """
        Calculate labor counts with weighted average and return as DataFrame.

        Returns:
        DataFrame: DataFrame with labor counts.
        """
        self.logger.info("Calculating average labor counts")
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
        self.logger.info("Average labor counts calculation completed")
        return df_cy

    def concat_avg(self):
        """
        Concatenate results from avg_log_wages, avg_labor_shares, and avg_labor_counts.

        Returns:
        DataFrame: Concatenated DataFrame with all averages.
        """
        self.logger.info("Concatenating all averages")
        df_cy = pd.concat(
            [self.avg_log_wages(), self.avg_labor_shares(), self.avg_labor_counts()],
            axis=1,
        )
        self.logger.info("Concatenation completed")
        return df_cy

    def change_10_years(self):
        """
        Calculate decadal changes for each zone.

        Returns:
        DataFrame: Decadal differences in specified metrics.
        """
        self.logger.info("Calculating 10-year changes")
        df_cy = self.concat_avg()
        cols = df_cy.columns.to_list()
        df_cy = df_cy.reset_index().pivot_table(index="czone", columns="year")

        for c in cols:
            df_cy["D{}".format(c), 1990] = df_cy[c, 2000] - df_cy[c, 1990]
            df_cy["D{}".format(c), 2000] = (df_cy[c, 2008] - df_cy[c, 2000]) * (10 / 7)

        df_cy = df_cy.stack().drop(columns=cols)
        self.logger.info("10-year changes calculation completed")
        return df_cy

    def rename_change_10_years(self):
        """
        Rename columns for change data according to ADH naming conventions.

        Returns:
        DataFrame: Renamed DataFrame for decadal change analysis.
        """
        self.logger.info("Renaming 10-year change columns")
        df_cy = self.change_10_years()

        # Multiplying certain columns by 100 for easier interpretation
        for c in self.share_cols:
            for mask_list in [""] + self.AVG_MASKS:
                suffix = "_" + "_".join(mask_list) if mask_list else ""
                col_name = f"D{c}{suffix}"
                if col_name in df_cy.columns:
                    df_cy[col_name] = df_cy[col_name] * 100.0

        cols_mask = df_cy.columns.str.contains("Dln")
        for c in df_cy.columns[cols_mask]:
            df_cy[c] = df_cy[c] * 100.0

        self.ADHnames = {}
        for base, rename_base in zip(
            ["manuf_share", "nonmanuf_share", "unemp_share", "nilf_share"],
            ["d_sh_empl_mfg", "d_sh_empl_nmfg", "d_sh_unempl", "d_sh_nilf"],
        ):
            self.ADHnames[f"D{base}"] = rename_base
            for mask_list in self.AVG_MASKS:
                suffix = "_".join(mask_list)
                col_name = f"D{base}_{suffix}"
                rename_col = f"{rename_base}_{suffix.replace('native', 'native').replace('col', 'edu_c').replace('ncol', 'edu_nc')}"
                self.ADHnames[col_name] = rename_col

        for mask_list in [""] + self.AVG_MASKS:
            suffix = "_" + "_".join(mask_list) if mask_list else ""
            col_name = f"Dlnwkwage{suffix}"
            rename_suffix = "_".join(
                ["d_avg_lnwkwage"]
                + [
                    (
                        "c"
                        if mask == "col"
                        else (
                            "nc"
                            if mask == "ncol"
                            else (
                                "m"
                                if mask == "male"
                                else "f" if mask == "female" else "native"
                            )
                        )
                    )
                    for mask in mask_list
                ]
            )
            self.ADHnames[col_name] = rename_suffix

        for col, rename_col in zip(
            ["lnmanuf", "lnnonmanuf", "lnunemp", "lnnilf"],
            [
                "lnchg_no_empl_mfg",
                "lnchg_no_empl_nmfg",
                "lnchg_no_unempl",
                "lnchg_no_nilf",
            ],
        ):
            self.ADHnames[f"D{col}"] = rename_col
            for mask_list in self.AVG_MASKS:
                suffix = "_".join(mask_list)
                col_name = f"D{col}_{suffix}"
                rename_suffix = f"{rename_col}_{suffix.replace('native', 'native').replace('col', 'edu_c').replace('ncol', 'edu_nc')}"
                self.ADHnames[col_name] = rename_suffix

        df_cy.rename(columns=self.ADHnames, inplace=True)
        self.logger.info("Renaming completed")
        return df_cy

    def merge_with_dorn_data(self):
        """
        Merge the computed data with Dorn data for further analysis.

        Returns:
        DataFrame: Merged dataset.
        """
        self.logger.info("Merging with Dorn data")
        df_cy = self.rename_change_10_years()
        df_NCA = self.CENSUS_PIPELINE.load_data("controls")

        CA_cols = [v for k, v in self.ADHnames.items()]
        other_cols = df_NCA.columns.difference(CA_cols)

        df_CA = pd.merge(
            df_cy,
            df_NCA[other_cols],
            left_on=["czone", "year"],
            right_on=["czone", "yr"],
            how="inner",
        )
        self.logger.info("Merging with Dorn data completed")
        return df_CA

    def run(self):
        """
        Run the full pipeline to process and analyze outcomes data.

        Returns:
        DataFrame: Final processed and merged dataset.
        """
        self.logger.info("Running the full OutcomesDataPipeline")
        result = self.merge_with_dorn_data()
        self.logger.info("OutcomesDataPipeline run completed")
        return result


def main():

    DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")

    OutcomesDataPipeline(CA=True).run().to_csv(
        os.path.join(DATA_DIR, "outcomes_CA.csv")
    )

    OutcomesDataPipeline(CA=False).run().to_csv(os.path.join(DATA_DIR, "outcomes.csv"))


if __name__ == "__main__":
    main()
