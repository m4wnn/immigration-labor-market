# %%
import pickle
import os
import pandas as pd
import numpy as np
import linearmodels as lm
from rich import inspect
from statsmodels.iolib.summary2 import Summary
from statsmodels.iolib.table import SimpleTable
from functools import reduce
from toolz import pipe

from typing import Any, Union, Optional, List, Dict, Tuple, Callable, Set, Iterable
from numpy.typing import NDArray
from src.data_processing import DataPipeline

# %%
data_pipeline = DataPipeline(data_path="data")
data = data_pipeline.load_data()
# %%
with open(os.path.join(data_pipeline.PATHS.DATA_PATH, "iv_results.pkl"), "rb") as f:
    iv_results = pickle.load(f)


# %%
def get_first_stage_results(
    iv: lm.iv.results.IVResults,
) -> lm.iv.results.FirstStageResults:
    return iv.first_stage


def get_simple_table(first_stage: lm.iv.results.FirstStageResults) -> SimpleTable:
    return first_stage.summary.tables[0]


class FirstStageResTab:
    fields: List[str]
    values: List[str]

    def __init__(self, fields: List[str], values: List[str]):
        self.fields = fields
        self.values = values

    @classmethod
    def from_SimpleTable(cls, table: SimpleTable) -> "FirstStageResTab":
        fields = [row[0] if row[0] != "" else np.nan for row in table.data]

        for i, row in enumerate(table.data):
            if i == 0:
                fields[i] = "endog"
            elif row[0] == "":
                fields[i] = fields[i - 1] + "_SE"

        values = [row[1] for row in table.data]
        return cls(fields, values)

    def __str__(self):
        return pd.DataFrame({"field": self.fields, "values": self.values}).to_string()


def find_index_of_x_in_y(x: Iterable[str], y: Iterable[str]) -> List[Tuple[int, int]]:
    x_list = list(x)  # Convertimos x a lista para poder usar .index()
    return [(i, x_list.index(elem)) for i, elem in enumerate(y) if elem in x_list]


def concat_FirstStageResTab(
    tables: Union[List[FirstStageResTab], NDArray[FirstStageResTab]]
) -> pd.DataFrame:
    ref_tab = tables[np.argmax([len(tab.fields) for tab in tables])]

    concat_tab = pd.DataFrame({"stats": ref_tab.fields})

    for i, tab in enumerate(tables):
        tmp_index = find_index_of_x_in_y(tab.fields, ref_tab.fields)
        tmp_index_ref_tab = [j[0] for j in tmp_index]
        tmp_index_tab = [j[1] for j in tmp_index]
        concat_tab[f"ex_{i}"] = np.repeat("-", len(concat_tab))
        concat_tab.loc[tmp_index_ref_tab, [f"ex_{i}"]] = [
            tab.values[j] for j in tmp_index_tab
        ]

    return concat_tab


# %%
concat_FirstStageResTab(
    iv_results.query("z == 'z_1' and y == 'Dln_wage'")
    .iv.apply(
        lambda iv: pipe(
            iv,
            get_first_stage_results,
            get_simple_table,
            FirstStageResTab.from_SimpleTable,
        )
    )
    .values
)
# %%
concat_FirstStageResTab(
    iv_results.query("z == 'z_2' and y == 'Dln_wage'")
    .iv.apply(
        lambda iv: pipe(
            iv,
            get_first_stage_results,
            get_simple_table,
            FirstStageResTab.from_SimpleTable,
        )
    )
    .values
)
