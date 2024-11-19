# %%
import pandas as pd
import linearmodels

from statsmodels.api import add_constant
from rich import inspect
from itertools import product
from toolz import pipe

import pickle
import os

import src.concat_simple_tables as cst

import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)
# %%
data = add_constant(pd.read_csv(os.path.join("data", "data.csv")))
# %%
_instruments = ["z_1", "z_2"]
_shock = ["x"]
_controls = [
    "const",
    "fborn_share_80",
    "manuf_share_80",
    "female_share_80",
    "col_share_80",
    "lnpop_80",
]
_dependent = ["Dln_wage", "Dunemp_rate", "Dnilf_rate"]
# %%
controls_subsets = {
    "const": ["const"],
    "key": ["const", "fborn_share_80"],
    "all": _controls,
}
# %%
grid = {
    "dependent": ["Dln_wage", "Dunemp_rate", "Dnilf_rate"],
    "controls": controls_subsets.keys(),
    "instrument": ["z_1", "z_2"],
}
iv_results = {"y": [], "controls": [], "z": [], "iv": []}

for y, controls, z in product(*grid.values()):
    iv_results["y"].append(y)
    iv_results["controls"].append(controls)
    iv_results["z"].append(z)
    iv_results["iv"].append(
        linearmodels.IV2SLS(
            dependent=data[y],
            endog=data[_shock],
            exog=data[controls_subsets[controls]],
            instruments=data[z],
        ).fit(cov_type="clustered", clusters=data.statefip)
    )

iv_results = pd.DataFrame(iv_results)

iv_results["partial_f_stat"] = [
    fs.first_stage.diagnostics["f.stat"]["x"] for fs in iv_results.iv
]

iv_results["partial_f_pval"] = [
    fs.first_stage.diagnostics["f.pval"]["x"] for fs in iv_results.iv
]

iv_results["weak_instrument_pval"] = [
    True if w > 0.005 else False for w in iv_results.partial_f_pval
]

iv_results["weak_instrument_approx"] = [
    True if w < 10 else False for w in iv_results.partial_f_stat
]

iv_results["pval_and_aprox"] = [
    a and b
    for (a, b) in zip(
        iv_results.weak_instrument_pval, iv_results.weak_instrument_approx
    )
]
# %%
with open(os.path.join("data", "iv_results.pkl"), "wb") as file:
    pickle.dump(iv_results, file)

# %%
cst.concat_FirstStageResTab(
    iv_results.query("z == 'z_1' and y == 'Dln_wage'")
    .iv.apply(
        lambda iv: pipe(
            iv,
            cst.get_first_stage_results,
            cst.get_simple_table,
            cst.FirstStageResTab.from_SimpleTable,
        )
    )
    .values
)
# %%
cst.concat_FirstStageResTab(
    iv_results.query("z == 'z_2' and y == 'Dln_wage'")
    .iv.apply(
        lambda iv: pipe(
            iv,
            cst.get_first_stage_results,
            cst.get_simple_table,
            cst.FirstStageResTab.from_SimpleTable,
        )
    )
    .values
)
# %%
tmp = iv_results.query(
    "z == 'z_2' and y == 'Dln_wage' and controls == 'const'"
).iv.values[0]
print(tmp.summary)
