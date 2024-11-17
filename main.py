# %%
import pandas as pd
import linearmodels

from rich import inspect
from itertools import product
import pickle

from src.data_processing import DataPipeline

import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)
# %%
data_pipeline = DataPipeline("data")
data = data_pipeline.load_data(add_const=True)
# %%
controls_subsets = {
    "const": ["const"],
    "key": ["const", "fborn_share_80"],
    "all": data_pipeline._controls.columns,
}
# %%
grid = {
    "dependent": ["Dln_wage", "Dunemp_rate", "Dnilf_rate"],
    "controls": controls_subsets.keys(),
    "instrument": ["z_1", "z_2"],
}

results = {"y": [], "controls": [], "z": [], "iv": []}

for y, controls, z in product(*grid.values()):
    results["y"].append(y)
    results["controls"].append(controls)
    results["z"].append(z)
    results["iv"].append(
        linearmodels.IV2SLS(
            dependent=data_pipeline._outcomes[y],
            endog=data_pipeline._shock,
            exog=data_pipeline._controls[controls_subsets[controls]],
            instruments=data_pipeline._instruments[z],
        ).fit(cov_type="clustered", clusters=data.statefip)
    )

results = pd.DataFrame(results)

results["partial_f_stat"] = [
    fs.first_stage.diagnostics["f.stat"]["x"] for fs in results.iv
]

results["partial_f_pval"] = [
    fs.first_stage.diagnostics["f.pval"]["x"] for fs in results.iv
]

results["weak_instrument_pval"] = [
    True if w > 0.005 else False for w in results.partial_f_pval
]

results["weak_instrument_approx"] = [
    True if w < 10 else False for w in results.partial_f_stat
]

results["pval_and_aprox"] = [
    a and b
    for (a, b) in zip(results.weak_instrument_pval, results.weak_instrument_approx)
]

# %%

with open("data/iv_results.pkl", "wb") as file:
    pickle.dump(results, file)
