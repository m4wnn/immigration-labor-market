# %%
from src.data_processing import DataPipeline
import linearmodels

import warnings
import statsmodels.api as sm
from itertools import product
import pandas as pd

warnings.simplefilter(action="ignore", category=FutureWarning)


# %%
data_pipeline = DataPipeline("data")
data = data_pipeline.load_data(add_const=True)
# %%
grid = {
    "dependent": ["Dln_wage", "Dunemp_rate", "Dnilf_rate"],
    "instrument": ["z_1", "z_2"],
    "cov_type": ["robust", "kernel", "unadjusted"],
}
# %%
results = {"y": [], "z": [], "cov_type": [], "iv": []}
for y, z, cov_type in product(*grid.values()):
    results["y"].append(y)
    results["z"].append(z)
    results["cov_type"].append(cov_type)
    results["iv"].append(
        linearmodels.IV2SLS(
            dependent=data_pipeline._outcomes[y],
            endog=data_pipeline._shock,
            exog=sm.add_constant(data_pipeline._controls),
            instruments=data_pipeline._instruments[z],
        ).fit(cov_type=cov_type)
    )
results = pd.DataFrame(results)

results["partial_f_stat"] = [
    fs.first_stage.diagnostics["f.stat"]["x"] for fs in results.iv
]
results["weak_instrument"] = [True if w < 10 else False for w in results.partial_f_stat]
# %%
results
