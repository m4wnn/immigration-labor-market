# %%
from src.data_processing import CensusDataPipeline, OutcomesDataPipeline

# %%
census = CensusDataPipeline()
data = census.cz_merge().reset_index().set_index(["czone", "year", "groups"])
# %%
N_c_1990 = data.query("year == 1990").groupby("czone")["perwt"].sum()
N_c_1990
# %%
I_c_1990 = data.query("year == 1990 and native == 0").groupby("czone").sum()["perwt"]
I_c_1990
# %%
I_c_2008 = data.query("year == 2008 and native == 0").groupby("czone").sum()["perwt"]
I_c_2008
# %%
I_c_1990_s = (
    data.query("year == 1990 and native == 0").groupby(["czone", "bpl"]).sum()["perwt"]
)
I_c_1990_s
# %%
I_c_2008_s = (
    data.query("year == 2008 and native == 0").groupby(["czone", "bpl"]).sum()["perwt"]
)
I_c_2008_s
# %%
I_1990_s = data.query("year == 1990 and native == 0").groupby(["bpl"]).sum()["perwt"]
I_1990_s
# %%
I_2008_s = data.query("year == 2008 and native == 0").groupby(["bpl"]).sum()["perwt"]
I_2008_s
# %%
f_c_1990_s = I_c_1990_s / I_1990_s
f_c_1990_s
# %%
x_c = (1 / N_c_1990) * (I_c_2008 - I_c_1990)
x_c = x_c.rename("x_c").reset_index()
x_c
# %%
z_c_1 = (f_c_1990_s * (I_2008_s - I_1990_s)).groupby("czone").sum() * (1 / N_c_1990)
z_c_1 = z_c_1.rename("z_c_1").reset_index()
z_c_1
# %%
z_c_2 = (f_c_1990_s * (I_2008_s - I_1990_s)).groupby("czone").sum() * (1 / I_c_1990)
z_c_2 = z_c_2.rename("z_c_2").reset_index()
z_c_2
# %%
outcomes = OutcomesDataPipeline()
outcomes_data = outcomes.run()
# %%
complete_data = (
    outcomes_data.merge(x_c, on="czone", how="inner")
    .merge(z_c_1, on="czone", how="inner")
    .merge(z_c_2, on="czone", how="inner")
)
# %%
complete_data.to_csv("data/complete_data.csv")
