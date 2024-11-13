# %%
from src.data_processing import DataPipeline

# %%
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)
# %%
data_pipeline = DataPipeline("data")
data = data_pipeline.load_data()
# %% Useful attributes in `data_pipeline`
# data_pipeline._shock
# data_pipeline._outcomes
# data_pipeline._controls
# data_pipeline._instruments
data.head()
