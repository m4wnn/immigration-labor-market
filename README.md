# Immigration and Labor-Market Outcomes

What is the impact of large inflows of immigrants on inequality across regions in the United States?

## Data Sources

- DATA I: 1980 Census and "2007" data from the American Community Survey (ACS) (i.e. 2006-2008 3 year ACS) [https://usa.ipums.org/usa/](https://usa.ipums.org/usa/)
- DATA II: Comuting zones (CZ) from [https://ddorn.net/data.htm](https://ddorn.net/data.htm)

### Data Transformation

#### Step I

By CZ $`c`$ and year $`y`$, native average wages, native unemployment and labor force participation rates are constructed.

#### Step II

The immigration inflow is defined and calculated as:

```math
x_c = \dfrac{1}{N_{c,1980}}\left(I_{c,2007} - I_{c,1980}\right)
```
where
- $`N_{c,1980}`$ is the total population of $`c`$ in 1980.
- $`I_{c,y}`$ is the population of immigrant in $`c`$ in year $`y`$.

#### Step III-a

The first instrument is constructed as:

```math
z_c^{(1)} = \dfrac{1}{N_{c, 1980}} \sum_{s} f_{c, 1980}^{s} \left(I_{2007}^s - I_{1980}^s\right)
```
where
- $`I_{y}^s`$ is the number of immigrants from source region $`s`$ in US in year $`y`$.
- $`f_{c,1980}^{s}` = \dfrac{I_{c, 1980}^s}{I_{1980}^s}`$ is the share of immigrants from source $` s`$ who are in $`c`$ in year 1980.

#### Step III-b

The second instrument is constructed as:

```math
z_{c}^{(2)} = \dfrac{1}{I_{c, 1980}} \sum_{s} f_{c, 1980}^{s} \left(I_{2007}^s - I_{1980}^s\right)
```
#### Step IV

Using 2SLS, project changes in CZ outcomes (percentage point for unemployment and labor force participation rates; percent or log for wages) on immigrant inflow.
- Instrument for $`x_c`$, with either $`z_c^{(1)}`$ or $`z_c^{(2)}`$, depending on which has a stronger first stage when controls are included.
- Controls are included  (like in Author Dorn, and Hanson) measured in 1980 data: key control is the share of population that is immigrant in 1980.

## Create the `conda` Environment from the `environment.yml` File

To run the code in this repository, cloning the conda environment is recommended. To do so, run the following command in the terminal:

```bash
conda env create -f environment.yaml
```

This will create a new conda environment called 'imm-mk-env'. To activate the environment, run the following command in the terminal:

```bash
conda activate imm-mk-env
```

For updating the environment from the `environment.yml` file, run the following command in the terminal:

```bash
conda env update -f environment.yml --prune
```

This action should be done if the `environment.yml` file is updated.

> [!WARNING]
> Before running the code, make sure to activate (or create) the conda environment by running the command `conda activate imm-mk-env`.
