# %% [markdown]
# # Assignment 2: Impact of immigration on local labor-market outcomes
# #### By: Augusto Ospital. First version: May 3, 2022. This version: November 5th, 2023 (Mallika Chandra)

# %% [markdown]
# Same data as Assignment 1, but only 1980 census and 2008 3-year ACS
# 
# __Step I__. By CZ $c$ and year $y$: Construct native average wages, native unemployment
# and labor force participation rates, share of native employment in manufacturing.
# 
# __Step II__. Construct immigrant inflow as a fraction of initial population: $$ x_c = \frac{1}{N_{c,1980}} \left( I_{c,2007} - I_{c,1980}\right)$$
# where $N_{c,1980} = $ population of $c$ in 1980, and $I_{c,1980} = $ population of immigrants in $c$ in 1980.
# 
# __Step III__. Construct “Card instrument:" $$ z_c = \frac{1}{N_{c,1980}} \sum_s f_{cs,1980} \left( I_{s,2007} - I_{s,1980}\right)$$
# where $I_{s,year} = $ number of immigrants from source region $s$ in US in $year$, and $f_{cs,1980} = I_{sc,year}\big/I_{s,year} = $ fraction of immigrants from $s$ who are in $c$ in 1980.
# 
# __Step IV__. Using 2SLS, project changes in CZ outcomes (percentage point for
# unemployment, LFP, and manufacturing share; percent for wage) on
# immigrant inflow
# - instrument for immigrant inflow, $x_c$ , with Card instrument, $z_c$
# - include controls (like in Autor, Dorn, and Hanson) measured in 1980

# %% [markdown]
# ## Code Preliminaries

# %%
from pathlib import Path
import pandas as pd
from econtools import group_id
import numpy as np
import os

# %%
# For regressions
import statsmodels.api as sm
from stargazer.stargazer import Stargazer #nice tables with statsmodels
from linearmodels.iv import IV2SLS, compare #2sls with clustered SEs
import matplotlib.pyplot as plt
%matplotlib inline
plt.rcParams['font.size'] = 11
plt.rcParams['font.family'] = 'serif'

# %%
mainp = os.path.dirname(os.path.abspath(__file__))

# %% [markdown]
# ## Prepare the Data

# %% [markdown]
# #### Load the data from IPUMS

# %%
# Take a look at column definitions:
column_definition = pd.read_stata(
    os.path.join(mainp, 'data', 'usa_00121.dta'),
    iterator=True
).variable_labels()
column_definition

# %%
df = pd.read_stata(
    os.path.join(mainp, 'data', 'usa_00121.dta'),
    convert_categoricals=False
)
# Keep those aged 20-60 and not in group quarters:
df = df[(df.age>=20) & (df.age<=60) & (df.gq<=2)].copy()
#Katrina data issue:
df.loc[(df.statefip==22)&(df.puma==77777),'puma'] = 1801 
df
# %%
# Summary statistics
def human_format(num):
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    # add more suffixes if you need them
    return '%.2f%s' % (num, ['', 'K', 'Million', 'Trillion', 'G', 'P'][magnitude])

df.describe().applymap(human_format).T

# %% [markdown]
# #### Merge geographies to CZs

# %%
df2008 = df[df.year==2008].copy()
# 2008 PUMAs:
df2008['PUMA'] = df2008['statefip'].astype(str).str.zfill(2) + \
                 df2008['puma'].astype('int').astype(str).str.zfill(4)
df2008['PUMA'] = df2008['PUMA'].astype('int')
# Merge to CZs:
df2008 = pd.merge(
    df2008,
    pd.read_stata(
        os.path.join(mainp, 'data', 'cw_puma2000_czone.dta'),
    ),
    left_on='PUMA', right_on='puma2000'
)
# %%
df1980 = df[df.year==1980].copy()
# 1980 county groups:
df1980['ctygrp1980'] = (
    df1980['statefip'].astype(str).str.zfill(2) +
    df1980['cntygp98'].astype(int).astype(str).str.zfill(3)
)
df1980['ctygrp1980'] = pd.to_numeric(df1980['ctygrp1980'])
# Merge:
df1980 = pd.merge(
    df1980,
    pd.read_stata(
        os.path.join(mainp, 'data', 'cw_ctygrp1980_czone_corr.dta'),
    ),
    on='ctygrp1980'
)
# %%
df = pd.concat([df2008, df1980])
# Create new individual weights at the CZ level:
df['weight'] = df['perwt'] * df['afactor']
df.drop(columns=['perwt','afactor'])

df

# %% [markdown]
# #### Construct elements of the numerator of the key shock and instrument
# I.e. $I_{c,2007}$, $I_{c,1980}$, $f_{cs,1980}$, $I_{s,2007}$, and $I_{s,1980}$

# %%
imm = df.copy()

# Drop US birth and n.e.c. and missing (etc.):
imm = imm[(imm.bpl>120) & (imm.bpl<900)]

# Create aggregate regions given inconsistencies in codes across time:
imm.loc[(imm.bpl>=150)&(imm.bpl<200),'nativity'] = 150
imm.loc[imm.bpl==200,'nativity'] = 200
imm.loc[imm.bpl==210,'nativity'] = 210
imm.loc[imm.bpl==250,'nativity'] = 250
imm.loc[imm.bpl==260,'nativity'] = 260
imm.loc[imm.bpl==300,'nativity'] = 300
imm.loc[(imm.bpl>=400)&(imm.bpl<410),'nativity'] = 400
imm.loc[(imm.bpl>=410)&(imm.bpl<415),'nativity'] = 410
imm.loc[(imm.bpl>=420)&(imm.bpl<=429),'nativity'] = 420
imm.loc[(imm.bpl>=430)&(imm.bpl<=440),'nativity'] = 430
imm.loc[(imm.bpl>=450)&(imm.bpl<=459),'nativity'] = 450
imm.loc[(imm.bpl>=460)&(imm.bpl<=465),'nativity'] = 460
imm.loc[imm.bpl==499,'nativity'] = 499
imm.loc[imm.bpl==500,'nativity'] = 500
imm.loc[imm.bpl==501,'nativity'] = 501
imm.loc[imm.bpl==502,'nativity'] = 502
imm.loc[imm.bpl==509,'nativity'] = 509
imm.loc[(imm.bpl>=510)&(imm.bpl<=519),'nativity'] = 510
imm.loc[(imm.bpl>=520)&(imm.bpl<=529),'nativity'] = 520
imm.loc[(imm.bpl>=530)&(imm.bpl<=550),'nativity'] = 530
imm.loc[imm.bpl==599,'nativity'] = 599
imm.loc[imm.bpl==600,'nativity'] = 600
imm.loc[(imm.bpl>=700)&(imm.bpl<=710),'nativity'] = 700

imm.dropna(subset=['nativity'], inplace=True)
imm = group_id(imm, cols=['nativity'], merge=True, name='source')

imm = imm[['czone','source','year','weight']].copy()

# %%
# Compute immigrant counts at different levels of aggregation:
imm['weight'] = imm['weight'].astype(float) #to ensure precision in sum
imm['I_csy'] = imm.groupby(['czone','source','year'])[['weight']].transform('sum')
imm['I_cy'] = imm.groupby(['czone','year'])[['weight']].transform('sum')
imm['I_sy'] = imm.groupby(['source','year'])[['weight']].transform('sum')
imm = imm[['czone','source','year','I_cy','I_csy','I_sy']].copy()
imm.drop_duplicates(inplace=True)

# %%
# Construct the fraction of immigrants from a source who are in a CZ in 1980:
imm1980 = imm[imm.year==1980].copy()
imm1980['share_cs80'] = imm1980['I_csy'] / imm1980['I_sy']
imm1980 = imm1980.groupby(['czone','source'])[['share_cs80']].sum()
imm = pd.merge(imm, imm1980, on=['czone','source'], how='left')

# %%
imm.drop(columns=['I_csy'], inplace=True)
imm.rename(columns={'I_cy':'I_c', 'I_sy':'I_s'}, inplace=True)

# %%
# Reshape to wide format:
imm = imm.pivot_table(index=['czone','source'], columns='year')

# Fill in missing values:
for y in [1980,2008]:
    imm['I_s',y] = imm.groupby(level=['source']).transform(np.nanmax)['I_s',y]
    imm['I_c',y] = imm.groupby(level=['czone']).transform(np.nanmax)['I_c',y]
    imm.loc[imm['share_cs80',y].isnull(), ('share_cs80',y)] = 0.0    

#Compute the time differences:
for c in ['I_s','I_c']:
    imm['D{}'.format(c)] = imm[c,2008] - imm[c,1980]

# %%
# Construct numerator of shock and instrument:
imm['fDI_s'] = imm['DI_s'] * imm['share_cs80',1980]

num_c = pd.concat([imm.groupby(level=['czone'])['DI_c'].max(),
                   imm.groupby(level=['czone'])['fDI_s'].sum()
                  ], axis=1)
del imm

# %%
num_c.head()

# %% [markdown]
# #### Construct denominator of key shock and instrument + controls + outcomes

# %%
def MySum(mask, newname, col = 'weight'):
    return df[mask].groupby('czone')[[col]].sum().rename(columns={col:newname})

# %%
# Controls:
is_1980 = df.year==1980
is_manuf = (df.ind1990>=100) & (df.ind1990<400)
is_emp = (df.empstat==1)
is_fem = (df.sex==2)
is_col = (df.educ>=10)
is_fborn = (df.bpl>120) & (df.bpl<900)

df_c = pd.concat([
    MySum(is_1980, 'pop_80'),
    MySum(is_1980 & is_manuf & is_emp, 'manuf_80'),
    MySum(is_1980 & is_fem & is_emp, 'female_80'),
    MySum(is_1980 & is_emp, 'emp_80'),
    MySum(is_1980 & is_col, 'col_80'),
    MySum(is_1980 & is_fborn, 'fborn_80'),
    MySum(is_1980 & (df.bpl<900), 'fborn_denom_80')
], axis=1)

df_c['manuf_share_80'] = df_c.manuf_80/df_c.emp_80         # manufacturing share of employed
df_c['female_share_80'] = df_c.female_80/df_c.emp_80       # female share of employed
df_c['col_share_80'] = df_c.col_80/df_c.pop_80             # college share of population
df_c['lnpop_80'] = np.log(df_c.pop_80)                     # log of population (in age range)
df_c['fborn_share_80'] = df_c.fborn_80/df_c.fborn_denom_80 # foreign-born share of employed

# %%
# Outcomes for natives

# Filling in weeks worked for 2008 ACS (using midpoint):
df.loc[(df.year==2008) & (df.wkswork2==1), 'wkswork1'] = 7
df.loc[(df.year==2008) & (df.wkswork2==2), 'wkswork1'] = 20
df.loc[(df.year==2008) & (df.wkswork2==3), 'wkswork1'] = 33
df.loc[(df.year==2008) & (df.wkswork2==4), 'wkswork1'] = 43.5
df.loc[(df.year==2008) & (df.wkswork2==5), 'wkswork1'] = 48.5
df.loc[(df.year==2008) & (df.wkswork2==6), 'wkswork1'] = 51
df['hours'] = df['uhrswork'] * df['wkswork1']

df = df[df.bpl<100].copy() #excluding US OUTLYING AREAS/TERRITORIES

df['incwage'] = df['incwage'] * df['weight']
df['hours'] = df['hours'] * df['weight']

is_emp = df.empstat==1
is_unemp = df.empstat==2
is_nilf = df.empstat==3

for y in [1980,2008]:
    df_c = pd.concat([df_c, 
                      MySum((df.year==y) & is_nilf, 'nilf_num_{}'.format(y)),   
                      MySum((df.year==y) & (is_emp|is_unemp|is_nilf), 'nilf_denom_{}'.format(y)),
                      
                      MySum((df.year==y) & is_unemp, 'unemp_num_{}'.format(y)),
                      MySum((df.year==y) & (is_emp|is_unemp), 'unemp_denom_{}'.format(y)),
                      
                      MySum(df.year==y, 'inc_{}'.format(y), 'incwage'),
                      MySum(df.year==y, 'hours_{}'.format(y), 'hours')
                     ], axis=1)

    df_c['nilf_rate_{}'.format(y)] = df_c['nilf_num_{}'.format(y)] / df_c['nilf_denom_{}'.format(y)]
    df_c['unemp_rate_{}'.format(y)] = df_c['unemp_num_{}'.format(y)] / df_c['unemp_denom_{}'.format(y)]
    df_c['ln_wage_{}'.format(y)] = np.log(df_c['inc_{}'.format(y)] / df_c['hours_{}'.format(y)])
    
    df_c.drop(columns=['nilf_num_{}'.format(y), 'nilf_denom_{}'.format(y), 
                       'unemp_num_{}'.format(y), 'unemp_denom_{}'.format(y),
                       'inc_{}'.format(y), 'hours_{}'.format(y)], inplace=True)

# %%
# Construct the shock and the instrument
df_c = pd.concat([num_c, df_c], axis=1)

df_c['x'] = df_c['DI_c']/df_c['pop_80']
df_c['z'] = df_c['fDI_s']/df_c['pop_80']
df_c.drop(columns=['DI_c','fDI_s'], inplace=True)


for v in ['ln_wage','unemp_rate','nilf_rate']:
    df_c['D{}'.format(v)] = df_c['{}_2008'.format(v)] - df_c['{}_1980'.format(v)]
    df_c.drop(columns=['{}_2008'.format(v),'{}_1980'.format(v)], inplace=True)

# Merge in state associated with each czone for clustering
df_c = pd.merge(df_c, pd.read_stata(os.path.join(mainp,'data','cz_state.dta')),
                on='czone', how='left')
# Assign Alaska and Hawaii the same cluster:
df_c.loc[df_c.statefip.isnull(), 'statefip'] = 99

# %% [markdown]
# ## Now use the above to run regressions!

# %% [markdown]
# Define the controls, and have the first regression with no controls, second with key controls, and last with all controls. 
# 
# You are supposed to run all three of these regression for the outcome variables, 1) x, 2) NILF, 3) Unemployment Rate, and 4) Wage 

# %%
df_c
