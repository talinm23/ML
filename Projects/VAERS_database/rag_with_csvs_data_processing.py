


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
from collections import Counter
import pandas as pd
import pandas as pd
from collections import Counter
import openai

pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)

path1 = r"T:\projects_data\VAERS_data\combined_csvs\VAERSDATA_1990_2025_COMBINED.csv"
adverse = pd.read_csv(path1,encoding='latin1') #'utf8'
path2 = r"T:\projects_data\VAERS_data\combined_csvs\VAERSSYMPTOMS_1990_2025_COMBINED.csv"
symptoms = pd.read_csv(path2,encoding='latin1')
path3 = r"T:\projects_data\VAERS_data\combined_csvs\VAERSVAX_1990_2025_COMBINED.csv"
vax = pd.read_csv(path3,encoding='latin1')

#----------------------------------------------------------------------------------------------------------
# Looking at ages < 1:
#----------------------------------------------------------------------------------------------------------
adverse_1 = adverse[adverse.AGE_YRS<1]

#----------------------------------------------------------------------------------------------------------
# we take this vax information table and just take those vax types strings separated by commas into a new column in the adverse table:
# Build one row per VAERS_ID (index = VAERS_ID)
vax_map = (vax
           .groupby('VAERS_ID')['VAX_TYPE']
           .agg(', '.join) )

# Add it to the adverse DataFrame, aligned by ID:
adverse_1 = adverse_1.copy()
adverse_1['VAX_TYPE_LIST'] = adverse_1['VAERS_ID'].map(vax_map)
adverse_1.loc[:, 'VAX_TYPE_LIST'] = adverse_1['VAERS_ID'].map(vax_map)

duplicates0 = adverse_1[adverse_1.duplicated('VAERS_ID')]
#There are some duplicates. We can just pick the one with the latest "order" of the vaccine.

symptom_cols = ['SYMPTOM1', 'SYMPTOM2', 'SYMPTOM3', 'SYMPTOM4', 'SYMPTOM5']
sym_long = (symptoms
            .melt(id_vars='VAERS_ID',
                  value_vars=symptom_cols,
                  value_name='SYMPTOM')  # new col with symptom string
            .dropna(subset=['SYMPTOM']))
symptoms_map = (sym_long
                .groupby('VAERS_ID')['SYMPTOM']
                .agg(', '.join))
adverse_1['SYMPTOM_LIST'] = adverse_1['VAERS_ID'].map(symptoms_map)

dupe_ids = duplicates0['VAERS_ID'].unique()
is_dupe_row = adverse_1['VAERS_ID'].isin(dupe_ids)
best_idx = (adverse_1[is_dupe_row]
            .groupby('VAERS_ID')['ORDER']
            .idxmax())

adverse_1_dedup = pd.concat([
        adverse_1.loc[best_idx],      # keep one “best” row per duplicate ID
        adverse_1[~is_dupe_row]       # keep rows that were unique
    ])#.sort_index()

#check that there are no dupes:
adverse_1_dedup[adverse_1_dedup.duplicated('VAERS_ID')]
adverse_1_dedup = adverse_1_dedup[['YEAR', 'VAERS_ID', 'RECVDATE', 'STATE', 'AGE_YRS', 'SEX', 'SYMPTOM_TEXT', 'DIED', 'DATEDIED',
       'VAX_DATE', 'ONSET_DATE', 'NUMDAYS', 'ORDER', 'VAX_TYPE_LIST', 'SYMPTOM_LIST']]
print(len(adverse_1_dedup))
print('Dataframe is ready to be used.')

adverse_1_dedup['VAX_DATE_d'] = pd.to_datetime(adverse_1_dedup['VAX_DATE'])
adverse_1_dedup['ONSET_DATE_d'] = pd.to_datetime(adverse_1_dedup['ONSET_DATE'])
adverse_1_dedup['DURATION'] = adverse_1_dedup['ONSET_DATE_d'] - adverse_1_dedup['VAX_DATE_d']
adverse_1_dedup ['DELTA_DAYS'] = adverse_1_dedup['DURATION'] .dt.days
adverse_1_dedup = adverse_1_dedup[
    (adverse_1_dedup['DELTA_DAYS'] >= 0) &
    (adverse_1_dedup['DELTA_DAYS'] <= 1825)]

#trimming down some columns:
df_save = adverse_1_dedup[['SYMPTOM_TEXT','DIED','VAX_TYPE_LIST', 'SYMPTOM_LIST','DURATION', 'DELTA_DAYS']]
df_save.to_csv('VAERS_DATA.csv', index=False)


print('done')





