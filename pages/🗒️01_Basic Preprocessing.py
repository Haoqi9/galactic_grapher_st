import streamlit as st
import pandas as pd
import sys
from Functions.my_funcs import describe_custom

###############################################################################################################################

# PAGE CONFIGURATION
st.set_page_config(
  page_title='01_Basic Preprocessing',
  page_icon="./Images/astr.jpg",
  layout='wide',
  initial_sidebar_state='expanded'
)
  
###############################################################################################################################

# BODY:
title1, title2 = st.columns([1,4])
title1.image('./Images/camino.jpg')
title2.write('')
title2.write('')
title2.write('')
title2.write('')
title2.write('# 01_Basic Preprocessing')
st.write('___')
st.header('Upload data file')

# *********** Receive data file and convert to df ***********

data_file = st.file_uploader(
  label='',
  accept_multiple_files=False,
  type=['csv', 'xlsx', 'pkl']
)

@st.cache_data
def convert_to_df(data_file):
  if data_file.name.endswith('.csv'):
      df = pd.read_csv(data_file)
  elif data_file.name.endswith('.pkl'):
      df = pd.read_pickle(data_file)
  elif data_file.name.endswith('.xlsx'):
      df = pd.read_excel(data_file)
  else:
    raise Exception('This APP currently accepts only one of these file extensions: [csv, pkl, xlsx]')
  return df

if data_file is None:
  st.info('Please upload a data file to begin with!')
  st.info('- Allowed extensions so far: csv, xlsx, pkl.')
  st.stop()
else:
  df = convert_to_df(data_file)
  st.success(f'**{data_file.name}** has been successfully uploaded!')

###############################################################################################################################

# SIDEBAR WIDGETS:
with st.sidebar:
  st.write('___')
  st.write('> `Switch between pages above 👆`')
  st.markdown(
    """
    # Table of Contents
    1. [Upload data file](#upload-data-file)
    1. [DataFrame of the data file](#dataframe-of-the-data-file)
    1. [Separation of feature data types](#separation-of-feature-data-types)
    1. [Duplicates](#duplicates)
    1. [Missing Values](#missing-values)
    1. [Preprocessed DataFrame](#preprocessed-dataframe)
    1. [Basic Descriptive statistics](#basic-descriptive-statistics)
    1. [Download preprocessed data](#download-preprocessed-data)
    """)
  
  # Sidebar annotations:
  st.write('___')
  st.write('## ANNOTATIONS')
  
  with st.form('annotations'):
    annotations = st.text_area(
      label='Annotations on visualizations',
      height=20,
      placeholder='Text to save as text file...',
      help="Please note that in order to save annotations as a text file, you must **first SAVE them**. Once saved, script is rerun!"
    )
    
    submit_annotations = st.form_submit_button('SAVE')
  
  if submit_annotations is True:
    st.success('Annotations have been saved!')
  # Option to donwload text file.
    download_annotations = st.download_button(
      label='💾 DOWNLOAD annotations as txt file',
      data=annotations,
      file_name = f"{data_file.name.split('.')[0]}_prepro.txt"
    )

    if download_annotations is True:
      st.success(f"**{data_file.name.split('.')[0]}_prepro.txt** has been successfully downloaded")

# *********** Dataframe of datafile ***********

st.write('___')
st.header('DataFrame of the data file')

# Add df shape info.
st.caption(f'- The following DataFrame contains **{df.shape[0]} rows** and **{df.shape[1]} columns**.')

# Show df.
st.dataframe(df)

# *********** Identify categorical and numerical columns ***********

st.write('___')
st.header('Separation of feature data types')
st.write(
  """
  - Separation of numerical and categorical features in this page *only matters for the imputation of missing values* for the section [Missing Values](#missing-values).
  - Here are the **default separation (automatic)** of feature types for all features:
  """
  )

cat_list = []
num_list = []

for col in df.columns:
    if pd.api.types.is_numeric_dtype(df[col]) and (df[col].nunique() > 2):
      num_list.append(col)
    else:
      cat_list.append(col)

num_list_default = sorted(num_list)
cat_list_default = sorted(cat_list)

list1, list2 = st.columns(2)
with list1:
  st.write(f'**Numerical features**:')
  placeholder_num = st.empty()
  placeholder_num.write(num_list_default)
with list2:
  st.write(f'**Categorical features**:')
  placeholder_cat = st.empty()
  placeholder_cat.write(cat_list_default)

# *********** Add and handle duplicates info ***********

st.write('___')
st.header('Duplicates')
n_duplicates = df.duplicated().sum()
st.caption(f'- Number of duplicated rows: **{n_duplicates}**.')

# Set drop duplicate flag.
drop_duplicate = False

# If any duplicate entry:
if n_duplicates > 0:
  st.caption('- DataFrame with duplicated entries:')
  st.dataframe(df.loc[df.duplicated()])
  
  placeholder_drop = st.empty()
  if placeholder_drop.checkbox(label='Drop duplicated rows') is True:
    # Flag changes.
    drop_duplicate = True
    df.drop_duplicates(inplace=True)
    placeholder_drop.empty()
    st.success('Duplicated rows have been deleted!')

# *********** Add and handle missing info ***********

st.write('___')
st.header("Missing Values")
n_missing_rows = df.apply(lambda x: x.isna().sum(), axis=1).gt(0).sum()
st.caption(f'- Number of rows with missing values (at least 1 feature missing): **{n_missing_rows}**.')
st.caption('- Percentage of missing values for each feature:')
df_missing_pct = pd.DataFrame([df[col].isna().mul(100).mean().round(2) for col in df.columns], index=df.columns, columns=['Missing (%)']).T
st.dataframe(df_missing_pct)
# Set impute missing flag.
missing_handling = None

# If any missing:
if n_missing_rows > 0:
  st.caption('- DataFrame with missing entries:')
  st.dataframe(df.loc[df.apply(lambda x: x.isna().sum(), axis=1).gt(0)])
  
  help_impute = """ For `Impute missing values`, it applies univariate imputation method: missing values in categorical features are replaced by the **mode** and **median** for numerical ones. Afterwards, the imputed numerical and categorical dfs are concatenated into a single DataFrame. Note that *identification of numerical and categorical features are based on the default or automatic version of the lists*."""
  with st.form('missing'):
    # Define placeholders for two ways of handling missing data.
    choose_one = st.radio(
      label='Choose one of the 2 options for handling missing values:',
      help=help_impute,
      options=['Drop missing values', 'Impute missing values']
    )
    
    submit_missing = st.form_submit_button('Submit choice')
  
  if submit_missing is True:
    if choose_one == 'Impute missing values':
      # Flag changes
      missing_handling = 'Imputed'
      df_num_imputed = df[num_list].apply(lambda x: x.fillna(x.median()))
      df_cat_imputed = df[cat_list].apply(lambda x: x.fillna(x.mode()[0]))
      # Concatenate to single df.
      df_imputed = pd.concat([df_num_imputed, df_cat_imputed], axis=1)
      num_missing_list = df[num_list].columns[df[num_list].apply(lambda x: x.isna().any())].to_list()
      cat_missing_list = df[cat_list].columns[df[cat_list].apply(lambda x: x.isna().any())].to_list()
      # Change variable name to df (standard) as the dataset might have only duplicates, missing or none!
      df = df_imputed
      st.success('Missing values have been imputed!')
      st.info(f'Mode imputation for categorical features with NaN: {cat_missing_list}')
      st.info(f'Median imputation for numerical features with NaN: {num_missing_list}')
      
    elif choose_one == 'Drop missing values':
      # Flag changes
      missing_handling = 'Dropped'
      df_imputed = df.dropna()
      df = df_imputed
      st.success(f'**{n_missing_rows} missing rows** have been deleted from df!')
    
# *********** Show preprocessed df ***********

st.markdown('___')
st.header('Preprocessed DataFrame')
st.caption(f'- The following DataFrame contains **{df.shape[0]} rows** and **{df.shape[1]} columns**.')
n_missing_rows_imp = df.apply(lambda x: x.isna().sum(), axis=1).gt(0).sum()
st.caption(f'- Number of rows after dropping duplicates **{df.shape[0]}**.')
st.caption(f'- Number of rows with missing values after imputation (at least 1 feature missing): **{n_missing_rows_imp}**.')
st.dataframe(df)

# *********** Show general descriptive analysis ***********

# Num features
st.markdown('___')
st.header('Basic Descriptive statistics')
st.write('- General descriptive analysis for **numerical features**:')
st.table(describe_custom(df[num_list]))
st.write('- General descriptive analysis for **categorical features**:')

# Cat features
n_cols_per_row = 5
cols_list = [f'col{i}' for i in range(1, len(cat_list) + 1)]
for i in range(0, len(cat_list) + 1, n_cols_per_row):
  cols_list_each = cols_list[i:i+n_cols_per_row]
  cat_list_each  = cat_list[i:i+n_cols_per_row]
  cols_list_each = st.columns(n_cols_per_row)
  for col, cat in zip(cols_list_each, cat_list_each):
    col.caption(f'**{cat}** (n = {df[cat].shape[0]})')
    n_unique_cats = df[cat].nunique()
    if n_unique_cats > 20:
      col.write(f'**{df[cat].name}** has too many categories **{n_unique_cats}** > 20')
    else:
      col.table(df[cat].value_counts(dropna=False, normalize=True))

# *********** Download preprocessed df as a csv file ***********

st.write('___')
st.header('Download preprocessed data')
@st.cache_data
def convert_df_to_csv(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode('utf-8')
  
csv_prepro = convert_df_to_csv(df)

download_prepro = st.download_button(
    label='💾 DOWNLOAD preprocessed data as csv file',
    data=csv_prepro,
    file_name=f"{data_file.name.split('.')[0]}_prepro.csv",
    mime='text/csv',
)
st.info('Changes to uploaded data:')
st.info(f'- Duplicates dropped: **{drop_duplicate}**')
st.info(f'- Missing values imputed: **{missing_handling}**')

if download_prepro is True:
  st.success(f"**{data_file.name.split('.')[0]}_prepro.csv** has been successfully downloaded!")
  st.balloons()