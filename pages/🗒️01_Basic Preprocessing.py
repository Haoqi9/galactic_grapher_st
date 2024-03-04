import streamlit as st
import pandas as pd
import pickle
import csv
from Functions.my_funcs import describe_custom

###############################################################################################################################

# PAGE CONFIGURATION
st.set_page_config(
  page_title='01_Basic Preprocessing',
  page_icon="./Images/astr.jpg",
  layout='wide',
  initial_sidebar_state='auto'
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
    chunk = data_file.read(8000)
    dialect = csv.Sniffer().sniff(str(chunk))
    inf_delimiter = dialect.delimiter
    data_file.seek(0)
    df = pd.read_csv(data_file, sep=inf_delimiter)
    return (df, inf_delimiter)
    
  elif data_file.name.endswith('.pkl'):
    df = pd.read_pickle(data_file)
    return df
  
  elif data_file.name.endswith('.xlsx'):
    df = pd.read_excel(data_file)
    return df

if data_file is None:
  st.info('Please upload a data file to begin with!')
  st.info('- Allowed extensions so far: csv, xlsx, pkl.')
  st.stop()
else:
  # csv files
  if data_file.name.endswith('.csv'):
    df, inf_delimiter = convert_to_df(data_file)
    st.success(f"**{data_file.name}** (Inferred delimiter = '**{inf_delimiter}**') has been successfully uploaded!")
  # Other extensions
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
    1. [Choose modifications to the DataFrame](#choose-modifications-to-the-dataframe)
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

# Show original df.
df_original = df.copy()
st.dataframe(df_original)

# *********** Identify categorical and numerical columns ***********

st.write('___')
st.header('Separation of feature data types')
st.write(
  """
  - Separation of numerical and categorical features in this page *only matters for the imputation of missing values* for the section [Missing Values](#missing-values).
  - Here are the **default separation (automatic)** of feature types for all features:
  """
  )

df_extra_info = pd.DataFrame(pd.concat([df_original.dtypes, df_original.nunique()], axis=1))
df_extra_info.columns = ['dtype', 'n_unique']

cat_list = []
num_list = []

for col in df.columns:
    if pd.api.types.is_numeric_dtype(df[col]) and (df[col].nunique() > 2):
      num_list.append(col)
    else:
      cat_list.append(col)

num_list_default = sorted(num_list)
cat_list_default = sorted(cat_list)

list1, list2, list3 = st.columns(3)
with list1:
  st.dataframe(df_extra_info)
with list2:
  st.write(f'**Numerical features**:')
  placeholder_num = st.empty()
  placeholder_num.write(num_list)
with list3:
  st.write(f'**Categorical features**:')
  placeholder_cat = st.empty()
  placeholder_cat.write(cat_list)

# *********** Add and handle duplicates info ***********

st.write('___')
st.header('Duplicates')
n_duplicates = df.duplicated().sum()
st.caption(f'- Number of duplicated rows: **{n_duplicates}**.')

# If any duplicate entry:
if n_duplicates > 0:
  st.caption('- DataFrame with duplicated entries:')
  st.dataframe(df.loc[df.duplicated()])

# *********** Add and handle missing info ***********

st.write('___')
st.header("Missing Values")
n_missing_rows = df.apply(lambda x: x.isna().sum(), axis=1).gt(0).sum()
st.caption(f'- Number of rows with missing values (at least 1 feature missing): **{n_missing_rows}**.')

# If any missing:
if n_missing_rows > 0:
  st.caption('- DataFrame with missing entries:')
  st.dataframe(df.loc[df.apply(lambda x: x.isna().sum(), axis=1).gt(0)])
  st.caption('- Percentage of missing values for each feature:')
  df_missing_pct = pd.DataFrame([df[col].isna().mul(100).mean().round(2) for col in df.columns], index=df.columns, columns=['Missing (%)'])
  st.dataframe(df_missing_pct)

# *********** Choose modifications to df ***********

st.markdown('___')
st.header('Choose modifications to the DataFrame')

# Set flags.
drop_duplicate   = False
cols_drop        = False
change_dtypes    = False
missing_handling = None

# Drop cols options
drop_cols_box = st.multiselect(
  label='**Columns to drop from DataFrame**',
  options=df.columns,
  placeholder='Choose column/-s'
)

if (len(drop_cols_box) > 0):
  # Flag changes
  cols_drop = True
  
  df.drop(columns=drop_cols_box, inplace=True)
  st.success(f"**{len(drop_cols_box)} columns dropped** from the df!")
  
  # Make sure update num and cat list (some columns are not there anymore)
  num_list = [num for num in num_list if num in df.columns]
  cat_list = [cat for cat in cat_list if cat in df.columns]

help_impute = """**If there are no missing rows, neither of the two options will modify the DataFrame**. For `Impute missing values`, it applies univariate imputation method: missing values in categorical features are replaced by the **mode** and **median** for numerical ones. Afterwards, the imputed numerical and categorical dfs are concatenated into a single DataFrame. Note that *identification of numerical and categorical features are based on the default or the modified version of the lists*."""

with st.form('modifications'):
  # Duplicate options
  dup_checkbox = st.checkbox(label='**Drop duplicated rows?**')

  # Change identification of columns dtype
  num_box = st.multiselect(
    label='Choose **numerical features**:',
    options=df.columns.tolist(),
    default=num_list,
    placeholder='Choose num variables'
  )

  cat_box = st.multiselect(
    label='Choose **categorical features**:',
    options=df.columns.tolist(),
    default=cat_list,
    placeholder='Choose cat variables'
  )
  
  # Missing options
  choose_one = st.radio(
    label='**Choose one of the 2 options for handling missing values:**',
    help=help_impute,
    options=['Drop missing values', 'Impute missing values']
  )
  
  submit_changes = st.form_submit_button('SUBMIT')

if submit_changes is True:
  # Changes: duplicates.
  if dup_checkbox is True:
    # Flag changes.
    drop_duplicate = True
    
    df.drop_duplicates(inplace=True)
    st.success(f'**{n_duplicates} duplicated rows** have been deleted!')

  # Changes: variables dtype.
  num_list = num_box
  cat_list = cat_box
  
  # Check if there is changes
  if (sorted(num_list) != num_list_default) or (sorted(cat_list) != cat_list_default):
    # Flag changes
    change_dtypes = True
  
  # Changes: missing.
  if choose_one == 'Impute missing values':
    # Flag changes
    missing_handling = 'Imputed'
    
    df_num_imputed = df[num_list].apply(lambda x: x.fillna(x.median()))
    df_cat_imputed = df[cat_list].apply(lambda x: x.fillna(x.mode()[0]))
    
    # Flag first cols with changes before and after imputation.
    num_missing_list = [num for num in num_list if not df[num].equals(df_num_imputed[num])]
    cat_missing_list = [cat for cat in cat_list if not df[cat].equals(df_cat_imputed[cat])]
    
    # Concatenate to single df.
    df = pd.concat([df_num_imputed, df_cat_imputed], axis=1)
    
    st.success('**Missing values** have been imputed!')
    
  elif choose_one == 'Drop missing values':
    # Flag changes
    missing_handling = 'Dropped'
    # When dropping cols n missing values could have changed!
    n_missing_rows_drop = df.apply(lambda x: x.isna().sum(), axis=1).gt(0).sum()
    df.dropna(inplace=True)
    st.success(f'**{n_missing_rows_drop} missing rows** have been deleted!')
    
# *********** Show preprocessed df ***********

st.markdown('___')
st.header('Preprocessed DataFrame')
st.write(f'- The original DataFrame contains **{df_original.shape[0]} rows** and **{df_original.shape[1]} columns**.')
st.write(f'- The preprocessed DataFrame contains **{df.shape[0]} rows** and **{df.shape[1]} columns**.')
n_missing_rows_imp = df.apply(lambda x: x.isna().sum(), axis=1).gt(0).sum()
st.caption(f'-Number of rows with missing values after treatment (at least 1 feature missing): **{n_missing_rows_imp}**.')
st.dataframe(df)

# *********** Show general descriptive analysis ***********

# Num features
st.markdown('___')
st.header('Basic Descriptive statistics')
st.write('- General descriptive analysis for **numerical features**:')
st.dataframe(describe_custom(df[num_list]).round(2))
st.write('- General descriptive analysis for **categorical features**:')

# Cat features
n_cols_per_row = 4
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
      col.dataframe(df[cat].value_counts(dropna=False, normalize=True))

# *********** Download preprocessed df as a csv file ***********

st.write('___')
st.header('Download preprocessed data')

# Downloader as csv file 
download_prepro = st.download_button(
    label='💾 DOWNLOAD preprocessed data as csv file',
    data=df.to_csv(index=False),
    file_name=f"{data_file.name.split('.')[0]}_prepro.csv",
    mime='text/csv',
)

df_object = df.copy()
df_object[cat_list] = df_object[cat_list].astype('object')

# Donwloader as pkl file (dtype conversion)
download_prepro_pkl = st.download_button(
    label='💾 DOWNLOAD preprocessed data as pkl file',
    data=pickle.dumps(df_object),
    file_name=f"{data_file.name.split('.')[0]}_prepro.pkl",
    help='This option cast **features in cat_list to object dtype** and saves the DataFrame as a pickle file to maintain dtypes. Variables of *object dtype are identified as categorical variables in this web app*'
)

# Info on modifications made to the original df:
st.warning('Modifications made to the uploaded data:')
st.info(f'- Duplicates dropped: **{drop_duplicate}**')

st.info(f'- Missing values imputed: **{missing_handling}**')
if missing_handling == 'Imputed':
  miss1, miss2 = st.columns(2)
  with miss1:
    st.write(f'**Median imputation** applied to:')
    st.write(num_missing_list)
  with miss2:
    st.write(f'**Mode imputation** applied to:')
    st.write(cat_missing_list)
    
cols_dropped = df_original.columns.difference(df.columns).tolist()
st.info(f"- Columns dropped: **{cols_drop}**")
if cols_drop is True:
  st.write(cols_dropped)

st.info(f"- Changes to identification of column data type: **{change_dtypes}**")
if change_dtypes is True:
  change1, change2 = st.columns(2)
  with change1:
    st.write('**Numerical features** (customized)')
    st.write(num_list)
  with change2:
    st.write('**Categorical features** (customized)')
    st.write(cat_list)

# Download process:
if download_prepro is True:
  st.success(f"**{data_file.name.split('.')[0]}_prepro.csv** has been successfully downloaded!")
  st.balloons()

if download_prepro_pkl is True:
  st.success(f"**{data_file.name.split('.')[0]}_prepro.pkl** has been successfully downloaded!")
  st.balloons()
