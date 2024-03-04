import streamlit as st
import pandas as pd
import numpy as np
import csv
from Functions.my_funcs import histogram_boxplot, barh_plot

###############################################################################################################################

# PAGE CONFIGURATION
st.set_page_config(
  page_title='02_Descriptive Analysis',
  page_icon="./Images/astr.jpg",
  layout='wide',
  initial_sidebar_state='auto'
)

###############################################################################################################################

# BODY TITLE
title1, title2 = st.columns([1,4])
title1.image('./Images/cohete.jpg')
title2.write('')
title2.write('')
title2.write('')
title2.write('')
title2.write('# 02_Univariate Descriptive Analysis')

# Receive data file and convert to df
st.write('___')
st.header('Upload data file')

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
  
  # Df info.
  df_info = st.info(f'**{data_file.name}** contains **{df.shape[0]}** rows and **{df.shape[1]}** columns!')
  # Check presence of NaN.
  n_missing = df.isna().sum().sum()
  if n_missing > 0:
    df_info.empty()
    df.dropna(inplace=True)
    st.warning(f"- **{data_file.name}** contains **{n_missing} missing values**! For visualization purposes, they are dropped.")
    st.warning(f"- After dropping NaN: **{df.shape[0]}** rows and **{df.shape[1]}** columns.")

# Show df (expander)
with st.expander('**See data file content**'):
  st.dataframe(df)

###############################################################################################################################

# Separation of features by data types:
cat_list = []
num_list = []

for col in df.columns:
    if pd.api.types.is_numeric_dtype(df[col]) and (df[col].nunique() > 2):
      num_list.append(col)
    else:
      cat_list.append(col)

num_list_default = sorted(num_list)
cat_list_default = sorted(cat_list)

###############################################################################################################################

# SIDEBAR WIDGETS:
with st.sidebar:
  st.write('___')
  st.write('> `Switch between pages above 👆`')
  
  st.write('___')
  st.header('SELECT VARIABLES', help=f'List of **all variables** from data: {df.columns.tolist()}')
  # Define Xs
  with st.form('variables'):
    num_box = st.multiselect(
      label='Choose **numerical features** of interest:',
      options=df.columns.tolist(),
      default=num_list_default,
      placeholder='Choose num variables'
    )
    
    cat_box = st.multiselect(
      label='Choose **categorical features** of interest:',
      options=df.columns.tolist(),
      default=cat_list_default,
      placeholder='Choose cat variables'
    )
    
    submit = st.form_submit_button('Submit')
  
  if submit is True:
    cat_list = cat_box
    num_list = num_box

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
      file_name = f"{data_file.name.split('.')[0]}_desc.txt"
    )

    if download_annotations is True:
      st.success(f"**{data_file.name.split('.')[0]}_desc.txt** has been successfully downloaded")

###############################################################################################################################

# BODY:
placeholder_init = st.empty()
placeholder_init.warning("Please select numerical and categorical variables of interest for distribution analysis **from the sidebar**, then click **Submit**!")

if submit is True:
  
  placeholder_init.empty()
  
  st.write('___')
  st.write('## Univariate Distribution Plots')
  
  # Implement progress bar:
  progress_text = '`CREATING VISUALIZATIONS...` (**0%**).'
  progress_bar  = st.progress(0, text=progress_text)
  
  all_list = num_list + cat_list
  progress_per_var = 1.0 / len(all_list)
  n_vars_processed = 1  # Avoid first iteration 0.
  
  tab1, tab2 = st.tabs(['📈 Numerical features', '📶 Categorical features'])
    
  with tab1:
    n_cols_per_row = 2
    cols_list = [f'col{i}' for i in range(1, len(num_list) + 1)]
    for i in range(0, len(num_list) + 1, n_cols_per_row):
      cols_list_each_2 = cols_list[i:i+n_cols_per_row]
      num_list_each_2  = num_list[i:i+n_cols_per_row]
      cols_list_each_2 = st.columns(n_cols_per_row)
      
      for col, num in zip(cols_list_each_2, num_list_each_2):
        
        if pd.api.types.is_numeric_dtype(df[num]):
          fig_hist = histogram_boxplot(
            series=df[num],
            figsize=(8,6),
            kde=True,
          )
          col.pyplot(fig_hist, use_container_width=True)
        else:
          col.write(f"`{df[num].name}` feature is not numerical!")
          col.write(f'Unique categories (**{df[num].nunique()}**):')
          col.write(pd.DataFrame(df[num].unique(), columns=['Category']).T)

        # Track progress bar (num plots).
        progress__actual = np.round(n_vars_processed * progress_per_var, 2)
        progress_text_r = f"⌛ `CREATING VISUALIZATIONS...` (**{progress__actual:.0%}**)."
        progress_bar.progress(progress__actual, text=progress_text_r)
        n_vars_processed += 1
        
  with tab2:
    n_cols_per_row = 2
    cols_list = [f'col{i}' for i in range(1, len(cat_list) + 1)]
    for i in range(0, len(cat_list) + 1, n_cols_per_row):
      cols_list_each_2 = cols_list[i:i+n_cols_per_row]
      cat_list_each_2  = cat_list[i:i+n_cols_per_row]
      cols_list_each_2 = st.columns(n_cols_per_row)
      
      for col, cat in zip(cols_list_each_2, cat_list_each_2):
        
        n_unique_cats = df[cat].nunique()
        if n_unique_cats > 25:
          col.write(f"`{df[cat].name}` has too many categories **{n_unique_cats}** > 20")
          col.write('Unique categories:')
          col.dataframe(pd.DataFrame(df[cat].unique(), columns=['Category']))
        else:
          fig_barh = barh_plot(
            series=df[cat],
            xlim_expansion=1.15,
            palette='tab10',
            figsize=(8,6)
          )
          col.pyplot(fig_barh, use_container_width=True)
        
        # Track progress bar (cat plots).
        progress__actual = np.round(n_vars_processed * progress_per_var, 2)
        progress_text_r = f"⌛ `CREATING VISUALIZATIONS...` (**{progress__actual:.0%}**)."
        progress_bar.progress(progress__actual, text=progress_text_r)
        n_vars_processed += 1
        
  # Ended!
  st.balloons()