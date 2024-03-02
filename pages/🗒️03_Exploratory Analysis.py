import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats
from Functions.my_funcs import (get_cramersV,
                                get_cramersV_matrix,
                                association_barplot,
                                scatteplots_wrt_y,
                                cat_num_plots,
                                class_balance_barhplot)

###############################################################################################################################

# PAGE CONFIGURATION
st.set_page_config(
  page_title='03_Exploratory Analysis',
  page_icon="./Images/astr.jpg",
  layout='wide',
  initial_sidebar_state='expanded'
)

###############################################################################################################################

# BODY TITLE
title1, title2 = st.columns([1,4])
title1.image('./Images/llega.jpg')
title2.write('')
title2.write('')
title2.write('')
title2.write('')
title2.write('# 03_Exploratory Analysis')

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
      df = pd.read_csv(data_file)
  elif data_file.name.endswith('.pkl'):
      df = pd.read_pickle(data_file)
  elif data_file.name.endswith('.xlsx'):
      df = pd.read_excel(data_file)
  else:
    raise Exception('This APP currently accepts only one of these file extensions: [csv, pkl, xlsx]')
  return df

if data_file is None:
  st.info('Please, upload a data file to begin with!')
  st.info('- Allowed extensions so far: csv, xlsx, pkl.')
  st.stop()
else:
  df = convert_to_df(data_file)
  df_info = st.info(f'**{data_file.name}** contains **{df.shape[0]}** rows and **{df.shape[1]}** columns! ')
  # Check NaN.
  n_missing = df.isna().sum().sum()
  if n_missing > 0:
    df_info.empty()
    df.dropna(inplace=True)
    st.warning(f'**{data_file.name}** contains **{n_missing} missing values**! For visualization purposes, they are dropped. Rows after dropping NaN: **{df.shape[0]}**')

###############################################################################################################################

# SIDEBAR WIDGETS:
with st.sidebar:
  st.write('___')
  st.write('> `Switch between pages above 👆`')
  
  st.write('___')
  st.header('SELECT VARIABLES', help=f'List of **all variables** from data: {df.columns.tolist()}')
  
  # Define y.
  y_box = st.selectbox(
    label='Choose **target variable** (y)',
    options=df.columns.tolist(),
  )
  
  # Feature df.
  X = df.drop(columns=y_box)
  # y series.
  y = df[y_box]
  
  # Determine y dtype.
  if pd.api.types.is_numeric_dtype(y) and (y.nunique() > 2):
    y_dtype = 'numerical'
  else:
    y_dtype = 'categorical'
    
  st.info(f'**{y_box}** is **{y_dtype}** (nunique = {y.nunique()}).')
  
  # Separation of features (X) by data types:
  cat_list = []
  num_list = []

  for col in X.columns:
      if pd.api.types.is_numeric_dtype(X[col]) and (X[col].nunique() > 2):
        num_list.append(col)
      else:
        cat_list.append(col)

  num_list_default = sorted(num_list)
  cat_list_default = sorted(cat_list)
  
  # Define Xs.
  with st.form('variables'):
    num_box = st.multiselect(
      label='Choose **numerical features** of interest:',
      options=X.columns.tolist(),
      default=num_list_default,
    )
    
    cat_box = st.multiselect(
      label='Choose **categorical features** of interest:',
      options=X.columns.tolist(),
      default=cat_list_default,
    )
    
    submit = st.form_submit_button('SUBMIT')
  
  if submit is True:
    cat_list = cat_box
    num_list = num_box
    y_name   = y.name
    
    # New X (filtered by only selected features).
    X = X[cat_list + num_list]

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
      file_name=f'{data_file.name.split('.')[0]}_EDA.txt',
    )

    if download_annotations is True:
      st.success(f'**{data_file.name.split('.')[0]}_EDA.txt** has been successfully downloaded')

###############################################################################################################################

# BODY:
placeholder_init = st.empty()
placeholder_init.warning("Please select target variable, numerical and categorical variables of interest for visualizations **from the sidebar**, then click 'SUBMIT'!")
    
# Variables chose, visualizations running...
if submit is True:
  placeholder_init.empty()
  
  # Implement progress bar:
  progress_text = '`CREATING VISUALIZATIONS...` (**0%**).'
  progress_bar  = st.progress(0, text=progress_text)
  
  all_list = num_list + cat_list
  # +2: 'Best Xs' and 'Among Xs' plots.
  progress_per_var = 1.0 / (len(all_list) +2)
  n_vars_processed = 1  # Avoid first iteration 0.

  # Visualizations running...
  tab1, tab2, tab3, tab4 = st.tabs(['🎯Best Xs', '♻️Among Xs', '📈Numerical features', '📶Categorical features'])
  
  # Best Xs (cramersV)
  with tab1:
    df_wide_fmt = X.apply(lambda x: get_cramersV(x, y))
    fig_barh_y  = association_barplot(
      df_widefmt=df_wide_fmt,
      y=df[y_name],
      text_right_size=0.0005,
    )
    st.pyplot(fig_barh_y, use_container_width=True)

    progress__actual = np.round(n_vars_processed * progress_per_var, 2)
    progress_text_r = f'⌛ `CREATING VISUALIZATIONS...` (**{progress__actual:.0%}**).'
    progress_bar.progress(progress__actual, text=progress_text_r)
    n_vars_processed += 1
  
  # Assoc among Xs
  with tab2:
    # Crammer corr matrix
    df_crammer_matrix = get_cramersV_matrix(
      df=X
    )
    
    fig_cramer, ax = plt.subplots()
    sns.heatmap(
      data=df_crammer_matrix.round(2),
      annot=True,
      annot_kws={'fontsize': 4},
      cmap='Reds',
      vmin=0, vmax=1,
      ax=ax
    )
    fig_cramer.suptitle("Cramer's V correlation matrix among Xs (all)", fontweight='bold')
    st.pyplot(fig_cramer, use_container_width=True)
    
    if len(num_list) > 1:
      df_corr_matrix = X[num_list].corr()
      fig_pearson, ax = plt.subplots()
      sns.heatmap(
        data=df_corr_matrix.round(2),
        annot=True,
        annot_kws={'fontsize': 4},
        cmap='coolwarm',
        vmin=-1, vmax=1,
        ax=ax
      )
      fig_pearson.suptitle("Pearson correlation matrix among Xs (numerical)", fontweight='bold')
      st.pyplot(fig_pearson, use_container_width=True)
      
    progress__actual = np.round(n_vars_processed * progress_per_var, 2)
    progress_text_r = f'⌛ `CREATING VISUALIZATIONS...` (**{progress__actual:.0%}**).'
    progress_bar.progress(progress__actual, text=progress_text_r)
    n_vars_processed += 1
  
  # Numerical Xs
  with tab3:
    n_cols_per_row = 2
    cols_list = [f'col{i}' for i in range(1, len(num_list) + 1)]
    for i in range(0, len(num_list) + 1, n_cols_per_row):
      cols_list_each_2 = cols_list[i:i+n_cols_per_row]
      num_list_each_2  = num_list[i:i+n_cols_per_row]
      cols_list_each_2 = st.columns(n_cols_per_row)
      
      for col, num in zip(cols_list_each_2, num_list_each_2):
        if y_dtype == 'numerical':
          fig_num = fig_hist = scatteplots_wrt_y(
            data=df,
            x_name=num,
            y_name=y_name,
          )
        elif y_dtype == 'categorical':
          fig_num, ax = plt.subplots()
          sns.kdeplot(
            data=df,
            x=num,
            hue=y_name,
            fill=True,
            palette='tab10',
            ax=ax
          )
          ax.set_ylabel('')
          ax.set_xlabel('')
          fig_num.suptitle(num, fontweight='bold')
          
        col.pyplot(fig_num, use_container_width=True)
        
        progress__actual = np.round(n_vars_processed * progress_per_var, 2)
        progress_text_r = f'⌛ `CREATING VISUALIZATIONS...` (**{progress__actual:.0%}**).'
        progress_bar.progress(progress__actual, text=progress_text_r)
        n_vars_processed += 1
  
  # Categorical Xs
  with tab4:
    n_cols_per_row = 2
    cols_list = [f'col{i}' for i in range(1, len(cat_list) + 1)]
    for i in range(0, len(cat_list) + 1, n_cols_per_row):
      cols_list_each_2 = cols_list[i:i+n_cols_per_row]
      cat_list_each_2  = cat_list[i:i+n_cols_per_row]
      cols_list_each_2 = st.columns(n_cols_per_row)
      
      for col, cat in zip(cols_list_each_2, cat_list_each_2):
        if y_dtype == 'numerical':
          fig_cat = cat_num_plots(
            data=df,
            x=cat,
            y=y_name,
          )
        elif y_dtype == 'categorical':
          n_unique_cats = df[cat].nunique()
          if n_unique_cats > 25:
            col.write(f'`{df[cat].name}` has too many categories **{n_unique_cats}** > 20')
            col.write('Unique categories:')
            col.write(pd.DataFrame(df[cat].unique(), columns=['Category']).T)
          else:
            fig_cat = class_balance_barhplot(
              x=df[cat],
              y=y,
            )
            
        col.pyplot(fig_cat, use_container_width=True)
        
        progress__actual = np.round(n_vars_processed * progress_per_var, 2)
        progress_text_r = f'⌛ `CREATING VISUALIZATIONS...` (**{progress__actual:.0%}**).'
        progress_bar.progress(progress__actual, text=progress_text_r)
        n_vars_processed += 1
  
  # Ended!
  st.balloons()