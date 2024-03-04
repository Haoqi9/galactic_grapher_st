import streamlit as st

###############################################################################################################################

# PAGE CONFIGURATION:
st.set_page_config(
  page_title='Homepage',
  page_icon="./Images/astr.jpg",
  layout='wide',
  initial_sidebar_state='expanded'
)

###############################################################################################################################

# SIDEBAR WIDGETS:
with st.sidebar:
  st.write('> `Switch between pages above 👆`')
  st.markdown(
    """
    # Table of Contents
    1. [01_Basic Preprocessing](#01-basic-preprocessing)
    1. [02_Descriptive Analysis](#02-descriptive-analysis)
    1. [03_Exploratory Analysis](#03-exploratory-analysis)
    """)  

###############################################################################################################################

# BODY:
# Title
title1, title2 = st.columns([1, 3])
with title1:
  st.image("./Images/astr.jpg", use_column_width=True)
with title2:
  st.write("# 💫 GALACTIC GRAPHER 💫")
  st.write('___')
  st.write('✨ A descriptive data analysis web app that encompasses from basic data preprocessing to EDA! ✨')
  st.caption('💡 **Created by**: Hao, Qi')
  st.caption('🛠️ **Source code**: https://github.com/Haoqi9/galactic_grapher_st')
  st.caption('📩 **Contact**: kamitttt98@gmail.com')
  
st.write('___')

# Description
desc1, desc2 = st.columns([2, 1])
with desc1:
  st.write('# DESCRIPTION')
  st.caption('*-Note that each page is independent in the sense that a **data file uploaded in one page cannot be accessed by another page simultaneously**. If you wish go through all 3 analysis process with the same data file, then you just need to upload given file in each analysis process (page).*')
  st.caption('-All pages feature a **text box for annotations** which can be found on the sidebar. Once saved, users can download it as a text file.')
  st.markdown(
    """
    **Galatic Grapher** is a web application designed to assist users in analyzing data. It offers three distinct webpages/platforms, each tailored to a specific aspect of data analysis:""")
    
  st.header('01_Basic Preprocessing')
  st.markdown(
    """ 
    - The first page, titled "01_Basic Preprocessing," encompasses all fundamental **steps involved in preparing the data for analysis**.
    - Users can upload data files, and the application guides them through the process of checking for and handling duplicates and missing values through basic univariate imputation methods.
    - It also displays basic summary statistics for preprocessed numerical and categorical features.
    - Once completed, users have the option to download the modified dataframe.
   >  `Very important`: *when switching pages, progress in the current page will be deleted and data file needs to be reuploaded*.
    """
    )
    
  st.header('02_Descriptive Analysis')
  st.markdown(
    """
    - The second page, "02_Descriptive Analysis," is dedicated to **visualizing the distribution of both numerical and categorical variables** within the dataframe.
    - The platform initially identifies numerical and categorical features present in the data file, but users have the flexibility to assign the data type for each feature as needed.
    - For selected numerical features, the platform displays box plots and histograms, while for categorical features, it presents horizontal bar plots. In cases where categorical features have more than 20 categories (high cardinality), a wide format DataFrame is displayed instead.
    - Visualizations in this page are **divided into 2 tabs**: one for numerical and the other for categorical plots.
    """
  )
    
  st.header('03_Exploratory Analysis')
  st.markdown(
    """
    - The third page, shifts the focus to **examining the relationship between feature variables (predictors) and the target variable (y)**. Users define the target variable as well as the data type for each feature in the data file.
    - This page is **divided into 4 tabs**:
      1. **Best Xs**: a bar plot with most predictive features wrt y based on Cramers' V.
      1. **Among Xs**: a heatmap of Cramer's-correlation matrix among all features. If the are at least 2 numerical features, a heatmap of pearson correlation matrix for these variables are displayed as well. 
      1. **Numerical features**: Kernel density distribution plots are shown *if target variable is categorical* and scatterplots with fitted regression line *if target is numerical* as well.
      1. **Categorical features**: Stacked horizontal bar plots are shown *if target is categorical* too and box plots *if target is numerical*.
    """
  )
with desc2:
  st.image('./Images/galaxy.jpg', use_column_width=True)
