# 💫 GALACTIC GRAPHER 💫

![Galactic Grapher](./Images/astr.jpg)

✨ A descriptive data analysis web app that encompasses from basic data preprocessing to EDA! ✨

👉 [Click here to access web app](https://galacticgrapherst-lj3kqinzmrckvsdu6sslnd.streamlit.app/) <br>
- NOTE: This app **goes into sleep mode if it's been idle for 7 days**. If this is the case, feel free to reach out to me via email (kamitttt98@gmail.com) so I can get it back up and running!

- **Alternatively, you can clone this repository to your local machine**, set the folder as your working directory in the conda terminal, ensure you have installed the modules listed in the `requirements.txt file`, and then run the command `streamlit run Homepage.py`.

---

## DESCRIPTION

*-Note that each page is independent in the sense that a **data file uploaded in one page cannot be accessed by another page simultaneously**. If you wish go through all 3 analysis process with the same data file, then you just need to upload given file in each analysis process (page).*
  
-All pages feature a **text box for annotations** which can be found on the sidebar. Once saved, users can download it as a text file.

**Galactic Grapher** is a web application designed to assist users in analyzing data. It offers three distinct webpages/platforms, each tailored to a specific aspect of data analysis:

### 01_Basic Preprocessing

- The first page, titled "01_Basic Preprocessing," encompasses all fundamental **steps involved in preparing the data for analysis**.
- Users can upload data files, and the application guides them through the process of checking for and handling duplicates and missing values through basic univariate imputation methods.
- It also displays basic summary statistics for preprocessed numerical and categorical features.
- Once completed, users have the option to download the modified dataframe.
  
`Very important`: *when switching pages, progress in the current page will be deleted and data file needs to be reuploaded*.

### 02_Descriptive Analysis

- The second page, "02_Descriptive Analysis," is dedicated to **visualizing the distribution of both numerical and categorical variables** within the dataframe.
- The platform initially identifies numerical and categorical features present in the data file, but users have the flexibility to assign the data type for each feature as needed.
- For selected numerical features, the platform displays box plots and histograms, while for categorical features, it presents horizontal bar plots. In cases where categorical features have more than 20 categories (high cardinality), a wide format DataFrame is displayed instead.
- Visualizations in this page are **divided into 2 tabs**: one for numerical and the other for categorical plots.

### 03_Exploratory Analysis

- The third page, shifts the focus to **examining the relationship between feature variables (predictors) and the target variable (y)**. Users define the target variable as well as the data type for each feature in the data file.
- This page is **divided into 4 tabs**:
  1. **Best Xs**: a bar plot with most predictive features wrt y based on Cramers' V.
  2. **Among Xs**: a heatmap of Cramer's-correlation matrix among all features. If there are at least 2 numerical features, a heatmap of Pearson correlation matrix for these variables are displayed as well. 
  3. **Numerical features**: Kernel density distribution plots are shown *if the target variable is categorical* and scatter plots with fitted regression line *if the target is numerical* as well.
  4. **Categorical features**: Stacked horizontal bar plots are shown *if the target is categorical* too and box plots *if the target is numerical*.

---

Created by: **Hao, Qi**
