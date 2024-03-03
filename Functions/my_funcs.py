"""
@author: Hao Qi
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats as stats
import statsmodels.api as sm
from typing import Literal


###############################################################################################################################

def describe_custom(df,
                    decimals=2,
                    sorted_nunique=True
                    ) -> pd.DataFrame:
    """
    Generate a custom summary statistics DataFrame for the input DataFrame.

    Parameters:
    ---
    - `df (pd.DataFrame)`: Input DataFrame for which summary statistics are calculated.
    - `decimals (int, optional)`: Number of decimal places to round the results to (default is 2).
    - `sorted_nunique (bool, optional)`: If True, sort the result DataFrame based on the 'nunique' column
      in descending order; if False, return the DataFrame without sorting (default is True).

    Returns:
    ---
    pd.DataFrame: A summary statistics DataFrame with counts, unique counts, minimum, 25th percentile,
    median (50th percentile), mean, standard deviation, coefficient of variation, 75th percentile, and maximum.
    """
    def q1_25(ser):
        return ser.quantile(0.25)

    def q2_50(ser):
        return ser.quantile(0.50)

    def q3_75(ser):
        return ser.quantile(0.75)
    
    def CV(ser):
        return ser.std()/ser.mean()

    df = df.agg(['count','nunique', 'mean', 'std', CV, q1_25, q2_50, q3_75, 'min', 'max']).round(decimals).T    
    if sorted_nunique is False:
        return df
    else:
        return df.sort_values('nunique', ascending=False)

###############################################################################################################################

def histogram_boxplot(series,
                      figsize=(8,6),
                      show_qqplot=False,
                      extra_title=None,
                      IQR_multiplier=3,
                      show_IQR=False,
                      show_maxmin=True,
                      kde=True,
                      **kwargs
                      ):
    # Get n + missing
    n = series.shape[0] - series.isna().sum()
    total_n = series.shape[0]
    # Crear ventana para los subgráficos
    f2, (ax_box2, ax_hist2) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": (.15, .85)}, figsize=figsize)
    # Crear boxplot
    sns.boxplot(x=series, whis=IQR_multiplier, ax=ax_box2)
    # Crear histograma
    sns.histplot(x=series, ax=ax_hist2, kde=kde, **kwargs)
    
    mean = np.mean(series)
    median = series.quantile(.5)
    p_25 = series.quantile(.25)
    p_75 = series.quantile(.75)
    IQR = IQR_multiplier * (p_75 - p_25)
    std = series.std()
    min = np.min(series)
    max = np.max(series)
    
    if show_maxmin is True:
        ax_hist2.axvline(min, color='grey', lw=1.5, alpha=0.15, label= 'Min: ' + f'{min:.2f}')
    if show_IQR is True:
        ax_hist2.axvline(p_25 - IQR, color='orange', lw=1.2, alpha=0.25, label= f'Q1 - {IQR_multiplier}*IQR: ' + f'{p_25 - IQR:.2f}')
    ax_hist2.axvline(p_25,color='black', lw=1.2, linestyle='--', alpha=0.35, label='Q1: ' + f'{p_25:.2f}')
    ax_hist2.axvline(mean,color='r', lw=2.5, alpha=0.45, label= 'Mean: ' + f'{mean:.2f}')
    ax_hist2.axvline(median,color='black', lw=1.5, linestyle='--', alpha=0.6, label='Q2: ' + f'{median:.2f}')
    ax_hist2.axvline(p_75,color='black', lw=1.2, linestyle='--', alpha=0.35, label='Q3: ' + f'{p_75:.2f}')
    if show_IQR is True:
        ax_hist2.axvline(p_75 + IQR, color='orange', lw=1.2, alpha=0.25, label= f'Q3 + {IQR_multiplier}*IQR: ' + f'{p_75 + IQR:.2f}')
        ax_hist2.axvspan(p_25 - IQR, p_75 + IQR, facecolor='yellow', alpha=0.15)
    if show_maxmin is True:
        ax_hist2.axvline(max, color='grey', lw=1.5, alpha=0.15, label= 'Max: ' + f'{max:.2f}')
      
    suptitle_text = f'"{series.name}"'
    if extra_title:
        suptitle_text += f" | {extra_title}"
        
    ax_box2.set_title(f'n: {n}/{total_n} | n_unique: {series.nunique():.0f} | std: {std:.2f} | skew: {series.skew():.2f} | kurt: {series.kurt():.2f}',
                fontsize=10)
    ax_hist2.set_xlabel(None)
    ax_box2.set_xlabel(None)
    # Mostrar gráfico
    f2.suptitle(suptitle_text, fontsize='medium', fontweight='bold')
    plt.legend(bbox_to_anchor=(1,1), fontsize='small')
    
    if show_qqplot is True:
        sm.qqplot(series.dropna(), fit=True, line='45', alpha=0.25)
        
    return f2
        
###############################################################################################################################

def barh_plot(series,
              sort=False,
              extra_title=None,
              figsize=(7,6),
              xlim_expansion=1.15,
              palette='hsv',
              **kwargs
              ):
    fig, axes = plt.subplots(figsize=figsize)
    
    sns.countplot(y=series,
                width=0.5,
                order=series.value_counts(sort=sort).index,
                palette=palette,
                ax=axes,
                **kwargs
                )
    
    counts_no_order = series.value_counts(sort=sort)
    props_no_order = series.value_counts(sort=sort, normalize=True)
        
    for i, (count, prop) in enumerate(zip(counts_no_order, props_no_order)):
        axes.annotate(f' ({count}, {prop:.0%})', (count, i), fontsize=8)

    suptitle_text = f"'{series.name}'"
    if extra_title:
        suptitle_text += f" | {extra_title}"
    
    axes.set_ylabel('')
    fig.suptitle(suptitle_text, fontsize=15, fontweight='bold')
    axes.set_title(f"n = {series.count()}/{series.size} | sort = {sort}")
    # Set xlimit
    _, xlim_r = axes.get_xlim()
    axes.set_xlim(right=xlim_r*xlim_expansion)
    
    return fig

###############################################################################################################################

def cat_num_plots(
    data: pd.DataFrame,
    y: str,
    x: str,
    plot_type: Literal['box', 'violin']='box',
    log_yscale=False,
    n_adj_param=0.1,
    n_size=6.5,
    bar_alpha=0.1,
    extra_title=None,
    palette='tab10',
    **kwargs
):

    fig, axes = plt.subplots(figsize=(8,6))

    if plot_type == 'violin':
        ax = sns.violinplot(data=data, y=y, x=x, hue=x, legend=False, palette=palette, ax=axes, **kwargs)
    elif plot_type == 'box':
        ax = sns.boxplot(data=data, y=y, x=x, hue=x, legend=False, palette=palette, ax=axes, **kwargs)
    else:
        raise Exception("Must choose between plot_type: ['violin', 'box']!")

    if log_yscale is True:
        ax.set_yscale('log')
    
    if pd.api.types.is_categorical_dtype(data[x]):
        order = None
    else:
        order = data[x].unique()
    
    if data[x].dtype.name == 'object':
        sort = False
    elif data[x].dtype.name == 'category':
        sort = True
    else:
        # Cover dummified cats.
        sort = False
        # raise TypeError(f"{x} variable is neither of object nor category dtype. It is {data[x].dtype.name}!")
    
    sns.countplot(data=data, x=x, ax=ax.twinx(), color='gray', order=order, alpha=bar_alpha)
    for i, (category_v, group) in enumerate(data.groupby(x, sort=sort)):
        n = len(group)
        plt.annotate(n, (i+n_adj_param, n), fontsize=n_size, color='blue')

    suptitle_text = f"{data[y].name} by {data[x].name}"
    if extra_title:
        suptitle_text += f" | {extra_title}"
      
    plt.suptitle(suptitle_text, fontsize=15, fontweight='bold')
    plt.title(f"n = {data[x].count()}/{data[x].size} | n_unique = {data[x].nunique()}", fontsize=10)
    axes.tick_params(axis='x', rotation=45, labelsize=9.5, labelrotation=45)

    return fig

###############################################################################################################################

def class_balance_barhplot(x,
                           y,
                           text_size=7.5,
                           figsize=(8,6)
                           ):
    
    fig, ax = plt.subplots(figsize=figsize)
    
    df_pct = pd.crosstab(x, y, normalize='index').sort_index(ascending=False)
    df_count = pd.crosstab(x, y).sort_index(ascending=False)
    
    # Binary or multiclass classification:
    n_y_classes = y.nunique()
    if n_y_classes > 2:
        df_pct.plot.barh(stacked=True, alpha=0.7, ax=ax)
        
        for i in range(0, len(df_pct.index)):
            pct_list = [str(np.round(pct, 2)) for pct in df_pct.iloc[i,:]]
            ax.annotate(
                text='p: ' + ' / '.join(pct_list),
                xy=(0.1, i + 0.14),
                alpha=0.8,
                color='black'
            )
            
        for i in range(0, len(df_count.index)):
            pct_list = [str(np.round(pct, 2)) for pct in df_count.iloc[i,:]]
            ax.annotate(
                text='n: ' + ' / '.join(pct_list),
                xy=(0.1, i - 0.14),
                alpha=0.8,
                color='black'
            )
        
    elif n_y_classes == 2:
        df_pct.plot.barh(stacked=True, color=['red', 'green'], alpha=0.7, ax=ax)

        for i, category in enumerate(df_pct.index):
            pct_0 = df_pct.iloc[:,0][category]
            pct_1 = df_pct.iloc[:,1][category]
            ax.annotate(text=f"{pct_0:.2f} | n={df_count.iloc[:,0][category]}",
                        xy=(0 + 0.01, i),
                        fontsize=text_size,
                        alpha=0.8,
                        color='blue'
                        )
            ax.annotate(text=f"{pct_1:.2f} | n={df_count.iloc[:,1][category]}",
                        xy=(0.92 - 0.1, i),
                        fontsize=text_size,
                        alpha=0.8,
                        color='blue'
                        )
            
    ax.legend(bbox_to_anchor=(1,1))
    ax.set_title(f"Class distribution of '{y.name}' for categories in '{x.name}'")
    fig.suptitle(x.name, fontsize=15, fontweight='bold')
    
    return fig

###############################################################################################################################

def get_cramersV(x,
                y,
                n_bins=5,
                return_scalar=False
                ):
    # Discretizar x continua
    if pd.api.types.is_numeric_dtype(x) and (not (x.nunique() == 2)):
        x= pd.cut(x, bins=min(n_bins, x.nunique()))
            
    # Discretizar y continua
    if pd.api.types.is_numeric_dtype(y) and (not (y.nunique() == 2)):
        y = pd.cut(y, bins=min(n_bins, y.nunique()))
    
    name = f'CramersV: min(nunique, {n_bins}) bins'
        
    data = pd.crosstab(x, y).values
    vCramer = stats.contingency.association(data, method='cramer')
    
    if return_scalar is True:
        return vCramer
    else:
        return pd.Series({name:vCramer}, name=x.name)
    
###############################################################################################################################

def get_cramersV_matrix(df,
                        n_bins=5,
                        n_decimals=3
                        ) -> pd.DataFrame:
        
    # Initialize an empty DataFrame for Cramer's V matrix
    num_cols = len(df.columns)
    cramer_matrix = pd.DataFrame(np.zeros((num_cols, num_cols)), columns=df.columns, index=df.columns)
    
    # Iterate over each pair of columns and calculate Cramer's V
    for col1 in df.columns:
        for col2 in df.columns:
            cramers_v = get_cramersV(df[col1], df[col2],
                                     n_bins=n_bins,
                                     return_scalar=True
                                     )
            cramer_matrix.at[col1, col2] = cramers_v
    
    return cramer_matrix.round(n_decimals)

###############################################################################################################################

def association_barplot(df_widefmt: pd.DataFrame,
                        y: pd.Series=None,
                        abs_value=False,
                        extra_title=None,
                        xlim_expansion=1.15,
                        text_size=8,
                        text_right_size=0.0003,
                        palette='coolwarm',
                        figsize=(6,5),
                        title_size=14,
                        no_decimals=False,
                        ascending=False,
                        sort=True,
                        **kwargs
                        ):

    metric_col = df_widefmt.T.columns[0]
    
    if sort is True:
        df_longfmt = df_widefmt.T.sort_values(metric_col, ascending=ascending)
    else:
        df_longfmt = df_widefmt.T
    
    hue = None
    if abs_value is True:
        df_longfmt['Sign'] = df_longfmt[metric_col].apply(lambda row: 'Negative' if row < 0 else 'Positive')
        df_longfmt[metric_col] = abs(df_longfmt[metric_col])
        df_longfmt.sort_values(metric_col, ascending=False, inplace=True)
        hue = df_longfmt['Sign']
        
    fig, ax = plt.subplots(figsize=figsize)
    
    sns.barplot(x=df_longfmt[metric_col], y=df_longfmt.index,
                hue=hue, hue_order=['Negative', 'Positive'],
                palette=palette, ax=ax, **kwargs)
    
    if no_decimals is True:
        for i, col in enumerate(df_longfmt.index):
            ax.annotate(text=f'{df_longfmt[metric_col][col]}',
                    xy=(df_longfmt[metric_col][col] + text_right_size, i),
                    fontsize=text_size)
    else:
        for i, col in enumerate(df_longfmt.index):
            ax.annotate(text=f'{df_longfmt[metric_col][col]:.3f}',
                        xy=(df_longfmt[metric_col][col] + text_right_size, i),
                        fontsize=text_size)
    
    if y is not None:
        title_text = f"{len(df_longfmt.index)} Predictors association wrt '{y.name}'"
    else:
        title_text = f"{len(df_longfmt.index)} Predictors"
        
    if extra_title:
        title_text += f" | {extra_title}"
    
    ax.set_ylabel('Predictors')
    fig.suptitle(title_text, fontsize=title_size, fontweight='bold')
    _, xlim_r = ax.get_xlim()
    ax.set_xlim(right=xlim_r*xlim_expansion)
    if abs_value is True:
        ax.legend(loc='lower right', fontsize=9)
    
    return fig

###############################################################################################################################

def scatteplots_wrt_y(
    data: pd.DataFrame,
    x_name: pd.Series,
    y_name: str,
    figsize=(8,6),
    **kwargs
):
    fig, ax = plt.subplots(figsize=figsize)
    
    sns.regplot(
        data=data,
        y=y_name,
        x=x_name,
        scatter_kws={'alpha':0.2},
        line_kws={'color':'red', 'alpha':0.4},
        ax=ax,
        **kwargs
    )
    
    pearson_corr = np.corrcoef(data[x_name], data[y_name])[0,1]
    ax.set_title(f"ρ = {pearson_corr:.2f}")
    ax.set_ylabel(y_name)
    ax.set_xlabel('')
    fig.suptitle(f"{x_name}", fontweight='bold')
    
    return fig
