from prefect import task, flow
from prefect.logging import get_run_logger
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import pearsonr
import seaborn as sns

filepath = ['2015','2016','2017','2018','2019','2020','2021','2022','2023','2024']

@task(retries=3, retry_delay_seconds=2)
def merge_data():
    filepath = ['2015','2016','2017','2018','2019','2020','2021','2022','2023','2024']
    all_df = []
    for path in filepath:
        df = pd.read_csv(f'happiness_project/world_happiness_{path}.csv',sep = ';')
        df['year'] = int(path)
        if path == '2024':
            df = df.rename(columns={'Ladder score':'Happiness score'})
        all_df.append(df)
    
    new_df = pd.concat(all_df, ignore_index=True)
    names = ['Happiness score', 'GDP per capita', 'Social support','Healthy life expectancy',
             'Freedom to make life choices','Generosity', 'Perceptions of corruption'] 
    for col in names: 
        new_df[col] = new_df[col].astype(str).str.replace(",", ".", regex=False).astype(float)
    new_df.to_csv('assignments_01/outputs/merged_happiness.csv',index=False)
    return new_df

@task
def desc_stat(df):
    logger = get_run_logger()
    all_mean = df['Happiness score'].mean()
    all_median = df['Happiness score'].median()
    all_std = df['Happiness score'].std()

    logger.info(f"Overall mean: {all_mean}")
    logger.info(f"Overall median: {all_median}")
    logger.info(f"Overall standard deviation: {all_std}")

    mean_year = df.groupby('year')['Happiness score'].mean()
    mean_region = df.groupby('Regional indicator')['Happiness score'].mean()

    logger.info(f"Mean by Year: {mean_year}")
    logger.info(f"Mean by Region: {mean_region}")
    return mean_region

@task
def happiness_visual(df):
    logger = get_run_logger()
    plt.hist(df['Happiness score'])
    plt.title('Happiness Score Hist')
    plt.ylabel('Count')
    plt.xlabel('Happiness Scores')
    plt.savefig('assignments_01\outputs\happiness_histogram.png')
    plt.show()
    logger.info(f'Histogram created')

    df.boxplot(column = 'Happiness score',by = 'year')
    plt.title('Happiness Score by Year')
    plt.xlabel('Year')
    plt.ylabel('Happiness score')
    plt.savefig('assignments_01\outputs\happiness_by_year.png')
    plt.show()
    logger.info(f'Box plot created')

    plt.scatter(x = df['GDP per capita'],y = df['Happiness score'], label = 'GDP vs Happiness')
    plt.xlabel('GDP')
    plt.ylabel('Happiness score')
    plt.savefig('assignments_01\outputs\gdp_vs_happiness.png')
    plt.show()
    logger.info(f'Scatter plot created')

    corr_matr = df.corr(numeric_only=True)
    sns.heatmap(corr_matr, annot=True)
    plt.title('Correlation Heatmap')
    plt.savefig('assignments_01\outputs\correlation_heatmap.png')
    plt.show()
    logger.info(f'Heatmap created')

@task
def hypothesis_test(df):
    logger = get_run_logger()
    hap_2019 = df[df['year']==2019]['Happiness score']
    hap_2020 = df[df['year']==2020]['Happiness score']
    mean_2019 = hap_2019.mean()
    mean_2020 = hap_2020.mean()
    t_stat_y, p_value_y = stats.ttest_ind(hap_2019, hap_2020, equal_var=False)
    logger.info(f'Mean for 2019: {mean_2019}')
    logger.info(f'Mean for 2020: {mean_2020}')
    logger.info(f't-statistics: {t_stat_y}')
    logger.info(f'p-value: {p_value_y}')
    if p_value_y<0.05:
        logger.info('There is a statistically significant difference in Happiness Score between 2019 and 2020')
    else:
        logger.info('The difference between 2019 and 2020 is NOT statistically significant.')

    reg_w = df[df['Regional indicator']=='Western Europe']['Happiness score']
    reg_ce = df[df['Regional indicator']=='Central and Eastern Europe']['Happiness score']
    mean_w = reg_w.mean()
    mean_ce= reg_ce.mean()
    t_stat_r, p_value_r = stats.ttest_ind(reg_w, reg_ce, equal_var=False)
    logger.info(f'Mean for Western Europe: {mean_w}')
    logger.info(f'Mean for Central and Eastern Europe: {mean_ce}')
    logger.info(f't-statistics: {t_stat_r}')
    logger.info(f'p-value: {p_value_r}')
    if p_value_r<0.05:
        logger.info('There is a statistically significant difference in Happiness Score between Western Europe and Central and Eastern Europe')
    else:
        logger.info('The difference between Western Europe and Central and Eastern Europe is NOT statistically significant.')
    return p_value_y

@task
def corr_and_mult_comp(df):
    logger = get_run_logger()
    columns = ['GDP per capita', 'Social support','Healthy life expectancy',
             'Freedom to make life choices','Generosity', 'Perceptions of corruption'] 
    for col in columns:
        coef, p_val = pearsonr(df[col], df['Happiness score'])
        logger.info(f'Unadjusted correlation between {col} and Happiness score: \n  coef: {coef};  p-val: {p_val}')
    
    number_of_tests = len(columns)
    adjusted_alpha = 0.05 / number_of_tests
    good_cor = []
    for col in columns:
        coef, p_val = pearsonr(df[col], df['Happiness score'])
        if p_val<adjusted_alpha:
            logger.info(f'Relation between {col} and Happiness score are significant for both alpha = 0.05 and adjusted alpha = {adjusted_alpha}')
            good_cor.append(col)
    return good_cor

@task
def final_results(df, mean_reg,p_val,good_cor):
    logger = get_run_logger()

    logger.info(f'The Happiness dataset included information from {len(df["Country"].unique())} countries and 9 years spanning from 2015 to 2024. ')
    logger.info(f'When comparing the mean happiness score by region. The top 3 happiest regions are:{",".join(mean_reg.nlargest(3).index.tolist())}. The top 3 least happy regions are: {",".join(mean_reg.nsmallest(3).index.tolist())} ')
    logger.info(f'Based on the preformed t-test comparing 2019 and 2020 happiness level with p-value{p_val}. There is no evidence that there is a statistically significant difference in Happines Score in 2019 and 2020')
    logger.info(f'The variables that are significantly correlated with Happiness score after Bonferroni correction are: {",".join(good_cor)}.')


@flow
def happiness_pipeline():
    df = merge_data()
    desc_s = desc_stat(df)
    hypothesis_t = hypothesis_test(df)
    corr_and_mult = corr_and_mult_comp(df)
    return merge_data(),happiness_visual(df),hypothesis_test(df),corr_and_mult_comp(df),final_results(df,desc_s,hypothesis_t,corr_and_mult)

if __name__ == "__main__":
    happiness_pipeline()

