from prefect import task, flow
import numpy as np
import pandas as pd

@task
def create_series(giv_array):
    return pd.Series(giv_array,name = 'values')
@task
def clean_data(series):
    return series.dropna()
@task
def summarize_data(series):
    dic = {
        'mean':series.mean(),
        'median':series.median(),
        'std':series.std(),
        'mode': series.mode()[0]
    }
    return dic
@flow
def pipeline_flow(arr):
    ser = create_series(arr)
    clean = clean_data(ser)
    summary = summarize_data(clean)
    return summary

if __name__ == "__main__":
    arr1 = np.array([12.0, 15.0, np.nan, 14.0, 10.0, np.nan, 18.0, 14.0, 16.0, 22.0, np.nan, 13.0])
    res = pipeline_flow(arr1)
    for key, value in res.items(): 
        print(key, ":", value)

#1. The cleaning process here could be done with just regular functions and there's no complex tasks where prefect could be usefull for.
#2. If such simple task should be done at a certain time every day/week the Prefect could be usefull because you wouldn't have to be monitored by a person.
