# Pandas Q1
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import pearsonr
import seaborn as sns

data = {
    "name":   ["Alice", "Bob", "Carol", "David", "Eve"],
    "grade":  [85, 72, 90, 68, 95],
    "city":   ["Boston", "Austin", "Boston", "Denver", "Austin"],
    "passed": [True, True, True, False, True]
}
df = pd.DataFrame(data)
print('First 3 rows:\n', df.head(3),'\nShape: ',df.shape,'\nData types:\n',df.dtypes)

# Pandas Q2
print(f"Students passed above 80:\n {df[(df["passed"] == True) & (df["grade"] > 80)]}")

# Pandas Q3
df['grade_curved']= df['grade']+5
print(f'New df:\n {df}')

# Pandas Q4
df['name_upper'] = df['name'].str.upper()
print(f'Names:\n {df[['name','name_upper']]}')

# Pandas Q5
print('Grade by city:\n',df.groupby('city')['grade'].mean())

# Pandas Q6
df['city'] = df['city'].replace('Austin','Houston')
print('New city:\n', df[['name','city']])

# Pandas Q7
df = df.sort_values('grade', ascending=False)
print('Sorted df\n',df.head(3))


# NumPy Q1
array = np.array([[10, 20, 30, 40, 50]])
print(f'Shape: {array.shape} \n DataType: {array.dtype}\n Ndim: {array.ndim}')

# NumPy Q2
arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(f'Shape: {arr.shape} \n Size: {arr.size}')

# NumPy Q3
print(f'Sliced arr:\n{arr[:2, :2]}')

# NumPy Q4
array2 = np.zeros((2,5),dtype=int)
array3 = np.ones((2,5),dtype=int)
print(f'Zeros:\n{array2}\nOnes:\n{array3}')

# NumPy Q5
array4 = np.arange(0, 50, 5)
print(f'Array:\n {array4}\nShape: {array4.shape}\nMean: {array4.mean()}\nSum: {array4.sum()}\nSt.D.: {array4.std()}')

# NumPy Q6
array5 = np.random.normal(0, 1, 200)
print(f'Mean: {array5.mean()}\nSt.D: {array5.std()}')


# Matplotlib Q1
x = [0, 1, 2, 3, 4, 5]
y = [0, 1, 4, 9, 16, 25]
plt.plot(x,y)
plt.title('Squares')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

# Matplotlib Q2
subjects = ["Math", "Science", "English", "History"]
scores   = [88, 92, 75, 83]
plt.bar(subjects,scores)
plt.title('Subject Scores')
plt.xlabel('subjects')
plt.ylabel('scores')
plt.show()

# Matplotlib Q3
x1, y1 = [1, 2, 3, 4, 5], [2, 4, 5, 4, 5]
x2, y2 = [1, 2, 3, 4, 5], [5, 4, 3, 2, 1]
plt.scatter(x1, y1, label="Dataset 1") 
plt.scatter(x2, y2, label="Dataset 2") 
plt.xlabel("x") 
plt.ylabel("y") 
plt.legend() 
plt.show()

# Matplotlib Q4
plt.subplot(1, 2, 1)
plt.plot(x,y)
plt.title('Squares')
plt.subplot(1, 2, 2)
plt.bar(subjects,scores)
plt.title('Subject Scores')
plt.tight_layout()
plt.show()


# Descriptive Stats Q1
data = [12, 15, 14, 10, 18, 22, 13, 16, 14, 15]
data_array = np.array(data)
print(f'Mean: {np.mean(data_array)}\n',
      f'Median: {np.median(data_array)}\n',
      f'Variance: {np.var(data_array)}\n',
      f'St.D: {np.std(data_array)}\n')

# Descriptive Stats Q2
rand_num = np.random.normal(65, 10, 500)
plt.hist(rand_num, bins= 20)
plt.title('Distribution of Scores')
plt.ylabel('Count')
plt.xlabel('Scores')
plt.show()

# Descriptive Stats Q3
group_a = [55, 60, 63, 70, 68, 62, 58, 65]
group_b = [75, 80, 78, 90, 85, 79, 82, 88]
plt.boxplot([group_a,group_b],labels=["Group A", "Group B"])
plt.title('Score Comparison')
plt.show()

# Descriptive Stats Q4
normal_data = np.random.normal(50, 5, 200)
skewed_data = np.random.exponential(10, 200)
plt.boxplot([normal_data,skewed_data],labels=["Normal", "Exponential"])
plt.title('Distribution Comparison')
plt.show()
# The exponential distribution is more skewed and median will provide a better statistics due to a lot of outliers

# Descriptive Stats Q5
data1 = [10, 12, 12, 16, 18]
data2 = [10, 12, 12, 16, 150]
print(f'Mean data1: {np.mean(data1)}.    Mean data2: {np.mean(data2)}\n',
      f'Median data1: {np.median(data1)}.   Median data2: {np.median(data2)}\n',
      f'Mode data1: {pd.Series(data1).mode()}.   Mode data2: {pd.Series(data2).mode()}\n')
#Median and mean are different due to the big outlier in data2, which highly affects the median and mean


# Hypothesis Q1
group_a = [72, 68, 75, 70, 69, 73, 71, 74]
group_b = [80, 85, 78, 83, 82, 86, 79, 84]
t_st1, p_val1 = stats.ttest_ind(group_a, group_b) 
print(f't-statistic: {t_st1} \np-value: {p_val1}')

# Hypothesis Q2
if p_val1 <0.05:
    print('The result IS statistically significant')
else:
    print('The result IS NOT statistically significant')

# Hypothesis Q3
before = [60, 65, 70, 58, 62, 67, 63, 66]
after  = [68, 70, 76, 65, 69, 72, 70, 71]
t_st2, p_val2 = stats.ttest_rel(before, after) 
print(f't-statistic: {t_st2} \np-value: {p_val2}')

# Hypothesis Q4
scores = [72, 68, 75, 70, 69, 74, 71, 73]
t_st3, p_val3 = stats.ttest_1samp(scores, 70) 
print(f't-statistic: {t_st3} \np-value: {p_val3}')

# Hypothesis Q5
t_st4, p_val4 = stats.ttest_ind(group_a, group_b, alternative="less") 
print(f'p-value: {p_val4}')

# Hypothesis Q6
print('The results are statistically significant, which showes that there is a strong evidence that group a has lower score than group b.')


# Correlation Q1
x = [1, 2, 3, 4, 5]
y = [2, 4, 6, 8, 10]
corr1 = np.corrcoef(x,y)
print(corr1,f'\nCorrelation coefficient: {corr1[0,1]}')
# There's going to be a good linear correlation because with each increase of x results in y increasing by 2x

# Correlation Q2
x = [1,  2,  3,  4,  5,  6,  7,  8,  9, 10]
y = [10, 9,  7,  8,  6,  5,  3,  4,  2,  1]
coef, p_val = pearsonr(x, y)
print(f'Correlation coefficient: {coef} \np value: {p_val}')

# Correlation Q3
people = {
    "height": [160, 165, 170, 175, 180],
    "weight": [55,  60,  65,  72,  80],
    "age":    [25,  30,  22,  35,  28]
}
df_p = pd.DataFrame(people)
corr_matr = df_p.corr()
print(corr_matr)

# Correlation Q4
x = [10, 20, 30, 40, 50]
y = [90, 75, 60, 45, 30]
plt.scatter(x, y, label="Negative Correlation") 
plt.xlabel("x") 
plt.ylabel("y") 
plt.show()

# Correlation Q5
sns.heatmap(corr_matr, annot=True)
plt.title('Correlation Heatmap')
plt.show()


# Pipeline Q1
arr1 = np.array([12.0, 15.0, np.nan, 14.0, 10.0, np.nan, 18.0, 14.0, 16.0, 22.0, np.nan, 13.0])

def create_series(giv_array):
    return pd.Series(giv_array,name = 'values')

def clean_data(series):
    return series.dropna()

def summarize_data(series):
    dic = {
        'mean':series.mean(),
        'median':series.median(),
        'std':series.std(),
        'mode': series.mode()[0]
    }
    return dic

def data_pipeline(arr):
    return summarize_data(clean_data(create_series(arr)))

res = data_pipeline(arr1)
for key, value in res.items(): 
    print(key, ":", value)