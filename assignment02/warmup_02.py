import os
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# --- scikit-learn API ---
#Q1
years  = np.array([1, 2, 3, 5, 7, 10]).reshape(-1, 1)
salary = np.array([45000, 50000, 60000, 75000, 90000, 120000])
new_years = np.array([4,8]).reshape(-1, 1)
model = LinearRegression()
model.fit(years, salary)
pred_salary = model.predict(new_years)
print(f'Slope: {model.coef_[0]}\nIntercept: {model.intercept_}\nPredicted Salary for 4 Years: {pred_salary[0]}\nPredicted Salary for 8 Years: {pred_salary[1]}')

#Q2
x = np.array([10, 20, 30, 40, 50])
reshape_x = x.reshape(-1, 1) 
print(f'Shape: {x.shape}\nReshaped x: {reshape_x.shape}')
#The scikit-learn requires to have information about how many samples(rows) in the data as well as how many features(columns) are there. The 1-D array stores information only for one of those parameters

#Q3
X_clusters, _ = make_blobs(n_samples=120, centers=3, cluster_std=0.8, random_state=7)
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X_clusters)
labels = kmeans.predict(X_clusters)  
print(f'Cluster Centers: {kmeans.cluster_centers_}\nPoints in each cluster: {np.bincount(labels)}')
plt.scatter(X_clusters[:, 0], X_clusters[:, 1], c=labels, cmap='viridis', s=60, alpha=0.7)
plt.title("Clusters by K-Means")
plt.savefig("outputs/kmeans_clusters.png")
plt.show()

# --- Linear Regression ---
np.random.seed(42)
num_patients = 100
age    = np.random.randint(20, 65, num_patients).astype(float)
smoker = np.random.randint(0, 2, num_patients).astype(float)
cost   = 200 * age + 15000 * smoker + np.random.normal(0, 3000, num_patients)

#Q1
plt.scatter(age,cost,c=smoker,cmap="coolwarm")
plt.title('Medical Cost vs Age')
plt.xlabel('Age')
plt.ylabel('Cost')
plt.savefig("outputs/cost_vs_age.png")
plt.show()
#There are visibly 2 distinct groups on the graph, which indicates that smoker variable does affect the medical cost

#Q2
age_shaped = age.reshape(-1, 1) 
x_train, x_test,y_train, y_test = train_test_split(age_shaped,cost, test_size=0.2, random_state=42)
print(f'Age train shape: {x_train.shape}\nAge test shape: {x_test.shape}\nCost train shape: {y_train.shape}\nCost test shape: {y_test.shape}')

#Q3
model = LinearRegression()
model.fit(x_train, y_train)
print(f'Slope: {model.coef_[0]} \nIntercept: {model.intercept_}')
y_pred = model.predict(x_test)
print(f'RMSE: {np.sqrt(np.mean((y_pred - y_test) ** 2))} \nR sqr: {model.score(x_test, y_test)}')
# The slope showes that there is a positive relationship between age and cost. Meaning with each increase in age the medical cost will increase by 196.58

#Q4
X_full = np.column_stack([age, smoker])
X_train, X_test,Y_train, Y_test = train_test_split(X_full,cost, test_size=0.2, random_state=42)
model_full = LinearRegression()
model_full.fit(X_train, Y_train)
Y_pred = model_full.predict(X_test)
print(f'RMSE with smoke variable: {np.sqrt(np.mean((Y_pred - Y_test) ** 2))} \nR sqr with smoke variable: {model_full.score(X_test, Y_test)}')
#Adding a smoker variable improved the model prediction, meaning the smoke variable heavily impacts medical costs
print("age coefficient:    ", model_full.coef_[0])
print("smoker coefficient: ", model_full.coef_[1])
# The smoker coef. showes the difference in medical cost between the smoker and non smoker

#Q5
plt.scatter(Y_pred,Y_test)
plt.axline((0, 0), slope=1)
plt.title('Predicted vs Actual')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.savefig("outputs/predicted_vs_actual.png")
plt.show()
#If the point falls above the diagonal meaning the model underpredicted this value, when below it over predicted it