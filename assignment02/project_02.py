# First of all the seperator used in this document is ; not , which would be important to include. 
# The grades for G1,G2,G3 are not in the 0-100 system, but in 0-20
# Medu and Fedu indicates the level of education for a parents(categorical variable)
# Travel time also has it's own categories
# In conclusion, some numerical variables are categorical, so they have to be handled carefully

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

#Part 1
students = pd.read_csv('student_performance_math.csv', sep = ';')
print(f'Shape: {students.shape}\nFirst 5 rows:\n{students.head(5)}\nData types: {students.dtypes}')
plt.hist(students['G3'],bins=21)
plt.title('Distribution of Final Math Grades')
plt.xlabel('Grade')
plt.ylabel('Number of Students')
plt.savefig('outputs/g3_distribution.png')
plt.show()

#Part 2
new_std = students[students['G3']!=0]
print(f'Old Shape: {students.shape};  New Shape: {new_std.shape}')
# If we kept the rows with 0's in G3, the whole model would be skewed and give us underestimated results.
new_std['sex'] = new_std['sex'].replace({'M':1,'F':0})
y_n_col = ['schoolsup','internet','higher','activities']
new_std[y_n_col] = new_std[y_n_col].replace({'yes':1,'no':0})
print(f'Pearson corr. original: {students["G3"].corr(students["absences"])}\nNew corr.: {new_std["G3"].corr(new_std["absences"])}')
# To see why there's a big difference here is the scatter plot with G3 grade and number of absences
plt.scatter(students['G3'],students['absences'])
plt.show()
# In the plot there is a lot of points on 0,0 thought it's not clearly shown. 
# We can now understand that a lot of points on 0,0 put a lot of 'weight', making a weaker linear relation. 
# Removing the 0's results in a stronger linear relation that is not affected by outliers

#Part 3
pear_corr = new_std.corrwith(new_std['G3']).sort_values()
print(f'Pearson Corr: \n{pear_corr}')
# Columnt G1 and G2 have very strong correlation with G3 which is not suprizing because the values in G3 depends on G1 and G2
# Failures, absences, and school suply are not as strongly linearly correlated but they have higher negative linear correlation than other variables

fig, axes = plt.subplots(3, 1,figsize=(8, 8))
new_std.boxplot(column = 'G3', by = 'failures',ax=axes[0])
axes[0].set_title('Number Failures')
axes[0].set_xlabel('Failures')
axes[0].set_ylabel('G3')
new_std.boxplot(column = 'G3', by = 'schoolsup',ax=axes[1])
axes[1].set_title('Additional support from school')
axes[1].set_ylabel('G3')
axes[1].set_xlabel('School Support')
axes[2].scatter(new_std['G3'], new_std['absences'])
axes[2].set_xlabel('Absences')
axes[2].set_ylabel('G3')
axes[2].set_title('Number of absences')
plt.suptitle("")
plt.tight_layout()
plt.savefig('outputs/relation_plots.png')
plt.show()
# I decided to create 3 graphs to see if there is a clear relation between G3 and absences, school support, failures
# On the box plots(falures and support) there is not strong but negative relationship
# On absence graph there is no clear linear relationship

#Part 4
x_train, x_test,y_train, y_test = train_test_split(new_std[['failures']],new_std[['G3']], test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
print(f'Slope: {model.coef_[0]} \nRMSE: {np.sqrt(np.mean((y_pred - y_test) ** 2))} \nR-sq: {model.score(x_test, y_test)}')
# The slope tells that with each failure, the grade will decrease by about 1.4 points
# The predictions made by a model could be off by around 3 points 
# The R-sqr is really small, which was expected because the Pearson correlation was closer to 0 than 1, meaning the model poorly predicts the G3 scores

#Part 5
feature_cols = ["failures", "Medu", "Fedu", "studytime", "higher", "schoolsup",
                "internet", "sex", "freetime", "activities", "traveltime"]
X = new_std[feature_cols].values
y = new_std["G3"].values

X_train, X_test,Y_train, Y_test = train_test_split(X,y, test_size=0.2, random_state=42)
model_new = LinearRegression()
model_new.fit(X_train, Y_train)
Y_pred = model_new.predict(X_test)
print(f'Train R sq: {model_new.score(X_train, Y_train)}\nTest R-sq: {model_new.score(X_test, Y_test)}\nRMSE: {np.sqrt(np.mean((Y_pred - Y_test) ** 2))}')
# The R-sq significantly improved compared to the results in part 4, though the value is still too small
for name, coef in zip(feature_cols, model_new.coef_):
    print(f"{name:12s}: {coef:+.3f}")
# The suprizing coef are school support and sex. The coef for dchool support indicated that students who's reciving extra help from school would have lower grade bu around 2 points
# This cold be because students who recive extra help had worst grades before reciving help
# The positive 0.453 coef for the sex variable indicates that female students predicted to have a lower grade than male students. 
# The train and test R-sq are relative close, so there's no overfitting.
# If deploying this model I would definitely removed variables like: free time, activities, Medu, Fedu, travel time. Those variables are the ones that have both small coef and low pearson correlation
# The variables that definitely should be included are failures and school support which have a high impact on the model. Other variables could be included but they probably wouldn't make a big difference because the model itself is bad and would not be usefull to actually predict G3

#Part 6
plt.scatter(Y_pred,Y_test)
plt.axline((5, 5), slope=1)
plt.title('"Predicted vs Actual (Full Model)')
plt.xlabel('Predicted')
plt.ylabel('Actual')
# The warm up q5 also required to have the same name. I hope there won't be any confusions 
plt.savefig("outputs/predicted_vs_actual.png")
plt.show()
# Looking at the graph, the points above the line are underpredicted; the ones below are overpredicted. 
# The model itself has some trouble predicting high and low grades. 

# After removing the students who had G3=0, the filtered data set, instead of 395 rows, included 357 rows. And the test set with filtered data had 72 
# The best model had RMSE of 2.855, which indicated that the predicted value has an error +-2.9 grade points. The best R-sq is 0.15, which means that the model can only explain about 15% of predicted grade variations. 
# The highest positive coef are internet variable and higher education. This indicates that if a student has access to the internet and another one doesn't(assuming everything else is identical), the person with internet access will have a higher grade by 0.834. Same with higher education, but it will be higher by 0.61 
# The lowest 2 coefficients are school support and failures. Which means that if a person receiving school support has a lower grade by around 2.1 points. And the more failures a student has, the grade will decrease by 1.1 
# What surprised me the most is the fact that most of the variables that could be considered impactful on the first glance didn't actually affect the linear regression

feature_cols_new = ["G1","failures", "Medu", "Fedu", "studytime", "higher", "schoolsup",
                "internet", "sex", "freetime", "activities", "traveltime"]
x_new = new_std[feature_cols_new].values
y_new = new_std["G3"].values

Xn_train, Xn_test,Yn_train, Yn_test = train_test_split(x_new,y_new, test_size=0.2, random_state=42)
model_n = LinearRegression()
model_n.fit(Xn_train, Yn_train)
Yn_pred = model_n.predict(Xn_test)
print(f'New model Test R-sq: {model_n.score(Xn_test, Yn_test)}\nRMSE: {np.sqrt(np.mean((Yn_pred - Yn_test) ** 2))}')
# The G1 variable is not causing G3, those 2 variables are just closely related because usually students have roughly the same grades throughout the terms
# Using previous term can help identify students who would struggle in the G3 term, but LINEAR models without G1 or G2 would not be able to give certain information on which students could struggle.
