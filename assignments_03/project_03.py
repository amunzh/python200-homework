import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests
from io import BytesIO
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    ConfusionMatrixDisplay
)
from sklearn.inspection import DecisionBoundaryDisplay
warnings.filterwarnings("ignore", category=RuntimeWarning)

# --- PART 1 ---
COLUMN_NAMES = [
    "word_freq_make",        # 0   percent of words that are "make"
    "word_freq_address",     # 1
    "word_freq_all",         # 2
    "word_freq_3d",          # 3   almost never appears
    "word_freq_our",         # 4
    "word_freq_over",        # 5
    "word_freq_remove",      # 6   common in "remove me from this list"
    "word_freq_internet",    # 7
    "word_freq_order",       # 8
    "word_freq_mail",        # 9
    "word_freq_receive",     # 10
    "word_freq_will",        # 11
    "word_freq_people",      # 12
    "word_freq_report",      # 13
    "word_freq_addresses",   # 14
    "word_freq_free",        # 15  classic spam word
    "word_freq_business",    # 16
    "word_freq_email",       # 17
    "word_freq_you",         # 18
    "word_freq_credit",      # 19
    "word_freq_your",        # 20  often high in spam
    "word_freq_font",        # 21  HTML emails
    "word_freq_000",         # 22  "win $ x,000" style offers
    "word_freq_money",       # 23  money related
    "word_freq_hp",          # 24  HP specific
    "word_freq_hpl",         # 25
    "word_freq_george",      # 26  specific HP person
    "word_freq_650",         # 27  area code
    "word_freq_lab",         # 28
    "word_freq_labs",        # 29
    "word_freq_telnet",      # 30
    "word_freq_857",         # 31
    "word_freq_data",        # 32
    "word_freq_415",         # 33
    "word_freq_85",          # 34
    "word_freq_technology",  # 35
    "word_freq_1999",        # 36
    "word_freq_parts",       # 37
    "word_freq_pm",          # 38
    "word_freq_direct",      # 39
    "word_freq_cs",          # 40
    "word_freq_meeting",     # 41
    "word_freq_original",    # 42
    "word_freq_project",     # 43
    "word_freq_re",          # 44  reply threads
    "word_freq_edu",         # 45
    "word_freq_table",       # 46
    "word_freq_conference",  # 47
    "char_freq_;",           # 48  frequency of ';'
    "char_freq_(",           # 49  frequency of '('
    "char_freq_[",           # 50  frequency of '['
    "char_freq_!",           # 51  exclamation marks (often big)
    "char_freq_$",           # 52  dollar sign (money related)
    "char_freq_#",           # 53  hash character
    "capital_run_length_average",  # 54  average length of capital letter runs
    "capital_run_length_longest",  # 55  longest capital run
    "capital_run_length_total",    # 56  total number of capital letters
    "spam_label"                    # 57  1 = spam, 0 = not spam
]
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data"
response = requests.get(url)
response.raise_for_status()

df = pd.read_csv(BytesIO(response.content), header=None)
df.columns = COLUMN_NAMES
print(f'{df.head()}\n{len(df)}')

col = ['word_freq_free', 'char_freq_!', 'capital_run_length_total']
for c in col:
    df.boxplot(column=c, by='spam_label')
    plt.title(f'{c} by email')
    plt.suptitle('')
    plt.xlabel('Email')
    plt.ylabel(c)
    plt.savefig(f'outputs/{c}_distribution')
    plt.show()
# There's a dramatic difference between classes. The spam has much higher values for all 3 categories
# Because there's a lot of 0's for word_freq the data is heavily skewed which could highly effect the knn or PCA model
# Some variables represent percentages, some lengths, and others count. This will matter because the logistic regression and knn have to be scaled first

# --- PART 2 ---
X = df.drop("spam_label", axis=1)
y = df["spam_label"]
X_train, X_test,y_train, y_test = train_test_split(X,y, test_size=0.2,stratify=y, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
# We need to scale features so that one feature will not dominate other ones. We do it on the X train set so that information from the test sets would not leak to the model

pca = PCA()
pca.fit(X_train_scaled)
cum_var = np.cumsum(pca.explained_variance_ratio_)
plt.plot(range(1, len(cum_var) + 1), cum_var)
plt.xlabel("number of components")
plt.ylabel("cumulative explained variance")
plt.title("cumulative explained variance vs. number of components")
plt.savefig('outputs/pca_var_project03.png')
plt.show()
n=np.argmax(cum_var >= 0.90)+1
print(f'n: {n}')
X_train_pca = pca.transform(X_train_scaled)[:, :n]
X_test_pca  = pca.transform(X_test_scaled)[:, :n]


# --- Part 3 ---
#knn unscaled
knn1 = KNeighborsClassifier(n_neighbors=5)
knn1.fit(X_train, y_train)
pred_knn_unscl = knn1.predict(X_test)
print(f'Accuracy knn unscaled: {accuracy_score(y_test, pred_knn_unscl)}\n Report knn unscaled:\n{classification_report(y_test, pred_knn_unscl)}\n')

#knn scaled
knn2 = KNeighborsClassifier(n_neighbors=5)
knn2.fit(X_train_scaled, y_train)
pred_knn_scl = knn2.predict(X_test_scaled)
print(f'Accuracy knn scaled: {accuracy_score(y_test, pred_knn_scl)}\n Report knn scaled:\n{classification_report(y_test, pred_knn_scl)}\n')

#knn PCA
knn3 = KNeighborsClassifier(n_neighbors=5)
knn3.fit(X_train_pca, y_train)
pred_knn_scl_pca = knn3.predict(X_test_pca)
print(f'Accuracy knn scaled PCA: {accuracy_score(y_test, pred_knn_scl_pca)}\n Report knn scaled PCA:\n{classification_report(y_test, pred_knn_scl_pca)}\n')
# There's a really small difference in preformance of full knn and PCA knn

#Decision Tree
depth = [3, 5, 10, None]
for d in depth:
    dec_tree = DecisionTreeClassifier(max_depth=d, random_state=42)
    dec_tree.fit(X_train, y_train)
    y_train_pred = dec_tree.predict(X_train)
    y_test_pred = dec_tree.predict(X_test)
    print(f"max_depth={d}, train accuracy={accuracy_score(y_train, y_train_pred)}, test accuracy={accuracy_score(y_test, y_test_pred)}")
# With increase in depth increase the accuracy for both train and test increases as well. But big depth and train accuracy could indicate that the model start overfitting the data.
# The depth of 10 would be the best to use. The unlimited depth is definitely overfits the data while depth of 5 have lower accuracy
dec_tree = DecisionTreeClassifier(max_depth=10, random_state=42)
dec_tree.fit(X_train, y_train)
tree_pred = dec_tree.predict(X_test)
print(f'Accuracy decision tree:, {accuracy_score(y_test, tree_pred)}\n Report decision tree:\n{classification_report(y_test, tree_pred)}\n')

#Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
pred_rf = rf.predict(X_test)
print(f'Accuracy random forest:, {accuracy_score(y_test, pred_rf)}\n Report random forest:\n{classification_report(y_test, pred_rf)}\n')

dt_feat = pd.Series(dec_tree.feature_importances_, index=X.columns)
print(f'Decision Tree features:{dt_feat.sort_values(ascending=False).head(10)}')
rf_feat = pd.Series(rf.feature_importances_, index=X.columns)
top10 = rf_feat.sort_values(ascending=False).head(10)
print(f'Top 10 Random Forest features:{top10}')
top10.plot.bar(title = 'RF features')
plt.tight_layout()
plt.savefig('outputs/feature_importances.png')
plt.show()
# Decision tree and random forest have a little different important features. Characters like !,$ or words remove are important in both models. Thought some words like you, your important for RF but not for DT
# The results kind of matches the intuition on what people would consider spam but some words like george is unexpected to see

#Logistic Regression
log_reg = LogisticRegression(C=1.0, max_iter=1000, solver='liblinear')
log_reg.fit(X_train_scaled, y_train)
pred_log_scaled = log_reg.predict(X_test_scaled)
print(f'Accuracy logistic scaled:, {accuracy_score(y_test, pred_log_scaled)}\nReport logistic scaled:\n{classification_report(y_test, pred_log_scaled)}')

log_reg_pca = LogisticRegression(C=1.0, max_iter=1000, solver='liblinear')
log_reg_pca.fit(X_train_pca, y_train)
pred_log_scaled_pca = log_reg_pca.predict(X_test_pca)
print(f'Accuracy logistic scaled PCA:, {accuracy_score(y_test, pred_log_scaled_pca)}\nReport logistic scaled PCA:\n{classification_report(y_test, pred_log_scaled_pca)}')

# The PCA makes models a little less accurate. After looking at all model the Random Forest preformed best in terms of accuracy and precision.
# For the spam filter I would try to minimize false positives. I think it is more important for the important email to not be accidentally marked as spam even if it will result spending a little more time sorting emails from spam.
# And looking at the precision variable in this case the random forest model would be the best choice

cm = confusion_matrix(y_test, pred_rf)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(colorbar=False)
plt.title("RF Confusion Matrix")
plt.savefig('outputs/best_model_confusion_matrix.png')
plt.show()
# The model more often labeling email that are spam to being not spam. So this model more likely to mark spam email as han than marking regular email as spam


# --- PART 4 ---
cv_knn_uns = cross_val_score(knn1, X_train, y_train, cv=5)
print(f'Score KNN Unscaled: {cv_knn_uns}\nMean KNN Unscaled: {cv_knn_uns.mean():.3f}\nStd KNN Unscaled:  {cv_knn_uns.std():.3f}\n')
cv_knn_s = cross_val_score(knn2, X_train_scaled, y_train, cv=5)
print(f'Score KNN Scaled: {cv_knn_s}\nMean KNN Scaled: {cv_knn_s.mean():.3f}\nStd KNN Scaled:  {cv_knn_s.std():.3f}\n')
cv_knn_pca = cross_val_score(knn3, X_train_pca, y_train, cv=5)
print(f'Score KNN PCA: {cv_knn_pca}\nMean KNN PCA: {cv_knn_pca.mean():.3f}\nStd KNN PCA:  {cv_knn_pca.std():.3f}\n')
cv_dec_tree= cross_val_score(dec_tree, X_train, y_train, cv=5)
print(f'Score Decision Tree: {cv_dec_tree}\nMean KNN PCA: {cv_dec_tree.mean():.3f}\nStd Decision Tree:  {cv_dec_tree.std():.3f}\n')
cv_rf= cross_val_score(rf, X_train, y_train, cv=5)
print(f'Score Decision Tree: {cv_rf}\nMean KNN PCA: {cv_rf.mean():.3f}\nStd Decision Tree:  {cv_rf.std():.3f}\n')
cv_log_reg= cross_val_score(log_reg, X_train_scaled, y_train, cv=5)
print(f'Score Log Reg: {cv_log_reg}\nMean Log Reg: {cv_log_reg.mean():.3f}\nStd Log Reg:  {cv_log_reg.std():.3f}\n')
cv_log_reg_pca= cross_val_score(log_reg_pca, X_train_pca, y_train, cv=5)
print(f'Score Log Reg PCA: {cv_log_reg_pca}\nMean Log Reg PCA: {cv_log_reg_pca.mean():.3f}\nStd Log Reg PCA:  {cv_log_reg_pca.std():.3f}\n')

# Based on the results the most accurate model is random forest, while the most stable is PCA logistic regression. And if the most accurate model is RF for both cases, it was unclear with only one split that the pca log reg was the most stable model


# --- PART 5 ---
log_reg_pipeline = Pipeline([
    ("scaler",     StandardScaler()),
    ("classifier", LogisticRegression(C=1.0, max_iter=1000, solver='liblinear'))
])
rf_pipeline = Pipeline([
    ("classifier", RandomForestClassifier(n_estimators=100, random_state=42))
])
log_reg_pipeline.fit(X_train, y_train)
rf_pipeline.fit(X_train, y_train)
lr_pred_pipeline = log_reg_pipeline.predict(X_test)
rf_pred_pipeline = rf_pipeline.predict(X_test)
print(f'Logistic Regression Classification Report Pipeline: \n{classification_report(y_test, lr_pred_pipeline)}')
print(f'Random Forest Classification Report Pipeline: \n{classification_report(y_test, rf_pred_pipeline)}')
# The pipelines are a little bit different because the logistic regression includes the scaler function and could've also included the PCA. 
# This is a much more helpful way to create a model, especially when you would give it to someone else. It would avoid mistakes like leaking the test data set, and it makes sure that everything stays in order and all the steps are clearly visible to other people.