import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris, load_digits
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

iris = load_iris(as_frame=True)
X = iris.data
y = iris.target

# --- Preprocessing ---
# Q1
X_train, X_test,y_train, y_test = train_test_split(X,y, test_size=0.2,stratify=y, random_state=42)
print(f'X train: {X_train.shape}\nY train: {y_train.shape}\nX test: {X_test.shape}\nY test: {y_test.shape}\n')

#Q2
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print(X_train_scaled.mean(axis=0),'\n')
# So the model doesn't lean on the training data as well, and the test data stays truly unseen


# --- KNN ---
# Q1
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
preds = knn.predict(X_test)
print(f'Accuracy:, {accuracy_score(y_test, preds)}\n Report: {classification_report(y_test, preds)}')

#Q2
knn.fit(X_train_scaled, y_train)
preds_scaled = knn.predict(X_test_scaled)
print(f'Accuracy Scaled:, {accuracy_score(y_test, preds_scaled)}\n')
# The scaling made accuracy worse, this happened because the dataset already had clear natural seperation.

#Q3
cv_scores = cross_val_score(knn, X_train, y_train, cv=5)
print(f'Score: {cv_scores}\nMean: {cv_scores.mean():.3f}\nStd:  {cv_scores.std():.3f}')
# The results are more trustworthy because it test several splits and show that not all of them show an accuracy of 1. So there are some variations depending on which split is used

#Q4
k_val = [1, 3, 5, 7, 9, 11, 13, 15]
for k in k_val:
    knn_t = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn_t, X_train, y_train, cv=5)
    print(f"k={k:2d}:  mean={scores.mean():.3f}  std={scores.std():.3f}")
# k=7 Would be the best choice because it has highest mean accuracy and lower variation


# --- Classifier Evaluation ---
# Q1
cm = confusion_matrix(y_test, preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=iris.target_names)
disp.plot(colorbar=False)
plt.title("Iris's Confusion Matrix")
plt.savefig('outputs/knn_confusion_matrix.png')
plt.show()
# The doesn't confuse any species which is expected based on the previous accuracy


# --- Decision Trees ---
dec_tree = DecisionTreeClassifier(max_depth=3, random_state=42)
dec_tree.fit(X_train, y_train)
tree_pred = dec_tree.predict(X_test)
print(f'Accuracy:, {accuracy_score(y_test, tree_pred)}\n Report: {classification_report(y_test, tree_pred)}')
# The decision tree has good accuracy, better than scaled knn, though the unscaled knn performed much better. 
# The decision tree doesn't rely on the distance as much as knn. There still could be some differences in the scaled vs. unscaled data, but really small


# --- Logistic Regression ---
c_val = [0.01,1,100]
for c in c_val:
    log_reg = OneVsRestClassifier(LogisticRegression(C=c, max_iter=1000, solver='liblinear'))
    log_reg.fit(X_train_scaled, y_train)
    print(f'C={c}, Coefficient size={sum(np.abs(est.coef_).sum()for est in log_reg.estimators_)}')
#  The penalty of C=0.01 puts heavier regularization to the coef, which shrinks it toward 0, while 100 puts almost no regularization on a coef. It controls complexity of the model doesn't letting it overfit the data. 

# --- PCA ---
digits = load_digits()
X_digits = digits.data
y_digits = digits.target
images   = digits.images 
# Q1
print(f'Shape X_digits: {X_digits.shape}\nShape images: {images.shape}')
fig, axes = plt.subplots(1, 10)
for num in range(10):
    mean_image = images[y_digits == num].mean(axis=0)
    axes[num].imshow(mean_image, cmap='gray_r')
    axes[num].set_title(str(num))
    axes[num].axis('off')
plt.tight_layout()
plt.savefig('outputs/sample_digits.png')
plt.show()

# Q2
pca = PCA()
pca.fit(X_digits)
scores = pca.transform(X_digits)
scatter = plt.scatter(scores[:, 0], scores[:, 1], c=y_digits, cmap='tab10', s=10)
plt.colorbar(scatter, label='Digit')
plt.savefig('outputs/pca_2d_projection.png')
plt.show()
# Yes but it's not always clear. 0's clearly are cluster together while 5 looks a bit spread

# Q3
cum_var = np.cumsum(pca.explained_variance_ratio_)
plt.plot(range(1, len(cum_var) + 1), cum_var)
plt.xlabel("number of components")
plt.ylabel("cumulative explained variance")
plt.title("cumulative explained variance vs. number of components")
plt.savefig('outputs/pca_variance_explained.png')
plt.show()
# You need around 12-13 components to explain 80% of the variance

# Q4
def reconstruct_digit(sample_idx, scores, pca, n_components):
    """Reconstruct one digit using the first n_components principal components."""
    reconstruction = pca.mean_.copy()
    for i in range(n_components):
        reconstruction = reconstruction + scores[sample_idx, i] * pca.components_[i]
    return reconstruction.reshape(8, 8)

n = [2, 5, 15, 40]
dig = X_digits[:5]
fig, axes = plt.subplots(5, 5)
for col, row in enumerate(dig):
    axes[0, col].imshow(row.reshape(8, 8), cmap='gray_r')
    axes[0, col].set_title(y_digits[col])
    axes[0, col].axis('off')

for r, num_comp in enumerate(n, start=1):
    for col in range(5):
        axes[r, col].imshow(reconstruct_digit(col, scores, pca, num_comp), cmap='gray_r')
        axes[r, col].axis('off')
plt.savefig('outputs/pca_reconstructions.png')
plt.tight_layout()
plt.show()
# At n = 5, the numbers could be recognized but not clearly, at n = 15, the numbers become very recognizable, which matches the variance component