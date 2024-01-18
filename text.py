import numpy as np
from sklearn.ensemble import RandomForestClassifier
from torchvision.datasets import ImageFolder
from torchvision import transforms
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import joblib
from skimage.feature import hog
import cv2

def grad_data(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    fe = hog(gray, orientations=9, pixels_per_cell=(16, 16), cells_per_block=(4, 4),
        block_norm='L2-Hys', transform_sqrt=True, feature_vector=True)

    return fe

dataset = ImageFolder('data4_test', transform=None)
dataset_size = len(dataset)
print(dataset_size)
print(dataset)

X = []
y = []

for imgs, labes in dataset:
    image_array = np.array(imgs)
    arr_transposed = np.transpose(image_array, (1, 2, 0))
    X.append(grad_data(image_array))
    y.append(labes)




model = joblib.load('clf.pkl')

y_pred = model.predict(X)

from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score

# 打印分类准确率
accuracy = np.mean(y_pred == y)
precision = precision_score(y, y_pred, average='weighted')
recall = recall_score(y, y_pred, average='weighted')
f1 = f1_score(y, y_pred, average='weighted')

# 计算混淆矩阵
cm = confusion_matrix(y, y_pred)

# 打印混淆矩阵


plt.figure(figsize=(8, 8))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()

classes = ['Boots', 'Sandals', 'Shoes', 'Slippers']
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=45)
plt.yticks(tick_marks, classes)

plt.xlabel('Predicted Label')
plt.ylabel('True Label')

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, str(cm[i, j]), ha="center", va="center")

plt.show()

print("Confusion Matrix:")
print(cm)
print("\nPrecision: {:.4f}".format(precision))
print("Recall: {:.4f}".format(recall))
print("Accuracy: {:.4f}".format(accuracy))
print("F1-Score: {:.4f}".format(f1))
