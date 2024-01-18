import joblib
import cv2
import matplotlib.pyplot as plt
from skimage.feature import hog
import numpy as np

def grad_data(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    fe = hog(gray, orientations=9, pixels_per_cell=(16, 16), cells_per_block=(4, 4),
        block_norm='L2-Hys', transform_sqrt=True, feature_vector=True)

    return fe

# 读取图片
image_path = "data4_test/Slippers/105197.20.jpg"

image = cv2.imread(image_path)

lables = ['Boots', 'Sandals', 'Shoes', 'Slippers']

image_array = np.array(image)
X = grad_data(image_array)
model = joblib.load('clf.pkl')
y_pred = model.predict([X])

plt.imshow(image)
plt.title(lables[y_pred[0]])
plt.show()

