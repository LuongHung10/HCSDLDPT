import os
import joblib
import numpy as np
from PIL import Image
import cv2
from skimage.morphology import skeletonize
from sklearn.metrics.pairwise import cosine_similarity

# Load data
all_images = joblib.load('all_imag.jbl')
attribute_list = joblib.load('attribute.jbl')

def extract_hu_moments(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    moments = cv2.moments(gray)
    hu = cv2.HuMoments(moments).flatten()
    return hu

def extract_skeleton_features(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 127, 1, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    skeleton = skeletonize(binary).astype(np.uint8)
    # Use the sum of skeleton pixels as a simple feature (or flatten for more detail)
    skeleton_sum = np.sum(skeleton)
    return np.array([skeleton_sum], dtype=np.float32)

def extract_hsv_histogram(img, bins=(8, 8, 8)):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, bins,
                        [0, 180, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    return hist

def extract_features(img_path):
    if not os.path.exists(img_path):
        print(f"File does not exist: {img_path}")
        return None
    try:
        img = cv2.imread(img_path)
        if img is None:
            print(f"Cannot read image: {img_path}")
            return None
        hu = extract_hu_moments(img)
        skeleton = extract_skeleton_features(img)
        hsv_hist = extract_hsv_histogram(img)
        # Chuẩn hóa từng phần
        hu = (hu - np.min(hu)) / (np.max(hu) - np.min(hu) + 1e-8)
        skeleton = skeleton / (np.max(skeleton) + 1e-8)
        hsv_hist = hsv_hist / (np.sum(hsv_hist) + 1e-8)
        feature = np.concatenate([hu, skeleton, hsv_hist])
        return feature
    except Exception as e:
        print(f"Error processing {img_path}: {e}")
        return None


from sklearn.preprocessing import normalize

def similar_images(query_features, top_k=3):
    if query_features is None or len(query_features) == 0:
        print("Query features are invalid")
        return []
    # Chuẩn hóa vector truy vấn
    query_norm = normalize(query_features.reshape(1, -1))
    # Chuẩn hóa toàn bộ attribute_list
    attr_matrix = np.array(attribute_list)
    attr_norm = normalize(attr_matrix)
    # Tính cosine similarity
    sims = (attr_norm @ query_norm.T).flatten()
    top_idx = np.argsort(sims)[::-1][:top_k]
    results = []
    for idx in top_idx:
        img_path = all_images[idx].replace('\\', '/')
        leaf = os.path.basename(os.path.dirname(img_path))
        display_path = img_path.replace('static/', '')
        results.append({
            'path': display_path,
            'leaf': leaf,
            'score': float(sims[idx])  # Đã là cosine similarity, nằm trong [0,1]
        })
    return results