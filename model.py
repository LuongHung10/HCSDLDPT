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
    """Trích xuất Hu Moments từ ảnh xám."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    moments = cv2.moments(gray)
    hu = cv2.HuMoments(moments).flatten()
    # Chuyển đổi Hu Moments sang log-scale để ổn định giá trị
    hu_log = -np.sign(hu) * np.log10(np.abs(hu) + 1e-8)
    return hu_log

def extract_skeleton_features(img):
    """Trích xuất đặc trưng Skeleton từ ảnh nhị phân."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 127, 1, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    skeleton = skeletonize(binary).astype(np.uint8)
    # Tính tổng số pixel của skeleton
    skeleton_sum = np.sum(skeleton)
    return np.array([skeleton_sum], dtype=np.float32)

def extract_hsv_histogram(img, bins=(8, 8, 8)):
    """Trích xuất HSV histogram từ ảnh."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, bins, [0, 180, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    return hist

def extract_features(img_path):
    """Trích xuất các đặc trưng từ ảnh."""
    if not os.path.exists(img_path):
        print(f"[Error] File does not exist: {img_path}")
        return None
    try:
        # Đọc và kiểm tra ảnh
        img = cv2.imread(img_path)
        if img is None:
            print(f"[Error] Cannot read image: {img_path}")
            return None
        
        # Trích xuất từng đặc trưng
        hu = extract_hu_moments(img)
        skeleton = extract_skeleton_features(img)
        hsv_hist = extract_hsv_histogram(img)
        
        # Chuẩn hóa từng đặc trưng
        hu = (hu - np.min(hu)) / (np.max(hu) - np.min(hu) + 1e-8)  # Chuẩn hóa Hu Moments
        skeleton = skeleton / (np.max(skeleton) + 1e-8)           # Chuẩn hóa Skeleton
        hsv_hist = hsv_hist / (np.sum(hsv_hist) + 1e-8)           # Chuẩn hóa HSV Histogram
        
        # Kết hợp các đặc trưng thành một vector duy nhất
        feature = np.concatenate([hu, skeleton, hsv_hist])
        return feature
    except Exception as e:
        print(f"[Error] Error processing {img_path}: {e}")
        return None


from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans

def initialize_kmeans(attribute_list, n_clusters=5):
    """
    Khởi tạo và huấn luyện K-Means trên attribute_list.
    """
    attr_matrix = np.array(attribute_list)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(attr_matrix)
    return kmeans

# Tìm kiếm ảnh tương tự kết hợp K-Means và Cosine Similarity
def similar_images_kmeans(query_features, kmeans, top_k=3):
    """
    Tìm kiếm ảnh tương tự bằng cách kết hợp K-Means và Cosine Similarity.
    """
    if query_features is None or len(query_features) == 0:
        print("Query features are invalid")
        return []
    
    # Chuẩn hóa vector truy vấn
    query_norm = normalize(query_features.reshape(1, -1))
    
    # Tìm cụm gần nhất bằng K-Means
    cluster_label = kmeans.predict(query_norm)[0]
    
    # Lọc các ảnh trong cụm tương ứng
    attr_matrix = np.array(attribute_list)
    cluster_indices = np.where(kmeans.labels_ == cluster_label)[0]
    cluster_attr = attr_matrix[cluster_indices]
    cluster_images = [all_images[i] for i in cluster_indices]

    # Chuẩn hóa các vector đặc trưng trong cụm
    cluster_attr_norm = normalize(cluster_attr)

    # Tính cosine similarity trong cụm
    sims = (cluster_attr_norm @ query_norm.T).flatten()
    top_idx = np.argsort(sims)[::-1][:top_k]

    # Trả về kết quả
    results = []
    for i, idx in enumerate(top_idx):
        img_path = cluster_images[idx].replace('\\', '/')
        leaf = os.path.basename(os.path.dirname(img_path))
        display_path = img_path.replace('static/', '')
        results.append({
            'path': display_path,
            'leaf': leaf,
            'score': float(sims[idx])  # Cosine similarity, nằm trong [0,1]
        })

    return results