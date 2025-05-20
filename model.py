import os
import joblib
import numpy as np
from PIL import Image
import torch
from torchvision import models, transforms
from sklearn.metrics.pairwise import cosine_similarity
import psycopg2

# Load model ResNet18
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet18(pretrained=True)
model = torch.nn.Sequential(*list(model.children())[:-1])  # Bỏ layer phân loại
model = model.to(device)
model.eval()

# Transform cho ResNet18
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Load dữ liệu đã lưu
all_images = joblib.load('all_imag.jbl')
attribute_list = joblib.load('attribute.jbl')  # Đã là list các vector 512

def extract_features(img_path):
    """Trích xuất đặc trưng từ ảnh bằng ResNet18, xử lý trực tiếp nếu là ảnh đã cắt"""
    try:
        if not os.path.exists(img_path):
            print(f"File does not exist: {img_path}")
            return None

        img = Image.open(img_path).convert('RGB')
        img_tensor = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            result = model(img_tensor)
        features = result.cpu().numpy().flatten()  # 512-dim vector
        print(f"Features extracted for {img_path}")
        return features

    except Exception as e:
        print(f"General error processing image {img_path}: {str(e)}")
        return None

def find_similar_images(query_features):
    """Tìm kiếm 3 ảnh tương tự nhất và trả về đường dẫn cùng tên nhân vật"""
    if query_features is None or len(query_features) == 0:
        print("Query features are invalid")
        return []

    similarities = []
    for i in range(len(attribute_list)):
        sim_score = cosine_similarity(query_features.reshape(1, -1), np.array(attribute_list[i]).reshape(1, -1))[0][0]
        similarities.append((i, sim_score))

    top_indices = sorted(similarities, key=lambda x: x[1], reverse=True)[:3]

    results = []
    for idx, score in top_indices:
        img_path = all_images[idx]
        img_path = img_path.replace('\\', '/')
        character_name = os.path.basename(os.path.dirname(img_path))
        display_path = img_path.replace('static/', '')
        results.append({
            'path': display_path,
            'character': character_name,
            'score': float(score)
        })

    return results

def find_similar_images_sql(query_features, top_k=3):
    conn = psycopg2.connect(
        host=os.getenv('DB_HOST'),
        dbname=os.getenv('DB_NAME'),
        user=os.getenv('DB_USER'),
        password=os.getenv('DB_PASS')
    )
    with conn.cursor() as cur:
        # Chuyển vector numpy sang chuỗi cho truy vấn pgvector
        vector_str = '[' + ','.join(str(float(x)) for x in query_features) + ']'
        cur.execute('''
            SELECT i.image_path, i.name, f.feature_vector <=> %s::vector AS distance
            FROM image_features f
            JOIN images i ON f.image_id = i.id
            WHERE f.model_name = 'resnet18'
            ORDER BY distance ASC
            LIMIT %s;
        ''', (vector_str, top_k))
        results = cur.fetchall()
    conn.close()
    # Trả về danh sách dict giống hệt find_similar_images
    formatted_results = []
    for row in results:
        img_path = row[0].replace('\\', '/').replace('static/', '')
        character_name = row[1]
        display_path = img_path  # đã loại static/ ở trên
        formatted_results.append({
            'path': display_path,
            'character': character_name,
            'score': float(1-row[2])
        })
    return formatted_results