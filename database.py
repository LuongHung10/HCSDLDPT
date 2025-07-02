import os
import psycopg2
from dotenv import load_dotenv
import joblib
import numpy as np
from transformers import CLIPTokenizer, CLIPTextModel
import torch

# Load environment variables
load_dotenv()
DB_HOST = os.getenv('DB_HOST')
DB_NAME = os.getenv('DB_NAME')
DB_USER = os.getenv('DB_USER')
DB_PASS = os.getenv('DB_PASS')

IMAGE_DIR = os.path.join('static', 'test')

def connect_db():
    return psycopg2.connect(
        host=DB_HOST,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASS
    )

def create_tables(conn, feature_dim):
    with conn.cursor() as cur:
        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS images (
                id SERIAL PRIMARY KEY,
                name TEXT NOT NULL,
                image_path TEXT NOT NULL UNIQUE,
                image_type TEXT,
                uploaded_at TIMESTAMP DEFAULT NOW(),
                description TEXT,
                description_vector VECTOR(512)
            );
        """)
        cur.execute(f"""
            CREATE TABLE hu_features (
                id SERIAL PRIMARY KEY,
                image_id INTEGER NOT NULL REFERENCES images(id) ON DELETE CASCADE,
                feature_vector VECTOR(7),
                extracted_at TIMESTAMP DEFAULT NOW()
            );
        """)
        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS sk_features (
                id SERIAL PRIMARY KEY,
                image_id INTEGER NOT NULL REFERENCES images(id) ON DELETE CASCADE,
                feature_vector VECTOR(1),
                extracted_at TIMESTAMP DEFAULT NOW()
            );
        """)
        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS hsv_features (
                id SERIAL PRIMARY KEY,
                image_id INTEGER NOT NULL REFERENCES images(id) ON DELETE CASCADE,
                feature_vector VECTOR(512),
                extracted_at TIMESTAMP DEFAULT NOW()
            );
        """)
        conn.commit()

def insert_image(conn, name, image_path, image_type, description, description_vector):
    with conn.cursor() as cur:
        cur.execute("""
            INSERT INTO images (name, image_path, image_type, description, description_vector)
            VALUES (%s, %s, %s, %s, %s)
            RETURNING id;
        """, (name, image_path, image_type, description, description_vector))
        image_id = cur.fetchone()[0]
        conn.commit()
        return image_id

def insert_hu_features(conn, image_id, feature_vector):
    with conn.cursor() as cur:
        cur.execute("""
            INSERT INTO hu_features (image_id, feature_vector)
            VALUES (%s, %s)
        """, (image_id, feature_vector))
        conn.commit()

def insert_sk_features(conn, image_id, feature_vector):
    with conn.cursor() as cur:
        cur.execute("""
            INSERT INTO sk_features (image_id, feature_vector)
            VALUES (%s, %s)
        """, (image_id, feature_vector))
        conn.commit()

def insert_hsv_features(conn, image_id, feature_vector):
    with conn.cursor() as cur:
        cur.execute("""
            INSERT INTO hsv_features (image_id, feature_vector)
            VALUES (%s, %s)
        """, (image_id, feature_vector))
        conn.commit()


def get_clip_text_embedding(text, tokenizer, text_model):
    inputs = tokenizer([text], padding=True, return_tensors="pt")
    with torch.no_grad():
        outputs = text_model(**inputs)
        return outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy()

def upload_all_images():
    conn = connect_db()

    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch16")
    text_model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch16")

    # Load dữ liệu đặc trưng từ file duy nhất
    attributes = joblib.load(open("attribute.jbl", 'rb'))

    create_tables(conn, 256)  # Tạo bảng với cấu trúc cần thiết
    image_idx = 0

    for name in os.listdir(IMAGE_DIR):
        char_dir = os.path.join(IMAGE_DIR, name)
        if not os.path.isdir(char_dir):
            continue
        for filename in os.listdir(char_dir):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_path = os.path.join(char_dir, filename)
                image_type = filename.split('.')[-1]
                description = f"Face of {name}"

                # CLIP embedding
                desc_vec = get_clip_text_embedding(description, tokenizer, text_model)
                desc_vec = [float(x) for x in desc_vec]

                print(f'Uploading {image_path}...')
                image_id = insert_image(conn, name, image_path, image_type, description, desc_vec)

                # Lấy đặc trưng cho ảnh hiện tại
                attribute = attributes[image_idx]

                # Lưu đặc trưng vào bảng tương ứng
                insert_hu_features(conn, image_id, attribute["hu"])
                insert_sk_features(conn, image_id, attribute["sk"])
                insert_hsv_features(conn, image_id, attribute["hsv"])

                image_idx += 1

    conn.close()
    print('All images uploaded.')

if __name__ == '__main__':
    upload_all_images()