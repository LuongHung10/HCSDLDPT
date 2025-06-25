from flask import Flask, render_template, request, redirect, url_for, flash, session
import os
from werkzeug.utils import secure_filename
from model import extract_features, similar_images_kmeans, initialize_kmeans
import secrets
import joblib

# Load data và khởi tạo K-Means
all_images = joblib.load('all_imag.jbl')
attribute_list = joblib.load('attribute.jbl')
kmeans_model = initialize_kmeans(attribute_list, n_clusters=5)

app = Flask(__name__)
app.secret_key = os.getenv('FLASK_SECRET_KEY', secrets.token_hex(32))

UPLOAD_FOLDER = 'static/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def index():
    results = session.get('results', [])
    uploaded_image = session.get('uploaded_image', None)
    error_message = None

    if request.method == 'POST':
        print("Received POST request")

        # Xử lý upload ảnh
        if 'file' in request.files and request.files['file'].filename != '':
            file = request.files['file']
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(file_path)
                uploaded_image = filename
                print(f"Uploaded image saved at: {file_path}")

                # Trích xuất đặc trưng
                features = extract_features(file_path)
                if features is not None:
                    print("Features extracted successfully")
                    # Sử dụng K-Means để tìm kiếm ảnh
                    similar_imgs = similar_images_kmeans(features, kmeans_model, top_k=3)
                    results = similar_imgs
                    print(f"Found similar images: {results}")
                else:
                    error_message = "Không thể xử lý ảnh đầu vào. Vui lòng đảm bảo ảnh hợp lệ."
                    flash(error_message, 'error')
                    print("Failed to extract features")

                # Lưu kết quả vào session
                session['results'] = results
                session['uploaded_image'] = uploaded_image

        # Xử lý reset
        if 'reset' in request.form:
            print("Reset requested")
            # Xóa ảnh đã upload
            for f in os.listdir(app.config['UPLOAD_FOLDER']):
                try:
                    os.remove(os.path.join(app.config['UPLOAD_FOLDER'], f))
                except Exception as e:
                    print(f"Error removing file {f}: {e}")
            # Xóa session
            session.pop('results', None)
            session.pop('uploaded_image', None)
            flash("Đã xóa kết quả và ảnh upload.", 'info')
            return redirect(url_for('index'))

        return redirect(url_for('index'))

    print("Rendering template")
    return render_template('index.html', results=results, uploaded_image=uploaded_image)

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)
