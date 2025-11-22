import base64
from io import BytesIO
from fastapi import FastAPI, UploadFile
from contextlib import asynccontextmanager
from fastapi.params import File
from fastapi.responses import HTMLResponse
from keras.models import load_model
from PIL import Image
import numpy as np

ml_models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    ml_models['maize_disease_model_v5'] = load_model('app/model/maize_disease_model_v5.keras')
    yield
    ml_models.clear()

app = FastAPI(lifespan=lifespan)

class_names = [
    'Blight',
    'Common_Rust', 
    'Gray_Leaf_Spot', 
    'Healthy'
]

def preprocess_image(image):
    img = Image.open(BytesIO(image))
    img = img.resize((256, 256))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.post("/predict/", response_class=HTMLResponse)
async def predict(file: UploadFile = File(...)):

    content = await file.read()
    img_array = preprocess_image(content)

    predictions = ml_models['maize_disease_model_v5'].predict(img_array)
    predicted_class_index = np.argmax(predictions[0])
    pred_class = class_names[predicted_class_index]
    confidence = float(predictions[0][predicted_class_index]*100)

    # Convert image to base64 for showing in HTML
    encoded = base64.b64encode(content).decode("utf-8")

    return f"""
    <html>
    <head>
        <style>
            body {{
                font-family: Arial;
                background: #f2f7f5;
                display: flex;
                justify-content: center;
                align-items: center;
                height: 100vh;
                margin: 0;
            }}
            .result-card {{
                background: white;
                padding: 30px;
                border-radius: 15px;
                box-shadow: 0 4px 20px rgba(0,0,0,0.15);
                text-align: center;
                width: 400px;
            }}
            img {{
                width: 280px;
                border-radius: 10px;
                margin-bottom: 20px;
                box-shadow: 0 2px 12px rgba(0,0,0,0.1);
            }}
            .label {{
                font-size: 20px;
                color: #333;
                font-weight: bold;
            }}
            .confidence {{
                font-size: 18px;
                color: #555;
                margin-bottom: 20px;
            }}
            a {{
                display: inline-block;
                margin-top: 10px;
                text-decoration: none;
                padding: 10px 15px;
                background: #2196F3;
                color: white;
                border-radius: 8px;
            }}
            a:hover {{
                background: #0b7dda;
            }}
        </style>
    </head>
    <body>
        <div class="result-card">
            <h2>Prediction Result 🌾</h2>

            <img src="data:image/jpeg;base64,{encoded}" alt="Uploaded Image">

            <p class="label">Predicted Class: {pred_class}</p>
            <p class="confidence">Confidence: {confidence:.2f}%</p>

            <a href="/">🔙 Upload Another Image</a>
        </div>
    </body>
    </html>
    """

@app.get("/", response_class=HTMLResponse)
async def homepage():
    return """
    <html>
    <head>
        <style>
            body {
                font-family: Arial;
                background: #f2f7f5;
                display: flex;
                justify-content: center;
                align-items: center;
                height: 100vh;
                margin: 0;
            }
            .card {
                background: white;
                padding: 40px;
                border-radius: 15px;
                box-shadow: 0 4px 20px rgba(0,0,0,0.15);
                text-align: center;
                width: 350px;
            }
            h2 {
                color: #333;
                margin-bottom: 20px;
            }
            input[type="file"] {
                margin-top: 15px;
                margin-bottom: 20px;
            }
            button {
                background: #4CAF50;
                color: white;
                border: none;
                padding: 12px 20px;
                border-radius: 8px;
                cursor: pointer;
                font-size: 16px;
            }
            button:hover {
                background: #45a049;
            }
        </style>
    </head>
    <body>
        <div class="card">
            <h2>Maize Leaf Disease Classifier 🌿</h2>
            <form action="/predict/" method="post" enctype="multipart/form-data">
                <input name="file" type="file" accept="image/*" required>
                <br>
                <button type="submit">Predict</button>
            </form>
        </div>
    </body>
    </html>
    """

