# FastAPI
from fastapi import FastAPI
from pydantic import BaseModel
import base64, io
from PIL import Image
import numpy as np
from fastapi.responses import FileResponse
from MDP_function import MDP
import os


# TensorFlowモデルのテスト読み込み
try:
    from tensorflow.keras.models import load_model
    model = load_model("resnet_mnist.keras")
    print("Model loaded successfully!")
    MODEL_LOADED = True
except Exception as e:
    print(f"Error loading model: {e}")
    model = None
    MODEL_LOADED = False

app = FastAPI()

class ImageData(BaseModel):
    image: str

from fastapi.staticfiles import StaticFiles

# 静的ファイルを /public にマウント
app.mount("/public", StaticFiles(directory=os.path.join(os.path.dirname(__file__), "..", "public")), name="public")

# ルートで index.html を返す
@app.get("/", response_class=FileResponse)
async def root():
    return FileResponse("public/index.html", media_type="text/html")


@app.post("/predict")
def predict(data: ImageData):
    try:
        print("Starting prediction process...")
        
        # Base64デコード
        img_data = data.image.split(",")[1]
        img_bytes = base64.b64decode(img_data)
        img = Image.open(io.BytesIO(img_bytes)).convert("L")
        img_array = np.array(img, dtype=np.uint8)
        
        #print("Original img_array shape:", img_array.shape)
        #print("Original img_array mean:", np.mean(img_array))
        
        # 反転処理
        img_array = 255 - img_array
        #print("After inversion - img_array mean:", np.mean(img_array))
        
        # 画像保存（テスト用）
        #cv2.imwrite("processed_image.png", img_array)
        #print("Image saved successfully")
        
        # モデルが読み込まれているかテスト
        if not MODEL_LOADED:
            return {"result": "Model not loaded", "error": "TensorFlow model failed to load"}
        
        result = MDP(img_array)
        return {"result": result}

    except Exception as e:
        print(f"Error in predict function: {str(e)}")
        print(f"Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        return {"error": f"Prediction failed: {str(e)}"}