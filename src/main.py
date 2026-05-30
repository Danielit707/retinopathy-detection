# src/main.py
import io
import os
import math
import torch
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import torchvision.transforms as transforms
import torchvision.models as models
from fastapi.responses import FileResponse

app = FastAPI(title="APTOS 2019 - Retinopathy Detection API (Ensemble)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
# Ahora apuntamos a la carpeta, no a un solo archivo
WEIGHTS_DIR = os.path.join(PROJECT_ROOT, "weights")

CLASSES = {
    0: "No DR (Healthy)",
    1: "Mild DR",
    2: "Moderate DR",
    3: "Severe DR",
    4: "Proliferative DR"
}

device = torch.device("cpu")

# --- NUEVA FUNCIÓN AUXILIAR DE CARGA ---
def load_single_model(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Weights file not found: '{path}'")
    
    raw_state_dict = torch.load(path, map_location=device)
    cleaned_state_dict = {k.replace("backbone.", ""): v for k, v in raw_state_dict.items()}
    
    model = models.efficientnet_b0(pretrained=False)
    
    in_features = cleaned_state_dict["classifier.1.0.weight"].shape[1]
    hidden_features = cleaned_state_dict["classifier.1.0.weight"].shape[0]
    out_features = cleaned_state_dict["classifier.1.3.weight"].shape[0]
    
    model.classifier[1] = torch.nn.Sequential(
        torch.nn.Linear(in_features, hidden_features),
        torch.nn.Identity(),
        torch.nn.Identity(),
        torch.nn.Linear(hidden_features, out_features)
    )
    
    model.load_state_dict(cleaned_state_dict)
    model.to(device)
    model.eval()
    return model

# --- NUEVA FUNCIÓN PARA EL ENSEMBLE ---
def load_ensemble():
    models_ensemble = []
    print(f"-> Cargando Ensemble desde: {WEIGHTS_DIR}")
    for i in range(5):
        path = os.path.join(WEIGHTS_DIR, f"best_efficientnet_fold_{i}.pth")
        print(f"   - Cargando fold {i}...")
        models_ensemble.append(load_single_model(path))
    print("-> Ensemble cargado exitosamente.")
    return models_ensemble

# Inicializar los 5 modelos
models_ensemble = load_ensemble()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not models_ensemble:
        raise HTTPException(status_code=500, detail="Models not initialized.")
    
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")
    
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        tensor = transform(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            # 1. Obtenemos las 5 predicciones (una de cada fold)
            scores = [m(tensor).item() for m in models_ensemble]
            
            # 2. Promediamos el resultado (el corazón del Ensemble)
            regression_score = sum(scores) / len(scores)
            
            # 3. Lógica de redondeo y probabilidades igual a la que tenías
            prediction = int(max(0, min(4, round(regression_score))))
            
            raw_similarities = [math.exp(-((regression_score - i) ** 2) / 0.6) for i in range(5)]
            total_similarity = sum(raw_similarities)
            probabilities = [sim / total_similarity for sim in raw_similarities]
            
        return {
            "class_id": prediction,
            "diagnosis": CLASSES[prediction],
            "confidence": round(probabilities[prediction] * 100, 2),
            "probabilities": {CLASSES[i]: round(prob * 100, 2) for i, prob in enumerate(probabilities)}
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
        print("--- ERROR DETECTADO ---")
        traceback.print_exc() # Esto imprimirá el error real en los logs de HF
        return {"error": str(e)}, 500
    
@app.get("/", response_class=FileResponse)
async def serve_ui():
    ui_path = os.path.join("src", "static", "index.html")
    if not os.path.exists(ui_path):
        raise HTTPException(status_code=404, detail="UI file not found.")
    return ui_path