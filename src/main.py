import io
import os
import torch
import traceback
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
WEIGHTS_DIR = os.path.join(PROJECT_ROOT, "weights")

CLASSES = {
    0: "No DR (Healthy)",
    1: "Mild DR",
    2: "Moderate DR",
    3: "Severe DR",
    4: "Proliferative DR"
}

device = torch.device("cpu")

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

def load_ensemble():
    models_ensemble = []
    print(f"-> Loading Ensemble from target directory: {WEIGHTS_DIR}")
    for i in range(5):
        path = os.path.join(WEIGHTS_DIR, f"best_efficientnet_fold_{i}.pth")
        print(f"   - Loading evaluation check fold {i}...")
        models_ensemble.append(load_single_model(path))
    print("-> Ensemble architecture loaded successfully.")
    return models_ensemble

# Initialize ensemble array
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
            # Create placeholder for accumulating softmax probabilities across models
            ensemble_probs = torch.zeros((1, 5), device=device)
            
            for model in models_ensemble:
                logits = model(tensor)
                probs = torch.softmax(logits, dim=1)
                ensemble_probs += probs
            
            # Compute final mathematical mean distribution
            ensemble_probs /= len(models_ensemble)
            ensemble_probs = ensemble_probs.squeeze(0).tolist()
            
            # Final prediction is the index containing maximum density
            prediction = ensemble_probs.index(max(ensemble_probs))
            confidence = ensemble_probs[prediction] * 100
            
            # --- CLINICAL RISK ANALYSIS SYSTEM ---
            # Sum probability of any positive sign of Retinopathy (Classes 1, 2, 3, 4)
            total_retinopathy_risk = sum(ensemble_probs[1:])
            
            risk_warning = False
            warning_message = "Normal case parameter clearance."
            
            # Risk Mitigation Trigger: If predicted 'Healthy' but alternative DR traces exceed 15%
            RISK_THRESHOLD = 0.15 
            if prediction == 0 and total_retinopathy_risk > RISK_THRESHOLD:
                risk_warning = True
                warning_message = (
                    f"Warning: Borderline Clinical Detection. Although the primary classification is "
                    f"Healthy, the ensemble detects a cumulative risk of {total_retinopathy_risk * 100:.2f}% "
                    f"pointing toward early-stage diabetic retinopathy. Secondary screening is recommended."
                )
                
        return {
            "class_id": prediction,
            "diagnosis": CLASSES[prediction],
            "confidence": round(confidence, 2),
            "probabilities": {CLASSES[i]: round(prob * 100, 2) for i, prob in enumerate(ensemble_probs)},
            "risk_analysis": {
                "risk_warning_triggered": risk_warning,
                "cumulative_dr_probability": round(total_retinopathy_risk * 100, 2),
                "message": warning_message
            }
        }
    except Exception as e:
        print("--- 🔥 INFERENCE EXECUTION FAILURE ---")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
    
@app.get("/", response_class=FileResponse)
async def serve_ui():
    ui_path = os.path.join("src", "static", "index.html")
    if not os.path.exists(ui_path):
        raise HTTPException(status_code=404, detail="UI file not found.")
    return ui_path