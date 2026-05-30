# generate_dummy_data.py
import os
import pandas as pd
import numpy as np
from PIL import Image

def create_dummy_dataset():
    print("Generating dummy dataset for pipeline testing...")
    
    # Define paths
    raw_dir = os.path.join("data", "raw")
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs("weights", exist_ok=True) # Also ensure weights directory exists
    
    num_samples = 40
    ids = []
    diagnoses = []
    
    # Generate fake images and labels (0 to 4)
    for i in range(num_samples):
        img_id = f"dummy_{i:03d}"
        ids.append(img_id)
        # Random diagnosis class between 0 (No DR) and 4 (Proliferative DR)
        diagnoses.append(np.random.randint(0, 5)) 
        
        # Create a random RGB image (e.g., 500x500) to simulate raw data
        random_image_array = np.random.randint(0, 255, (500, 500, 3), dtype=np.uint8)
        img = Image.fromarray(random_image_array)
        img.save(os.path.join(raw_dir, f"{img_id}.png"))
        
    # Create DataFrame
    df = pd.DataFrame({
        "id_code": ids,
        "diagnosis": diagnoses
    })
    
    # Split into dummy train and validation sets (80% / 20%)
    train_df = df.sample(frac=0.8, random_state=42)
    val_df = df.drop(train_df.index)
    
    # Save CSV files
    train_df.to_csv(os.path.join("data", "train.csv"), index=False)
    val_df.to_csv(os.path.join("data", "val.csv"), index=False)
    
    print(f"Successfully generated {num_samples} dummy images in '{raw_dir}' and CSV files in 'data/'.")

if __name__ == "__main__":
    create_dummy_dataset()