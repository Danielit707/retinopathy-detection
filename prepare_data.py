import glob
import os
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm

print("Redimensionando imágenes en el disco a 224x224...")
images = glob.glob('data/raw/*.png')
for p in tqdm(images):
    try:
        with Image.open(p) as img:
            img.resize((224, 224), Image.Resampling.LANCZOS).save(p)
    except Exception as e:
        print(f"Error con la imagen {p}: {e}")

print("\nDividiendo el archivo CSV (Stratified Split)...")
df = pd.read_csv('data/train.csv')
train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['diagnosis'], random_state=42)

train_df.to_csv('data/train.csv', index=False)
val_df.to_csv('data/val.csv', index=False)
print(f"¡Dataset listo! Entrenamiento: {len(train_df)} | Validación: {len(val_df)}")