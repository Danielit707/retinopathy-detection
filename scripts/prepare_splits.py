import os
import pandas as pd
from sklearn.model_selection import StratifiedKFold

def main():
    INPUT_CSV = os.path.join("data", "train.csv")
    FOLDS_DIR = os.path.join("data", "folds")
    
    # Crear un directorio limpio para almacenar los CSVs de los folds
    os.makedirs(FOLDS_DIR, exist_ok=True)
    
    if not os.path.exists(INPUT_CSV):
        print(f"[-] Error: No se encontró el archivo base en {INPUT_CSV}")
        return

    df = pd.read_csv(INPUT_CSV)
    
    # Configurar el particionador estratificado
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    print("[+] Generando archivos físicos para las 5 particiones de Cross-Validation...")
    
    for fold_id, (train_idx, val_idx) in enumerate(skf.split(df, df['diagnosis'])):
        # Aislar la data correspondiente a este fold
        df_train = df.iloc[train_idx]
        df_val = df.iloc[val_idx]
        
        # Definir rutas específicas para este fold
        fold_train_csv = os.path.join(FOLDS_DIR, f"train_fold_{fold_id}.csv")
        fold_val_csv = os.path.join(FOLDS_DIR, f"val_fold_{fold_id}.csv")
        
        # Guardar de forma definitiva en el disco
        df_train.to_csv(fold_train_csv, index=False)
        df_val.to_csv(fold_val_csv, index=False)
        
        print(f" -> Fold {fold_id} exportado: {fold_train_csv} y {fold_val_csv}")

    print("[+] ¡Todas las particiones listas y almacenadas en data/folds/!")

if __name__ == "__main__":
    main()