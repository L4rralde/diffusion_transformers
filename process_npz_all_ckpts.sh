#!/bin/bash

# --- CONFIGURACIÓN DE SEGURIDAD ---
set -e  # Detiene el script si ocurre un error en Python (útil para no seguir si algo falla)

# 1. Validar argumentos
if [ -z "$1" ]; then
  echo "⚠️  Error: Debes especificar el tipo de modelo."
  echo "Uso: $0 <model_type>"
  echo "Ejemplo: $0 adaln"
  exit 1
fi

MODEL_TYPE=$1
NUM_SAMPLES=1000
BASE_DIR="outputs/${MODEL_TYPE}"
SCRIPT_NAME="samples_to_npz.py"

# Verificar que el directorio base existe
if [ ! -d "$BASE_DIR" ]; then
  echo "❌ Error: No existe el directorio $BASE_DIR"
  exit 1
fi

echo "========================================"
echo "Generando NPZ para modelo: $MODEL_TYPE"
echo "Buscando carpetas en: $BASE_DIR"
echo "========================================"

# 2. Buscar carpetas y procesar
# 'find' busca directorios (-type d) que empiecen con "samples_"
# 'sort -V' asegura que samples_epoch2 se procese antes que samples_epoch10
find "$BASE_DIR" -maxdepth 1 -type d -name "samples_*" -print0 | sort -z -V | while IFS= read -r -d '' folder_path; do
    
    folder_name=$(basename "$folder_path")
    
    # 3. Verificación de seguridad: ¿La carpeta está vacía?
    if [ -z "$(ls -A "$folder_path")" ]; then
        echo "⚠️  [SALTANDO] La carpeta está vacía: $folder_name"
        continue
    fi

    echo "------------------------------------------------"
    echo "📂 Procesando carpeta: $folder_name"
    
    # 4. Ejecutar el script de conversión
    python "$SCRIPT_NAME" \
        --image-folder "$folder_path" \
        --num-samples "$NUM_SAMPLES"

    # Opcional: Si el script de python genera el output en la misma carpeta, 
    # aquí podrías moverlo o renombrarlo si quisieras organizar mejor los .npz.
    
done

echo "------------------------------------------------"
echo "✅ ¡Procesamiento de NPZ terminado!"