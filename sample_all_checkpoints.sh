#!/bin/bash

# --- CONFIGURACIÓN DE SEGURIDAD ---
set -e  # Si ocurre cualquier error, el script se detiene inmediatamente.
# ----------------------------------

# 1. Validar argumentos
if [ -z "$1" ]; then
  echo "⚠️  Uso: $0 <model_type>"
  exit 1
fi

MODEL_TYPE=$1
RESULTS_DIR="outputs/${MODEL_TYPE}/results"
OUT_BASE="outputs/${MODEL_TYPE}"
NUM_SAMPLES=1000

# Verificar existencia de directorio de entrada
if [ ! -d "$RESULTS_DIR" ]; then
  echo "❌ Error crítico: No existe el directorio $RESULTS_DIR"
  exit 1
fi

echo "🛡️  Iniciando proceso SEGURO para modelo: $MODEL_TYPE"
echo "📂 Leyendo desde: $RESULTS_DIR"

# 2. Bucle Seguro (Maneja espacios y ordena numéricamente)
# Usamos 'find' + 'sort' + 'while read' para máxima seguridad en nombres de archivo
find "$RESULTS_DIR" -maxdepth 1 -name "*_ema.pt" -print0 | sort -z -V | while IFS= read -r -d '' ckpt_path; do
    
    filename=$(basename "$ckpt_path")
    
    # Detección de época
    if [[ "$filename" =~ epoch([0-9]+) ]]; then
        epoch="${BASH_REMATCH[1]}"
        out_dir="${OUT_BASE}/samples_${epoch}_ema"
    elif [[ "$filename" =~ last ]]; then
        epoch="last"
        out_dir="${OUT_BASE}/samples_last_ema"
    else
        echo "⚠️  Saltando archivo desconocido: $filename"
        continue
    fi

    # 3. Protección contra sobrescritura
    if [ -d "$out_dir" ]; then
        # Verificamos si la carpeta tiene archivos dentro
        if [ "$(ls -A "$out_dir")" ]; then
            echo "⏭️  [OMITIDO] La carpeta ya existe: $out_dir"
            continue
        fi
    fi

    echo "------------------------------------------------"
    echo "Processing: $filename"
    echo "Target: $out_dir"
    
    # 4. Ejecución
    # El comando 'python' se ejecuta con las rutas entre comillas para evitar errores de shell
    python sample_dit.py \
        --model-type "$MODEL_TYPE" \
        --num-samples "$NUM_SAMPLES" \
        --ckpt "$ckpt_path" \
        --out-dir "$out_dir" \
        --seed 1234

done

echo "------------------------------------------------"
echo "✅ Proceso completado exitosamente."