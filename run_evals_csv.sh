#!/bin/bash

# --- CONFIGURACIÓN ---
REF_FILE="ref_cifar10.npz"
EVAL_SCRIPT="evaluations/evaluator.py"

# --- VALIDACIONES ---
if [ -z "$1" ]; then
  echo "⚠️  Uso: $0 <model_type>"
  echo "Ejemplos: adaln, cross, incontext"
  exit 1
fi

MODEL_TYPE=$1
BASE_DIR="outputs/${MODEL_TYPE}"
MASTER_LOG="${BASE_DIR}/summary_evaluations.txt"
MASTER_CSV="${BASE_DIR}/results_table.csv"

if [ ! -f "$REF_FILE" ]; then
    echo "❌ Error: No se encuentra el archivo de referencia: $REF_FILE"
    exit 1
fi

if [ ! -d "$BASE_DIR" ]; then
    echo "❌ Error: No existe el directorio: $BASE_DIR"
    exit 1
fi

# --- INICIO ---
echo "========================================"
echo "Evaluando modelo: $MODEL_TYPE"
echo "Generando CSV en: $MASTER_CSV"
echo "========================================"

# 1. Preparar Archivos Maestros
# Texto
echo "Resultados de Evaluación - Modelo: $MODEL_TYPE" > "$MASTER_LOG"
echo "Fecha: $(date)" >> "$MASTER_LOG"
echo "----------------------------------------" >> "$MASTER_LOG"

# CSV: Creamos el encabezado
echo "Model,Checkpoint,Inception Score,FID,sFID,Precision,Recall" > "$MASTER_CSV"

# 2. Bucle de procesamiento
# Usamos sort -V para orden numérico (epoch2 antes que epoch10)
find "$BASE_DIR" -name "samples.npz" | sort -V | while read sample_file; do
    
    sample_dir=$(dirname "$sample_file")
    folder_name=$(basename "$sample_dir")
    local_log="${sample_dir}/eval_results.log"

    echo -n "📊 Procesando $folder_name... "
    
    # --- EJECUCIÓN PYTHON ---
    echo "Evaluación para $folder_name" > "$local_log"
    echo "----------------------------" >> "$local_log"
    
    # Ejecutamos y guardamos salida en log local
    if python "$EVAL_SCRIPT" "$REF_FILE" "$sample_file" >> "$local_log" 2>&1; then
        echo "✅ OK"
        
        # --- PARSEO DE DATOS PARA CSV ---
        # Usamos grep y awk para extraer los números del log local recién creado
        # Asumimos el formato: "Label: Valor"
        
        # Inception Score (campo 2 tras separar por ': ')
        val_is=$(grep "Inception Score:" "$local_log" | awk -F': ' '{print $2}')
        
        # FID (usamos ^FID para que no se confunda con sFID)
        val_fid=$(grep "^FID:" "$local_log" | awk -F': ' '{print $2}')
        
        # sFID
        val_sfid=$(grep "sFID:" "$local_log" | awk -F': ' '{print $2}')
        
        # Precision
        val_prec=$(grep "Precision:" "$local_log" | awk -F': ' '{print $2}')
        
        # Recall
        val_rec=$(grep "Recall:" "$local_log" | awk -F': ' '{print $2}')

        # Escribir fila en el CSV
        # Si algún valor está vacío (error de script), aparecerá como hueco en el CSV
        echo "${MODEL_TYPE},${folder_name},${val_is},${val_fid},${val_sfid},${val_prec},${val_rec}" >> "$MASTER_CSV"

    else
        echo "❌ FALLÓ"
        echo "${MODEL_TYPE},${folder_name},ERROR,ERROR,ERROR,ERROR,ERROR" >> "$MASTER_CSV"
    fi

    # --- GUARDAR EN RESUMEN DE TEXTO ---
    echo "" >> "$MASTER_LOG"
    echo "########################################" >> "$MASTER_LOG"
    echo "Checkpoint: $folder_name" >> "$MASTER_LOG"
    echo "########################################" >> "$MASTER_LOG"
    cat "$local_log" >> "$MASTER_LOG"

done

echo "========================================"
echo "¡Terminado!"
echo "📄 Resumen texto: $MASTER_LOG"
echo "📈 Tabla CSV:     $MASTER_CSV"