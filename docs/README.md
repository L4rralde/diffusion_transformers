# Generacion de imagenes 64x64 CIFAR-10 con Diffusion Transformer DiT-S

<div align="center">
<img src="https://github.com/L4rralde/diffusion_transformers/blob/main/docs/image.jpg" width="600"/>
</div>

## Instalacion

```sh
git clone --recursive https://github.com/L4rralde/diffusion_transformers.git
cd diffusion_transformers
python -m venv dit.venv
source dit.venv/bin/activate
pip install -r requirements.txt
```

## Entrenar modelos

Implementamos 3 arquitecturas:

- [x] AdaLN-Zero
- [x] Cross-Attention
- [x] In-context conditioning

Valores posibles de `<model_type>`: adaln, cross, incontext


```sh
python train_dit.py --model-type <model_type>  --results-dir outputs/<model_type>/results
```

Ejemplo (adaln):

```sh
python train_dit.py --model-type adaln  --results-dir outputs/adaln/results
```

O puedes descargar nuestros *checkpoints*:

- [dit_adaln_epoch100_ema.pt](https://github.com/L4rralde/diffusion_transformers/releases/download/weights/dit_adaln_epoch100_ema.pt)
- [dit_cross_epoch100_ema.pt](https://github.com/L4rralde/diffusion_transformers/releases/download/weights/dit_cross_epoch100_ema.pt)
- [dit_incontext_epoch100_ema.pt](https://github.com/L4rralde/diffusion_transformers/releases/download/weights/dit_incontext_epoch100_ema.pt)

## Evaluacion

### Genera muestras para evaluación


```sh
python sample_dit.py --model-type <model_type> --num-samples 1000 --ckpt outputs/<model_type>/results/dit_last_ema.pt --out-dir outputs/<model_type>/samples
```

### Convierte muestras en npz para evaluación


```sh
python samples_to_npz.py --image-folder outputs/<model_type>/samples --num-samples 1000
```

### Obtén métricas de generación

**Primero** Instala dependencias para el script de evaluacion:

```sh
pip install -r eval_requirements.txt
```

Tambien **es necesario** descargar el [archivo de referencia](https://github.com/L4rralde/diffusion_transformers/releases/download/cifar10_test.npz/ref_cifar10.npz)

Script de evaluacion:


```sh
python evaluations/evaluator.py ref_cifar10.npz outputs/<model_type>/samples/samples.npz
```

