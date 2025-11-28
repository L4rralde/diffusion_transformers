

### Entrenar modelos

**DiT adaLN-Zero**

```sh
python train_dit.py --model-type adaln  --results-dir outputs/adaln/results
```

**DiT Cross Attention**

```sh
python train_dit.py --model-type cross  --results-dir outputs/cross/results
```

**DiT In Context**

```sh
python train_dit.py --model-type incontext  --results-dir outputs/incontext/results
```

### Genera muestras para evaluación

**DiT adaLN-Zero**

```sh
python sample_dit.py --model-type adaln --num-samples 1000 --ckpt outputs/adaln/results/dit_last_ema.pt --out-dir outputs/adaln/samples
```

### Convierte muestras en npz para evaluación

**DiT adaLN-Zero**

```sh
python samples_to_npz.py --image-folder outputs/adaln/samples --num-samples 1000
```

### Obtén métricas de generación

**DiT adaLN-Zero**

```sh
python evaluations/evaluator.py ref_cifar10.npz outputs/adaln/samples/samples.npz
```
