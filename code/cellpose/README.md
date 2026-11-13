# Cellpose-SAM — axón + mielina emparejados

Dos modelos Cellpose-SAM (`cpsam`) fine-tuneados:

| modelo | qué segmenta | etiqueta de entrenamiento |
|---|---|---|
| `fiber` | fibra completa (axón ∪ mielina) | `X // 2` para `X >= 2` |
| `axon` | solo el axón | `X // 2` para `X` par `>= 2` |

En inferencia, **mielina = fibra − axón**, así que cada axón `2k` sale emparejado
con su mielina `2k+1` por construcción. Si el modelo de fibras fusiona dos
fibras (dos axones dentro), cada píxel se asigna al axón más cercano y salen dos
pares. Las imágenes se reducen ×0.25 (las fibras quedan en el rango de tamaños
de Cellpose) y las máscaras se devuelven a 3008×4096.

## Archivos

| archivo | qué hace |
|---|---|
| `train.py` | entrena `fiber`, `axon` o los dos (mismo split train/val) |
| `predict.py` | test → TIFF uint16 con el formato de entrega |
| `evaluate.py` | validación con las métricas oficiales (`analysis/metrics.py`) |
| `common.py` | datos, escalado, emparejado fibra/axón, comprobación de formato |
| `wandb_log.py` | registro opcional en W&B (`--wandb`) |

## Instalación

```bash
pip install -r code/cellpose/requirements.txt
```

## Uso (desde la raíz del repo)

```bash
# 1. entrenar (validación = 15 % de las imágenes, seed 0)
python code/train.py --model cellpose --target both --epochs 200 --wandb

# 2. evaluar en validación (Dice, HD95 en nm, error de g-ratio)
python code/cellpose/evaluate.py --wandb

# 3. modelo final con TODAS las imágenes (va a code/checkpoints/cellpose_all)
python code/train.py --model cellpose --target both --epochs 200 --all

# 4. predicciones del test
python code/cellpose/predict.py --images /ruta/test --out predictions \
       --ckpt code/checkpoints/cellpose_all
```

`predict.py` y `evaluate.py` leen la escala de entrenamiento de
`<ckpt>/train_fiber.json`, así que no hay que repetirla. W&B usa el proyecto
`WANDB_PROJECT` (por defecto `tem-challenge`); sin `--wandb` no se envía nada.

## Inferencia: umbral y multiescala

El error dominante en validación eran **fibras enteras no detectadas** (156 de
1094, sobre todo grandes): rellenarlas bajaría el HD95 de 669 a 24 nm, mientras
que quitar falsos positivos o perfeccionar bordes apenas lo mueve. El modelo de
fibras "casi" las ve (cellprob mediano −0.7 frente a 3.4 en las detectadas), así
que por defecto la inferencia:

* usa `cellprob_threshold = -1` (Cellpose usa 0) — `--cellprob`
* hace una 2ª pasada a escala/2 y añade las instancias que no solapan más de un
  20 % con lo ya detectado — `--no-multiscale` para desactivarla

Validación (39 imágenes, métricas oficiales, mismos pesos):

| inferencia | Dice medio | HD95 medio | MAE g-ratio |
|---|---|---|---|
| Cellpose por defecto | 0.828 | 669 nm | 0.0167 |
| multiescala | 0.859 | 528 nm | 0.0163 |
| **cellprob −1 + multiescala (por defecto)** | **0.865** | **535 nm** | **0.0151** |

Los parámetros se eligieron entre 5 variantes mirando estas mismas 39 imágenes,
así que la cifra puede ser algo optimista. La inferencia tarda ~2× (2 pasadas).

## Pesos

`<ckpt>/models/cpsam_tem_fiber` y `cpsam_tem_axon` (~1.2 GB cada uno, no están
en git).
