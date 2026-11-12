# Experimentos (todos)

Dataset 505 si no se indica. HD95 global en nm; fold 0 en los runs con validación.

| # | qué probamos | resultado |
|---|---|---|
| 1 | nnU-Net base (sin extras), 20 ep | 750.0 nm, Dice 0.862. Referencia. |
| 2 | Backbone ResEncM | 838.9 nm. Peor: su patch más chico pierde contexto. |
| 3 | Resolución completa (dataset 501) | OOM y divergencia (NaN). Descartado. |
| 4 | 3 canales a resolución completa (dataset 506) | 760.6 raw / 629.0 con borde. Peor que media resolución. |
| 5 | Cabeza radial tipo Stardist | 765.9 nm (cluster). Peor que el baseline. |
| 6 | Stardist alternativo | 778 nm. Peor. |
| 7 | Cierre de anillo (RingClosure) | 885.7 nm. Peor. |
| 8 | Radial + aceptación (NestedAccept) | 755.6 nm. Sumar la aceptación no mejora. |
| 9 | Aceptación con target IoU, 60 ep | ⚠️ AUC 0.861 **INVÁLIDO** (cabeza aleatoria, ver abajo). |
| 10 | Aceptación con target regla (anillo roto), 60 ep | 721.4 raw / 574.6 borde. El AUC 0.876 era **ruido**; el resultado del borde sí es válido. |
| 11 | Comité: juez de anillo + juez de borde | ⚠️ Comparación **INVÁLIDA** (comparaba cabezas aleatorias). El juez de borde tampoco descartaba nada. |
| 12 | Crítico adversarial | No aprendió. Descartado. |
| 13 | Descartar como cuarta clase | Falla: fondo y artefacto son idénticos. |
| 14 | Consenso de 3 modelos + puertas | 520 nm, Dice 0.891. Tres modelos, no es entregable. |
| 15 | Post-proceso: score τ + borde | τ=0.2 + borde: 565.7 nm. Pero el score no rechazaba nada (mínimo 0.217): **todo el mérito era del borde**. |
| 16 | Ranking por fibra (idea 1), 20 ep | AUC 0.797 vs 0.829 del control. No ayuda. |
| 17 | Árbitro aprendido con features del CSV (idea 2), 20 ep | AUC 0.922; 598.2 nm, Dice 0.8629. Gana HD95, pierde Dice. |
| 18 | GNN sobre grafo de fibras (idea 3), 20 ep | AUC 0.904; 648.9 nm, Dice 0.8658. Su filtro colapsa: 820.8. |
| 19 | Cabeza con incertidumbre (idea 4), 20 ep | AUC 0.842. No supera al control. |
| 20 | Presupuesto de épocas (mismo código) | 20 ep 653.9; **150 ep 591.3 (óptimo)**; 250 ep 634.6. |
| 21 | GNN a 150 épocas | 615.5 nm / Dice 0.8788. Pierde contra el benchmark (591.3 / 0.8815). |

## Lecciones

- El **borde** elimina la mayor parte del error (721 → 574 nm): los FP grandes pesan más que miles de chicos.
- El **AUC no es la métrica**: las ideas 2 y 3 suben el AUC y no bajan el HD95.
- El cuello de botella es la **etiqueta**, no la capacidad: 156 fibras con anillo roto explican 270 nm.

## ⚠️ Corrección importante (bug de la cabeza perezosa)

`AcceptHead` creaba sus pesos **en el primer forward**, y nnU-Net construye el
optimizador **justo antes**. Resultado: **la cabeza nunca estuvo en el
optimizador y nunca se entrenó** en todos los runs anteriores. Verificado: el
score por fibra tenía **mínimo 0.217**, así que a τ=0.2 rechazaba **cero fibras**
(`head@0.2` = `borde`, idénticos). Arreglado con `materialize()` en
`benchmark.py`. Esto invalida las filas 9 y 11, y explica la 15.

## Priors geométricos/visuales como filtro (SIN fuga de GT)

Features por fibra, probe logístico con GroupKFold por caso (54 casos, 2625
fibras). Evaluado en el modelo de 150 ep con cabeza rota:

| prior | AUC (sin GT) | HD95 a su mejor τ |
|---|---|---|
| clásico (cobertura angular de la mielina **predicha**) | **0.910** | 599.9 |
| DINOv2 congelado (384 dims) | 0.833 | — |
| perfil radial de **la imagen** (10 feats) | 0.724 | 590.9 |
| **gate de borde (regla trivial)** | — | **579.8** |
| oráculo (usa el GT) | — | 234.9 |

**Ningún prior supera a no filtrar.** Ni con AUC 0.910. Los falsos positivos que
dominan el HD95 son **indistinguibles** de ciertas fibras verdaderas con
features de imagen: cualquier umbral que los cace mata fibras reales.

> **Lección central:** el AUC no solo es mala guía, puede ser **activamente
> engañosa**. Un filtro con AUC 0.910 hizo *daño*. La métrica depende del
> **punto de operación**, no del ordenamiento global.

*(Un intento previo dio 320.2 nm con "perfil radial": era **fuga de GT** — el
feature 11 era `(gt[axon]==0).mean()`, literalmente el criterio del oráculo.
Retirado.)*
