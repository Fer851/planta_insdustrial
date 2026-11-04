# Planta MuJoCo con 4 cintas transportadoras

Simulación procedural en Python, **sin programación orientada a objetos**.

## Diseño

```text
               CINTA 3  (-X)
        <--------------------------
        |                         ^
        |                         |
CINTA 4 |                         | CINTA 2
  (-Y)  |                         |  (+Y)
        v                         |
        -------------------------->
               CINTA 1  (+X)
```

El recorrido es horario:

1. Cinta inferior: `+X`
2. Cinta derecha: `+Y`
3. Cinta superior: `-X`
4. Cinta izquierda: `-Y`
5. Regreso a la cinta inferior.

Cada cinta dispone de una cámara superior. Las cuatro imágenes se muestran a la vez en una ventana OpenCV 2×2.

## Instalación

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Ejecución

```bash
python planta_4_cintas.py
```

## Controles

- `Q` o `ESC`: salir.
- `P`: pausa/reanuda el transporte.
- `R`: reinicia las piezas.
- `S`: guarda una captura de las cuatro cámaras.

## Notas de simulación

MuJoCo no dispone de una primitiva nativa de cinta con velocidad tangencial de superficie. En este ejemplo la acción de cada cinta se modela como una fuerza tangencial con realimentación de velocidad, manteniendo además la pieza centrada y realizando la transición entre tramos en las esquinas.

Los rodillos sí se animan mediante actuadores `velocity`, para que el equipo sea visualmente reconocible como una línea de transporte.

## Parámetros principales

En la cabecera de `planta_4_cintas.py` se pueden cambiar:

```python
BELT_SPEED = 0.55
VEL_KP = 7.0
CENTER_KP = 12.0
TURN_KP = 16.0
CAM_W = 360
CAM_H = 260
CAM_FPS = 20.0
```

## Estructura del código

El código no define ninguna `class`. Se organiza mediante funciones:

- `build_mjcf()`
- `get_ids()`
- `reset_objects()`
- `update_segment()`
- `transport_force()`
- `apply_transport()`
- `render_camera_mosaic()`
- `run()`

Esto permite utilizarlo directamente como ejemplo de programación procedural.
