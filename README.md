# Planta Industrial Simulada - Python, MuJuco y OpenCV

## 1. Descripción
El sistema simula un entorno automatizado con 5 cintas transportadoras, un horno industrial y piezas mecánicas (cajas y cilindros) utilizando el motor de físicas **MuJoCo**. Además, integra un sistema de visión artificial mediante **OpenCV** que simula cámaras. Estas cámaras realizan interpolación y análisis de contornos en tiempo real para detectar las piezas y detener automáticamente los rodillos de las cintas correspondientes. El proyecto está diseñado para ser modular y escalable.

## 2. Estructura del repositorio
```text
.
|--- main.py                  # Código fuente principal 
|--- requirements.txt         # Dependencias del proyecto
|--- src/
|    |--- planta.py           # Clase PlantaIndustrial (Controlador principal)
|    |--- configuracion.py    # Clase Configuracion (Constantes físicas)
|    |--- cinta.py            # Clase CintaTransportadora (Lógica de fuerzas)
|    |--- camara.py           # Clase CamaraInspeccion (Visión artificial)
|    |--- objetos.py          # Clase ObjetosIndustrial (Cinemática)
|    |--- horno.py            # Clase Horno 
|    |--- robot.py            # Módulo de control del brazo Franka Panda
|--- Practica_1_Planta_Industrial.pdf # Informe técnico del proyecto
|--- README.md                # Este documento


## Diseño de la planta 

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

## 3. Requisitos del  sistema 
* Sistemas operativo: Windows 11 / linux
* Intérprete: Python 13.10 o superior
* Entorno: Recomendable uso de un entorno virtual (.venv)

## 4. Dependencias Externas 
* El proyecto requiere herramientas de simulación física, visión por computador y manipulación de numérica
* Dependencias para instalar: pip install -r requirements.txt (mujoco, opencv-python, numpy)

## 5. Ejecución 
* El proyecto se ejecuta directamente llamando al archivo principal.
* Activar entorno virtual (opcional pero recomendado): python main.py

## 6. Salida esperada 
* Al ejecutar correctamente el programa, se abrirá la venta gráfica del visor de cámaras de OpenCV.
* Salida en terminal:
  === PLANTA ROBOTIZADA: 4 CINTAS + PANDA + HORNO ===
Circuito: inferior -> derecha -> superior -> izquierda -> inferior
Estacion robot: Cinta 4 -> Franka Panda -> Cinta 5 -> horno
Piezas que recoge el Panda: [0]
El desvio dinamico anterior se ha eliminado para evitar el atasco en la interseccion.
Q/ESC: salir | P: pausa global | 1..5: ON/OFF cintas | R: reset | S: captura
Modo IK TEST: T=pose objetivo configurada | H=pose HOME cartesiana
Target IK: pos=[-1.450, 0.050, 0.620] rpy_deg=[0.0, 0.0, 20.0] frame=home
Banderas iniciales: C1=ON C2=ON C3=ON C4=ON C5=ON
Camaras: FOV=32 deg, altura=0.92 m
Bloqueo de rotacion de objetos: True
Cilindros centrados: True
Panda base=(-1.02, 0.20, 0.16) yaw=-90 deg
Programacion Orientada a Objetos: Fase 1 conectada.
..........

* Controles (desde la ventana de OpenCV):
  * 1 a 5: Alternar el estado de cada cinta (on/off).
  * p/p: Pausar la simulación física.
  * q o ESC: Cerrar el simulador de forma segura. 

## 7. Problemas comunes 
* El teclado no responde: Es indispensable hacer clic izquierdo sobre la ventana gris de OpenCV ("Camaras de la planta") para que esta tenga el foco; de lo contrario, las teclas modificarán las vistas internas de MuJoCo en lugar de activar las cintas.
* Error mujoco.FatalError al iniciar: Verificar que el archivo XML del modelo del robot Franka Panda se encuentra en la ruta esperada dentro de la carpeta src/vendor/mujoco_menagerie.
  

# Planta Mujoco con 4 cintas transportadora 

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
