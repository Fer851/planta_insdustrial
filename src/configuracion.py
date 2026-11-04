#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Planta industrial con 4 cintas transportadoras formando un circuito cuadrado.
- MuJoCo + Python
- Programación procedural: NO usa clases / POO
- 4 cintas: inferior, derecha, superior, izquierda
- Objetos transferidos automáticamente entre cintas
- Orientación de los objetos bloqueada: se trasladan pero no rotan
- Los cilindros se mantienen exactamente sobre la línea central de cada banda
- 4 cámaras de inspección, una por cinta, más una para la rama del horno
- Isla robotizada junto a la mitad de la cinta 4
- Franka Emika Panda oficial de MuJoCo Menagerie sobre la isla
- Control del Panda separado en robot.py (procedural, sin POO)
- IK cartesiana general del Panda: target XYZ + orientación RPY o quaternion
- Modo de prueba IK manual para mover el efector final a una pose dada
- El pick-and-place automático puede reactivarse deshabilitando el modo de prueba IK
- Cinta lateral de salida hacia un horno
- Mosaico OpenCV con las cámaras de la planta
- Ventana 3D de MuJoCo opcional

Controles:
    Q / ESC : salir
    P       : pausar/reanudar toda la simulación
    1       : arrancar/detener Cinta 1
    2       : arrancar/detener Cinta 2
    3       : arrancar/detener Cinta 3
    4       : arrancar/detener Cinta 4
    5       : arrancar/detener Cinta 5 (horno)
    R       : reiniciar objetos
    S       : guardar captura del mosaico
    T       : enviar de nuevo la pose IK de prueba
    H       : mover el TCP a la pose cartesiana HOME
"""

import time
from pathlib import Path

import cv2
import numpy as np
import mujoco
import mujoco.viewer

import src.robot as robot

class Configuracion:
    def __init__(self):
        # -----------------------------------------------------------------------------
        # CONFIGURACIÓN GENERAL
        # -----------------------------------------------------------------------------
        self.SHOW_MUJOCO_VIEWER = True
        self.SHOW_CAMERA_MOSAIC = True

        self.CAM_W = 360
        self.CAM_H = 260
        self.CAM_FPS = 20.0

        self.BELT_SPEED = 0.55          # m/s, velocidad objetivo de los objetos

        # -----------------------------------------------------------------------------
        # BANDERAS DE MARCHA / PARO DE CADA CINTA
        # -----------------------------------------------------------------------------
        # True  -> cinta en marcha
        # False -> cinta detenida
        # Estas variables pueden modificarse directamente antes de ejecutar el programa
        # y también con las teclas 1..5 durante la simulación.
        self.BANDERA_CINTA_1 = True      # cinta inferior (+X)
        self.BANDERA_CINTA_2 = True      # cinta derecha (+Y)
        self.BANDERA_CINTA_3 = True      # cinta superior (-X)
        self.BANDERA_CINTA_4 = True      # cinta izquierda (-Y)
        self.BANDERA_CINTA_5 = True      # cinta hacia el horno (-X)

        self.VEL_KP = 7.0               # control de velocidad
        self.CENTER_KP = 12.0           # mantiene el objeto centrado en la cinta
        self.TURN_KP = 16.0             # ayuda a realizar el giro en las esquinas
        self.MAX_FORCE = 12.0           # N por eje

        self.BELT_HALF_LENGTH = 1.45
        self.BELT_HALF_WIDTH = 0.28
        self.BELT_Z = 0.12
        self.BELT_TOP_Z = 0.18
        self.LANE = 1.45                 # coordenada de la línea central de cada cinta
        self.TURN_START = 1.08           # comienza a orientar el objeto hacia la siguiente cinta
        self.CORNER_SWITCH = 1.27        # cambio efectivo de cinta dentro de la placa de esquina
        self.RAIL_END_GAP = 0.06       # separación uniforme antes de cada zona de transferencia
        self.RAIL_HALF_LENGTH = self.LANE - self.BELT_HALF_WIDTH - self.RAIL_END_GAP
        self.RAIL_THICKNESS = 0.035
        self.RAIL_HALF_HEIGHT = 0.10
        self.RAIL_CENTER_Z = 0.22

        # Cámaras de inspección. El poste queda fuera de la cinta, pero el cabezal
        # óptico queda sobre el borde de la banda y mira casi perpendicularmente hacia
        # la superficie. El FOV estrecho evita ver gran parte del suelo/planta.
        self.CAM_HEIGHT = 0.92
        self.CAM_FOVY = 32.0
        self.CAM_LENS_SIDE_OFFSET = 0.10
        self.CAM_POLE_SIDE_OFFSET = self.BELT_HALF_WIDTH + 0.45

        # Cámara específica de la cinta del horno: el cabezal queda completamente
        # FUERA del ancho de la banda y mira hacia su línea central.
        self.CAM_OVEN_HEIGHT = 0.78
        self.CAM_OVEN_SIDE_OFFSET = 0.46
        self.CAM_OVEN_FOVY = 34.0

        # Cámara 4 (izquierda): se coloca completamente FUERA de la banda y además
        # desplazada hacia la mitad superior de la Cinta 4. De esta forma no invade
        # ni la banda ni la intersección Cinta 4 -> Cinta 5/Panda.
        self.CAM_LEFT_INSPECTION_Y = 0.82
        self.CAM_LEFT_HEAD_CLEARANCE = 0.20
        self.CAM_LEFT_POLE_CLEARANCE = 0.52

        # Evita que las piezas roten por contacto, rozamiento o durante los giros.
        # Conservamos freejoint para que sigan teniendo dinámica XYZ, pero forzamos
        # quaternion identidad y velocidad angular nula después de cada paso físico.
        self.LOCK_OBJECT_ROTATION = True

        # Objetos cilíndricos del modelo (azul y verde). En tramos rectos se fija
        # exactamente su coordenada transversal a la línea central de la banda.
        self.CYLINDER_OBJECT_IDS = {1, 3}
        self.LOCK_CYLINDERS_TO_CENTERLINE = True

        # Dimensiones de las piezas. Los cilindros azul (obj1) y verde (obj3)
        # conservan su radio pero tienen ahora la MITAD de la altura anterior.
        self.OBJECT_HALF = 0.085
        self.OBJ1_RADIUS = 0.075
        self.OBJ1_HALF_HEIGHT = 0.045      # antes 0.090 -> altura total 0.09 m
        self.OBJ2_HALF_Z = 0.070
        self.OBJ3_RADIUS = 0.085
        self.OBJ3_HALF_HEIGHT = 0.0375     # antes 0.075 -> altura total 0.075 m
        self.OBJ4_HALF_Z = 0.065
        self.OBJECT_CLEARANCE_Z = 0.01
        self.OBJECT_REST_Z = [
            self.BELT_TOP_Z + self.OBJECT_HALF + self.OBJECT_CLEARANCE_Z,
            self.BELT_TOP_Z + self.OBJ1_HALF_HEIGHT + self.OBJECT_CLEARANCE_Z,
            self.BELT_TOP_Z + self.OBJ2_HALF_Z + self.OBJECT_CLEARANCE_Z,
            self.BELT_TOP_Z + self.OBJ3_HALF_HEIGHT + self.OBJECT_CLEARANCE_Z,
            self.BELT_TOP_Z + self.OBJ4_HALF_Z + self.OBJECT_CLEARANCE_Z,
        ]

        # Rama lateral desde la Cinta 4 hacia el horno.
        # La boca se deja completamente libre: el robot deposita la pieza sobre la
        # Cinta 5 y la banda se encarga del resto. Ya no se hace un desvío dinámico
        # desde la Cinta 4, que era la causa principal del atasco observado.
        self.BRANCH_SPEED = 0.48
        self.BRANCH_Y = 0.0
        self.BRANCH_HALF_LENGTH = 1.15
        self.BRANCH_HALF_WIDTH = 0.32
        self.BRANCH_CENTER_X = -(self.LANE + self.BRANCH_HALF_LENGTH - 0.05)
        self.BRANCH_OPEN_HALF = 0.38
        self.BRANCH_RAIL_HALF = (self.RAIL_HALF_LENGTH - self.BRANCH_OPEN_HALF) / 2.0
        self.BRANCH_RAIL_CENTER_Y = (self.RAIL_HALF_LENGTH + self.BRANCH_OPEN_HALF) / 2.0
        self.BRANCH_RAIL_START_X = -2.00
        self.BRANCH_RAIL_END_X = -3.48
        self.BRANCH_RAIL_CENTER_X = 0.5 * (self.BRANCH_RAIL_START_X + self.BRANCH_RAIL_END_X)
        self.BRANCH_RAIL_HALF_LENGTH = 0.5 * abs(self.BRANCH_RAIL_END_X - self.BRANCH_RAIL_START_X)

        # Apertura de la barandilla interior de Cinta 4 para que el Panda pueda
        # alcanzar el cubo sin atravesar una protección física.
        self.PICK_RAIL_OPEN_Y_MIN = -0.10
        self.PICK_RAIL_OPEN_Y_MAX = 0.72
        self.LEFT_RAIL_UPPER_CENTER_Y = 0.5 * (self.RAIL_HALF_LENGTH + self.PICK_RAIL_OPEN_Y_MAX)
        self.LEFT_RAIL_UPPER_HALF = 0.5 * (self.RAIL_HALF_LENGTH - self.PICK_RAIL_OPEN_Y_MAX)
        self.LEFT_RAIL_LOWER_CENTER_Y = 0.5 * (-self.RAIL_HALF_LENGTH + self.PICK_RAIL_OPEN_Y_MIN)
        self.LEFT_RAIL_LOWER_HALF = 0.5 * (self.PICK_RAIL_OPEN_Y_MIN + self.RAIL_HALF_LENGTH)

        # Configuración geométrica del Panda. La lógica completa del brazo está en
        # robot.py y sigue siendo 100 % procedural (sin clases / sin POO).
        self.ROBOT_CONFIG = robot.make_config(
            lane=self.LANE,
            branch_y=self.BRANCH_Y,
            branch_speed=self.BRANCH_SPEED,
        )

        # -----------------------------------------------------------------------------
        # PRUEBA DE IK CARTESIANA DEL PANDA
        # -----------------------------------------------------------------------------
        # True: desactiva temporalmente el pick-and-place y controla el TCP con una
        # pose objetivo 6D. False: recupera el comportamiento automático de la planta.
        self.ROBOT_IK_TEST_MODE = True

        # Posición del TCP en coordenadas WORLD [x, y, z] (metros).
        self.ROBOT_IK_TARGET_POSITION = np.array([-1.45, 0.05, 0.62], dtype=float)

        # Orientación [roll, pitch, yaw] en grados.
        # Con frame="home", estos ángulos son un incremento respecto a la orientación
        # inicial del efector. Esto hace la primera prueba mucho más segura.
        # Para una orientación absoluta respecto a WORLD, cambia el frame a "world".
        self.ROBOT_IK_TARGET_RPY_DEG = np.array([0.0, 0.0, 20.0], dtype=float)
        self.ROBOT_IK_ORIENTATION_FRAME = "home"   # "home" o "world"
        self.ROBOT_IK_GRIPPER_OPEN = True

        # Isla robotizada: la plataforma toma la pose base directamente de robot.py
        # para que no haya dos configuraciones distintas que puedan desalinearse.
        self.ISLAND_X = self.ROBOT_CONFIG["base_x"]
        self.ISLAND_Y = self.ROBOT_CONFIG["base_y"]
        self.ISLAND_Z = self.ROBOT_CONFIG["base_z"]
        self.ISLAND_HALF_X = 0.18
        self.ISLAND_HALF_Y = 0.38

        # Horno al final de la nueva cinta.
        self.OVEN_CENTER_X = -4.05
        self.OVEN_ENTRY_X = -3.58

        self.OUTPUT_DIR = Path(__file__).resolve().parent / "capturas"
        self.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
