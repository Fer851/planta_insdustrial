import time
from pathlib import Path

import cv2
import numpy as np
import mujoco
import mujoco.viewer
import src.robot as robot

class Xml:
    def __init__(self, cfg):
        self.cfg = cfg
    # -----------------------------------------------------------------------------
    # GEOMETRÍA XML DE LA PLANTA
    # -----------------------------------------------------------------------------
    def camera_xyaxes(self, cam_pos, target):
        """Devuelve los ejes X/Y de una cámara MuJoCo que mira hacia `target`."""
        self.cam = np.asarray(cam_pos, dtype=float)
        self.tgt = np.asarray(target, dtype=float)

        # La cámara de MuJoCo mira a lo largo de su eje local -Z.
        self.z_axis = self.cam - self.tgt
        self.z_axis /= np.linalg.norm(self.z_axis)

        self.up = np.array([0.0, 0.0, 1.0])
        self.x_axis = np.cross(self.up, self.z_axis)
        if np.linalg.norm(self.x_axis) < 1e-8:
            self.up = np.array([0.0, 1.0, 0.0])
            self.x_axis = np.cross(self.up, self.z_axis)
        self.x_axis /= np.linalg.norm(self.x_axis)
        self.y_axis = np.cross(self.z_axis, self.x_axis)
        self.y_axis /= np.linalg.norm(self.y_axis)

        self.vals = np.concatenate([self.x_axis, self.y_axis])
        return " ".join(f"{v:.6f}" for v in self.vals)


    def build_mjcf(self):
        """Construye el MJCF como texto. Todo se genera proceduralmente."""

        # Cámaras de inspección: el soporte/poste está a un lado de cada cinta,
        # mientras que el objetivo óptico queda cerca del borde de la banda. Así
        # la imagen contiene esencialmente banda + pieza y resulta adecuada para
        # reconocimiento posterior de color y forma.
        self.cam_bottom = (0.0, -(self.cfg.LANE + self.cfg.CAM_LENS_SIDE_OFFSET), self.cfg.CAM_HEIGHT)
        self.cam_right  = (self.cfg.LANE + self.cfg.CAM_LENS_SIDE_OFFSET, 0.0, self.cfg.CAM_HEIGHT)
        self.cam_top    = (0.0, self.cfg.LANE + self.cfg.CAM_LENS_SIDE_OFFSET, self.cfg.CAM_HEIGHT)

        # La cámara 4 NO queda encima de la banda. Su x está más allá del borde
        # exterior de la Cinta 4 y su y se aleja de la intersección con Cinta 5.
        self.cam_left = (
            -(self.cfg.LANE + self.cfg.BELT_HALF_WIDTH + self.cfg.CAM_LEFT_HEAD_CLEARANCE),
            self.cfg.CAM_LEFT_INSPECTION_Y,
            self.cfg.CAM_HEIGHT,
        )
        self.cam_oven   = (self.cfg.BRANCH_CENTER_X, self.cfg.BRANCH_Y - self.cfg.CAM_OVEN_SIDE_OFFSET, self.cfg.CAM_OVEN_HEIGHT)

        # Se apunta al centro de una zona local de inspección de la banda. El
        # pequeño offset hace que la cámara siga estando físicamente a un lado,
        # pero la dirección de visión sea casi cenital.
        self.target_bottom = (0.0, -self.cfg.LANE, self.cfg.BELT_TOP_Z + 0.015)
        self.target_right  = (self.cfg.LANE, 0.0, self.cfg.BELT_TOP_Z + 0.015)
        self.target_top    = (0.0, self.cfg.LANE, self.cfg.BELT_TOP_Z + 0.015)
        self.target_left   = (-self.cfg.LANE, self.cfg.CAM_LEFT_INSPECTION_Y, self.cfg.BELT_TOP_Z + 0.015)
        self.target_oven   = (self.cfg.BRANCH_CENTER_X, self.cfg.BRANCH_Y, self.cfg.BELT_TOP_Z + 0.015)

        self.pole_bottom_y = -(self.cfg.LANE + self.cfg.CAM_POLE_SIDE_OFFSET)
        self.pole_right_x  =  (self.cfg.LANE + self.cfg.CAM_POLE_SIDE_OFFSET)
        self.pole_top_y    =  (self.cfg.LANE + self.cfg.CAM_POLE_SIDE_OFFSET)
        self.pole_left_x   = -(self.cfg.LANE + self.cfg.BELT_HALF_WIDTH + self.cfg.CAM_LEFT_POLE_CLEARANCE)
        self.pole_left_y   = self.cfg.CAM_LEFT_INSPECTION_Y
        self.pole_oven_y   = self.cfg.BRANCH_Y - (self.cfg.BRANCH_HALF_WIDTH + 0.52)

        self.xy_bottom = self.camera_xyaxes(self.cam_bottom, self.target_bottom)
        self.xy_right  = self.camera_xyaxes(self.cam_right, self.target_right)
        self.xy_top    = self.camera_xyaxes(self.cam_top, self.target_top)
        self.xy_left   = self.camera_xyaxes(self.cam_left, self.target_left)
        self.xy_oven   = self.camera_xyaxes(self.cam_oven, self.target_oven)

        self.xml = f"""
    <mujoco model="square_conveyor_factory">
        <compiler angle="degree" autolimits="true"/>
        <option timestep="0.002" gravity="0 0 -9.81" integrator="RK4"/>

        <visual>
            <global offwidth="1280" offheight="960"/>
            <quality shadowsize="2048"/>
            <headlight ambient="0.35 0.35 0.35" diffuse="0.8 0.8 0.8" specular="0.2 0.2 0.2"/>
        </visual>

        <default>
            <geom contype="1" conaffinity="1" friction="1.2 0.02 0.002" condim="4"/>
            <joint damping="0.15"/>
        </default>

        <asset>
            <texture name="floor_tex" type="2d" builtin="checker"
                    rgb1="0.18 0.20 0.22" rgb2="0.23 0.25 0.27" width="512" height="512"/>
            <material name="floor_mat" texture="floor_tex" texrepeat="8 8" reflectance="0.10"/>
            <material name="belt_mat" rgba="0.08 0.10 0.12 1"/>
            <material name="belt_edge_mat" rgba="0.28 0.32 0.36 1"/>
            <material name="transfer_mat" rgba="0.18 0.20 0.23 1"/>
            <material name="camera_mat" rgba="0.07 0.07 0.08 1"/>
            <material name="pole_mat" rgba="0.30 0.33 0.36 1"/>
            <material name="roller_mat" rgba="0.13 0.15 0.17 1"/>
            <material name="island_mat" rgba="0.24 0.28 0.31 1"/>
            <material name="island_panel_mat" rgba="0.10 0.38 0.52 1"/>
            <material name="oven_mat" rgba="0.22 0.24 0.26 1"/>
            <material name="oven_dark_mat" rgba="0.06 0.06 0.07 1"/>
            <material name="oven_hot_mat" rgba="0.95 0.22 0.05 1" emission="0.8"/>
        </asset>

        <worldbody>
            <light pos="0 0 6" dir="0 0 -1" diffuse="0.85 0.85 0.85"/>
            <light pos="3 -3 4" dir="-0.5 0.5 -1" diffuse="0.45 0.45 0.45"/>
            <geom name="floor" type="plane" size="6 6 0.1" material="floor_mat"/>

            <!-- CINTA 1: INFERIOR, flujo +X -->
            <geom name="belt_bottom" type="box" pos="0 -{self.cfg.LANE} {self.cfg.BELT_Z}"
                size="{self.cfg.BELT_HALF_LENGTH} {self.cfg.BELT_HALF_WIDTH} 0.06" material="belt_mat"/>
            <geom type="box" pos="0 -{self.cfg.LANE-self.cfg.BELT_HALF_WIDTH-self.cfg.RAIL_THICKNESS} {self.cfg.RAIL_CENTER_Z}"
                size="{self.cfg.RAIL_HALF_LENGTH} {self.cfg.RAIL_THICKNESS} {self.cfg.RAIL_HALF_HEIGHT}" material="belt_edge_mat"/>
            <geom type="box" pos="0 -{self.cfg.LANE+self.cfg.BELT_HALF_WIDTH+self.cfg.RAIL_THICKNESS} {self.cfg.RAIL_CENTER_Z}"
                size="{self.cfg.RAIL_HALF_LENGTH} {self.cfg.RAIL_THICKNESS} {self.cfg.RAIL_HALF_HEIGHT}" material="belt_edge_mat"/>

            <!-- CINTA 2: DERECHA, flujo +Y -->
            <geom name="belt_right" type="box" pos="{self.cfg.LANE} 0 {self.cfg.BELT_Z}"
                size="{self.cfg.BELT_HALF_WIDTH} {self.cfg.BELT_HALF_LENGTH} 0.06" material="belt_mat"/>
            <geom type="box" pos="{self.cfg.LANE-self.cfg.BELT_HALF_WIDTH-self.cfg.RAIL_THICKNESS} 0 {self.cfg.RAIL_CENTER_Z}"
                size="{self.cfg.RAIL_THICKNESS} {self.cfg.RAIL_HALF_LENGTH} {self.cfg.RAIL_HALF_HEIGHT}" material="belt_edge_mat"/>
            <geom type="box" pos="{self.cfg.LANE+self.cfg.BELT_HALF_WIDTH+self.cfg.RAIL_THICKNESS} 0 {self.cfg.RAIL_CENTER_Z}"
                size="{self.cfg.RAIL_THICKNESS} {self.cfg.RAIL_HALF_LENGTH} {self.cfg.RAIL_HALF_HEIGHT}" material="belt_edge_mat"/>

            <!-- CINTA 3: SUPERIOR, flujo -X -->
            <geom name="belt_top" type="box" pos="0 {self.cfg.LANE} {self.cfg.BELT_Z}"
                size="{self.cfg.BELT_HALF_LENGTH} {self.cfg.BELT_HALF_WIDTH} 0.06" material="belt_mat"/>
            <geom type="box" pos="0 {self.cfg.LANE-self.cfg.BELT_HALF_WIDTH-self.cfg.RAIL_THICKNESS} {self.cfg.RAIL_CENTER_Z}"
                size="{self.cfg.RAIL_HALF_LENGTH} {self.cfg.RAIL_THICKNESS} {self.cfg.RAIL_HALF_HEIGHT}" material="belt_edge_mat"/>
            <geom type="box" pos="0 {self.cfg.LANE+self.cfg.BELT_HALF_WIDTH+self.cfg.RAIL_THICKNESS} {self.cfg.RAIL_CENTER_Z}"
                size="{self.cfg.RAIL_HALF_LENGTH} {self.cfg.RAIL_THICKNESS} {self.cfg.RAIL_HALF_HEIGHT}" material="belt_edge_mat"/>

            <!-- CINTA 4: IZQUIERDA, flujo -Y -->
            <geom name="belt_left" type="box" pos="-{self.cfg.LANE} 0 {self.cfg.BELT_Z}"
                size="{self.cfg.BELT_HALF_WIDTH} {self.cfg.BELT_HALF_LENGTH} 0.06" material="belt_mat"/>

            <!-- Barandilla interior de Cinta 4: abierta delante de la isla robotizada. -->
            <geom type="box" pos="-{self.cfg.LANE-self.cfg.BELT_HALF_WIDTH-self.cfg.RAIL_THICKNESS} {self.cfg.LEFT_RAIL_UPPER_CENTER_Y} {self.cfg.RAIL_CENTER_Z}"
                size="{self.cfg.RAIL_THICKNESS} {self.cfg.LEFT_RAIL_UPPER_HALF} {self.cfg.RAIL_HALF_HEIGHT}" material="belt_edge_mat"/>
            <geom type="box" pos="-{self.cfg.LANE-self.cfg.BELT_HALF_WIDTH-self.cfg.RAIL_THICKNESS} {self.cfg.LEFT_RAIL_LOWER_CENTER_Y} {self.cfg.RAIL_CENTER_Z}"
                size="{self.cfg.RAIL_THICKNESS} {self.cfg.LEFT_RAIL_LOWER_HALF} {self.cfg.RAIL_HALF_HEIGHT}" material="belt_edge_mat"/>

            <!-- La barandilla exterior queda abierta en y=0: esa abertura ya no
                hace el desvío; sirve solo como paso hacia la Cinta 5. -->
            <geom type="box" pos="-{self.cfg.LANE+self.cfg.BELT_HALF_WIDTH+self.cfg.RAIL_THICKNESS} {self.cfg.BRANCH_RAIL_CENTER_Y} {self.cfg.RAIL_CENTER_Z}"
                size="{self.cfg.RAIL_THICKNESS} {self.cfg.BRANCH_RAIL_HALF} {self.cfg.RAIL_HALF_HEIGHT}" material="belt_edge_mat"/>
            <geom type="box" pos="-{self.cfg.LANE+self.cfg.BELT_HALF_WIDTH+self.cfg.RAIL_THICKNESS} -{self.cfg.BRANCH_RAIL_CENTER_Y} {self.cfg.RAIL_CENTER_Z}"
                size="{self.cfg.RAIL_THICKNESS} {self.cfg.BRANCH_RAIL_HALF} {self.cfg.RAIL_HALF_HEIGHT}" material="belt_edge_mat"/>

            <!-- CINTA 5: RAMA LATERAL HACIA EL HORNO, flujo -X. -->
            <geom name="belt_oven" type="box" pos="{self.cfg.BRANCH_CENTER_X} {self.cfg.BRANCH_Y} {self.cfg.BELT_Z}"
                size="{self.cfg.BRANCH_HALF_LENGTH} {self.cfg.BRANCH_HALF_WIDTH} 0.06" material="belt_mat"/>
            <geom name="branch_transfer" type="box" pos="-{self.cfg.LANE} {self.cfg.BRANCH_Y} {self.cfg.BELT_Z}"
                size="{self.cfg.BELT_HALF_WIDTH+0.06} {self.cfg.BRANCH_HALF_WIDTH} 0.06" material="transfer_mat"/>

            <!-- Las guías de Cinta 5 empiezan después de la zona de descarga del
                robot. Así el cubo no cae contra una pared al ser depositado. -->
            <geom type="box" pos="{self.cfg.BRANCH_RAIL_CENTER_X} {self.cfg.BRANCH_Y-self.cfg.BRANCH_HALF_WIDTH-self.cfg.RAIL_THICKNESS} {self.cfg.RAIL_CENTER_Z}"
                size="{self.cfg.BRANCH_RAIL_HALF_LENGTH} {self.cfg.RAIL_THICKNESS} {self.cfg.RAIL_HALF_HEIGHT}" material="belt_edge_mat"/>
            <geom type="box" pos="{self.cfg.BRANCH_RAIL_CENTER_X} {self.cfg.BRANCH_Y+self.cfg.BRANCH_HALF_WIDTH+self.cfg.RAIL_THICKNESS} {self.cfg.RAIL_CENTER_Z}"
                size="{self.cfg.BRANCH_RAIL_HALF_LENGTH} {self.cfg.RAIL_THICKNESS} {self.cfg.RAIL_HALF_HEIGHT}" material="belt_edge_mat"/>

            <!-- ISLA ROBOTIZADA. La plataforma es baja y queda alineada con la
                intersección Cinta 4 / Cinta 5. El Panda se adjunta después con MjSpec. -->
            <body name="process_island" pos="{self.cfg.ISLAND_X} {self.cfg.ISLAND_Y} 0">
                <geom name="island_base" type="box" pos="0 0 {self.cfg.ISLAND_Z/2}"
                    size="{self.cfg.ISLAND_HALF_X} {self.cfg.ISLAND_HALF_Y} {self.cfg.ISLAND_Z/2}"
                    material="island_mat" contype="0" conaffinity="0"/>
                <geom name="island_panda_plate" type="cylinder" pos="0 0 {self.cfg.ISLAND_Z+0.025}"
                    size="0.16 0.025" material="belt_edge_mat" contype="0" conaffinity="0"/>
            </body>

            <!-- HORNO INDUSTRIAL AL FINAL DE LA RAMA -->
            <body name="oven" pos="{self.cfg.OVEN_CENTER_X} 0 0">
                <!-- base y fondo -->
                <geom type="box" pos="0 0 0.12" size="0.55 0.62 0.10" material="oven_mat"/>
                <geom type="box" pos="-0.50 0 0.68" size="0.08 0.62 0.58" material="oven_mat"/>
                <!-- paredes laterales, dejando el frontal abierto hacia +X -->
                <geom type="box" pos="0 -0.56 0.68" size="0.55 0.06 0.58" material="oven_mat"/>
                <geom type="box" pos="0  0.56 0.68" size="0.55 0.06 0.58" material="oven_mat"/>
                <geom type="box" pos="0 0 1.23" size="0.55 0.62 0.07" material="oven_mat"/>
                <!-- marco frontal -->
                <geom type="box" pos="0.50 -0.48 0.68" size="0.05 0.08 0.58" material="oven_dark_mat"/>
                <geom type="box" pos="0.50  0.48 0.68" size="0.05 0.08 0.58" material="oven_dark_mat"/>
                <geom type="box" pos="0.50 0 1.16" size="0.05 0.40 0.10" material="oven_dark_mat"/>
                <!-- interior incandescente visible desde la cinta -->
                <geom name="oven_interior" type="box" pos="-0.42 0 0.58" size="0.025 0.39 0.31" contype="0" conaffinity="0"/>
                <light pos="-0.18 0 0.82" dir="1 0 -0.15" diffuse="1.0 0.25 0.05"/>
            </body>

            <!-- TRANSFERENCIAS EN LAS CUATRO ESQUINAS -->
            <geom name="corner_br" type="box" pos=" {self.cfg.LANE} -{self.cfg.LANE} {self.cfg.BELT_Z}"
                size="{self.cfg.BELT_HALF_WIDTH} {self.cfg.BELT_HALF_WIDTH} 0.06" material="transfer_mat"/>
            <geom name="corner_tr" type="box" pos=" {self.cfg.LANE}  {self.cfg.LANE} {self.cfg.BELT_Z}"
                size="{self.cfg.BELT_HALF_WIDTH} {self.cfg.BELT_HALF_WIDTH} 0.06" material="transfer_mat"/>
            <geom name="corner_tl" type="box" pos="-{self.cfg.LANE}  {self.cfg.LANE} {self.cfg.BELT_Z}"
                size="{self.cfg.BELT_HALF_WIDTH} {self.cfg.BELT_HALF_WIDTH} 0.06" material="transfer_mat"/>
            <geom name="corner_bl" type="box" pos="-{self.cfg.LANE} -{self.cfg.LANE} {self.cfg.BELT_Z}"
                size="{self.cfg.BELT_HALF_WIDTH} {self.cfg.BELT_HALF_WIDTH} 0.06" material="transfer_mat"/>

            <!-- RODILLOS VISUALES: 2 POR CINTA -->
            <body name="roller_b1" pos="-1.15 -{self.cfg.LANE} 0.20">
                <joint name="jr_b1" type="hinge" axis="0 1 0"/>
                <geom type="cylinder" size="0.08 0.25" euler="90 0 0" material="roller_mat" contype="0" conaffinity="0"/>
            </body>
            <body name="roller_b2" pos="1.15 -{self.cfg.LANE} 0.20">
                <joint name="jr_b2" type="hinge" axis="0 1 0"/>
                <geom type="cylinder" size="0.08 0.25" euler="90 0 0" material="roller_mat" contype="0" conaffinity="0"/>
            </body>

            <body name="roller_r1" pos="{self.cfg.LANE} -1.15 0.20">
                <joint name="jr_r1" type="hinge" axis="1 0 0"/>
                <geom type="cylinder" size="0.08 0.25" euler="0 90 0" material="roller_mat" contype="0" conaffinity="0"/>
            </body>
            <body name="roller_r2" pos="{self.cfg.LANE} 1.15 0.20">
                <joint name="jr_r2" type="hinge" axis="1 0 0"/>
                <geom type="cylinder" size="0.08 0.25" euler="0 90 0" material="roller_mat" contype="0" conaffinity="0"/>
            </body>

            <body name="roller_t1" pos="1.15 {self.cfg.LANE} 0.20">
                <joint name="jr_t1" type="hinge" axis="0 1 0"/>
                <geom type="cylinder" size="0.08 0.25" euler="90 0 0" material="roller_mat" contype="0" conaffinity="0"/>
            </body>
            <body name="roller_t2" pos="-1.15 {self.cfg.LANE} 0.20">
                <joint name="jr_t2" type="hinge" axis="0 1 0"/>
                <geom type="cylinder" size="0.08 0.25" euler="90 0 0" material="roller_mat" contype="0" conaffinity="0"/>
            </body>

            <body name="roller_l1" pos="-{self.cfg.LANE} 1.15 0.20">
                <joint name="jr_l1" type="hinge" axis="1 0 0"/>
                <geom type="cylinder" size="0.08 0.25" euler="0 90 0" material="roller_mat" contype="0" conaffinity="0"/>
            </body>
            <body name="roller_l2" pos="-{self.cfg.LANE} -1.15 0.20">
                <joint name="jr_l2" type="hinge" axis="1 0 0"/>
                <geom type="cylinder" size="0.08 0.25" euler="0 90 0" material="roller_mat" contype="0" conaffinity="0"/>
            </body>

            <!-- Rodillos de la cinta hacia el horno. -->
            <body name="roller_o1" pos="-2.18 {self.cfg.BRANCH_Y} 0.20">
                <joint name="jr_o1" type="hinge" axis="0 1 0"/>
                <geom type="cylinder" size="0.08 0.25" euler="90 0 0" material="roller_mat" contype="0" conaffinity="0"/>
            </body>
            <body name="roller_o2" pos="-3.25 {self.cfg.BRANCH_Y} 0.20">
                <joint name="jr_o2" type="hinge" axis="0 1 0"/>
                <geom type="cylinder" size="0.08 0.25" euler="90 0 0" material="roller_mat" contype="0" conaffinity="0"/>
            </body>

            <!-- CÁMARAS DE INSPECCIÓN: SOPORTE LATERAL + VISIÓN SOBRE LA BANDA -->
            <!-- El poste permanece fuera de la cinta; un brazo corto sitúa el cabezal
                óptico junto al borde. El FOV estrecho muestra casi exclusivamente
                la superficie de la banda para visión artificial. -->
            <body name="cam_station_bottom" pos="0 0 0">
                <geom type="cylinder" pos="0 {self.pole_bottom_y} {self.cfg.CAM_HEIGHT/2}"
                    size="0.025 {self.cfg.CAM_HEIGHT/2}" material="pole_mat" contype="0" conaffinity="0"/>
                <geom type="box" pos="0 {(self.pole_bottom_y + self.cam_bottom[1])/2} {self.cfg.CAM_HEIGHT}"
                    size="0.025 {abs(self.cam_bottom[1]-self.pole_bottom_y)/2} 0.022" material="pole_mat" contype="0" conaffinity="0"/>
                <geom type="box" pos="{self.cam_bottom[0]} {self.cam_bottom[1]} {self.cam_bottom[2]}"
                    size="0.075 0.055 0.055" material="camera_mat" contype="0" conaffinity="0"/>
                <camera name="cam_bottom" pos="{self.cam_bottom[0]} {self.cam_bottom[1]} {self.cam_bottom[2]}"
                        xyaxes="{self.xy_bottom}" fovy="{self.cfg.CAM_FOVY}"/>
            </body>

            <body name="cam_station_right" pos="0 0 0">
                <geom type="cylinder" pos="{self.pole_right_x} 0 {self.cfg.CAM_HEIGHT/2}"
                    size="0.025 {self.cfg.CAM_HEIGHT/2}" material="pole_mat" contype="0" conaffinity="0"/>
                <geom type="box" pos="{(self.pole_right_x + self.cam_right[0])/2} 0 {self.cfg.CAM_HEIGHT}"
                    size="{abs(self.cam_right[0]-self.pole_right_x)/2} 0.025 0.022" material="pole_mat" contype="0" conaffinity="0"/>
                <geom type="box" pos="{self.cam_right[0]} {self.cam_right[1]} {self.cam_right[2]}"
                    size="0.055 0.075 0.055" material="camera_mat" contype="0" conaffinity="0"/>
                <camera name="cam_right" pos="{self.cam_right[0]} {self.cam_right[1]} {self.cam_right[2]}"
                        xyaxes="{self.xy_right}" fovy="{self.cfg.CAM_FOVY}"/>
            </body>

            <body name="cam_station_top" pos="0 0 0">
                <geom type="cylinder" pos="0 {self.pole_top_y} {self.cfg.CAM_HEIGHT/2}"
                    size="0.025 {self.cfg.CAM_HEIGHT/2}" material="pole_mat" contype="0" conaffinity="0"/>
                <geom type="box" pos="0 {(self.pole_top_y + self.cam_top[1])/2} {self.cfg.CAM_HEIGHT}"
                    size="0.025 {abs(self.cam_top[1]-self.pole_top_y)/2} 0.022" material="pole_mat" contype="0" conaffinity="0"/>
                <geom type="box" pos="{self.cam_top[0]} {self.cam_top[1]} {self.cam_top[2]}"
                    size="0.075 0.055 0.055" material="camera_mat" contype="0" conaffinity="0"/>
                <camera name="cam_top" pos="{self.cam_top[0]} {self.cam_top[1]} {self.cam_top[2]}"
                        xyaxes="{self.xy_top}" fovy="{self.cfg.CAM_FOVY}"/>
            </body>

            <body name="cam_station_left" pos="0 0 0">
                <geom type="cylinder" pos="{self.pole_left_x} {self.pole_left_y} {self.cfg.CAM_HEIGHT/2}"
                    size="0.025 {self.cfg.CAM_HEIGHT/2}" material="pole_mat" contype="0" conaffinity="0"/>
                <geom type="box" pos="{(self.pole_left_x + self.cam_left[0])/2} {self.pole_left_y} {self.cfg.CAM_HEIGHT}"
                    size="{abs(self.cam_left[0]-self.pole_left_x)/2} 0.025 0.022" material="pole_mat" contype="0" conaffinity="0"/>
                <geom type="box" pos="{self.cam_left[0]} {self.cam_left[1]} {self.cam_left[2]}"
                    size="0.055 0.075 0.055" material="camera_mat" contype="0" conaffinity="0"/>
                <camera name="cam_left" pos="{self.cam_left[0]} {self.cam_left[1]} {self.cam_left[2]}"
                        xyaxes="{self.xy_left}" fovy="{self.cfg.CAM_FOVY}"/>
            </body>

            <!-- Cámara de inspección de la cinta 5 hacia el horno. -->
            <body name="cam_station_oven" pos="0 0 0">
                <geom type="cylinder" pos="{self.cfg.BRANCH_CENTER_X} {self.pole_oven_y} {self.cfg.CAM_OVEN_HEIGHT/2}"
                    size="0.025 {self.cfg.CAM_OVEN_HEIGHT/2}" material="pole_mat" contype="0" conaffinity="0"/>
                <geom type="box" pos="{self.cfg.BRANCH_CENTER_X} {(self.pole_oven_y + self.cam_oven[1])/2} {self.cfg.CAM_OVEN_HEIGHT}"
                    size="0.025 {abs(self.cam_oven[1]-self.pole_oven_y)/2} 0.022" material="pole_mat" contype="0" conaffinity="0"/>
                <geom type="box" pos="{self.cam_oven[0]} {self.cam_oven[1]} {self.cam_oven[2]}"
                    size="0.075 0.055 0.055" material="camera_mat" contype="0" conaffinity="0"/>
                <camera name="cam_oven" pos="{self.cam_oven[0]} {self.cam_oven[1]} {self.cam_oven[2]}"
                        xyaxes="{self.xy_oven}" fovy="{self.cfg.CAM_OVEN_FOVY}"/>
            </body>

            <!-- CÁMARA GENERAL DE LA PLANTA -->
            <camera name="overview" pos="4.6 -4.6 5.2" xyaxes="0.707 0.707 0 -0.49 0.49 0.72" fovy="50"/>

            <!-- OBJETOS -->
            <body name="obj0" pos="-0.95 -{self.cfg.LANE} {self.cfg.OBJECT_REST_Z[0]}">
                <freejoint name="obj0_free"/>
                <geom name="obj0_geom" type="box" size="{self.cfg.OBJECT_HALF} {self.cfg.OBJECT_HALF} {self.cfg.OBJECT_HALF}"
                    mass="0.18" rgba="0.90 0.16 0.12 1"/>
            </body>

            <body name="obj1" pos="{self.cfg.LANE} -0.70 {self.cfg.OBJECT_REST_Z[1]}">
                <freejoint name="obj1_free"/>
                <geom name="obj1_geom" type="cylinder" size="{self.cfg.OBJ1_RADIUS} {self.cfg.OBJ1_HALF_HEIGHT}"
                    mass="0.08" rgba="0.12 0.45 0.95 1"/>
            </body>

            <body name="obj2" pos="0.55 {self.cfg.LANE} {self.cfg.OBJECT_REST_Z[2]}">
                <freejoint name="obj2_free"/>
                <geom name="obj2_geom" type="box" size="0.10 0.07 0.07"
                    mass="0.15" rgba="0.95 0.76 0.10 1"/>
            </body>

            <body name="obj3" pos="-{self.cfg.LANE} 0.45 {self.cfg.OBJECT_REST_Z[3]}">
                <freejoint name="obj3_free"/>
                <geom name="obj3_geom" type="cylinder" size="{self.cfg.OBJ3_RADIUS} {self.cfg.OBJ3_HALF_HEIGHT}"
                    mass="0.085" rgba="0.18 0.75 0.30 1"/>
            </body>

            <body name="obj4" pos="0.15 -{self.cfg.LANE} {self.cfg.OBJECT_REST_Z[4]}">
                <freejoint name="obj4_free"/>
                <geom name="obj4_geom" type="box" size="0.07 0.11 0.065"
                    mass="0.14" rgba="0.72 0.28 0.90 1"/>
            </body>
        </worldbody>

        <actuator>
            <velocity name="belt_act_b1" joint="jr_b1" kv="3" ctrlrange="-9 9"/>
            <velocity name="belt_act_b2" joint="jr_b2" kv="3" ctrlrange="-9 9"/>
            <velocity name="belt_act_r1" joint="jr_r1" kv="3" ctrlrange="-9 9"/>
            <velocity name="belt_act_r2" joint="jr_r2" kv="3" ctrlrange="-9 9"/>
            <velocity name="belt_act_t1" joint="jr_t1" kv="3" ctrlrange="-9 9"/>
            <velocity name="belt_act_t2" joint="jr_t2" kv="3" ctrlrange="-9 9"/>
            <velocity name="belt_act_l1" joint="jr_l1" kv="3" ctrlrange="-9 9"/>
            <velocity name="belt_act_l2" joint="jr_l2" kv="3" ctrlrange="-9 9"/>
            <velocity name="belt_act_o1" joint="jr_o1" kv="3" ctrlrange="-9 9"/>
            <velocity name="belt_act_o2" joint="jr_o2" kv="3" ctrlrange="-9 9"/>
        </actuator>
    </mujoco>
    """
        return self.xml