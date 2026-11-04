import time
from pathlib import Path

import cv2
import numpy as np
import mujoco
import mujoco.viewer

from src.configuracion import Configuracion
from src.xml import Xml
from src.camara import CamaraInspeccion 
from src.objetos import ObjetosIndustrial 
import src.robot as robot
from src.cinta import CintaTransportadora
from src.horno import Horno


 
class PlantaIndustrial:
    def __init__(self):
        # COMPOSICIÓN: Ejecucion "tiene" una configuración, un generador XML, etc.
        self.cfg = Configuracion()
        self.generador_xml = Xml(self.cfg) # Le pasamos la config si Xml la necesita
    
        self.gestor_objetos = ObjetosIndustrial(self.cfg)  
        self.cintas = [
            CintaTransportadora(0, "Cinta 1 · Inferior",  self.cfg.BELT_SPEED,   "+X", ["belt_act_b1", "belt_act_b2"],  5.5, self.cfg),
            CintaTransportadora(1, "Cinta 2 · Derecha",   self.cfg.BELT_SPEED,   "+Y", ["belt_act_r1", "belt_act_r2"], -5.5, self.cfg),
            CintaTransportadora(2, "Cinta 3 · Superior",  self.cfg.BELT_SPEED,   "-X", ["belt_act_t1", "belt_act_t2"], -5.5, self.cfg),
            CintaTransportadora(3, "Cinta 4 · Izquierda", self.cfg.BELT_SPEED,   "-Y", ["belt_act_l1", "belt_act_l2"],  5.5, self.cfg),
            CintaTransportadora(4, "Cinta 5 · Horno",     self.cfg.BRANCH_SPEED, "-X", ["belt_act_o1", "belt_act_o2"], -5.5, self.cfg),
        ]
        self.camaras = [
            CamaraInspeccion("cam_bottom", self.cintas[0]),
            CamaraInspeccion("cam_right",  self.cintas[1]),
            CamaraInspeccion("cam_top",    self.cintas[2]),
            CamaraInspeccion("cam_left",   self.cintas[3]),
            CamaraInspeccion("cam_oven",   self.cintas[4]),
        ]
        

    # -----------------------------------------------------------------------------
    # ACTUADORES DE LOS RODILLOS DE LA PLANTA
    # -----------------------------------------------------------------------------
    def estado_cintas(self):
        """Estado actual de las cintas"""
        return[cinta.esta_activa() for cinta in self.cintas]
    
    def initialize_belt_rollers(self, model, data):
        """Cada objeto CintaTransportadora actualiza sus propios rodillos"""
        for cinta in self.cintas:
            cinta.initialize_belt_rollers(model, data)
            
    # -----------------------------------------------------------------------------
    # BUCLE PRINCIPAL
    # -----------------------------------------------------------------------------
    def run(self):
        self.model = robot.build_model_with_franka(self.generador_xml.build_mjcf(), self.cfg.ROBOT_CONFIG)
        self.data = mujoco.MjData(self.model)

        self.overview_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, "overview")

        self.body_ids, self.dof_addrs = self.gestor_objetos.get_ids(self.model)
        self.robot_ids = robot.get_robot_ids(self.model)
        self.robot_state = robot.new_robot_state()
        self.segments = [3, 1, 2, 3, 0]
        geom_id_horno = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "oven_interior")
        self.horno = Horno(geom_id_horno, temperatura_inicial=20.0)

        self.gestor_objetos.reset_objects(self.model, self.data, self.segments)
        robot.initialize_robot(self.model, self.data, self.robot_ids, self.robot_state)

        if self.cfg.ROBOT_IK_TEST_MODE:
            robot.set_ik_target_rpy(
                self.model, self.robot_ids, self.robot_state,
                self.cfg.ROBOT_IK_TARGET_POSITION,
                self.cfg.ROBOT_IK_TARGET_RPY_DEG,
                orientation_frame=self.cfg.ROBOT_IK_ORIENTATION_FRAME,
            )
            mujoco.mj_forward(self.model, self.data)
        self.initialize_belt_rollers(self.model, self.data)
        

        renderer = None
        if self.cfg.SHOW_CAMERA_MOSAIC:
            renderer = mujoco.Renderer(self.model, height=self.cfg.CAM_H, width=self.cfg.CAM_W)
            cv2.namedWindow("Camaras de la planta", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Camaras de la planta", self.cfg.CAM_W*3, self.cfg.CAM_H*2)

        self.paused = False
        self.running = True
        self.last_cam_time = -1.0

        print("\n=== PLANTA ROBOTIZADA: 4 CINTAS + PANDA + HORNO ===")
        print("Circuito: inferior -> derecha -> superior -> izquierda -> inferior")
        print("Estacion robot: Cinta 4 -> Franka Panda -> Cinta 5 -> horno")
        print(f"Piezas que recoge el Panda: {sorted(self.cfg.ROBOT_CONFIG['pick_object_ids'])}")
        print("El desvio dinamico anterior se ha eliminado para evitar el atasco en la interseccion.")
        print("Q/ESC: salir | P: pausa global | 1..5: ON/OFF cintas | R: reset | S: captura")
        if self.cfg.ROBOT_IK_TEST_MODE:
            print("Modo IK TEST: T=pose objetivo configurada | H=pose HOME cartesiana")
            print("Target IK:", robot.ik_target_text(self.robot_state))
        else:
            print("Modo robot: pick-and-place automático")
        print("Banderas iniciales:", " ".join(
            f"C{i+1}={'ON' if v else 'OFF'}" for i, v in enumerate(self.estado_cintas())
        ))
        print(f"Camaras: FOV={self.cfg.CAM_FOVY:.0f} deg, altura={self.cfg.CAM_HEIGHT:.2f} m")
        print(f"Bloqueo de rotacion de objetos: {self.cfg.LOCK_OBJECT_ROTATION}")
        print(f"Cilindros centrados: {self.cfg.LOCK_CYLINDERS_TO_CENTERLINE}")
        print(f"Panda base=({self.cfg.ROBOT_CONFIG['base_x']:.2f}, {self.cfg.ROBOT_CONFIG['base_y']:.2f}, {self.cfg.ROBOT_CONFIG['base_z']:.2f}) yaw={self.cfg.ROBOT_CONFIG['base_yaw_deg']:.0f} deg")
        print("Programacion Orientada a Objetos: Fase 1 conectada.\n")

        self.viewer_ctx = None
        if self.cfg.SHOW_MUJOCO_VIEWER:
            self.viewer_ctx = mujoco.viewer.launch_passive(self.model, self.data)
            self.viewer_ctx.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
            self.viewer_ctx.cam.fixedcamid = self.overview_id

        try:
            while self.running:
                if self.viewer_ctx is not None and not self.viewer_ctx.is_running():
                    break

                self.step_start = time.perf_counter()

                if not self.paused:
                    if self.cfg.ROBOT_IK_TEST_MODE:
                        # Modo de validación: se ignora temporalmente la máquina de estados
                        # pick-and-place y se manda directamente una pose 6D al TCP.
                        self.gestor_objetos.apply_transport(
                            self.model, self.data, self.body_ids, self.dof_addrs, self.segments, self.robot_state,self.cintas,self.horno
                        )
                        robot.control_manual_ik(
                            self.model, self.data, self.robot_ids, self.robot_state,
                            gripper_open=self.cfg.ROBOT_IK_GRIPPER_OPEN,
                        )
                    else:
                        # Modo automático original: Cinta 4 -> Panda -> Cinta 5.
                        robot.update_robot_state_machine(
                            self.model, self.data, self.body_ids, self.robot_ids, self.robot_state, self.segments,
                            self.cfg.OBJECT_REST_Z, self.cfg.ROBOT_CONFIG
                        )
                        self.gestor_objetos.apply_transport(
                            self.model, self.data, self.body_ids, self.dof_addrs, self.segments, self.oven_counts, self.robot_state, self.cintas
                        )
                        robot.control_robot_ik(self.model, self.data, self.robot_ids, self.robot_state, self.cfg.ROBOT_CONFIG), self.cintas
                        robot.follow_tcp_with_grasped_object(self.model, self.data, self.robot_ids, self.robot_state)

                    # Física.
                    mujoco.mj_step(self.model, self.data)

                    # Estabilización de piezas.
                    self.gestor_objetos.lock_object_rotations(self.model, self.data)
                    self.gestor_objetos.lock_cylinders_to_belt_center(self.model, self.data, self.segments)
                    self.horno.actualizar_color(self.model) 
                    if not self.cfg.ROBOT_IK_TEST_MODE:
                        robot.follow_tcp_with_grasped_object(self.model, self.data, self.robot_ids, self.robot_state)
                    mujoco.mj_forward(self.model, self.data)

                if self.viewer_ctx is not None:
                    self.viewer_ctx.sync()

                if self.cfg.SHOW_CAMERA_MOSAIC and renderer is not None:
                    if self.last_cam_time < 0.0 or (self.data.time - self.last_cam_time) >= (1.0 / self.cfg.CAM_FPS) or self.paused:
                        mosaic = self.render_camera_mosaic(
                            renderer, self.data, self.paused, self.data.time, self.horno.numero_piezas, self.robot_state, self.estado_cintas()
                        )
                        cv2.imshow("Camaras de la planta", mosaic)
                        self.last_cam_time = self.data.time

                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord('q'), 27):
                        self.running = False
                    elif key == ord('p'):
                        self.paused = not self.paused
                        print("PAUSA GLOBAL" if self.paused else "SIMULACIÓN REANUDADA")
                    elif key in (ord('1'), ord('2'), ord('3'), ord('4'), ord('5')):
                        numero_cinta = key - ord('0')
                        activa = self.cintas[numero_cinta -1].conmutar()
                        self.initialize_belt_rollers(self.model, self.data)
                        print(f"Cinta {numero_cinta}: {'ON' if activa else 'OFF'}")
                    elif key == ord('r'):
                        self.gestor_objetos.reset_objects(self.model, self.data, self.segments)
                        self.robot_state.clear()
                        self.robot_state.update(robot.new_robot_state())
                        robot.initialize_robot(self.model, self.data, self.robot_ids, self.robot_state)
                        if self.cfg.ROBOT_IK_TEST_MODE:
                            robot.set_ik_target_rpy(
                                self.model, self.robot_ids, self.robot_state,
                                self.cfg.ROBOT_IK_TARGET_POSITION, self.cfg.ROBOT_IK_TARGET_RPY_DEG,
                                orientation_frame=self.cfg.ROBOT_IK_ORIENTATION_FRAME,
                            )
                            mujoco.mj_forward(self.model, self.data)
                        self.initialize_belt_rollers(self.model, self.data)
                        print("Planta y Panda reiniciados")
                    elif key == ord('t') and self.cfg.ROBOT_IK_TEST_MODE:
                        robot.set_ik_target_rpy(
                            self.model, self.robot_ids, self.robot_state,
                            self.cfg.ROBOT_IK_TARGET_POSITION, self.cfg.ROBOT_IK_TARGET_RPY_DEG,
                            orientation_frame=self.cfg.ROBOT_IK_ORIENTATION_FRAME,
                        )
                        mujoco.mj_forward(self.model, self.data)
                        print("[IK] Target configurado:", robot.ik_target_text(self.robot_state))
                    elif key == ord('h') and self.cfg.ROBOT_IK_TEST_MODE:
                        robot.set_ik_target_home(self.model, self.robot_ids, self.robot_state)
                        mujoco.mj_forward(self.model, self.data)
                        print("[IK] Target HOME cartesiano activado")
                    
                    elif key in (ord('+') ,ord('m')):
                        self.horno.aumentar_temperatura(10.0)
                        print(f"Temperatura horno sube: {self.horno.obtener_temperatura()} °C")
                    elif key in (ord('-'), ord('n')):
                        self.horno.disminuir_temperatura(10.0)
                        print(f"Temperatura horno baja: {self.horno.obtener_temperatura()} °C")
                    
                    elif key == ord('s'):
                        if 'mosaic' in locals():
                            filename = self.cfg.OUTPUT_DIR / f"mosaico_{int(time.time())}.png"
                            cv2.imwrite(str(filename), mosaic)
                            print(f"Captura guardada: {filename}")

                self.elapsed = time.perf_counter() - self.step_start
                self.sleep_time = self.model.opt.timestep - self.elapsed
                if self.sleep_time > 0:
                    time.sleep(self.sleep_time)

        finally:
            if self.viewer_ctx is not None:
                self.viewer_ctx.close()
            if renderer is not None:
                renderer.close()
            cv2.destroyAllWindows()
            
            
    # -----------------------------------------------------------------------------
    # VISUALIZACIÓN DE LAS 4 CÁMARAS
    # -----------------------------------------------------------------------------
    def add_label(self,image, text, status_text=None):
        img = image.copy()
        cv2.rectangle(img, (0, 0), (img.shape[1], 36), (20, 20, 20), -1)
        cv2.putText(img, text, (12, 25), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (245, 245, 245), 2, cv2.LINE_AA)
        if status_text:
            cv2.putText(img, status_text, (12, img.shape[0]-12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (245, 245, 245), 1, cv2.LINE_AA)
        return img


    def render_camera_mosaic(self, renderer, data, paused, sim_time, oven_total, robot_state,flags):
        frames = []
        flags_txt = " ".join(f"C{i+1}:{'ON' if v else 'OFF'}" for i, v in enumerate(flags))
        status = (f"t={sim_time:6.1f}s · {'PAUSA' if paused else 'EN MARCHA'} · "
                f"{flags_txt} · Panda={robot_state['state']} · "
                f"ciclos={robot_state['cycles']} · horno={oven_total}")

        titulos = ["CINTA 1 · INFERIOR", "CINTA 2 · DERECHA", "CINTA 3 · SUPERIOR", "CINTA 4 · IZQ", "CINTA 5 · HORNO"]
        
        for i, cam in enumerate(self.camaras):
            img_procesada = cam.actualizar(renderer, data)
            frames.append(self.add_label(img_procesada, titulos[i], status if i ==0 else None))
            
        renderer.update_scene(data, camera="overview")
        rgb = renderer.render()
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        frames.append(self.add_label(bgr, "Vista General", status))

        top = np.hstack((frames[0], frames[1], frames[2]))
        bottom = np.hstack((frames[3],frames[4], frames[5]))
        mosaic = np.vstack((top, bottom))

        cv2.putText(mosaic, "Q/ESC salir | P pausa | 1-5 cintas ON/OFF | R reset | S captura",
                    (12, mosaic.shape[0]-12), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (255, 255, 255), 2, cv2.LINE_AA)
        return mosaic