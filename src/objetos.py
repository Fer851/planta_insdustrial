import time
from pathlib import Path

import cv2
import numpy as np
import mujoco
import mujoco.viewer

import src.robot as robot

class ObjetosIndustrial:
    def __init__(self, cfg):
        self.cfg = cfg

    # -----------------------------------------------------------------------------
    # UTILIDADES DE OBJETOS
    # -----------------------------------------------------------------------------
    def get_ids(self, model):
        self.body_ids = []
        self.dof_addrs = []
        for i in range(5):
            self.bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"obj{i}")
            self.jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"obj{i}_free")
            self.body_ids.append(self.bid)
            self.dof_addrs.append(model.jnt_dofadr[self.jid])
        return self.body_ids, self.dof_addrs


    def reset_objects(self, model, data, segments):
        """Reinicia las 5 piezas. El cubo rojo comienza cerca de la isla."""
        starts = [
            (-self.cfg.LANE, 0.95, self.cfg.OBJECT_REST_Z[0], 3),   # cubo rojo: llega pronto a la estación Panda
            ( self.cfg.LANE, -0.72, self.cfg.OBJECT_REST_Z[1], 1),
            ( 0.58,  self.cfg.LANE, self.cfg.OBJECT_REST_Z[2], 2),
            (-0.62, -self.cfg.LANE, self.cfg.OBJECT_REST_Z[3], 0),  # cilindro verde lejos de la estación robot
            (-0.45,  self.cfg.LANE, self.cfg.OBJECT_REST_Z[4], 2),
        ]

        for i, (x, y, z, seg) in enumerate(starts):
            self.jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"obj{i}_free")
            self.qadr = model.jnt_qposadr[self.jid]
            self.dadr = model.jnt_dofadr[self.jid]

            data.qpos[self.qadr:self.qadr+3] = [x, y, z]
            data.qpos[self.qadr+3:self.qadr+7] = [1.0, 0.0, 0.0, 0.0]
            data.qvel[self.dadr:self.dadr+6] = 0.0
            segments[i] = seg

        data.xfrc_applied[:] = 0.0
        mujoco.mj_forward(model, data)


    def lock_object_rotations(self, model, data):
        """Mantiene todas las piezas con orientación fija respecto al mundo."""
        if not self.cfg.LOCK_OBJECT_ROTATION:
            return

        for i in range(5):
            self.jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"obj{i}_free")
            self.qadr = model.jnt_qposadr[self.jid]
            self.dadr = model.jnt_dofadr[self.jid]
            data.qpos[self.qadr+3:self.qadr+7] = [1.0, 0.0, 0.0, 0.0]
            data.qvel[self.dadr+3:self.dadr+6] = 0.0


    def lock_cylinders_to_belt_center(self, model, data, segments):
        """Mantiene los cilindros en la línea central de los tramos rectos."""
        if not self.cfg.LOCK_CYLINDERS_TO_CENTERLINE:
            return

        for i in self.cfg.CYLINDER_OBJECT_IDS:
            self.jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"obj{i}_free")
            self.qadr = model.jnt_qposadr[self.jid]
            self.dadr = model.jnt_dofadr[self.jid]
            self.x = float(data.qpos[self.qadr])
            self.y = float(data.qpos[self.qadr + 1])
            self.seg = segments[i]

            if self.seg == 0 and self.x < self.cfg.TURN_START:
                data.qpos[self.qadr + 1] = -self.cfg.LANE
                data.qvel[self.dadr + 1] = 0.0
            elif self.seg == 1 and self.y < self.cfg.TURN_START:
                data.qpos[self.qadr] = self.cfg.LANE
                data.qvel[self.dadr] = 0.0
            elif self.seg == 2 and self.x > -self.cfg.TURN_START:
                data.qpos[self.qadr + 1] = self.cfg.LANE
                data.qvel[self.dadr + 1] = 0.0
            elif self.seg == 3 and self.y > -self.cfg.TURN_START:
                data.qpos[self.qadr] = -self.cfg.LANE
                data.qvel[self.dadr] = 0.0
            elif self.seg == 4:
                data.qpos[self.qadr + 1] = self.cfg.BRANCH_Y
                data.qvel[self.dadr + 1] = 0.0


    def update_segment(self, pos, seg, obj_index):
        """Máquina de estados del circuito. La Cinta 5 solo la activa el robot."""
        self.x, self.y = float(pos[0]), float(pos[1])

        if seg == 0 and self.x >= self.cfg.CORNER_SWITCH:
            return 1
        if seg == 1 and self.y >= self.cfg.CORNER_SWITCH:
            return 2
        if seg == 2 and self.x <= -self.cfg.CORNER_SWITCH:
            return 3
        if seg == 3 and self.y <= -self.cfg.CORNER_SWITCH:
            return 0
        return seg



    def send_object_back_to_line(self, model, data, i, dadr, segments):
        """Al entrar en el horno, reaparece en Cinta 1 para mantener flujo continuo."""
        self.jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"obj{i}_free")
        self.qadr = model.jnt_qposadr[self.jid]
        self.x0 = -1.05 + 0.18 * (i % 3)
        data.qpos[self.qadr:self.qadr+3] = [self.x0, -self.cfg.LANE, self.cfg.OBJECT_REST_Z[i]]
        data.qpos[self.qadr+3:self.qadr+7] = [1.0, 0.0, 0.0, 0.0]
        data.qvel[dadr:dadr+6] = 0.0
        segments[i] = 0


    def apply_transport(self, model, data, body_ids, dof_addrs, segments, robot_state,cintas, horno):
        """Aplica transporte; el cubo reservado se inmoviliza hasta que lo toma el Panda."""
        data.xfrc_applied[:] = 0.0

        for i, bid in enumerate(body_ids):
            self.pos = data.xpos[bid]
            self.dadr = dof_addrs[i]
            self.vel = data.qvel[self.dadr:self.dadr+3]

            # Si el robot ya porta la pieza, la cinta no debe aplicar ninguna fuerza.
            if i == robot_state["grasped_obj"]:
                continue

            # Cubo reservado: mantenerlo centrado y detenido en la zona de pick.
            if i == robot_state["target_obj"] and robot_state["state"] != "idle":
                self.fx, self.fy = robot.robot_hold_force(self.pos, self.vel, self.cfg.ROBOT_CONFIG, self.cfg.MAX_FORCE)
            else:
                segments[i] = self.update_segment(self.pos, segments[i], i)
                seg_actual = segments[i]
                self.fx, self.fy = cintas[seg_actual].transport_force(self.pos, self.vel)

            data.xfrc_applied[bid, 0] = self.fx
            data.xfrc_applied[bid, 1] = self.fy

            if segments[i] == 4 and self.pos[0] <= self.cfg.OVEN_ENTRY_X:
                horno.registrar_pieza() 
                self.send_object_back_to_line(model, data, i, self.dadr, segments)
                continue

            if self.pos[2] < -0.1 or abs(self.pos[0]) > 5.0 or abs(self.pos[1]) > 2.4:
                self.jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"obj{i}_free")
                self.qadr = model.jnt_qposadr[self.jid]
                data.qpos[self.qadr:self.qadr+3] = [-0.8 + 0.25*i, -self.cfg.LANE, self.cfg.OBJECT_REST_Z[i]]
                data.qpos[self.qadr+3:self.qadr+7] = [1.0, 0.0, 0.0, 0.0]
                data.qvel[self.dadr:self.dadr+6] = 0.0
                segments[i] = 0