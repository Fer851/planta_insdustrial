import numpy as np 
import mujoco 

class CintaTransportadora:
    def __init__(self,id_cinta,nombre,velocidad,direccion,actuadores,vel_rodillo,cfg):
        self.id = id_cinta 
        self.nombre = nombre
        self.velocidad = float(velocidad) # m/s
        self.direccion = direccion 
        self.activa = True 

        # Parametros fisicos y actuadores asociados a Mujoco
        self.actuadores = actuadores 
        self.vel_rodillo = vel_rodillo
        self.cfg = cfg 
        
    def arrancar(self):
        self.activa = True
        
    def detener(self):
        self.activa = False 
    
    def conmutar(self):
        self.activa = not self.activa 
        return self.activa 
    
    def esta_activa(self):
        return self.activa 
    
    def initialize_belt_rollers(self, model, data):
            """Actualiza los rodillos según la bandera independiente de cada cinta."""
            valor = self.vel_rodillo if self.activa else 0.0
            
            for name in self.actuadores:
                self.aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
                if self.aid < 0:
                    raise RuntimeError(f"No se encontró el actuador de rodillo '{name}'.")
                data.ctrl[self.aid] = valor
    
    
    def transport_force(self, pos, vel):
            """Fuerza de transporte de las 4 cintas y de la rama al horno."""
            self.x, self.y = float(pos[0]), float(pos[1])
            self.vx, self.vy = float(vel[0]), float(vel[1])
    
            self.fx = 0.0
            self.fy = 0.0
    
            # Si la bandera de esta cinta está a False, se aplica frenado viscoso
            # sobre el plano XY. La física sigue activa y el resto de cintas/robot
            # continúan funcionando de forma independiente.
            if not self.esta_activa():
                self.fx = self.cfg.VEL_KP * (0.0 - self.vx)
                self.fy = self.cfg.VEL_KP * (0.0 - self.vy)
                return (
                    float(np.clip(self.fx, -self.cfg.MAX_FORCE, self.cfg.MAX_FORCE)),
                    float(np.clip(self.fy, -self.cfg.MAX_FORCE, self.cfg.MAX_FORCE)),
                )
    
            if self.id == 0:  # inferior: +X -> +Y
                alpha = np.clip((self.x - self.cfg.TURN_START) / max(self.cfg.CORNER_SWITCH - self.cfg.TURN_START, 1e-6), 0.0, 1.0)
                desired_vx = self.velocidad * (1.0 - 0.45 * alpha)
                desired_vy = self.velocidad * (0.85 * alpha)
                self.fx = self.cfg.VEL_KP * (desired_vx - self.vx)
                self.fy = self.cfg.VEL_KP * (desired_vy - self.vy) + (1.0 - alpha) * self.cfg.CENTER_KP * (-self.cfg.LANE - self.y)
    
            elif self.id == 1:  # derecha: +Y -> -X
                alpha = np.clip((self.y - self.cfg.TURN_START) / max(self.cfg.CORNER_SWITCH - self.cfg.TURN_START, 1e-6), 0.0, 1.0)
                desired_vx = -self.velocidad * (0.85 * alpha)
                desired_vy = self.velocidad * (1.0 - 0.45 * alpha)
                self.fx = self.cfg.VEL_KP * (desired_vx - self.vx) + (1.0 - alpha) * self.cfg.CENTER_KP * (self.cfg.LANE - self.x)
                self.fy = self.cfg.VEL_KP * (desired_vy - self.vy)
    
            elif self.id == 2:  # superior: -X -> -Y
                alpha = np.clip((-self.x - self.cfg.TURN_START) / max(self.cfg.CORNER_SWITCH - self.cfg.TURN_START, 1e-6), 0.0, 1.0)
                desired_vx = -self.velocidad * (1.0 - 0.45 * alpha)
                desired_vy = -self.velocidad * (0.85 * alpha)
                self.fx = self.cfg.VEL_KP * (desired_vx - self.vx)
                self.fy = self.cfg.VEL_KP * (desired_vy - self.vy) + (1.0 - alpha) * self.cfg.CENTER_KP * (self.cfg.LANE - self.y)
    
            elif self.id == 3:  # izquierda: -Y -> +X en esquina inferior
                alpha = np.clip((-self.y - self.cfg.TURN_START) / max(self.cfg.CORNER_SWITCH - self.cfg.TURN_START, 1e-6), 0.0, 1.0)
                desired_vx = self.velocidad * (0.85 * alpha)
                desired_vy = -self.velocidad * (1.0 - 0.45 * alpha)
                self.fx = self.cfg.VEL_KP * (desired_vx - self.vx) + (1.0 - alpha) * self.cfg.CENTER_KP * (-self.cfg.LANE - self.x)
                self.fy = self.cfg.VEL_KP * (desired_vy - self.vy)
    
            elif self.id == 4:  # Cinta 5 al horno: -X
                desired_vx = -self.velocidad
                desired_vy = 0.0
                self.fx = self.cfg.VEL_KP * (desired_vx - self.vx)
                self.fy = self.cfg.VEL_KP * (desired_vy - self.vy) + self.cfg.CENTER_KP * (self.cfg.BRANCH_Y - self.y)
    
            return float(np.clip(self.fx, -self.cfg.MAX_FORCE, self.cfg.MAX_FORCE)), float(np.clip(self.fy, -self.cfg.MAX_FORCE, self.cfg.MAX_FORCE))
    