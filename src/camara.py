import cv2 
import numpy as np
import src.robot as robot

class CamaraInspeccion: 
    def __init__(self,nombre,cinta_asociada):
        self.nombre = nombre 
        self.cinta = cinta_asociada 
        self.objeto_detectado = False 
        self.forma_detectada = None
        self.imagen_actual = None  
    
    def capturar(self,renderer,data):
        """Capturar la imagen de esta cámara específica en Mujoco"""
        renderer.update_scene(data, camera = self.nombre)
        rgb = renderer.render()
        # Convertimos de RGB a BGR
        self.imagen_actual = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        return self.imagen_actual
    
    
    def detectar_forma(self,imagen):
        """OpenCV para detectar formas como cilindro o caja"""
        # Convertimos a escala de grises
        gray = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
        # Ajustamos el umbral para destaca las piezas sobre el fondo 
        _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)
        # Buscamos contornos 
        contornos, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        self.objeto_detectadof = False 
        self.forma_detectada = None 
        # Centralizar objeto y camara 
        centro_img_x = imagen.shape[1] // 2
        centro_img_y = imagen.shape[0] // 2
        
        
        
        for cnt in contornos: 
            area = cv2.contourArea(cnt)
            # Filtramos ruido (áreas muy pequeñas) y el propio borde de la cinta (áreas gigantes)
            if 50 < area < 12000:
                # calcular el centro del contorno 
                M = cv2.moments(cnt)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    
                    # Solo detenemos la cinta si la pieza está cerca del centro de la cámara
                    distancia_al_centro = np.sqrt((cx - centro_img_x)**2 + (cy - centro_img_y)**2)
                    
                    if distancia_al_centro < 80.0:  # Rango de detección central
                        self.objeto_detectado = True
                        
                        # Aproximación poligonal
                        perimetro = cv2.arcLength(cnt, True)
                        aprox = cv2.approxPolyDP(cnt, 0.01 * perimetro, True)
                        
                        # Si tiene más de 6 vértices rectos, lo consideramos Cilindro
                        if len(aprox) > 6:
                            self.forma_detectada = "Cilindro"
                        else:
                            self.forma_detectada = "Caja"
                        
                        # Dibujamos el polígono en la imagen
                        cv2.drawContours(imagen, [aprox], -1, (0, 255, 0), 2)
                        cv2.putText(imagen, self.forma_detectada, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                        break
                        
                        
        return self.forma_detectada
                      
                
    
    def actualizar(self, renderer, data):
        img = self.capturar(renderer, data)
        self.detectar_forma(img)
        
        # Si detecta objeto, detiene su propia cinta
        if self.objeto_detectado:
            self.cinta.detener()
            
        return img
    
   
