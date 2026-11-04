

class Horno:
    def __init__(self,geom_id, temperatura_inicial = 20.0):
        self.numero_piezas = 0 
        self.temperatura = temperatura_inicial 
        self.geom_id = geom_id
        
    def registrar_pieza(self): 
        self.numero_piezas +=1
    
    def aumentar_temperatura(self, incremento):
        self.temperatura += incremento
    
    def disminuir_temperatura(self, decremento):
        self.temperatura -= decremento 
    
    def obtener_temperatura(self):
        return self.temperatura 

    def actualizar_color(self, model):
        if self.temperatura < 50.0:
         color = [0.1, 0.3, 1.0, 1.0]
        elif self.temperatura < 100.0:
          color = [1.0, 0.6, 0.1, 1.0]
        else:
           color = [1.0, 0.1, 0.1, 1.0]
        model.geom_rgba[self.geom_id] = color