import cv2
import torch
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.animation as animation
import numpy as np
from frames import FileVideoStream, FPS
import time

# PyTorch Hub Model
model = torch.hub.load('ultralytics/yolov5', 'yolov5n')  # or yolov5n - yolov5x6, custom
model.classes = [2] #coches y motos

class Target_car:
    def __init__(self, xmin, ymin, xmax, ymax):
        self.id_bb = None
        self.xmin= xmin
        self.ymin= ymin
        self.xmax= xmax
        self.ymax= ymax
        self.count= 0
        self.tracking = [] #lista de bounding boxs del tracking [{"xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax}, {"xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax}, ..., {"xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax}]

    def set_coord(self, xmin, ymin, xmax, ymax):
        self.xmin= xmin
        self.ymin= ymin
        self.xmax= xmax
        self.ymax= ymax

    def increase_counter(self):
        self.count+=1

    def zero_counter(self):
        self.count = 0

    def add_tracking(self, xmin, ymin, xmax, ymax):
        bb = {"xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax}
        self.tracking.append(bb)

    def IoU(self, xmin2, ymin2, xmax2, ymax2): #Función que calcula la IoU entre dos bounding boxes
        x_inter1 = max(self.xmin, xmin2)
        y_inter1 = max(self.ymin, ymin2)
        x_inter2 = min(self.xmax, xmax2)
        y_inter2 = min(self.ymax, ymax2)

        width_inter = abs(x_inter2 - x_inter1)
        height_inter = abs(y_inter2 - y_inter1)
        area_inter = width_inter * height_inter

        width_box1 = abs(self.xmax - self.xmin)
        height_box1 = abs(self.ymax - self.ymin)

        width_box2 = abs(xmax2 - xmin2)
        height_box2 = abs(ymax2 - ymin2)

        area_box1 = width_box1 * height_box1
        area_box2 = width_box2 * height_box2

        area_union = area_box1 + area_box2 - area_inter

        iou= area_inter / area_union

        return iou #valor entre (0 a 1)

class Frame:
    def __init__(self):
        self.list_target_cars = []
    
    def load_targets_car(self, df, confidence=0.5): 
        for i in df.index: 
            number_confidence = float(df["confidence"][i])
            if number_confidence>= confidence:  #Si la detección del yolo tiene una confianza superior a 0.75
                xmin= float(df["xmin"][i])
                ymin= float(df["ymin"][i])
                xmax= float(df["xmax"][i])
                ymax= float(df["ymax"][i])

                if ymin>=550 and ymax>=550 and xmin>=52 and xmax<=475: #Si la bounding box está en la región de interés
                    target_car = Target_car(xmin, ymin, xmax, ymax) #Creamos un objecto de coche
                    target_car.id_bb = i #Le assignamos una id característica
                    target_car.add_tracking(target_car.xmin, target_car.ymin, target_car.xmax, target_car.ymax)
                    self.list_target_cars.append(target_car) # [object_Target_car1, object_Target_car2, ..., object_Target_carn]


class Tracker:
    def __init__(self, video_path= "/videos/output7.mp4",fps = 30, output_path = ""):

        self.fps = fps
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print("Total frames: ", self.total_frames)
        self.list_tracking = []
        self.count_up = 0
        self.count_down = 0

        if output_path=="":
            self.saving = False
        else:
            self.saving = True
            self.output_path = output_path
            self.frames_array = []
    
    def generator_optimized(self, sample_rate=1, epsilon = 0.5, max_not_appear = 3):

        t = 0
        last_num_id = 0

        fvs = FileVideoStream(self.video_path,fps = self.fps).start()
        time.sleep(1.0)
        # start the FPS timer
        fps = FPS().start()

        #iters = 70 #numero de frames que vamos a hacerle el tracking (solo ha sido para hacer comprobaciones, luego desparecerá pq lo haremos con todos los frames)

        iteration = 1 #iteración en la que estamos
        same_targets_car = [] #lista de objetos de target_car que harán match con la "lista_original"
        lens = []
        name_car = 1
        while(fvs.more()):

            frame = fvs.read()

            # saltem els frames que no ens interessen
            if iteration%sample_rate != 0:# or iteration > 100:# or iteration > 100 or iteration < 35:
                iteration += 1
                continue

            #frame = imutils.resize(frame, width=450)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = np.dstack([frame, frame, frame])

            print(f" ---------- FRAME {iteration} ---------- ")

            #self.cap.set(cv2.CAP_PROP_POS_FRAMES, fno)
            #_, image = self.cap.read()
            t0 = time.time()
            result_frame = model(frame)
            t+= time.time()-t0

            resultPd_frame = result_frame.pandas().xyxy[0] #pandas del resultado del primer frame

            #if not resultPd_frame.empty:
               #print(f'PANDAS= {resultPd_frame}')
               # result_frame.show()

            
            frame_tmp = Frame() #creamos el "frame temporal" que se mirará si hace match con la lista de la iteración anterior (self.list_tracking)
            frame_tmp.load_targets_car(resultPd_frame)

            match = False #variable que nos dirá si ha hecho match o no

            for bb_f1 in self.list_tracking: #per totes les bb de la lista de tracking (iteració anterior)
                #print("Calculem IoU de les bb amb la bb amb id = ", bb_f1.id_bb)
                iou_max= 0 #partimos de un iou malo

                for bb_f2 in frame_tmp.list_target_cars: #per totes les bb del frame tmp
                    iou= bb_f1.IoU(bb_f2.xmin, bb_f2.ymin, bb_f2.xmax, bb_f2.ymax) #calculamos la IoU de las bb de la iteración anterior con los de la nueva

                    if iou > iou_max and iou >= epsilon:
                        iou_max = iou
                        bb_match = bb_f2 #hace match!!
                
                if iou_max != 0: #Si hemos encontrado un buen IoU, y por lo tanto ha hecho match
                    bb_f1.zero_counter() #assignamos el contador a 0
                    bb_f1.add_tracking(bb_match.xmin, bb_match.ymin, bb_match.xmax, bb_match.ymax) #añadimos las coordenadas al tracking del objeto
                    
                    bb_f1.set_coord(bb_match.xmin, bb_match.ymin, bb_match.xmax, bb_match.ymax) #cambiamos las nuevas coordenadas al objeto que hace referencia
                    same_targets_car.append(bb_match) #añadimos el objeto del frame temporal que hace match a la lista de same_targets_car

                    match = True #hemos hecho match

                
                else:
                    bb_f1.increase_counter() #Como no ha hecho match sumamos 1



            #Encontramos los elementos que no se han encontrado pasados max_not_appear (num de iteraciones maximas que damos por perdido el objeto)
            new_list_tracking= [] #crearemos la lista que será nuestra self.list_tracking al haver hecho toda la iteración
            for element in self.list_tracking:
                if element.count < max_not_appear: #Si el elemento se ha encontrado antes de max_not_appear iteraciones
                    new_list_tracking.append(element) #permanece en la lista

                elif element.count == max_not_appear or iteration== self.total_frames: #Si el elemento no ha aparecido "5 veces" o estamos en el último frame--> decimos que ha desaparecido
                    # AQUI ES DONDE SE HA DE PONER EL CONTADOR (PARA ABAJO Y PARA ARRIBA)
                    if len(element.tracking) > 3: #Si el elemento tiene tracking largo (que no sea un falso positivo)
                        bb_pre = element.tracking[0] #bb_pre (inicial)
                        bb_act = element.tracking[-1] #bb_act (final)
                        ymin_pre = bb_pre["ymin"]
                        ymin_act = bb_act["ymin"]
                        diff = ymin_act - ymin_pre

                        if diff < 0: 
                            self.count_up += 1
                            print(f'El coche {name_car} ha salido del parking')
                            #print("===============UNO SALIO!!!!===============")
                        if diff > 0: 
                            self.count_down += 1
                            print(f'El coche {name_car} ha entrado al parking')
                            #print("===============UNO ENTRO!!!!===============")
                        
                        name_car+=1
            
            if self.saving:
                # add the counters to the frame
                cv2.putText(frame, f"In: {self.count_down}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(frame, f"Out: {self.count_up}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                # add the cropping rectangle to the frame (yellow)
                xmax = 475
                xmin = 52
                ymax = 958
                ymin = 550
                cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), (0, 255, 255), 2)

                # add the bb of the targets to the frame
                for track in self.list_tracking:
                    bb = track.tracking[-1]
                    cv2.rectangle(frame, (int(bb["xmin"]),int(bb["ymin"])), (int(bb["xmax"]),int(bb["ymax"])), (0, 255, 0), 2)

                    # draw a line of the centroid track to show the direction of the car
                    for i in range(1, len(track.tracking)):
                        bb1 = track.tracking[i-1]
                        c1 = (int((bb1["xmin"]+bb1["xmax"])/2), int((bb1["ymin"]+bb1["ymax"])/2))
                        bb2 = track.tracking[i]
                        c2 = (int((bb2["xmin"]+bb2["xmax"])/2), int((bb2["ymin"]+bb2["ymax"])/2))
                        cv2.line(frame, c1, c2, (0, 255, 0), 2)
                # save the frame
                self.frames_array.append(frame)

            #Encontramos los bounding box del frame temporal que no han hecho match con los que ya había en la iteración anterior
            new_targets_car = [x for x in frame_tmp.list_target_cars if x not in same_targets_car] #son "nuevos cotxes" que no habían aparecido en el vídeo anteriormente

            for element in new_targets_car: #añadimos a la nueva lista los elementos nuevos
                last_num_id+=1
                element.id_bb = (last_num_id) #le asignamos un nuevo id característico
                new_list_tracking.append(element)
        
            self.list_tracking = new_list_tracking #nueva self.list_tracking que se utilizará en la próxima iter

            # actualitzar el FPS counter
            fps.update()
            iteration+=1

        total = self.count_up + self.count_down
        print(" ---------- RESULTATS ----------")
        print("AI model time: ", t, "\n")
        print(f"TOTAL COTXES DETECTATS: {self.count_up + self.count_down} \nCONTADOR SORTIDA: {self.count_up} i CONTADOR ENTRADA: {self.count_down}")

        # save the frame array as a video
        if self.saving:
            frame_shape = self.frames_array[0].shape
            print(frame_shape)
            out = cv2.VideoWriter(self.output_path, cv2.VideoWriter_fourcc(*'mp4v'), 30/sample_rate, (frame_shape[1], frame_shape[0]))
            
            for frame in self.frames_array:
                out.write(frame)
            out.release()


for path in ["videos/output7.mp4", "videos/output2.mp4", "videos/output3.mp4"]:
    print(f'\n\nVIDEO: {path}')
    tracking = Tracker(path,fps = 1, output_path="output_videos/output2_long.mp4")
    tracking.generator_optimized(sample_rate = 5, epsilon = 0.5, max_not_appear = 2)
    