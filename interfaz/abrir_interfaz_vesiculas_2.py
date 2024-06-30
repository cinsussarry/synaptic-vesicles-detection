import sys
from PyQt5.uic import loadUi
from PyQt5.QtWidgets import QApplication, QMainWindow, QSlider, QFileDialog, QInputDialog, QMessageBox
import pyqtgraph as pg
import tifffile
import cv2
import numpy as np
import shutil
import os
from predict_mito import multiply_mask, predict_mito
from functions import region_growing

class MyApp(QMainWindow):
    def __init__(self):
        super(MyApp, self).__init__()
        loadUi('/Users/catalinainsussarry/Downloads/prueba_interfaz_vesicles.ui', self)
        self.image_widget.ui.histogram.hide()
        self.image_widget.ui.roiBtn.hide()
        self.image_widget.ui.menuBtn.hide()
        self.current_index = 0
        self.actionNew.triggered.connect(self.create_new_project)
        self.actionOpen.triggered.connect(self.open_project)
        self.actionMitochondria.triggered.connect(self.predict_mitochondria)
        self.checkBox_neurons.stateChanged.connect(self.estado_mascara_neuronas)
        self.checkBox_mito.stateChanged.connect(self.estado_mascara_mitocondrias)
        self.actionMitochondria_Label.triggered.connect(self.label_mitochondria)
        self.mouse_click_connection = None
        self.actionMitochondria_Add.triggered.connect(self.add_mitochondria)
        self.actionFinish_Add.triggered.connect(self.finish_click)
        #self.actionMitochondria_Delete.triggered.connect(self.delete_mitochondria)
        self.actionMitochondria_Undo.triggered.connect(self.undo_mitochondria)

    def cargar_imagen(self, path_volumen):
        print("cargar imagen")
        self.tiff=tifffile.imread(path_volumen)
        self.tiff=np.transpose(self.tiff, axes=(0,2,1))
        #SOLO POR AHORA
        #self.tiff=self.tiff[9:,:,:]
        self.tiff=self.tiff[29:129,:,:]
        self.image_widget.getView().setAspectLocked(True)
        self.image_widget.getView().autoRange(padding=0.0)

        self.image_widget.timeLine.setPen((255,255,0,200))
        #self.image_widget.setImage(self.tiff)
        self.volumen_con_mascaras = self.tiff.copy()
        print(self.tiff.shape)
        size=os.path.getsize(path_volumen)
        self.GB=np.round(size/10**(9),1)
        self.type=self.tiff.dtype
        self.frames=self.tiff.shape[0]
        print(self.frames)
        self.label_dimension.setText(str(self.current_index+1)+"/"+str(self.frames)+";  "  +str(self.tiff.shape[2])+"x"+str(self.tiff.shape[1]) + ";  "+str(self.GB)+ " GB;  " + str(self.type))

    
        self.image_widget.getView().scene().sigMouseMoved.connect(self.on_mouse_moved)
        self.image_widget.timeLine.sigPositionChanged.connect(self.on_time_line_changed)
    
    def cargar_mascara(self, path_mascara):
        print("Cargar máscara")

        self.mascara = tifffile.imread(path_mascara)
        self.mascara = np.transpose(self.mascara, axes=(0, 2, 1))
        self.mascara = self.mascara[:100,:,:] #SOLO PARA REDUCIR LA MEMORIA
        print(self.mascara.shape)

        if self.mascara.shape != self.tiff.shape:
            QMessageBox.warning(self, "Error", "La máscara y el volumen no tienen las mismas dimensiones.")
            return
        self.mascara_exists=True
        # Crear una versión oscurecida de la imagen
        oscurecido_volumen = (self.tiff * 0.5).astype(np.uint8)

        # Aplicar la máscara: mantener la imagen original donde la máscara es blanca (>= 250), y usar la versión oscurecida donde la máscara es negra
        self.volumen_con_mascaras = np.where(self.mascara >= 250, self.tiff, oscurecido_volumen).astype(np.uint8)
  
    def estado_mascara_neuronas(self, state):
        if state==0:
            self.volumen_a_visualizar=self.tiff
        if state==2:
            self.volumen_a_visualizar=self.volumen_con_mascaras

        mitocondrias_state = 2 if self.checkBox_mito.isChecked() else 0
        self.estado_mascara_mitocondrias(mitocondrias_state)
    
    def estado_mascara_mitocondrias(self, state):
        if state==0 and self.checkBox_neurons.isChecked():
            self.volumen_a_visualizar=self.volumen_con_mascaras
        if state==2 and self.checkBox_neurons.isChecked():
            self.volumen_a_visualizar=self.volumen_con_mitocondrias
        if state==2 and not self.checkBox_neurons.isChecked():
            self.volumen_a_visualizar=self.volumen_con_mitocondrias_sin_neuronas 
        self.visualizar_imagen(start_index=self.current_index)
            
    def cargar_mascara_mitocondrias(self, path_mascara):
        
        self.mascara_mitocondrias = tifffile.imread(path_mascara)
        self.mascara_mitocondrias = np.transpose(self.mascara_mitocondrias, axes=(0, 2, 1))
        self.mascara_mitocondrias = self.mascara_mitocondrias[:100,:,:]
        print(self.mascara_mitocondrias.shape)

        if self.mascara_mitocondrias.shape != self.tiff.shape:
            QMessageBox.warning(self, "Error", "La máscara de mitocondrias y el volumen no tienen las mismas dimensiones.")
            return
        self.mascara_mitocondrias_exists=True
    
    def colorear_mascara_mitocondrias(self, volumen, mask, RGB):  
        # Definir el color rojo translúcido
        alpha = 0.2
        red_color = np.array([253, 60, 20], dtype=np.uint8)  # Color rojo

        if not RGB:
            # Convertir la imagen a color
            volumen_color = np.stack([volumen] * 3, axis=-1)
        else:
            volumen_color=volumen

        # Preparar el volumen con máscaras
        volumen_con_mitocondrias = np.copy(volumen_color)
        # Crear la superposición roja
        red_overlay = np.zeros_like(volumen_color)
        red_overlay[mask >= 250] = red_color
        # Aplicar la transparencia sobre toda la imagen
        #volumen_con_mitocondrias = (alpha * red_overlay + (1 - alpha) * volumen_color).astype(np.uint8)
        volumen_con_mitocondrias = (alpha * red_overlay + volumen_color).astype(np.uint8)
        return volumen_con_mitocondrias

    def visualizar_imagen(self, start_index):
        self.image_widget.getView().setAspectLocked(True)
        self.image_widget.getView().autoRange(padding=0.0)
        self.image_widget.timeLine.setPen((255,255,0,200))
        self.image_widget.setImage(self.volumen_a_visualizar)
        self.image_widget.setCurrentIndex(start_index)
        
    def on_time_line_changed(self):
        self.current_index = int(self.image_widget.currentIndex)
        #print(f"Current frame index: {self.current_index}")
        self.label_dimension.setText(str(self.current_index+1)+"/"+str(self.frames)+";  "  +str(self.tiff.shape[2])+"x"+str(self.tiff.shape[1]) + ";  "+str(self.GB)+ " GB;  " + str(self.type))
    

    def on_mouse_moved(self, pos):
        mouse_point = self.image_widget.getView().mapSceneToView(pos)
        self.x, self.y = int(mouse_point.x()), int(mouse_point.y())

        if 0 <= self.x < self.tiff.shape[1] and 0 <= self.y < self.tiff.shape[2]:
            self.label_coordenadas.setText(f"x= {self.x}, y= {self.y}, z= {self.current_index}")
            self.label_value.setText(f"value= {self.tiff[self.current_index, self.x, self.y]}")
            if self.mascara_etiquetada_mitochondria_exists:
                label=self.mascara_etiquetada_mitochondria[self.current_index,self.x,self.y]
                if label!=0:
                    self.label_mitocondria.setText(f"label mitochondria: {label}")
                else:
                    self.label_mitocondria.setText(f"label mitochondria:")

        else:
            self.label_coordenadas.setText("")
            self.label_value.setText("")
    
    def create_new_project(self):
        # Pedir nombre del proyecto
        project_name, ok = QInputDialog.getText(self, 'Nuevo Proyecto', 'Ingrese el nombre del proyecto:')
        if ok and project_name:
            
            # Seleccionar la carpeta donde se creará el proyecto
            project_dir = QFileDialog.getExistingDirectory(self, 'Seleccione la carpeta donde se creará el proyecto')
            if not project_dir:
                return  # El usuario canceló la selección de la carpeta
            
            project_dir = os.path.join(project_dir, project_name)
            
            try:
                os.makedirs(project_dir)
            except FileExistsError:
                QMessageBox.warning(self, "Error", "El proyecto ya existe.")
                return
            
            # Seleccionar el primer archivo (volumen a analizar)
            volume_file, _ = QFileDialog.getOpenFileName(self, "Seleccione el volumen a analizar", "", "Archivos (*.tif *.tiff)")
            if not volume_file:
                QMessageBox.warning(self, "Error", "Debe seleccionar un archivo de volumen.")
                return
            
            #Desea seleccionar una mascara de neuronas de interes?

            # Seleccionar el segundo archivo (máscara)
            mask_file, _ = QFileDialog.getOpenFileName(self, "Seleccione la máscara de neuronas de interes, en caso de tenerla", "", "Archivos (*.tif *.tiff)")
            if not mask_file:
                QMessageBox.warning(self, "Advertencia!", "No se ha seleccionado un archivo de mascara")
                #pass
            else:
                mask_dest = os.path.join(project_dir, 'mascara_neuronas' + os.path.splitext(mask_file)[1])
                shutil.copy2(mask_file, mask_dest)
            
            # Copiar los archivos a la nueva carpeta

            volume_dest = os.path.join(project_dir, 'volumen' + os.path.splitext(volume_file)[1])
            
            shutil.copy2(volume_file, volume_dest)
            
            QMessageBox.information(self, "Éxito", f"El proyecto '{project_name}' ha sido creado y los archivos han sido copiados.")
            self.analizar_carpeta(project_dir)
            self.path_carpeta=project_dir

    def open_project(self):
        # Seleccionar la carpeta del proyecto existente
        project_dir = QFileDialog.getExistingDirectory(self, 'Seleccione la carpeta del proyecto existente')
        if not project_dir:
            return None  # El usuario canceló la selección de la carpeta

        # Verificar la existencia de archivos clave en la carpeta
        volumen_path = None

        for archivo in os.listdir(project_dir):
            archivo_path = os.path.join(project_dir, archivo)
            
            if archivo.lower() == 'volumen.tif' or archivo.lower() == 'volumen.tiff':
                volumen_path = archivo_path

        if not volumen_path:
            QMessageBox.warning(self, "Error", "No se encontró el archivo 'volumen.tif' o 'volumen.tiff' en la carpeta seleccionada.")
            return None
        self.analizar_carpeta(project_dir)
        self.path_carpeta=project_dir

    def analizar_carpeta(self, project_dir):
    # Inicializamos los paths como None
        path_volumen = None
        path_mascara_neuronas = None
        path_mascara_mitocondrias = None
        path_mascara_vesiculas = None
        self.mascara_exists=False
        self.mascara_mitocondrias_exists=False
        self.multiplied_exists=False
        self.mascara_etiquetada_mitochondria_exists=False

        # Recorremos los archivos de la carpeta
        for archivo in os.listdir(project_dir):
            # Construimos el path completo del archivo
            archivo_path = os.path.join(project_dir, archivo)
            
            # Verificamos si el archivo es volumen.tif o volumen.tiff
            if archivo.lower() == 'volumen.tif' or archivo.lower() == 'volumen.tiff':
                path_volumen = archivo_path
                self.cargar_imagen(path_volumen) 

            # Verificamos si el archivo es mascaras_neuronas
            if archivo.lower() == 'mascara_neuronas.tif' or archivo.lower() == 'mascara_neuronas.tiff':
                path_mascara_neuronas = archivo_path
                self.checkBox_neurons.setChecked(True)
                self.cargar_mascara(path_mascara_neuronas)
                self.volumen_a_visualizar=self.volumen_con_mascaras

            # Verificamos si el archivo es mascaras_mitocondrias
            if archivo.lower() == 'mascara_mitocondrias.tif':
                path_mascara_mitocondrias = archivo_path
                self.checkBox_mito.setChecked(True)
                self.cargar_mascara_mitocondrias(path_mascara_mitocondrias)
                self.volumen_con_mitocondrias=self.colorear_mascara_mitocondrias(self.volumen_con_mascaras, self.mascara_mitocondrias,RGB=False)
                self.volumen_con_mitocondrias_sin_neuronas=self.colorear_mascara_mitocondrias(self.tiff,self.mascara_mitocondrias,RGB=False)
                self.volumen_a_visualizar=self.volumen_con_mitocondrias

            # Verificamos si el archivo es mascaras_vesiculas
            if archivo.lower() == 'mascara_vesiculas.tif':
                path_mascara_vesiculas = archivo_path
                print('Hay mascaras de vesiculas')
        
        self.visualizar_imagen(start_index=self.current_index)
    
    def cargar_multiplied(self):
        for archivo in os.listdir(self.path_carpeta):
            archivo_path = os.path.join(self.path_carpeta, archivo)
            if archivo.lower() == 'multiplied.tif':
                multiplied_path = archivo_path
                self.multiplied = tifffile.imread(multiplied_path)
                self.multiplied = np.transpose(self.multiplied, axes=(0, 2, 1))
                print('multiplied shape',self.multiplied.shape)
                self.multiplied_exists=True
        if not self.mascara_exists:
            QMessageBox.warning(self, "Error", "Debe tener una máscara de neuronas para predecir mitocondrias y vesiculas.")
            return None
        if not self.multiplied_exists or (self.multiplied.shape != self.tiff.shape):
            self.multiplied=multiply_mask(self.tiff, self.mascara, self.path_carpeta) #se crea la multiplied y se guarda
    
    def predict_mitochondria (self):
        self.cargar_multiplied()
        self.mascara_mitocondrias=predict_mito(self.multiplied, self.path_carpeta) #se predice y se guarda mascara mitocondrias
        print(np.unique(self.mascara_mitocondrias))
        self.checkBox_mito.setChecked(True)
        self.mascara_mitocondrias_exists=True
        self.volumen_con_mitocondrias=self.colorear_mascara_mitocondrias(self.volumen_con_mascaras, self.mascara_mitocondrias,RGB=False)
        self.volumen_con_mitocondrias_sin_neuronas=self.colorear_mascara_mitocondrias(self.tiff,self.mascara_mitocondrias,RGB=False)
        self.volumen_a_visualizar=self.volumen_con_mitocondrias
        self.visualizar_imagen(start_index=self.current_index)
    
    def label_mitochondria(self):
        #se llama a archivo que corre pyclesperanto 3D
        #POR AHORA CARGO DIRECTAMENTE EL VOLUMEN YA ETIQUETADO
        path='/Users/catalinainsussarry/Downloads/segmented_3d_mito_fullvolume.tif'
        self.mascara_etiquetada_mitochondria_exists=True
        self.mascara_etiquetada_mitochondria = tifffile.imread(path)
        self.mascara_etiquetada_mitochondria = np.transpose(self.mascara_etiquetada_mitochondria, axes=(0, 2, 1))
        self.mascara_etiquetada_mitochondria = self.mascara_etiquetada_mitochondria[:100,:,:]

    def add_mitochondria(self):
        if self.mouse_click_connection is None:
                    self.mouse_click_connection = self.image_widget.scene.sigMouseClicked.connect(self.on_mouse_clicked)
        print("Mouse click event connected.")
    
    def finish_click(self):
        # Desconectar el evento de clic del mouse de la función
        if self.mouse_click_connection is not None:
            self.image_widget.scene.sigMouseClicked.disconnect(self.on_mouse_clicked)
            self.mouse_click_connection = None
        print("Mouse click event disconnected.")
    
    def on_mouse_clicked(self, event):
        # Get the position of the mouse click in the image
        mouse_point = self.image_widget.getView().mapSceneToView(event.scenePos())
        self.x, self.y = int(mouse_point.x()), int(mouse_point.y())
        # Print the coordinates
        print(f"Mouse clicked at: x={self.x}, y={self.y}")
        new_mask=region_growing(self.tiff[self.current_index,:,:],self.mascara_mitocondrias[self.current_index,:,:],[(self.x,self.y)],30)
        self.mascara_mitocondrias_undo=(self.mascara_mitocondrias).copy()
        self.volumen_con_mitocondrias_undo=(self.volumen_con_mitocondrias).copy()
        self.volumen_con_mitocondrias_sin_neuronas_undo=(self.volumen_con_mitocondrias_sin_neuronas).copy()
        self.mascara_mitocondrias[self.current_index,:,:]=new_mask
        self.volumen_con_mitocondrias[self.current_index,:,:]=self.colorear_mascara_mitocondrias(self.volumen_con_mascaras[self.current_index,:,:],new_mask,RGB=False)
        self.volumen_con_mitocondrias_sin_neuronas[self.current_index,:,:]=self.colorear_mascara_mitocondrias(self.tiff[self.current_index,:,:],new_mask,RGB=False)
        self.visualizar_imagen(start_index=self.current_index)
    
    def undo_mitochondria(self):
        self.mascara_mitocondrias=self.mascara_mitocondrias_undo
        self.volumen_con_mitocondrias=self.volumen_con_mitocondrias_undo
        self.volumen_con_mitocondrias_sin_neuronas=self.volumen_con_mitocondrias_sin_neuronas_undo
        mitocondrias_state = 2 if self.checkBox_mito.isChecked() else 0
        self.estado_mascara_mitocondrias(mitocondrias_state)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MyApp()
    window.show()
    sys.exit(app.exec_())
