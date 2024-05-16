'''Isabel Valdes Luevanos A01025802.'''

import sys
from io import BytesIO

import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from matplotlib.backends.backend_agg import FigureCanvas
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QLabel,
    QPushButton,
    QSlider,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)
from scipy.signal import butter, lfilter


class AudioUploader(QWidget):
    '''Clase de inicialización de app.'''
    def __init__(self):
        '''Funcion de inicio.'''
        super().__init__()
        self.initUI()

    def initUI(self)
    '''Inicialización de los aspectos gráficos de la interfaz, como título y pestañas.'''
        self.setWindowTitle('HMI de señales')
        self.setGeometry(0, 0, 1200, 800)
        
        self.tabs = QTabWidget(self)
        self.homeTab = QWidget()  
        self.fourierTab = QWidget()
        self.filtersTab = QWidget()
        self.saveTab = QWidget()  
        
        self.tabs.addTab(self.homeTab, "Home")  
        self.tabs.addTab(self.fourierTab, "Fourier")
        self.tabs.addTab(self.filtersTab, "Filtros")
        self.tabs.addTab(self.saveTab, "Guardar") 

        self.setupHomeTab()
        self.setupFourierTab()
        self.setupFiltersTab()
        self.setupSaveTab()

        layout = QVBoxLayout(self)
        layout.addWidget(self.tabs)
        self.setLayout(layout)

    def setupHomeTab(self):
        '''Configuración de la página de Home, se establecen los botones con sus nombres.'''
        layout = QVBoxLayout()
        self.button = QPushButton('Subir Audio', self)
        self.button.clicked.connect(self.openFileDialog)
        self.label = QLabel('No se ha seleccionado archivo', self)
        self.imageLabelHome = QLabel("Gráfica de la señal aparecerá aquí", self)
        layout.addWidget(self.button)
        layout.addWidget(self.label)
        layout.addWidget(self.imageLabelHome)
        self.homeTab.setLayout(layout)
 
    def setupFourierTab(self):
        '''Configuración de pestaña de Fourier con sus botones.'''
        fourierTabLayout = QVBoxLayout()
        self.imageLabelFourier = QLabel("Gráfica de Fourier aparecerá aquí", self)
        self.generateFourierButton = QPushButton("Generar Fourier", self)
        self.generateFourierButton.clicked.connect(self.displayFourier)
        fourierTabLayout.addWidget(self.generateFourierButton)
        fourierTabLayout.addWidget(self.imageLabelFourier)
        self.fourierTab.setLayout(fourierTabLayout)

    def setupFiltersTab(self):
        '''Configuración de pestaña de Filtros.'''
        self.filtersTabLayout = QVBoxLayout(self.filtersTab)
        self.addFiltersControls()

    def addFiltersControls(self):
        '''Botones y opciones para configuración de filtros como menu desplegable, slider para rango de frecuencia y corte y selección de orden de filtro.'''
        for i in reversed(range(self.filtersTabLayout.count())): 
            widget = self.filtersTabLayout.itemAt(i).widget()
            if widget is not None:
                widget.setParent(None)

        self.filterTypeCombo = QComboBox()
        self.filterTypeCombo.addItems(["Filtro pasa-bajas", "Filtro pasa-altas", "Filtro pasa bandas"])
        self.filtersTabLayout.addWidget(QLabel("Seleccionar tipo de filtro:"))
        self.filtersTabLayout.addWidget(self.filterTypeCombo)

        self.filtersTabLayout.addWidget(QLabel("Frecuencia de corte inferior (Hz):"))
        self.lowerCutoffFrequencySlider = QSlider(Qt.Horizontal)
        self.lowerCutoffFrequencySlider.setRange(20, 19980)
        self.lowerCutoffFrequencySlider.setValue(500)
        self.lowerCutoffFrequencySlider.setTickInterval(1000)
        self.lowerCutoffFrequencySlider.setTickPosition(QSlider.TicksBelow)
        self.filtersTabLayout.addWidget(self.lowerCutoffFrequencySlider)

        self.upperCutoffFrequencySlider = QSlider(Qt.Horizontal)
        self.upperCutoffFrequencySlider.setRange(20, 20000)
        self.upperCutoffFrequencySlider.setValue(2000)
        self.upperCutoffFrequencySlider.setTickInterval(1000)
        self.upperCutoffFrequencySlider.setTickPosition(QSlider.TicksBelow)
        self.filtersTabLayout.addWidget(QLabel("Frecuencia de corte superior (Hz):"))
        self.filtersTabLayout.addWidget(self.upperCutoffFrequencySlider)

        self.filtersTabLayout.addWidget(QLabel("Orden del filtro:"))
        self.filterOrderSpinBox = QSpinBox()
        self.filterOrderSpinBox.setRange(1, 10)
        self.filterOrderSpinBox.setValue(1)
        self.filtersTabLayout.addWidget(self.filterOrderSpinBox)

        self.applyFilterButton = QPushButton("Aplicar filtro", self)
        self.applyFilterButton.clicked.connect(self.applyFilter)
        self.filtersTabLayout.addWidget(self.applyFilterButton)

        self.backButton = QPushButton("Atrás", self)
        self.backButton.clicked.connect(self.addFiltersControls)
        self.filtersTabLayout.addWidget(self.backButton)

        #Display de la grafica de la senal filtrada

        self.imageLabelFilters = QLabel("Gráfica de la señal filtrada aparecerá aquí", self)
        self.filtersTabLayout.addWidget(self.imageLabelFilters)

    def setupSaveTab(self):
        '''Pestaña de guardado de archivos, botones de selección de formato.'''
        layout = QVBoxLayout()
        self.outputFormatCombo = QComboBox()
        self.outputFormatCombo.addItems(["WAV - PCM_16", "FLAC - PCM_24"])
        layout.addWidget(QLabel("Seleccionar formato de salida:"))
        layout.addWidget(self.outputFormatCombo)

        self.saveButton = QPushButton("Save Result", self)
        self.saveButton.clicked.connect(self.saveFile)
        layout.addWidget(self.saveButton)

        self.saveTab.setLayout(layout)

    def displayFourier(self):
        ''' Display de gráfica de transformada de Fourier.'''
        if hasattr(self, 'sound') and self.sound is not None:
            fft_spectrum = np.fft.rfft(self.sound)
            freq = np.fft.rfftfreq(len(self.sound), d=1./self.sr)
            self.displayAudioGraph(freq, np.abs(fft_spectrum), "Transformada de Fourier", x_label="Frecuencia (Hz)")

    def applyFilter(self):
        '''Aplicar configuraciones de los filtros, rango para pasa bandas.'''
        filter_type = self.filterTypeCombo.currentText()
        lower_cutoff = self.lowerCutoffFrequencySlider.value()
        upper_cutoff = self.upperCutoffFrequencySlider.value()
        order = self.filterOrderSpinBox.value()
        if 'pasa bandas' in filter_type:
            cutoff_frequencies = [lower_cutoff, upper_cutoff]
            normalized_cutoff = [x / (self.sr / 2) for x in cutoff_frequencies]  
            b, a = butter(order, normalized_cutoff, btype='band', fs=self.sr)
        else:
            cutoff = lower_cutoff if 'pasa-bajas' in filter_type else upper_cutoff
            normalized_cutoff = cutoff / (self.sr / 2)
            b, a = butter(order, normalized_cutoff, btype='low' if 'pasa-bajas' in filter_type else 'high', fs=self.sr)
        self.filtered_sound = lfilter(b, a, self.sound)
        self.displayFilteredGraph(np.arange(len(self.filtered_sound)) / self.sr, self.filtered_sound, "Señal filtrada")

    def displayFilteredGraph(self, x, y, title):
        '''Display de gráfica de filtros.'''
        self.addFiltersControls()  
        fig, ax = plt.subplots()
        ax.plot(x, y)
        ax.set_title(title)
        ax.set_xlabel("Tiempo (s)")
        ax.set_ylabel("Amplitud")
        canvas = FigureCanvas(fig)
        canvas.draw()
        buf = BytesIO()
        canvas.print_png(buf)
        buf.seek(0)
        pixmap = QPixmap()
        pixmap.loadFromData(buf.getvalue(), "PNG")
        plt.close(fig)
        self.imageLabelFilters.setPixmap(pixmap)
        self.imageLabelFilters.setScaledContents(True)
 
    def openFileDialog(self):
        '''Clase para guardar archivo de audio.'''
        options = QFileDialog.Options()
        fileName, _ = QFileDialog.getOpenFileName(self, "Seleccione archivo de audio", "", "Archivos de audio (*.wav *.mp3 *.aac)", options=options)
        if fileName:
            self.label.setText(f'Archivo seleccionado: {fileName}')
            self.displayAudioGraphs(fileName)

    def displayAudioGraphs(self, filePath):
        '''Display grafica de audio'''
        self.sound, self.sr = sf.read(filePath)
        if self.sound.ndim > 1:
            self.sound = self.sound.mean(axis=1)
        self.displayAudioGraph(np.arange(len(self.sound)) / self.sr, self.sound, "Señal de audio")
 
    def displayAudioGraph(self, x, y, title, x_label="Tiempo (s)"):
        '''Función para graficar señales.'''
        fig, ax = plt.subplots()
        ax.plot(x, y)
        ax.set_title(title)
        ax.set_xlabel(x_label)
        ax.set_ylabel("Amplitud")
        canvas = FigureCanvas(fig)
        canvas.draw()
        buf = BytesIO()
        canvas.print_png(buf)
        buf.seek(0)
        pixmap = QPixmap()
        pixmap.loadFromData(buf.getvalue(), "PNG")
        plt.close(fig)
        if "Señal de audio" in title:
            self.imageLabelHome.setPixmap(pixmap)
        elif "Transformada de Fourier" in title:
            self.imageLabelFourier.setPixmap(pixmap)
        elif "Señal filtrada" in title:
            self.imageLabelFilters.setPixmap(pixmap)
            self.imageLabelFilters.setScaledContents(True)

    def saveFile(self):
        '''Mostrar error por formato de archivo de audio incorrecto.'''
        selected_format = self.outputFormatCombo.currentText().split(" - ")
        if len(selected_format) != 2:
            print("Error de formato")
            return

        format = selected_format[0].lower() 
        subtype = selected_format[1].lower() 

        options = QFileDialog.Options()
        file_extension = format if format != 'aiff' else 'aif' 
        fileName, _ = QFileDialog.getSaveFileName(self, "Guardar archivo", "", f"Archivo de audio (*.{file_extension})", options=options)
        if fileName:
            try:
                processedSignal = np.int16(
                    self.filtered_sound * 32767 / np.max(np.abs(self.filtered_sound))
                )
                sf.write(fileName, processedSignal, self.sr, format=format, subtype=subtype)
                print("Archivo guardado exitosamente.")
            except Exception as e:
                print(f"Error al guardar el archivo: {str(e)}")

def main():
    '''Inicialización de aplicación.'''
    app = QApplication(sys.argv)
    ex = AudioUploader()
    ex.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
