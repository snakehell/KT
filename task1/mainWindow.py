from settingsWindow import SettingsWindow
from functions import Functions
from mesh import Mesh
import numpy as np
import json
from scipy.integrate import odeint
from scipy.optimize import fsolve
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PyQt6.QtWidgets import (
    QMainWindow,
    QWidget,
    QGridLayout,
    QMenuBar
)
from PyQt6.QtGui import QAction


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("APP")
        self.setGeometry(100, 100, 1600, 800)

        menuBar = QMenuBar(self)
        self.setMenuBar(menuBar)
        
        settingsAction = QAction(self)
        settingsAction.setText("&Settings")
        settingsAction.triggered.connect(self.onSettingsClick)
        
        startAction = QAction(self)
        startAction.setText("&Start")
        startAction.triggered.connect(self.onStartClick)
        
        menuBar.addAction(settingsAction)
        menuBar.addAction(startAction)

        self.settingsWindow = SettingsWindow()
        self.settingsWindow.clicked.connect(self.onSettingsWindowClicked)
        
        main_widget = QWidget()
        self.setCentralWidget(main_widget)

        plot_layout = QGridLayout()
        main_widget.setLayout(plot_layout)

        self.figure = Figure()
        self.canvas = FigureCanvasQTAgg(self.figure)
        plot_layout.addWidget(self.canvas, 0, 0, 1, 1)
        
        self.ax0 = self.figure.add_subplot(231)
        self.ax1 = self.figure.add_subplot(232)
        self.ax2 = self.figure.add_subplot(233)
        self.ax3 = self.figure.add_subplot(234)
        self.ax4 = self.figure.add_subplot(235)
        self.ax_all = self.figure.add_subplot(236)
        
        self.ax = [self.ax0, self.ax1, self.ax2, self.ax3, self.ax4, self.ax_all]

        self.coef_filename = "DEFAULT"
        self.temp_filename = "DEFAULT"
        self.time_filename = "DEFAULT"
        self.outp_filename = "result.json"
        
        
    def onSettingsClick(self):
        self.settingsWindow.show()
    
    def onStartClick(self):
        if self.coef_filename == "DEFAULT":
            self.eps = [0.1, 0.1, 0.05, 0.02, 0.05]
            self.c = [900, 900, 520, 1930, 520]
            self.lambd = [[0, 240, 0, 0, 0], [240, 0, 130, 0, 0], [0, 130, 0, 118, 0], [0, 0, 118, 0, 10.5], [0, 0, 0, 10.5, 0]]
        else:
            self.c, self.eps, self.lambd = Functions.coef_init(self.coef_filename)
            
        self.C0 = 5.67
        self.A = 0.1
        self.model = Mesh("model2.obj")
        self.LS = self.model.s_ij*self.lambd
        self.EpSC = self.eps*self.model.s_i*self.C0
            
        if self.temp_filename == "DEFAULT":
            self.T = fsolve(Functions.functions_t0, np.zeros((5,), dtype=int), args=(self.LS, self.EpSC, self.A, self.c))
        else:
            self.T = Functions.temp_init(self.temp_filename)
        
        if self.time_filename == "DEFAULT":
            self.t_end = 1000
            self.t = np.linspace(1, self.t_end, 301)
        else:
            self.t, self.t_end = Functions.time_init(self.time_filename)
        

        self.T = [20, 0, 0, 10, 10]

        print("Initial temperature", self.T)
        
        self.sol1 = odeint(Functions.functions, self.T, self.t, args=(self.LS, self.EpSC, self.A, self.c))

        for i in range(5):
            self.ax[i].plot(self.t, self.sol1[:, i], label=f"y[{i}]")
            self.ax[5].plot(self.t, self.sol1[:, i], label=f"y[{i}]")
            self.ax[i].legend()
        self.ax[5].legend()
        self.canvas.draw_idle()

        dictionary = {
            "t": self.t.tolist(),
            "y0": self.sol1[:, 0].tolist(),
            "y1": self.sol1[:, 1].tolist(),
            "y2": self.sol1[:, 2].tolist(),
            "y3": self.sol1[:, 3].tolist(),
            "y4": self.sol1[:, 4].tolist()
        }
        json_object = json.dumps(dictionary, indent=6)
 
        with open(self.outp_filename, "w") as outfile:
            outfile.write(json_object)

    
    def onSettingsWindowClicked(self):
        if self.settingsWindow.coef_path_radio_default.isChecked():
            self.coef_filename = "DEFAULT"
        else:
            self.coef_filename = self.settingsWindow.coef_path_edit.text()
            
        if self.settingsWindow.temp_path_radio_default.isChecked():
            self.temp_filename = "DEFAULT"
        else:
            self.temp_filename = self.settingsWindow.temp_path_edit.text()
        
        if self.settingsWindow.time_path_radio_default.isChecked():
            self.time_filename = "DEFAULT"
        else:
            self.time_filename = self.settingsWindow.time_path_edit.text()
        
        if self.settingsWindow.outp_path_radio_default.isChecked():
            self.outp_filename = "DEFAULT"
        else:
            self.outp_filename = self.settingsWindow.outp_path_edit.text()
        print(self.coef_filename, self.temp_filename, self.time_filename, self.outp_filename)
