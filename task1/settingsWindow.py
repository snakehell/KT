from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QLineEdit,
    QMainWindow,
    QPushButton,
    QWidget,
    QGridLayout,
    QButtonGroup,
    QLabel,
    QRadioButton,
    QFileDialog
)


class SettingsWindow(QMainWindow):
    clicked = pyqtSignal()
    
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Settings")
        self.setGeometry(300, 300, 850, 300)
        
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        
        self.coef_path_label = QLabel("Path to coefficient file")
        self.coef_path_radio_path = QRadioButton("Select file:")
        self.coef_path_radio_default = QRadioButton("Default")
        self.coef_path_radio_path.setChecked(True)
        self.coef_path_edit = QLineEdit("input filepath")
        self.coef_path_button_browse = QPushButton("Browse")
        self.coef_path_button_browse.clicked.connect(self.onBrowseCoefClick)
        self.coef_path_radio_group = QButtonGroup(main_widget)
        self.coef_path_radio_group.addButton(self.coef_path_radio_path, 1)
        self.coef_path_radio_group.addButton(self.coef_path_radio_default, 2)
        
        self.temp_path_label = QLabel("Path to temperature start file")
        self.temp_path_radio_path = QRadioButton("Select file:")
        self.temp_path_radio_default = QRadioButton("Default")
        self.temp_path_radio_path.setChecked(True)
        self.temp_path_edit = QLineEdit("input filepath")
        self.temp_path_button_browse = QPushButton("Browse")
        self.temp_path_button_browse.clicked.connect(self.onBrowseTempClick)
        self.temp_path_radio_group = QButtonGroup(main_widget)
        self.temp_path_radio_group.addButton(self.temp_path_radio_path, 1)
        self.temp_path_radio_group.addButton(self.temp_path_radio_default, 2)
        
        self.time_path_label = QLabel("Path to time interval file")
        self.time_path_radio_path = QRadioButton("Select file:")
        self.time_path_radio_default = QRadioButton("Default")
        self.time_path_radio_path.setChecked(True)
        self.time_path_edit = QLineEdit("input filepath")
        self.time_path_button_browse = QPushButton("Browse")
        self.time_path_button_browse.clicked.connect(self.onBrowseTimeClick)
        self.time_path_radio_group = QButtonGroup(main_widget)
        self.time_path_radio_group.addButton(self.time_path_radio_path, 1)
        self.time_path_radio_group.addButton(self.time_path_radio_default, 2)
        
        self.outp_path_label = QLabel("Path to output file")
        self.outp_path_radio_path = QRadioButton("Select file:")
        self.outp_path_radio_default = QRadioButton("Default")
        self.outp_path_radio_path.setChecked(True)
        self.outp_path_edit = QLineEdit("input filepath")
        self.outp_path_button_browse = QPushButton("Browse")
        self.outp_path_button_browse.clicked.connect(self.onBrowseOutClick)
        self.outp_path_radio_group = QButtonGroup(main_widget)
        self.outp_path_radio_group.addButton(self.outp_path_radio_path, 1)
        self.outp_path_radio_group.addButton(self.outp_path_radio_default, 2)
        
        self.applyButton = QPushButton("Apply")
        self.applyButton.clicked.connect(self.clicked)
        
        self.input_layout = QGridLayout(main_widget)
        main_widget.setLayout(self.input_layout)
        
        self.input_layout.addWidget(self.coef_path_label, 0, 0)
        self.input_layout.addWidget(self.coef_path_radio_path, 0, 1)
        self.input_layout.addWidget(self.coef_path_radio_default, 1, 1)
        self.input_layout.addWidget(self.coef_path_edit, 0, 2, 1, 4)
        self.input_layout.addWidget(self.coef_path_button_browse, 0, 6)
        
        self.input_layout.addWidget(self.temp_path_label, 2, 0)
        self.input_layout.addWidget(self.temp_path_radio_path, 2, 1)
        self.input_layout.addWidget(self.temp_path_radio_default, 3, 1)
        self.input_layout.addWidget(self.temp_path_edit, 2, 2, 1, 4)
        self.input_layout.addWidget(self.temp_path_button_browse, 2, 6)
        
        self.input_layout.addWidget(self.time_path_label, 4, 0)
        self.input_layout.addWidget(self.time_path_radio_path, 4, 1)
        self.input_layout.addWidget(self.time_path_radio_default, 5, 1)
        self.input_layout.addWidget(self.time_path_edit, 4, 2, 1, 4)
        self.input_layout.addWidget(self.time_path_button_browse, 4, 6)
        
        self.input_layout.addWidget(self.outp_path_label, 6, 0)
        self.input_layout.addWidget(self.outp_path_radio_path, 6, 1)
        self.input_layout.addWidget(self.outp_path_radio_default, 7, 1)
        self.input_layout.addWidget(self.outp_path_edit, 6, 2, 1, 4)
        self.input_layout.addWidget(self.outp_path_button_browse, 6, 6)
        
        self.input_layout.addWidget(self.applyButton, 8, 6)

    def onBrowseCoefClick(self):
        filename, ok = QFileDialog.getOpenFileName(
            self,
            "Select coefficient file", 
            "./", 
            "Json files (*.json *.csv)"
        )
        if filename:
            self.coef_path_edit.setText(filename)
    
    def onBrowseTempClick(self):
        filename, ok = QFileDialog.getOpenFileName(
            self,
            "Select temperature file", 
            "./", 
            "Json files (*.json *.csv)"
        )
        if filename:
            self.temp_path_edit.setText(filename)
    
    def onBrowseTimeClick(self):
        filename, ok = QFileDialog.getOpenFileName(
            self,
            "Select time interval file", 
            "./", 
            "Json files (*.json *.csv)"
        )
        if filename:
            self.time_path_edit.setText(filename)
    
    def onBrowseOutClick(self):
        filename, ok = QFileDialog.getSaveFileName(
            self,
            "Select output file", 
            "./", 
            "Json files (*.json *.csv)"
        )
        if filename:
            self.outp_path_edit.setText(filename)
