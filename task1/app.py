import sys

from mainWindow import MainWindow
from PyQt6.QtWidgets import QApplication

app = QApplication(sys.argv)
window = MainWindow()
window.show()

app.exec()
