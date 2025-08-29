"""
MAIN ENTRY POINT: starts the application/GUI
o   this script will set up the Qt application (QApplicatioin),
o   express the main window, and
o   show the main window, which starts the event loop
"""

import sys
from PySide6.QtWidgets import QApplication
from gui.main_window import MainWindow

if __name__ == '__main__':

    # create QApplication
    app = QApplication(sys.argv)

    # express main window
    window = MainWindow()
    window.show()
    
    # start event loop
    sys.exit(app.exec())