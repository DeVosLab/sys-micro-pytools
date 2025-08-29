"""VISUALISATION TOOL WINDOW:
"""

from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton

class VisualiseWindow(QWidget):
    def __init__(self, back_callback, show_countplot, show_gridplot, show_channelplot):
        super().__init__()
        self.setWindowTitle("Visualise Plots")
        layout = QVBoxLayout()
        layout.addWidget(QLabel("Select an operation:"))

        self.countplot_btn = QPushButton('Count Plot')
        self.countplot_btn.clicked.connect(show_countplot)
        self.gridplot_btn = QPushButton('Grid Plot')
        self.gridplot_btn.clicked.connect(show_gridplot)
        self.channelplot_btn = QPushButton('Channel Plot')
        self.channelplot_btn.clicked.connect(show_channelplot)

        layout.addWidget(self.countplot_btn)
        layout.addWidget(self.gridplot_btn)
        layout.addWidget(self.channelplot_btn)

        self.back_btn = QPushButton("Back to Main Menu")
        self.back_btn.clicked.connect(back_callback)
        layout.addWidget(self.back_btn)
        
        self.setLayout(layout)