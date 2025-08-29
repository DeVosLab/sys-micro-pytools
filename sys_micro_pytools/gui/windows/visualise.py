"""VISUALISATION TOOL WINDOW:
"""

from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton

class VisualiseWindow(QWidget):
    def __init__(self, back_callback):
        super().__init__()
        self.setWindowTitle("Visualise Plots")
        layout = QVBoxLayout()
        layout.addWidget(QLabel("Select an operation:"))

        self.countplot_btn = QPushButton('Count Plot')
        self.gridplot_btn = QPushButton('Grid Plot')
        self.channelplot_btn = QPushButton('Channel Plot')

        layout.addWidget(self.countplot_btn)
        layout.addWidget(self.gridplot_btn)
        layout.addWidget(self.channelplot_btn)

        self.back_btn = QPushButton("Back")
        self.back_btn.clicked.connect(back_callback)
        layout.addWidget(self.back_btn)
        
        self.setLayout(layout)