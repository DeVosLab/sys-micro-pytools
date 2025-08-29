"""
MAIN APPLICATION WINDOW: defines the main window of the application, manages navigation between
different tools and pages
o   this script will inherit from QWidget
o   it contains and manages all other windows/pages via a QStackedWidget
o   it handles navigation logic: switching between main menu, tool windows, etc.
"""

from PySide6.QtWidgets import QWidget, QVBoxLayout, QStackedWidget
from .main_menu import MainMenu
from .grid2table import Grid2TableWindow
from .dose_response import DoseResponseWindow
from .visualise import VisualiseWindow
from .count_plot import CountPlotWindow
from .grid_plot import GridPlotWindow
from .channel_plot import ChannelPlotWindow

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Sys-Micro-PyTools GUI')

        # use QStackedWidget to manage navigation betwewen pages
        self.stacked = QStackedWidget()

        # instantiate windows
        self.menu = MainMenu(
            self.show_grid2table,
            self.show_dose_response,
            self.show_visualise)
        self.grid2table_window = Grid2TableWindow(self.show_menu)
        self.dose_response_window = DoseResponseWindow(self.show_menu)
        self.visualise_window = VisualiseWindow(
            self.show_menu,
            self.show_countplot,
            self.show_gridplot,
            self.show_channelplot)
        self.count_plot_window = CountPlotWindow(self.show_menu)
        self.grid_plot_window = GridPlotWindow(self.show_menu)
        self.channel_plot_window = ChannelPlotWindow(self.show_menu)



        # add windows to stack
        self.stacked.addWidget(self.menu)
        self.stacked.addWidget(self.grid2table_window)
        self.stacked.addWidget(self.dose_response_window)
        self.stacked.addWidget(self.visualise_window)
        self.stacked.addWidget(self.count_plot_window)
        self.stacked.addWidget(self.grid_plot_window)
        self.stacked.addWidget(self.channel_plot_window)

        layout = QVBoxLayout()
        layout.addWidget(self.stacked)
        self.setLayout(layout)
        self.show_menu()

    def show_menu(self):
        self.stacked.setCurrentWidget(self.menu)

    # navigation methods
    def show_grid2table(self):
        self.stacked.setCurrentWidget(self.grid2table_window)
    def show_dose_response(self):
        self.stacked.setCurrentWidget(self.dose_response_window)
    def show_visualise(self):
        self.stacked.setCurrentWidget(self.visualise_window)
    def show_countplot(self):
        self.stacked.setCurrentWidget(self.count_plot_window)
    def show_gridplot(self):
        self.stacked.setCurrentWidget(self.grid_plot_window)
    def show_channelplot(self):
        self.stacked.setCurrentWidget(self.channel_plot_window)