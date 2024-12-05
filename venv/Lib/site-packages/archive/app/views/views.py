"""Views utilized for rendering program data to the user.

Hydra follows `M-V-C structure <https://en.wikipedia.org/wiki/model–view–controller>`_,
this being the primary Views module. The GUI is built primarily with tkinter in order to
keep minimal dependencies.

:copyright: 2016, See AUTHORS for more details
:license: GNU General Public License, See LICENSE for more details.

"""

import tkinter as tk

from .render import PipesRenderer, ResultsRenderer
from .widgets import PropertyFrame, CustomListBox, SelectFrame


class CustomToplevel(tk.Toplevel):

    name = ''

    def __init__(self, master):
        """A base top level widget class to inherit.

        Args:
            master: A tkinter widget.

        """
        super(CustomToplevel, self).__init__(master)

        self.title('Manage ' + self.name)

        list_frame = tk.LabelFrame(self, text=self.name, padx=5, pady=5)
        list_frame.pack(padx=10, pady=10, side=tk.LEFT, fill=tk.Y)
        self.listbox = CustomListBox(list_frame)
        self.listbox.pack(fill=tk.Y, expand=tk.YES)

        control_frame = tk.LabelFrame(self, text='Controls', padx=5, pady=5)
        control_frame.pack(padx=10, pady=10, side=tk.TOP, fill=tk.X)
        self.create_button = tk.Button(control_frame, text='Create')
        self.create_button.pack(side=tk.LEFT)
        self.update_button = tk.Button(control_frame, text='Update')
        self.update_button.pack(side=tk.LEFT)
        self.delete_button = tk.Button(control_frame, text='Delete')
        self.delete_button.pack(side=tk.LEFT)

        self.data_frame = tk.LabelFrame(self, text='Data', padx=5, pady=5)
        self.data_frame.pack(padx=10, pady=10, side=tk.TOP, fill=tk.BOTH, expand=tk.YES)

        self.protocol('WM_DELETE_WINDOW', self.withdraw)


class NodeView(CustomToplevel):

    name = 'Nodes'

    def __init__(self, master):
        """A top level widget for node management.

        Args:
            master: A tkinter widget.

        """
        super(NodeView, self).__init__(master)
        self.name_frame = PropertyFrame(self.data_frame, 'Name')
        self.name_frame.pack(padx=10, pady=10, side=tk.TOP, fill=tk.X)
        self.elevation_frame = PropertyFrame(self.data_frame, 'Elev. (ft)')
        self.elevation_frame.pack(padx=10, pady=10, side=tk.TOP, fill=tk.X)


class SectionView(CustomToplevel):
    name = 'Sections'

    def __init__(self, master):
        """A top level widget for section management.

        Args:
            master: A tkinter widget.

        """
        super(SectionView, self).__init__(master)
        self.name_frame = PropertyFrame(self.data_frame, 'Name')
        self.name_frame.pack(padx=10, pady=10, side=tk.TOP, fill=tk.X)
        self.span_frame = PropertyFrame(self.data_frame, 'Span (in)')
        self.span_frame.pack(padx=10, pady=10, side=tk.TOP, fill=tk.X)
        self.rise_frame = PropertyFrame(self.data_frame, 'Rise (in)')
        self.rise_frame.pack(padx=10, pady=10, side=tk.TOP, fill=tk.X)
        self.mannings_frame = PropertyFrame(self.data_frame, 'Mannings n')
        self.mannings_frame.pack(padx=10, pady=10, side=tk.TOP, fill=tk.X)
        self.count_frame = PropertyFrame(self.data_frame, 'Count, #')
        self.count_frame.pack(padx=10, pady=10, side=tk.TOP, fill=tk.X)
        self.shape_frame = SelectFrame(self.data_frame, 'Shape')
        self.shape_frame.pack(padx=10, pady=10, side=tk.TOP, fill=tk.X)


class PipeView(CustomToplevel):

    name = 'Pipes'

    def __init__(self, master):
        """A top level widget for pipe management.

        Args:
            master: A tkinter widget.

        """
        super(PipeView, self).__init__(master)
        from_frame = tk.Frame(self.data_frame)
        from_frame.pack()
        self.node_1_frame = SelectFrame(from_frame, 'From Node')
        self.node_1_frame.pack(padx=10, pady=10, side=tk.LEFT, fill=tk.X)
        self.invert_1_frame = PropertyFrame(from_frame, 'Invert (ft)')
        self.invert_1_frame.pack(padx=10, pady=10, side=tk.LEFT, fill=tk.X)
        to_frame = tk.Frame(self.data_frame)
        to_frame.pack()
        self.node_2_frame = SelectFrame(to_frame, 'To Node')
        self.node_2_frame.pack(padx=10, pady=10, side=tk.LEFT, fill=tk.X)
        self.invert_2_frame = PropertyFrame(to_frame, 'Invert (ft)')
        self.invert_2_frame.pack(padx=10, pady=10, side=tk.LEFT, fill=tk.X)
        self.length_frame = PropertyFrame(self.data_frame, 'Length (ft)')
        self.length_frame.pack(padx=10, pady=10, side=tk.TOP, fill=tk.X)
        self.section_frame = SelectFrame(self.data_frame, 'Section')
        self.section_frame.pack(padx=10, pady=10, side=tk.TOP, fill=tk.X)


class BasinView(CustomToplevel):
    
    name = 'Basins'
    
    def __init__(self, master):
        """A top level widget for basin management.

        Args:
            master: A tkinter widget.

        """
        super(BasinView, self).__init__(master)
        self.node_frame = SelectFrame(self.data_frame, 'Node')
        self.node_frame.pack(padx=10, pady=10, side=tk.TOP, fill=tk.X)
        self.area_frame = PropertyFrame(self.data_frame, 'Area (ac)')
        self.area_frame.pack(padx=10, pady=10, side=tk.TOP, fill=tk.X)
        self.c_frame = PropertyFrame(self.data_frame, 'Runoff Coefficient, c')
        self.c_frame.pack(padx=10, pady=10, side=tk.TOP, fill=tk.X)
        self.tc_frame = PropertyFrame(self.data_frame, 'Time of Conc. (min)')
        self.tc_frame.pack(padx=10, pady=10, side=tk.TOP, fill=tk.X)


class IntensityPolynomialView(CustomToplevel):

    name = 'Intensity Polynomials'

    def __init__(self, master):
        """A top level widget for second polynomial management.

        Args:
            master: A tkinter widget.

        """
        super(IntensityPolynomialView, self).__init__(master)
        label = tk.Label(self.data_frame, text=u'i = a + b*log(t) + c*log(t)\u00B2 + d*log(t)\u00B3')
        label.pack(padx=10, pady=10, side=tk.TOP, fill=tk.X)
        self.name_frame = PropertyFrame(self.data_frame, 'Name')
        self.name_frame.pack(padx=10, pady=10, side=tk.TOP, fill=tk.X)
        self.a_frame = PropertyFrame(self.data_frame, 'coefficient a')
        self.a_frame.pack(padx=10, pady=10, side=tk.TOP, fill=tk.X)
        self.b_frame = PropertyFrame(self.data_frame, 'coefficient b')
        self.b_frame.pack(padx=10, pady=10, side=tk.TOP, fill=tk.X)
        self.c_frame = PropertyFrame(self.data_frame, 'coefficient c')
        self.c_frame.pack(padx=10, pady=10, side=tk.TOP, fill=tk.X)
        self.d_frame = PropertyFrame(self.data_frame, 'coefficient d')
        self.d_frame.pack(padx=10, pady=10, side=tk.TOP, fill=tk.X)


class AnalysisView(CustomToplevel):

    name = 'Analyses'

    def __init__(self, master):
        """A top level widget for analysis management.

        Args:
            master: A tkinter widget.

        """
        super(AnalysisView, self).__init__(master)
        self.name_frame = PropertyFrame(self.data_frame, 'Name')
        self.name_frame.pack(padx=10, pady=10, side=tk.TOP, fill=tk.X)
        self.node_frame = SelectFrame(self.data_frame, 'Node')
        self.node_frame.pack(padx=10, pady=10, side=tk.TOP, fill=tk.X)
        self.tw_frame = PropertyFrame(self.data_frame, 'Tailwater Elev. (ft)')
        self.tw_frame.pack(padx=10, pady=10, side=tk.TOP, fill=tk.X)
        self.intensity_polynomial_frame = SelectFrame(self.data_frame, 'Intensity Polynomial')
        self.intensity_polynomial_frame.pack(padx=10, pady=10, side=tk.TOP, fill=tk.X)
        self.workbook_frame = PropertyFrame(self.data_frame, 'Output Workbook')
        self.workbook_frame.pack(padx=10, pady=10, side=tk.TOP, fill=tk.X)

        self.workbook_button = tk.Button(self.data_frame, text='Select Workbook File')
        self.workbook_button.pack(side=tk.TOP)


class AnalyzerView(tk.Toplevel):

    def __init__(self, master):
        """A top level widget for running analyses.

        Args:
            master: A tkinter widget.

        """
        super(AnalyzerView, self).__init__(master)

        self.title('Hydraulic Analyzer')

        frame = tk.Frame(self, padx=5, pady=5)
        frame.pack(padx=10, pady=10, side=tk.TOP, fill=tk.BOTH)

        self.analysis_frame = SelectFrame(frame, 'Analysis')
        self.analysis_frame.pack(padx=10, pady=10, side=tk.TOP, fill=tk.X)

        self.run_button = tk.Button(frame, text='Run')
        self.run_button.pack(side=tk.LEFT)
        self.open_wb_button = tk.Button(frame, text='Open Workbook')
        self.open_wb_button.pack(side=tk.LEFT)

        self.protocol('WM_DELETE_WINDOW', self.withdraw)


class MainView(tk.Tk):

    def __init__(self, master):
        """A top level widget for the application.

        Args:
            master: A tkinter widget.

        """
        tk.Tk.__init__(self, master)

        self.title('Hydra')

        self.menu_bar = tk.Menu(self, tearoff=False)
        self.config(menu=self.menu_bar)

        self.database_menu = tk.Menu(self.menu_bar, tearoff=False)
        self.manage_menu = tk.Menu(self.menu_bar, tearoff=False)
        self.analysis_menu = tk.Menu(self.menu_bar, tearoff=False)
        self.report_menu = tk.Menu(self.menu_bar, tearoff=False)
        self.library_menu = tk.Menu(self.menu_bar, tearoff=False)
        self.help_menu = tk.Menu(self.menu_bar, tearoff=False)

        self.menu_bar.add_cascade(label='Database', menu=self.database_menu)
        self.menu_bar.add_cascade(label='Manage', menu=self.manage_menu)
        self.menu_bar.add_cascade(label='Analysis', menu=self.analysis_menu)
        self.menu_bar.add_cascade(label='Library', menu=self.library_menu)
        self.menu_bar.add_cascade(label='Help', menu=self.help_menu)

        control_frame = tk.LabelFrame(self, text='Controls', padx=5, pady=5)
        control_frame.pack(padx=10, pady=10, side=tk.RIGHT, fill=tk.Y)
        self.node_frame = SelectFrame(control_frame, 'Top Node')
        self.node_frame.pack(padx=10, pady=10, side=tk.TOP, fill=tk.X)
        self.analysis_frame = SelectFrame(control_frame, 'Analysis')
        self.analysis_frame.pack(padx=10, pady=10, side=tk.TOP, fill=tk.X)

        self.canvas = tk.Canvas(self, width=600, height=300)
        self.canvas.pack(side=tk.TOP, fill=tk.BOTH, expand=tk.YES)

        self.node_view = NodeView(self)
        self.node_view.withdraw()
        self.section_view = SectionView(self)
        self.section_view.withdraw()
        self.pipe_view = PipeView(self)
        self.pipe_view.withdraw()
        self.basin_view = BasinView(self)
        self.basin_view.withdraw()
        self.intensity_polynomial_view = IntensityPolynomialView(self)
        self.intensity_polynomial_view.withdraw()
        self.analysis_view = AnalysisView(self)
        self.analysis_view.withdraw()

        self.analyzer_view = AnalyzerView(self)
        self.analyzer_view.withdraw()

        self.disable_menu()

    def enable_menu(self):
        """Allow access to all menu items."""
        self.menu_bar.entryconfig('Manage', state='normal')
        self.menu_bar.entryconfig('Analysis', state='normal')
        self.menu_bar.entryconfig('Library', state='normal')

    def disable_menu(self):
        """Disable menu items that are database dependent."""
        self.menu_bar.entryconfig('Manage', state='disabled')
        self.menu_bar.entryconfig('Analysis', state='disabled')
        self.menu_bar.entryconfig('Library', state='disabled')

    def render_pipes(self, pipes):
        """Display a visual profile representation of a series of pipes.

        Args:
            pipes (list[pyflo.models.Pipe]): A series of pipes.

        """
        self.canvas.delete("all")
        pipes_renderer = PipesRenderer(self.canvas, pipes)
        pipes_renderer.draw()

    def render_results(self, results):
        """Display a visual profile representation of hydraulic results.

        Args:
            results: (list[pyflo.models.Result]): A series of results.

        """
        results_renderer = ResultsRenderer(self.canvas, results)
        results_renderer.draw()

    def render_no_node(self):
        """Display a placeholder, to be used when no valid nodes are selected."""
        self.canvas.delete("all")
        width = self.canvas.winfo_width()
        height = self.canvas.winfo_height()
        self.canvas.create_text(width / 2, height / 2, text='No Valid Node Selected')
