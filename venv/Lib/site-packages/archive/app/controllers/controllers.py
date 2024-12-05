"""Controllers utilized for interaction between models and views.

Hydra follows `M-V-C structure <https://en.wikipedia.org/wiki/model–view–controller>`_,
this being the primary Controller module.

:copyright: 2016, See AUTHORS for more details
:license: GNU General Public License, See LICENSE for more details.

"""

import datetime
import tkinter as tk
from tkinter import filedialog, messagebox

import peewee as pw
from app import system
from app import views

from archive.app import models
from archive.app.models import databases as db
from archive.app.models import write
from pyflo import build, sections
from pyflo import networks as nw
from pyflo.rational import hydrology as an


class BaseManager:

    model = pw.Model
    
    def __init__(self, view):
        """The base class for managing models and top level views."""
        self.view = view

    def bind_view(self):
        """Bind the listbox selection of the associated view."""

        def on_listbox_select(_):
            instance = self.get_selected_instance()
            if instance:
                self.set_entries(instance)

        self.view.listbox.bind('<<ListboxSelect>>', on_listbox_select)
    
    def refresh_names(self):
        """Clear and update the listbox from the display names of all model instances."""
        self.view.listbox.options(self.model.get_all_display_names())

    def show_view(self):
        """Display the associated view."""
        self.view.deiconify()
        self.refresh_names()
    
    def get_selected_instance(self):
        """Get the model instance from the display name currently selected in the listbox."""
        index = self.view.listbox.curselection()[0]
        name = self.view.listbox.get(index)
        query = self.model.select()
        for instance in query:
            if instance.display_name == name:
                return instance

    def set_entries(self, instance):
        """Update the view fields from the model properties."""
        return

    def set_model(self, instance):
        """Update the model properties from the view fields."""
        return

    def create_instance(self):
        instance = self.model()
        return instance

    def delete(self):
        try:
            instance = self.get_selected_instance()
            self.delete_related(instance)
            instance.delete_instance()
            list_length = len(self.view.listbox.get(0, tk.END))
            if list_length <= 1:
                index = None
            elif int(self.view.listbox.curselection()[0]) + 1 >= list_length:
                index = int(self.view.listbox.curselection()[0]) - 1
            else:
                index = int(self.view.listbox.curselection()[0])
            self.refresh_names()
            if index:
                self.view.listbox.select_set(index)
        except Exception as e:
            messagebox.showerror(
                "Delete Database Instance Error",
                e
            )

    def update(self):
        try:
            instance = self.get_selected_instance()
            self.set_model(instance)
            instance.save()
            self.refresh_names()
            self.view.listbox.set(instance.display_name)
            self.set_entries(instance)
        except Exception as e:
            messagebox.showerror(
                "Update Database Instance Error",
                e
            )

    def create(self):
        try:
            instance = self.create_instance()
            self.set_model(instance)
            instance.save()
            self.refresh_names()
            self.view.listbox.set(instance.display_name)
            self.set_entries(instance)
        except Exception as e:
            messagebox.showerror(
                'Create Database Instance Error',
                e
            )

    def delete_related(self, instance):
        return

    def delete_related_results(self, instance):
        return


class NodeManager(BaseManager):
    
    model = models.Node
    
    def __init__(self, view):
        """The base class for managing :class:`pyflo.models.Node` models and views."""
        super(NodeManager, self).__init__(view)

    def set_entries(self, instance):
        """Update the view fields from the node model properties."""
        self.view.name_frame.set(instance.name)
        self.view.elevation_frame.set(instance.elevation)

    def set_model(self, instance):
        instance.name = self.view.name_frame.entry.get()
        instance.elevation = float(self.view.elevation_frame.entry.get())

    def show_view(self):
        super(NodeManager, self).show_view()

    def delete_related(self, instance):
        instance.delete_related()
        self.delete_related_results(instance)

    def delete_related_results(self, instance):
        pipes = models.Pipe.select()
        out_pipe = build.pipes_ordered_down_from_node(instance, pipes)[-1]
        out_node = out_pipe.node_2
        out_node.delete_related_results()


class SectionManager(BaseManager):
    
    model = models.Section

    def __init__(self, view):
        """The base class for managing :class:`pyflo.models.Section` models and views."""
        super(SectionManager, self).__init__(view)

    def set_entries(self, instance):
        """Update the view fields from the model properties."""
        super(SectionManager, self).set_entries(instance)
        self.view.name_frame.set(instance.name)
        self.view.span_frame.set(instance.span)
        self.view.rise_frame.set(instance.rise)
        self.view.mannings_frame.set(instance.mannings)
        self.view.count_frame.set(instance.count)
        self.view.shape_frame.set(instance.shape_display)

    def set_model(self, instance):
        super(SectionManager, self).set_model(instance)
        instance.name = self.view.name_frame.entry.get()
        instance.span = self.view.span_frame.entry.get()
        instance.rise = self.view.rise_frame.entry.get()
        instance.mannings = float(self.view.mannings_frame.entry.get())
        instance.count = int(self.view.count_frame.entry.get())
        instance.shape_display = self.view.shape_frame.cb.get()

    def show_view(self):
        super(SectionManager, self).show_view()
        values = models.Section.get_all_shape_display_names()
        self.view.shape_frame.options(values)

    def delete_related(self, instance):
        self.delete_related_results(instance)

    def delete_related_results(self, instance):
        pipes = models.Section.select()
        for pipe in models.Pipe.select().where(models.Pipe.section == self):
            out_pipe = build.pipes_ordered_down_from_node(pipe.node_1, pipes)[-1]
            out_node = out_pipe.node_2
            out_node.delete_related_results()


class PipeManager(BaseManager):
    
    model = models.Pipe
    
    def __init__(self, view):
        """The base class for managing :class:`pyflo.models.Pipe` models and views."""
        super(PipeManager, self).__init__(view)
    
    def set_entries(self, instance):
        """Update the view fields from the model properties."""
        super(PipeManager, self).set_entries(instance)
        self.view.node_1_frame.set(instance.node_1.name)
        self.view.node_2_frame.set(instance.node_2.name)
        self.view.invert_1_frame.set(instance.invert_1)
        self.view.invert_2_frame.set(instance.invert_2)
        self.view.length_frame.set(instance.length)
        self.view.section_frame.set(instance.section.name)

    def set_model(self, instance):
        super(PipeManager, self).set_model(instance)
        node_1_name = self.view.node_1_frame.cb.get()
        node_2_name = self.view.node_2_frame.cb.get()
        instance.node_1 = models.Node.get(name=node_1_name)
        instance.node_2 = models.Node.get(name=node_2_name)
        instance.invert_1 = float(self.view.invert_1_frame.entry.get())
        instance.invert_2 = float(self.view.invert_2_frame.entry.get())
        instance.length = float(self.view.length_frame.entry.get())
        section_name = self.view.section_frame.cb.get()
        instance.section = models.Section.get(name=section_name)

    def show_view(self):
        super(PipeManager, self).show_view()
        node_names = models.Node.get_all_display_names()
        self.view.node_1_frame.options(node_names)
        self.view.node_2_frame.options(node_names)
        section_names = models.Section.get_all_display_names()
        self.view.section_frame.options(section_names)

    def delete_related(self, instance):
        self.delete_related_results(instance)

    def delete_related_results(self, instance):
        pipes = models.Pipe.select()
        out_pipe = build.pipes_ordered_down_from_node(instance.node_1, pipes)[-1]
        out_node = out_pipe.node_2
        out_node.delete_related_results()


class BasinManager(BaseManager):
    
    model = models.Basin
    
    def __init__(self, view):
        """The base class for managing :class:`pyflo.models.Basin` models and views."""
        super(BasinManager, self).__init__(view)

    def set_entries(self, instance):
        super(BasinManager, self).set_entries(instance)
        self.view.node_frame.set(instance.node.name)
        self.view.area_frame.set(instance.area)
        self.view.c_frame.set(instance.c)
        self.view.tc_frame.set(instance.tc)

    def set_model(self, instance):
        super(BasinManager, self).set_model(instance)
        node_name = self.view.node_frame.cb.get()
        instance.node = models.Node.get(name=node_name)
        instance.area = float(self.view.area_frame.entry.get())
        instance.c = float(self.view.c_frame.entry.get())
        instance.tc = float(self.view.tc_frame.entry.get())

    def show_view(self):
        super(BasinManager, self).show_view()
        self.view.node_frame.options(models.Node.get_all_display_names())

    def delete_related(self, instance):
        self.delete_related_results(instance)

    def delete_related_results(self, instance):
        pipes = models.Pipe.select()
        out_pipe = build.pipes_ordered_down_from_node(instance.node, pipes)[-1]
        out_node = out_pipe.node_2
        out_node.delete_related_results()


class IntensityPolynomialManager(BaseManager):
    
    model = models.IntensityPolynomial
    
    def __init__(self, view):
        """The base class for managing :class:`pyflo.models.IntensityPolynomial` models and views."""
        super(IntensityPolynomialManager, self).__init__(view)

    def set_entries(self, instance):
        """Update the view fields from the model properties."""
        super(IntensityPolynomialManager, self).set_entries(instance)
        self.view.name_frame.set(instance.name)
        self.view.a_frame.set(instance.a)
        self.view.b_frame.set(instance.b)
        self.view.c_frame.set(instance.c)
        self.view.d_frame.set(instance.d)

    def set_model(self, instance):
        super(IntensityPolynomialManager, self).set_model(instance)
        instance.name = self.view.name_frame.entry.get()
        instance.a = float(self.view.a_frame.entry.get())
        instance.b = float(self.view.b_frame.entry.get())
        instance.c = float(self.view.c_frame.entry.get())
        instance.d = float(self.view.d_frame.entry.get())

    def delete_related(self, instance):
        instance.delete_related()


class AnalysisManager(BaseManager):
    
    model = models.Analysis
    
    def __init__(self, view):
        """The base class for managing :class:`pyflo.models.Analysis` models and views."""
        super(AnalysisManager, self).__init__(view)

    def set_entries(self, instance):
        """Update the view fields from the model properties."""
        super(AnalysisManager, self).set_entries(instance)
        self.view.name_frame.set(instance.name)
        self.view.node_frame.set(instance.node.name)
        self.view.tw_frame.set(instance.tw)
        self.view.intensity_polynomial_frame.set(instance.intensity_polynomial.name)
        self.view.workbook_frame.set(instance.workbook)

    def set_model(self, instance):
        super(AnalysisManager, self).set_model(instance)
        instance.name = self.view.name_frame.entry.get()
        node_name = self.view.node_frame.cb.get()
        instance.node = models.Node.get(name=node_name)
        instance.tw = float(self.view.tw_frame.entry.get())
        intensity_polynomial_name = self.view.intensity_polynomial_frame.cb.get()
        instance.intensity_polynomial = models.IntensityPolynomial.get(name=intensity_polynomial_name)
        instance.workbook = self.view.workbook_frame.entry.get()

    def show_view(self):
        super(AnalysisManager, self).show_view()
        self.view.node_frame.options(models.Node.get_all_display_names())
        self.view.intensity_polynomial_frame.options(
            models.IntensityPolynomial.get_all_display_names())

    def bind_view(self):
        super(AnalysisManager, self).bind_view()
        
        def on_button_click(_):
            self.select_file()
        
        self.view.workbook_button.bind('<ButtonRelease-1>', on_button_click)

    def select_file(self):
        try:
            filename = filedialog.asksaveasfilename(defaultextension='.xls')
            if filename == '':  # return `None` if dialog closed with "cancel".
                return
            self.view.workbook_frame.set(filename)
        except Exception as e:
            messagebox.showerror(
                "Select Workbook Error",
                e
            )

    def delete_related(self, instance):
        instance.delete_related()


class Analyzer:

    def __init__(self, view):
        self.view = view

    def show_view(self):
        self.view.deiconify()
        self.view.analysis_frame.options(models.Analysis.get_all_display_names())

    def bind_view(self):
        
        def on_button_click(_):
            self.select_file()
        
        self.view.open_wb_button.bind('<ButtonRelease-1>', on_button_click)

    def get_selected_analysis(self):
        analysis_name = self.view.analysis_frame.cb.get()
        analysis = models.Analysis.get(name=analysis_name)
        return analysis

    def run(self):
        """Runs an HGL analysis and posts the results to the database.

        Note:
            Any old results associated with the analysis are removed and replaced.

        """
        try:
            model_analysis = self.get_selected_analysis()
            model_analysis.delete_related()
            analysis = self.get_analysis_from_model(model_analysis)
            data = analysis.hgl_solution_data()
            model_results = self.post_results_to_model(data, model_analysis)
            model_analysis.last_run = datetime.datetime.now()
            model_analysis.save()
            if model_analysis.workbook:
                write.storm_tabulation(model_analysis, model_results, model_analysis.workbook)
        except Exception as e:
            messagebox.showerror(
                "Run models.Analysis Error",
                e
            )

    @staticmethod
    def get_analysis_from_model(model_analysis):
        network = nw.Network()
        out_node = network.create_node()
        out_node.id = model_analysis.node.id
        model_i_poly = model_analysis.intensity_polynomial
        intensity_polynomial = an.IntensityPolynomial(
            a=model_i_poly.a,
            b=model_i_poly.b,
            c=model_i_poly.c,
            d=model_i_poly.d
        )
        model_pipes = build.pipes_ordered_up_from_node(model_analysis.node, models.Pipe.select())
        available_nodes = [out_node]
        for model_pipe in model_pipes:
            new_node = network.create_node()
            new_node.id = model_pipe.node_1.id
            basin_query = models.Basin.select().where(models.Basin.node == model_pipe.node_1)
            if basin_query.exists():
                model_basin = models.Basin.get(node=model_pipe.node_1)
                new_node.create_basin(model_basin.area, model_basin.c, model_basin.tc)
            for a_node in available_nodes:
                if hasattr(a_node, 'id') and a_node.id == model_pipe.node_2.id:
                    section = sections.Circle(model_pipe.section.span,
                                              model_pipe.section.mannings,
                                              model_pipe.section.count
                                              )
                    new_node.create_pipe(
                        node_2=a_node,
                        invert_1=model_pipe.invert_1,
                        invert_2=model_pipe.invert_2,
                        length=model_pipe.length,
                        section=section
                    )
                    break
            available_nodes.append(new_node)

        analysis = an.Analysis(node=out_node,
                               tw=model_analysis.tw,
                               intensity_polynomial=intensity_polynomial
                               )
        return analysis

    @staticmethod
    def post_results_to_model(data, model_analysis):
        for line in data:
            node = models.Node.get(id=line['pipe'].node_1.id)
            pipe = models.Pipe.get(node_1=node)
            basin = models.Basin.get(node=node)
            new_result = models.Result.create(
                analysis=model_analysis,
                pipe=pipe,
                basin=basin,
                area=line['area'],
                runoff=line['runoff_area'],
                intensity=line['second'],
                time=line['time'],
                flow=line['flow'],
                velocity_actual=line['velocity_actual'],
                velocity_physical=line['velocity_physical'],
                hgl_upper=line['hgl_1'],
                hgl_lower=line['hgl_2']
            )
            new_result.save()
        return models.Result.get(analysis=model_analysis)

    def select_file(self):
        try:
            analysis = self.get_selected_analysis()
            system.open_file_with_default(analysis.workbook)
        except Exception as e:
            messagebox.showerror(
                "Open Workbook Error",
                e
            )


class Controller:

    def __init__(self, root):
        self.view = views.MainView(root)

        self.node_manager = NodeManager(self.view.node_view)
        self.section_manager = SectionManager(self.view.section_view)
        self.pipe_manager = PipeManager(self.view.pipe_view)
        self.basin_manager = BasinManager(self.view.basin_view)
        self.intensity_polynomial_manager = IntensityPolynomialManager(self.view.intensity_polynomial_view)
        self.analysis_manager = AnalysisManager(self.view.analysis_view)
        self.analyzer = Analyzer(self.view.analyzer_view)

        self.view.database_menu.add_command(label='Create', command=self.create_database)
        self.view.database_menu.add_command(label='Connect', command=self.connect_database)

        self.view.manage_menu.add_command(label='Nodes', command=self.node_manager.show_view)
        self.view.manage_menu.add_command(label='Sections', command=self.section_manager.show_view)
        self.view.manage_menu.add_command(label='Pipes', command=self.pipe_manager.show_view)
        self.view.manage_menu.add_command(label='Basins', command=self.basin_manager.show_view)

        self.view.analysis_menu.add_command(label='Manage', command=self.analysis_manager.show_view)
        self.view.analysis_menu.add_command(label='Run', command=self.analyzer.show_view)

        self.view.library_menu.add_command(label='Intensity', command=self.intensity_polynomial_manager.show_view)

        self.node_manager.bind_view()
        self.section_manager.bind_view()
        self.pipe_manager.bind_view()
        self.basin_manager.bind_view()
        self.intensity_polynomial_manager.bind_view()
        self.analysis_manager.bind_view()
        self.analyzer.bind_view()

        self.init_bindings()

    def create_database(self):
        """Create and connect to database file"""
        try:
            filename = filedialog.asksaveasfilename(defaultextension='.db')
            if filename == '':  # return `None` if dialog closed with "cancel".
                return
            db.open_file(filename)
            db.connect()
            db.database.create_tables([models.Node, models.Section, models.Pipe, models.Basin,
                                       models.IntensityPolynomial, models.Analysis, models.Result])
            self.view.enable_menu()
        except Exception as e:
            messagebox.showerror(
                'Database Creation Error',
                e
            )

    def connect_database(self):
        """Connect to database file"""
        try:
            filename = filedialog.askopenfilename()
            if filename == '':  # return `None` if dialog closed with "cancel".
                return
            db.open_file(filename)
            db.connect()
            self.view.enable_menu()
            self.update_canvas()
            self.update_view_selects()
            self.bind_view_selects()
        except Exception as e:
            messagebox.showerror(
                'Database Connection Error',
                e
            )

    def init_bindings(self):
        
        def on_canvas_resize(_):
            self.update_canvas()
        
        def on_analyzer_run_button_click(_):
            self.analyzer.run()
            self.update_view()
        
        def on_manager_create_button_click(manager, _):
            manager.create()
            self.update_view()
        
        def on_manager_update_button_click(manager, _):
            manager.update()
            self.update_view()
        
        def on_manager_delete_button_click(manager, _):
            manager.delete()
            self.update_view()
        
        self.view.canvas.bind('<Configure>', on_canvas_resize)
        self.analyzer.view.run_button.bind('<ButtonRelease-1>', on_analyzer_run_button_click)
        managers = (
            self.node_manager,
            self.section_manager,
            self.pipe_manager,
            self.basin_manager,
            self.intensity_polynomial_manager,
            self.analysis_manager
        )
        for m in managers:
            m.view.create_button.bind('<ButtonRelease-1>', lambda evt, man=m: on_manager_create_button_click(man, evt))
            m.view.update_button.bind('<ButtonRelease-1>', lambda evt, man=m: on_manager_update_button_click(man, evt))
            m.view.delete_button.bind('<ButtonRelease-1>', lambda evt, man=m: on_manager_delete_button_click(man, evt))

    def update_view(self):
        self.update_canvas()
        self.update_view_selects()

    def update_canvas(self):
        if not db.database.is_closed():
            node_name = self.view.node_frame.cb.get()
            node_query = models.Node.select().where(models.Node.name == node_name)
            if node_query.exists():
                node = models.Node.get(name=node_name)
                pipes = models.Pipe.select()
                pipes = build.pipes_ordered_down_from_node(node, pipes)
                if pipes:
                    self.view.render_pipes(pipes)
                results = None
                analysis_name = self.view.analysis_frame.cb.get()
                analysis_query = models.Analysis.select().where(
                    models.Analysis.name == analysis_name)
                if analysis_query.exists():
                    analysis = models.Analysis.get(name=analysis_name)
                    results = analysis.get_pipe_results(pipes)
                if results:
                    self.view.render_results(results)
                return
        self.view.render_no_node()

    def update_view_selects(self):
        self.view.node_frame.options(models.Node.get_all_display_names())
        self.view.analysis_frame.options(models.Analysis.get_all_display_names())

    def bind_view_selects(self):

        def on_select(_):
            self.update_canvas()

        self.view.node_frame.cb.bind('<<ComboboxSelected>>', on_select)
        self.view.analysis_frame.cb.bind('<<ComboboxSelected>>', on_select)
