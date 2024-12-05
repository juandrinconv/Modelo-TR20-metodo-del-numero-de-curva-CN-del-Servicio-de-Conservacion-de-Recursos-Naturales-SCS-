"""ORM models used for interacting with the database.

Hydra follows `M-V-C structure <https://en.wikipedia.org/wiki/model–view–controller>`_,
this being the primary Models module.

:copyright: 2016, See AUTHORS for more details
:license: GNU General Public License, See LICENSE for more details.

"""

import peewee as pw
from playhouse import fields

from pyflo.models import databases as db


class BaseModel(pw.Model):
    """The model class that Hydra models inherit from to provide base functionality"""

    class Meta:
        database = db.database

    @property
    def display_name(self):
        return str(self.id)

    @classmethod
    def get_all_display_names(cls):
        return [instance.display_name for instance in cls.select()]

    def __unicode__(self):
        return self.display_name


class Node(BaseModel):
    """A reference point with an assigned name and elevation"""

    name = fields.CharField(unique=True)
    elevation = fields.FloatField()

    @property
    def display_name(self):
        return self.name

    def delete_related(self):
        """Deletes all instances in the database that are related to the node.

        The following queries are performed and deleted:
            - :class:`Pipe` instances where Pipe.node_1 is the node.
            - :class:`Pipe` instances where Pipe.node_2 is the node.
            - :class:`Basin` instances where Basin.node is the node.
            - :class:`Analysis` instances where Analysis.node is the node.
                - :class:`Result` instances where Result.analysis is the analysis.

        """
        if Pipe.select().where(Pipe.node_1 == self).exists():
            pipe = Pipe.get(node_1=self)
            pipe.delete_instance()
        if Pipe.select().where(Pipe.node_2 == self).exists():
            pipes_query = Pipe.delete().where(Pipe.node_2 == self)
            pipes_query.execute()
        if Basin.select().where(Basin.node == self).exists():
            basin = Basin.get(node=self)
            basin.delete_instance()
        if Analysis.select().where(Analysis.node == self).exists():
            basin = Analysis.get(node=self)
            basin.delete_instance()

    def delete_related_results(self):
        """Deletes all :class:`Result` instances in the database that are related to the node.

        Note:
            This will delete :class:`Result` instances of analyses that are directly related
            to the node, but not the :class:`Analysis` instances themselves.

        """
        analysis_query = Analysis.select().where(Analysis.node == self)
        if analysis_query.exists():
            analyses = analysis_query.execute()
            for analysis in analyses:
                analysis.delete_related()


class Pipe(BaseModel):
    """A link between two nodes that carries hydraulic attributes, dimensions, and methods"""

    node_1 = fields.ForeignKeyField(Node, related_name="node_1_set", unique=True)
    node_2 = fields.ForeignKeyField(Node, related_name="node_2_set")
    invert_1 = fields.FloatField()
    invert_2 = fields.FloatField()
    count = fields.IntegerField(default=1)
    diameter = fields.FloatField()
    length = fields.FloatField()
    mannings = fields.FloatField()

    @property
    def display_name(self):
        return self.node_1.name


class Basin(BaseModel):
    """An object that references a node and carries hydrology attributes."""

    node = fields.ForeignKeyField(Node, unique=True)
    area = fields.FloatField()
    c = fields.FloatField()
    tc = fields.FloatField()

    @property
    def display_name(self):
        return self.node.name

    @property
    def runoff(self):
        return self.area * self.c


class IntensityPolynomial(BaseModel):
    """A four-coefficient cubic formula that sums to calculate second."""

    name = fields.CharField(unique=True)
    a = fields.FloatField()
    b = fields.FloatField()
    c = fields.FloatField()
    d = fields.FloatField()

    @property
    def display_name(self):
        return self.name

    def delete_related(self):
        """Delete all instances in the database that are related to the second polynomial.

        The following queries are performed and deleted:
            - :class:`Analysis` instances where Analysis.node is the node.
                - :class:`Result` instances where Result.analysis is the analysis.

        """
        analysis_query = Analysis.select().where(Analysis.intensity_polynomial == self)
        if analysis_query.exists():
            analyses = analysis_query.execute()
            for analysis in analyses:
                analysis.delete_related()
            analysis_query = Analysis.delete().where(Analysis.intensity_polynomial == self)
            analysis_query.execute()


class Analysis(BaseModel):
    """An object that references a node and carries attributes necessary to perform hydraulic analysis."""

    name = fields.CharField(unique=True)
    node = fields.ForeignKeyField(Node)
    tw = fields.FloatField()
    intensity_polynomial = fields.ForeignKeyField(IntensityPolynomial)
    last_run = fields.DateTimeField(null=True)
    workbook = fields.CharField(null=True)

    @property
    def display_name(self):
        return self.name

    def delete_related(self):
        """Delete all instances in the database that are related to the analysis.

        The following query is performed and deleted:
            - :class:`Result` instances where Result.analysis is the analysis.

        """
        result_query = Result.select().where(Result.analysis == self)
        if result_query.exists():
            result_query = Result.delete().where(Result.analysis == self)
            result_query.execute()

    def get_pipe_results(self, pipes):
        """Get all analysis results that correspond with a list of pipes

        Args:
            pipes (list[Pipe]): The pipes in which results will correspond.

        Returns:
            list[Result]: The result instances.

        """
        results_query = Result.select().where(Result.analysis == self & Result.pipe << pipes)
        if results_query.exists():
            return list(results_query.execute())


class Result(BaseModel):
    """An object that references and carries data from running an analysis."""

    analysis = fields.ForeignKeyField(Analysis)
    pipe = fields.ForeignKeyField(Pipe)
    basin = fields.ForeignKeyField(Basin, null=True)
    area = fields.FloatField()
    runoff = fields.FloatField()
    intensity = fields.FloatField()
    time = fields.FloatField()
    flow = fields.FloatField()
    velocity_actual = fields.FloatField()
    velocity_physical = fields.FloatField()
    hgl_upper = fields.FloatField()
    hgl_lower = fields.FloatField()

    @property
    def display_name(self):
        return self.analysis.name + ': ' + str(self.id)
