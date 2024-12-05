"""Functions for mass manipulation of the initialized database's tables.

:copyright: 2016, See AUTHORS.md for mode details.
:license: GNU General Public License, See LICENSE.md for more details.

"""

from .analyses import get_pipes_ordered_down_from_node
from .models import Pipe, Analysis


# def delete_related_to_node(node):
#     if Pipe.select().where(Pipe.node_1 == node).exists():
#         pipe = Pipe.get(node_1=node)
#         pipe.delete_instance()
#     if Pipe.select().where(Pipe.node_2 == node).exists():
#         pipes_query = Pipe.delete().where(Pipe.node_2 == node)
#         pipes_query.execute()
#     if Basin.select().where(Basin.node == node).exists():
#         basin = Basin.get(node=node)
#         basin.delete_instance()
#     return


# def delete_related_to_analysis(analysis):
#     if Result.select().where(Result.analysis == analysis).exists():
#         result_query = Result.delete().where(Result.analysis == analysis)
#         result_query.execute()
#     return


# def delete_related_to_intensity_polynomial(intensity_polynomial):
#     analysis_query = Analysis.select().where(Analysis.intensity_polynomial == intensity_polynomial)
#     if analysis_query.exists():
#         analyses = analysis_query.execute()
#         for analysis in analyses:
#             delete_related_to_analysis(analysis)
#         analysis_query = Analysis.delete().where(Analysis.intensity_polynomial == intensity_polynomial)
#         analysis_query.execute()
#     return


def delete_results_related_to_node(node):
    pipes = Pipe.select()
    out_pipe = get_pipes_ordered_down_from_node(node, pipes)[-1]
    out_node = out_pipe.node_2
    analysis_query = Analysis.select().where(Analysis.node == out_node)
    if analysis_query.exists():
        analyses = analysis_query.execute()
        for analysis in analyses:
            # delete_related_to_analysis(analysis)
            analysis.delete_related()
    return


# def delete_analysis_results(analysis):
#     result_query = Result.delete().where(Result.analysis == analysis)
#     result_query.execute()
#     return
