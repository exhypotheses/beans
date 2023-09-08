"""Module dag"""
import os

import graphviz
import pymc

import config


class DAG:
    """
    Class DAG
    """

    def __init__(self):
        """
        The constructor

        """

        configurations = config.Config()
        self.warehouse = configurations.warehouse

    def exc(self, model: pymc.Model):
        """

        :param model:
        :return:
        """

        # The DAG
        diagram = pymc.model_graph.ModelGraph(model=model).make_graph()
        diagram.node_attr.update(shape='circle')
        diagram.graph_attr.update(size="11.3,11.9")

        # Diagrams
        diagram.save(os.path.join(self.warehouse, 'model.gv'))
        graphviz.render(engine='dot', format='pdf', filepath=os.path.join(self.warehouse, 'model.gv'))

        return graphviz.Source.from_file(filename=os.path.join(self.warehouse, 'model.gv'))
