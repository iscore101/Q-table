from one_op import q_learner
from modified_ds2 import Vertex
from modified_ds2 import DirectedGraph

    # Example 2
#      A
#    /   \
# Source   B -> Sink
#    \   /
#      C

graph2 = DirectedGraph(2000)
A2 = Vertex("A", 700, 2)
B2 = Vertex("B", 400, 5)
C2 = Vertex("C", 350, 4)
graph2.add_vertex(A2)
graph2.add_vertex(B2)
graph2.add_vertex(C2)
graph2.set_source(A2)
graph2.set_sink(B2)
graph2.add_edge(A2, B2)
graph2.add_edge(A2, C2)
graph2.add_edge(C2, B2)

graph2.sort_vertices()

q_learn = q_learner(len(graph2.vertices), 3, 3, 3, 3, 3, 3, 0.5, 0.5, 0, graph2) # len(graph2.vertices()) = 3
q_learn.populate_q_table_offline()
print(q_learn.Q)
