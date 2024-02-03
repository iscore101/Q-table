from one_op import q_learner
from modified_ds2 import Vertex
from modified_ds2 import DirectedGraph
from flask import Flask, request
import numpy as np
import json
import base64

app = Flask(__name__)

@app.route('/metrics', methods=['POST']) 
def receive_metrics():
    global json_data, graph, _q_learner
    print("===> receive_metrics is called")

    json_data = request.data.decode('utf-8')

    if graph is None and json_data is not None:
        print("===> first-time init")
        graph = build_graph(json_data)
        print(f"===> graph `{graph}` built")
        _q_learner = q_learner(len(graph.vertices), 3, 3, 5000, 3, 3, 5000, 0.5, 0.5, 0.3, graph)
        # _q_learner.populate_q_table_offline()

    print(f'===> json_data: {json_data}')
    return f'json_data: {json_data}'

@app.route('/parallelisms', methods=['GET'])
def send_action():
    global json_data, last_json_data, graph, _q_learner
    print("===> get_parallelism is called")

    if json_data != last_json_data and not (json_data is None or graph is None):
        print("===> Will compute new action")
        state = get_latest_metrics(json_data, graph.sorted_vertices)
        actions = _q_learner.online_generate_action(state)
        action = {graph.sorted_vertices[i]: a for i, a in enumerate(actions)}
        reward = get_reward(json_data, graph.sink.name)
        _q_learner.online_update_q_table(reward)

        last_json_data = json_data # set last_json_data after processing

        print(f'new action: {action}')
        return f'action: {action}'
    else:
        print('no new action')
        return 'no action'

def get_reward(json_data, sink_id):
        data = json.loads(json_data)

        all_metrics = data['metricHistory']
        latest_timestamp = max(all_metrics.keys())
        sink_metrics = all_metrics[latest_timestamp]['vertexMetrics'][sink_id]

        reward = sink_metrics['NUM_RECORDS_IN_PER_SECOND']
        return reward

def get_parallelisms(json_data):
        data = json.loads(json_data)

        parallelisms = data['jobTopology']['parallelisms']
        return parallelisms

def get_latest_metrics(json_data, index_to_id_dict):
    '''
        index json_data['metricHistory'] for processing_rate, parallelism, selectivity.
        state: (input_partitions, parralelism * num_operators,
            selectivity * num_operators, processing_rate * num_operators)
    '''
    data = json.loads(json_data)

    # Check if metric history is populated
    all_metrics = data['metricHistory']
    if all_metrics is None:
        raise ValueError('No metricHistory.')

    latest_timestamp = max(all_metrics.keys())
    vertex_metrics = all_metrics[latest_timestamp]['vertexMetrics']
    print(f'vertex metrics:{vertex_metrics}')

    num_ops = len(vertex_metrics)
    state = np.zeros(1 + (3 * num_ops))
    for i in range(num_ops):
        vertex_id = index_to_id_dict[i]
        curr_vertex_metrics = vertex_metrics[vertex_id]

        if 'SOURCE_DATA_RATE' in curr_vertex_metrics: # source op
            vertex_input_rate = curr_vertex_metrics['SOURCE_DATA_RATE']
            if vertex_input_rate == 'NaN':
                state[0] = 0
            else:
                state[0] = vertex_input_rate

        # vertex_input_rate = curr_vertex_metrics['NUM_RECORDS_IN_PER_SECOND']
        vertex_parallelism = get_parallelisms(json_data)[vertex_id]

        vertex_output_rate = curr_vertex_metrics['NUM_RECORDS_OUT_PER_SECOND']
        if vertex_output_rate == 'NaN':
            vertex_output_rate = 0

        vertex_processing_rate = curr_vertex_metrics['CURRENT_PROCESSING_RATE']
        if vertex_processing_rate == 0:
            vertex_processing_rate = vertex_output_rate + 1

        # state[0] = vertex_input_rate # weird indexing bc one_op(?)
        state[i + 1] = vertex_parallelism
        state[num_ops + (i + 1)] = vertex_output_rate / vertex_processing_rate
        state[2 * num_ops + (i + 1)] = vertex_processing_rate

    return state

def decode_vertex_bytes(vertex_bytes):
    ''' Decode the vertex ID from base64 to hexadecimal. '''
    decoded_bytes = base64.b64decode(vertex_bytes)
    hex_result = decoded_bytes.hex()
    return hex_result

def build_graph(json_data):
    '''
        build graph from jobTopology
        only used for consistent order of vertices
        and sink
    '''
    data = json.loads(json_data)
    topology = data['jobTopology']
    new_graph = DirectedGraph(0)

    sorted_vertices = []
    for vertex_dict in topology['verticesInTopologicalOrder']:
        vertex_id = decode_vertex_bytes(vertex_dict['bytes'])
        vertex = Vertex(vertex_id, 0, 0)
        sorted_vertices.append(vertex)
        new_graph.add_vertex(vertex)

    for from_vertex_name, to_vertices in topology['outputs'].items():
        from_vertex = new_graph.vertices[from_vertex_name]
        for to_vertex_dict in to_vertices:
            to_vertex_bytes = to_vertex_dict['bytes']
            to_vertex = new_graph.vertices[decode_vertex_bytes(to_vertex_bytes)]
            new_graph.add_edge(from_vertex, to_vertex)

    new_graph.set_source(sorted_vertices[0])
    new_graph.set_sink(sorted_vertices[-1])
    print(f'sv:{sorted_vertices}\nts:{new_graph.topological_sort()}')
    # assert(sorted_vertices == new_graph.topological_sort()) # fails

    print(f'finished graph: {new_graph}')
    return new_graph


if __name__ == '__main__':
    print("===> starting main")
    json_data = None
    last_json_data = None
    graph = None
    _q_learner = None
    app.run(host='0.0.0.0', port=5001) # , debug=True
