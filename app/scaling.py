from one_op import q_learner
from modified_ds2 import Vertex
from modified_ds2 import DirectedGraph
from flask import Flask, request
import numpy as np
import json
import base64

app = Flask(__name__)

# @app.route('/', methods=['POST', 'GET'])
@app.route('/metrics', methods=['POST']) 
#def update_json_data():
def receive_metrics():
    global json_data, graph, _q_learner
    print("===> receive_metrics is called")

    if json_data is None:
        print("===> first-time init")
        json_data = request.data.decode('utf-8')
        graph = build_graph(json_data)
        _q_learner = q_learner(len(graph.vertices), 3, 3, 3, 3, 3, 3, 0.5, 0.5, 0, graph)
        _q_learner.populate_q_table_offline()
    else:
        json_data = request.data.decode('utf-8') # update global var with metrics

    print(f'===> json_data: {json_data}')
    return f'json_data: {json_data}'

# @app.route('/', methods=['GET'])
@app.route('/parallelisms', methods=['GET'])
#def send_action():
def get_parallelism():
    global json_data, last_json_data, graph, _q_learner
    print("===> get_parallelism is called")

    if json_data != last_json_data:
        print("===> Will compute new action")
        state = get_latest_metrics(json_data, graph.sorted_vertices)
        actions = _q_learner.online_generate_action(state)
        action = {graph.sorted_vertices[i]: a for i, a in enumerate(actions)}
        reward = get_reward(json_data, graph.sink.name)
        _q_learner.online_update_q_table(reward)

        last_json_data = json_data # set last_json_data after processing

        print(f'action: {action}')
        return f'action: {action}'

def get_reward(json_data, sink_id):
    try:
        data = json.loads(json_data)

        all_metrics = data['metricHistory']
        latest_timestamp = max(all_metrics.keys())
        latest_metrics = all_metrics[latest_timestamp]

        reward = latest_metrics[sink_id]['NUM_RECORDS_IN_PER_SECOND']
        return reward

    except Exception as e:
        print(f'Error: {e}')

def get_parallelisms(json_data):
    try:
        data = json.loads(json_data)

        parallelisms = data['jobTopology']['parallelisms']
        return parallelisms

    except Exception as e:
        print(f'Error: {e}')

def get_latest_metrics(json_data, index_to_id_dict):
    '''
        index json_data['metricHistory'] for processing_rate, parallelism, selectivity.
        state: (input_partitions, parralelism * num_operators,
            selectivity * num_operators, processing_rate * num_operators)
    '''
    try:
        data = json.loads(json_data)

        # Check if metric history is populated
        all_metrics = data['metricHistory']
        if all_metrics is None:
            raise ValueError('No metricHistory.')

        latest_timestamp = max(all_metrics.keys())
        latest_metrics = all_metrics[latest_timestamp]

        state = np.zeros(1 + (3 * num_ops))
        num_ops = len(latest_metrics)
        for i in range(num_ops):
            vertex_id = index_to_id_dict[i]
            vertex_metrics = latest_metrics[vertex_id]

            if 'SOURCE_DATA_RATE' in vertex_metrics: # source op
                vertex_input_rate = vertex_metrics['SOURCE_DATA_RATE']
                if vertex_input_rate == 'NaN':
                    state[0] = 0
                else:
                    state[0] = vertex_input_rate

            # vertex_input_rate = vertex_metrics['NUM_RECORDS_IN_PER_SECOND']
            vertex_parallelism = get_parallelisms(data)[vertex_id]
            vertex_output_rate = vertex_metrics['CURRENT_PROCESSING_RATE']
            vertex_processing_rate = vertex_metrics['CURRENT_PROCESSING_RATE']

            # state[0] = vertex_input_rate # weird indexing bc one_op(?)
            state[i + 1] = vertex_parallelism
            state[2 * (i + 1)] = vertex_output_rate / vertex_processing_rate
            state[3 * (i + 1)] = vertex_processing_rate

        return state

    except Exception as e:
        print(f'Error: {e}')

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
    try:
        data = json.loads(json_data)

        topology = data['jobTopology']
        new_graph = DirectedGraph(0)
        for vertex_dict in topology['verticesInTopologicalOrder']:
            vertex_id = decode_vertex_bytes(vertex_dict['bytes'])
            new_graph.add_vertex(Vertex(vertex_id, 0, 0))
        sorted_vertices = new_graph.topological_sort()
        new_graph.set_source(new_graph.vertices[sorted_vertices[0]])
        new_graph.set_sink(new_graph.vertices[sorted_vertices[-1]])

        for from_vertex_name, to_vertices in topology['outputs'].items():
            from_vertex = new_graph.vertices[from_vertex_name]
            for to_vertex_dict in to_vertices:
                to_vertex_bytes = to_vertex_dict['bytes']
                to_vertex = new_graph.vertices[decode_vertex_bytes(to_vertex_bytes)]
                new_graph.add_edge(from_vertex, to_vertex)
        
        return new_graph

    except Exception as e:
        print(f'Error: {e}')

if __name__ == '__main__':
    print("===> starting main")
    json_data = None
    last_json_data = None
    graph = None
    _q_learner = None
    app.run(host='0.0.0.0', port=5001)

    # while json_data is None:
    #     # Add a delay to avoid continuous checking
    #     time.sleep(0.25)
    # print("===> before graph in main")
    # graph = build_graph(json_data)
    # q_learner = q_learner(len(graph.vertices), 3, 3, 3, 3, 3, 3, 0.5, 0.5, 0, graph)
    # q_learner.populate_q_table_offline()

    # last_json_data = None
    # # while True:
    # #     if json_data != last_json_data:
    # #         print("===> Will compute new action")
    # #         state = get_latest_metrics(json_data, graph.sorted_vertices)
    # #         actions = q_learner.online_generate_action(state)
    # #         action_dict = {graph.sorted_vertices[i]: a for i, a in enumerate(actions)}
    # #         action = action_dict # update global variable for get requesst
    # #         reward = get_reward(graph.sink.name)
    # #         q_learner.online_update_q_table(reward)

    # #         last_json_data = json_data # Reset last_json_data after processing

    # #     time.sleep(0.25) 
    # if json_data != last_json_data:
    #     print("===> Will compute new action")
    #     state = get_latest_metrics(json_data, graph.sorted_vertices)
    #     actions = q_learner.online_generate_action(state)
    #     action_dict = {graph.sorted_vertices[i]: a for i, a in enumerate(actions)}
    #     action = action_dict # update global variable for get requesst
    #     reward = get_reward(graph.sink.name)
    #     q_learner.online_update_q_table(reward)

    #     last_json_data = json_data # Reset last_json_data after processing 
