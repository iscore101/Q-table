from one_op import q_learner
from modified_ds2 import Vertex
from modified_ds2 import DirectedGraph
from flask import Flask, request
import numpy as np
import json
import time
import base64

app = Flask(__name__)

@app.route("/", methods=['POST', 'GET'])
def hello():
    global json_data
    if request.method == 'POST':
        # Access the request data
        data = request.data.decode('utf-8')
        json_data = data
        print(f"POST request body: {json_data}")
        return f"POST request body: {json_data}"
    else:
        return "Hello from Python!"

@app.route("/paralelisms", methods=['GET'])
def hello_parallelism():
    global json_data  # Use the global keyword to access the global variable
    if request.method == 'GET':
        # Access the global json_data
        # print(f"Global json_data: {json_data}")
        if json_data is not None:
            # Call the function and get the result
            parallelism_result = get_parallelisms(json_data)
            json_result = json.dumps(parallelism_result)
            print(f"Parallelism result: {json_result}")
            return f"Parallelism result: {json_result}"
        else:
            return "No data available to process."
    else:
        return "Hello from Python!"

def get_reward(json_data, sink_id):
    try:
        # Parse the JSON data
        data = json.loads(json_data)

        # Check if the expected structure is present
        if 'metricHistory' not in data:
            raise ValueError("Invalid JSON format. Missing 'metricHistory'.")

        # Extract parallelisms
        all_metrics = data['jobTopology']['parallelisms']
        latest_timestamp = max(all_metrics.keys())
        latest_metrics = all_metrics[latest_timestamp]

        reward = latest_metrics[sink_id]['NUM_RECORDS_IN_PER_SECOND']

        return reward

    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
    except KeyError as e:
        print(f"Key not found: {e}")
    except Exception as e:
        print(f"Error: {e}")

def get_parallelisms(json_data):
    try:
        # Parse the JSON data
        data = json.loads(json_data)

        # Check if the expected structure is present
        if 'jobTopology' not in data or 'parallelisms' not in data['jobTopology']:
            raise ValueError("Invalid JSON format. Missing 'jobTopology' or 'parallelisms'.")

        # Extract parallelisms
        parallelisms = data['jobTopology']['parallelisms']

        return parallelisms

    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
    except KeyError as e:
        print(f"Key not found: {e}")
    except Exception as e:
        print(f"Error: {e}")

def get_latest_metrics(json_data, index_to_id_dict):
    ''' index json_data["metricHistory"] for
    processing_rate, parallelism, selectivity.
    state: (input_partitions, parralelism * num_operators,
        selectivity * num_operators, processing_rate * num_operators) '''
    try:
        # Parse the JSON data
        data = json.loads(json_data)

        # Check if the expected structure is present
        if 'metricHistory' not in data:
            raise ValueError("Invalid JSON format. Missing 'metricHistory'.")

        all_metrics = json_data['metricHistory']
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
            vertex_parallelism = get_parallelisms(json_data)[vertex_id]
            vertex_output_rate = vertex_metrics['CURRENT_PROCESSING_RATE']
            vertex_processing_rate = vertex_metrics['CURRENT_PROCESSING_RATE']

            # state[0] = vertex_input_rate # weird indexing bc one_op(?)
            state[i + 1] = vertex_parallelism
            state[2 * (i + 1)] = vertex_output_rate / vertex_processing_rate
            state[3 * (i + 1)] = vertex_processing_rate

        return state

    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
    except KeyError as e:
        print(f"Key not found: {e}")
    except Exception as e:
        print(f"Error: {e}")

def decode_vertex_bytes(vertex):
    ''' Decode the vertex ID from base64 to hexadecimal. '''
    vertex_bytes = vertex['bytes']
    decoded_bytes = base64.b64decode(vertex_bytes)
    hex_result = decoded_bytes.hex()
    return hex_result

def build_graph(json_data):
    try:
        # Parse the JSON data
        data = json.loads(json_data)

        # Check if the expected structure is present
        if 'jobTopology' not in data:
            raise ValueError("Invalid JSON format. Missing 'jobTopology'.")

        topology = data['jobTopology']
        graph = DirectedGraph(0)
        for vertex in topology['verticesInTopologicalOrder']:
            vertex_id = decode_vertex_bytes(vertex['bytes'])
            graph.add_vertex(Vertex(vertex_id, 0, 0))
        sorted_vertex_names = graph.topological_sort()
        graph.set_source(graph.vertices[sorted_vertex_names[0]])
        graph.set_sink(graph.vertices[sorted_vertex_names[-1]])

        for from_vertex_name, to_vertices in topology['outputs'].items():
            from_vertex = graph.vertices[from_vertex_name]
            for to_vertex in to_vertices:
                to_vertex_bytes = to_vertex['bytes']
                to_vertex = graph.vertices[decode_vertex_bytes(to_vertex_bytes)]
                graph.add_edge(from_vertex, to_vertex)
        
        return graph

    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
    except KeyError as e:
        print(f"Key not found: {e}")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    
    json_data = None
    app.run(host='0.0.0.0', port=5001)

    while json_data is None:
        time.sleep(1)

    graph = build_graph(json_data)
    # index_to_id_dict = {} this is graph.sorted_vertices
    q_learner = q_learner(len(graph.vertices), 3, 3, 3, 3, 3, 3, 0.5, 0.5, 0, graph)
    q_learner.populate_q_table_offline()

    previous_json_data = None
    while True:
        if json_data is not None and json_data != previous_json_data:
            state = get_latest_metrics(json_data, graph.sorted_vertices)
            actions = q_learner.online_generate_action(state)
            action_dict = {graph.sorted_vertices[vert_ind]: action for vert_ind, action in enumerate(actions)}
            # send action_dict to orkhan
            reward = get_reward(graph.sink.name)
            q_learner.online_update_q_table(reward)

            previous_json_data = json_data # Reset previous_json_data after processing

        time.sleep(1) # Add a delay to avoid continuous checking

    # Example 2
    #      A
    #    /   \
    # Source   B -> Sink
    #    \   /
    #      C

    # graph2 = DirectedGraph(2000)
    # A2 = Vertex("A", 700, 2)
    # B2 = Vertex("B", 400, 5)
    # C2 = Vertex("C", 350, 4)
    # graph2.add_vertex(A2)
    # graph2.add_vertex(B2)
    # graph2.add_vertex(C2)
    # graph2.set_source(A2)
    # graph2.set_sink(B2)
    # graph2.add_edge(A2, B2)
    # graph2.add_edge(A2, C2)
    # graph2.add_edge(C2, B2)
    # graph2.sort_vertices()

    # print(graph2.vertices)

    # q_learn = q_learner(len(graph2.vertices), 3, 3, 3, 3, 3, 3, 0.5, 0.5, 0, graph2) # len(graph2.vertices()) = 3
    # q_learn.populate_q_table_offline()
    # print(q_learn.Q)
