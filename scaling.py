from one_op import q_learner
from modified_ds2 import Vertex
from modified_ds2 import DirectedGraph
from flask import Flask, request
import numpy as np
import json
import time

app = Flask(__name__)

@app.route("/", methods=['POST', 'GET'])
def hello():
    global json_data
    if request.method == 'POST':
        # Access the request data
        data = request.data.decode('utf-8')
        json_data = data
        print(f"POST request body: {json_data}")
        # return f"POST request body: {data}"
        return json_data
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
            parallelism_result = parse_json_and_extract_parallelism(json_data)
            # json_result = jsonify({"parallelism_result": parallelism_result})
            json_result = json.dumps(parallelism_result)
            print(f"Parallelism result: {json_result}")
            return f"Parallelism result: {json_result}"
        else:
            return "No data available to process."
    else:
        return "Hello from Python!"

def parse_json_and_extract_parallelism(json_data):
    try:
        # Parse the JSON data
        data = json.loads(json_data)

        # Check if the expected structure is present
        if 'jobTopology' not in data or 'parallelisms' not in data['jobTopology']:
            raise ValueError("Invalid JSON format. Missing 'jobTopology' or 'parallelisms'.")

        # Extract parallelisms
        parallelisms = data['jobTopology']['parallelisms']

        # # Create a hashmap to store vertex ID and parallelism
        # parallelism_hashmap = {}

        # # Iterate over vertices and extract vertex ID and parallelism
        # for vertex_id, parallelism in parallelisms.items():
        #     parallelism_hashmap[vertex_id] = parallelism

        return parallelisms

    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
    except KeyError as e:
        print(f"Key not found: {e}")
    except Exception as e:
        print(f"Error: {e}")

def parse_json_and_extract_latest_metrics(json_data, index_to_id_dict):
    ''' index json_data["metricHistory"] for
    processing_rate, parallelism, selectivity.
    state: (input_partitions, parralelism * num_operators,
        selectivity * num_operators, processing_rate * num_operators) '''
    try:
        # Parse the JSON data
        data = json.loads(json_data)

        # Check if the expected structure is present
        if 'metricHistory' not in data or 'parallelisms' not in data['jobTopology']:
            raise ValueError("Invalid JSON format. Missing 'jobTopology' or 'parallelisms'.")

        all_metrics = json_data['metricHistory']
        latest_timestamp = max(all_metrics.keys())
        latest_metrics = all_metrics[latest_timestamp]

        state = np.zeros(1 + (3 * num_ops))
        num_ops = len(latest_metrics)
        for i in range(num_ops):
            vertex_id = index_to_id_dict[i]
            vertex_metrics = latest_metrics[vertex_id]

            # if 'SOURCE_DATA_RATE' in vertex_metrics: # source op
            #     vertex_input_rate = vertex_metrics['SOURCE_DATA_RATE']
            #     if vertex_input_rate == 'NaN':
            #         state[0] = 0
            #     else:
            #         state[0] = vertex_input_rate

            vertex_input_rate = vertex_metrics['NUM_RECORDS_IN_PER_SECOND']
            vertex_parallelism = parse_json_and_extract_parallelism(json_data)[vertex_id]
            vertex_output_rate = vertex_metrics['CURRENT_PROCESSING_RATE']
            vertex_processing_rate = vertex_metrics['CURRENT_PROCESSING_RATE']

            state[0] = vertex_input_rate # weird indexing bc one_op(?)
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

def build_graph(metrics):
    ...

# def update_vertices(graph, metrics):
#     # use flask to get parameters of vertices (call parse... metrics)
#     ...

if __name__ == "__main__":
    graph = build_graph(...) # need dict of vertices somehow {id : }
    q_learner = q_learner(...)
    q_learner.populate_q_table_offline()

    # TODO: populate these (from graph?)
    id_to_index_dict = {}
    index_to_id_dict = {}

    json_data = None
    app.run(host='0.0.0.0', port=5001)

    previous_json_data = None  # Track the previous value of json_data

    # probably better ways of doing this
    while True:
        # Check if json_data has changed from None to containing data
        if json_data is not None and json_data != previous_json_data:

            state = parse_json_and_extract_latest_metrics(json_data, index_to_id_dict)
            actions = q_learner.online_generate_action(state)
            action_dict = {index_to_id_dict[vert_ind]: action for vert_ind, action in enumerate(actions)}
            reward = state[0] # assumes sink is last in index_to_id_dict (one_op)
            q_learner.online_update_q_table(reward)
            # update_vertices(graph, latest_metrics)

            # Reset previous_json_data after processing
            previous_json_data = json_data

        # Add a delay to avoid continuous checking
        time.sleep(1)

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
