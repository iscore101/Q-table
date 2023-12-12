from one_op import q_learner
from modified_ds2 import Vertex
from modified_ds2 import DirectedGraph
from flask import Flask, request
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

        # Create a hashmap to store vertex ID and parallelism
        parallelism_hashmap = {}

        # Iterate over vertices and extract vertex ID and parallelism
        for vertex_id, parallelism in parallelisms.items():
            parallelism_hashmap[vertex_id] = parallelism

        return parallelism_hashmap

    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
    except KeyError as e:
        print(f"Key not found: {e}")
    except Exception as e:
        print(f"Error: {e}")

def parse_json_and_extract_latest_metrics(json_data):
    '''index jsondata["metricHistory"] for
    processing_rate, parallelism, selectivity'''
    ...

def build_graph(metrics):
    ...

def update_vertices(graph, metrics):
    # use flask to get parameters of vertices (call parse... metrics)
    ...

if __name__ == "__main__":
    json_data = None
    app.run(host='0.0.0.0', port=5001)

    while json_data is None:
        time.sleep(1)

    initial_metrics = parse_json_and_extract_latest_metrics(json_data)
    graph = build_graph(initial_metrics)
    q_learner = q_learner(...)

    previous_json_data = None  # Track the previous value of json_data

    # probably better ways of doing this
    while True:
        # Check if json_data has changed from None to containing data
        if json_data is not None and json_data != previous_json_data:
            # Call when json_data changes

            latest_metrics = parse_json_and_extract_latest_metrics(json_data)
            q_learner.online_generate_action(latest_metrics)
            reward = ...
            q_learner.online_update_q_table(reward)
            update_vertices(graph, latest_metrics)

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
