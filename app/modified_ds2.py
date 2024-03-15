# Vertex Class Definition
class Vertex:
    def __init__(self, name, processing_rate, selectivity, parallelism=1):
        self.name = name
        self.processing_rate = processing_rate
        self.parallelism = parallelism
        self.selectivity = selectivity
        self.adjacent = set()

    def add_neighbor(self, vertex):
        if isinstance(vertex, Vertex):
            self.adjacent.add(vertex)

    def get_neighbors(self):
        return self.adjacent

# DirectedGraph Class Definition
class DirectedGraph:
    def __init__(self, input_rate):
        self.vertices = {}
        self.input_rate = input_rate
        self.source = None
        self.sink = None
        self.sorted_vertices = None

    def add_vertex(self, vertex):
        if isinstance(vertex, Vertex):
            self.vertices[vertex.name] = vertex

    def get_vertex(self, name):
        return self.vertices.get(name)

    def set_source(self, vertex):
        if isinstance(vertex, Vertex):
            self.source = vertex

    def set_sink(self, vertex):
        if isinstance(vertex, Vertex):
            self.sink = vertex

    def add_edge(self, from_vertex, to_vertex):
        if from_vertex.name in self.vertices and to_vertex.name in self.vertices:
            from_vertex.add_neighbor(to_vertex)

    def topological_sort(self):
        stack = []
        visited = set()

        def dfs(vertex):
            visited.add(vertex.name)
            for neighbor in vertex.get_neighbors():
                if neighbor.name not in visited:
                    dfs(neighbor)
            stack.append(vertex)

        dfs(self.source)
        sorted_vertices = [vertex for vertex in stack[::-1]]
        self.sorted_vertices = {index: vertex.name for index, vertex in enumerate(sorted_vertices)}
        return sorted_vertices

    def compute_output_rates(self):
        if not self.source:
            raise ValueError("Source node is not set. Cannot compute output rates.")
        output_rates = {}
        output_rates[self.source.name] = self.source.selectivity * min(self.input_rate, self.source.parallelism * self.source.processing_rate)

        for vertex in self.topological_sort():
            if vertex != self.source:
                aggregated_output = sum([output_rates[v.name] for v in self.vertices.values() if vertex in v.get_neighbors()])
                output_rates[vertex.name] = vertex.selectivity * min(vertex.parallelism * vertex.processing_rate, aggregated_output)

        return output_rates

    def get_sink_input_rate(self, output_rates):
        if not self.sink:
            raise ValueError("Sink node is not set. Cannot compute its input rate.")
        return output_rates.get(self.sink.name, None)

    # def sort_vertices(self):
    #     # Get the sorted keys
    #     sorted_keys = sorted(self.vertices.keys())
    #     # Create a new map that maps each key to its index (order)
    #     self.sorted_vertices = {index: key for index, key in enumerate(sorted_keys)}
    #     for key, value in self.sorted_vertices.items():
    #         print(f"{key}: {value}")

    def compute_output_rates_in_q_table(self, param_list):
        param_num = len(self.vertices)
        # print(param_list)
        # print(param_num)

        self.input_rate = param_list[0]

        for i in range(param_num):
            self.vertices[self.sorted_vertices[i]].parallelism = param_list[1+i]
            self.vertices[self.sorted_vertices[i]].selectivity = param_list[1+i+param_num]
            self.vertices[self.sorted_vertices[i]].processing_rate = param_list[1+i+param_num*2]

        output_rates2 = self.compute_output_rates()
        sink_input_rate2 = self.get_sink_input_rate(output_rates2)
        # print(sink_input_rate2)
        return sink_input_rate2
