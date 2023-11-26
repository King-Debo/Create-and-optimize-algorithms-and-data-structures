# data.py
# The file that will contain the code for the data processing module.

# Import the modules
import numpy as np
import scipy as sp
import networkx as nx
import csv
import json
import xml
import sqlite3
import requests

# Define the data processing class
class DataProcess:

    # Initialize the data processing
    def __init__(self, user_input):

        # Store the user input
        self.user_input = user_input

        # Store the problem domain
        self.problem_domain = user_input["problem_domain"]

        # Store the desired algorithm or data structure
        self.desired_algorithm_or_data_structure = user_input["desired_algorithm_or_data_structure"]

        # Store the evaluation criteria
        self.evaluation_criteria = user_input["evaluation_criteria"]

    # Define the generate data function
    def generate_data(self):

        # Check the problem domain
        if self.problem_domain == "Sorting":

            # Generate a random array of integers
            self.input_data = np.random.randint(0, 100, size=10)

            # Sort the array in ascending order
            self.output_data = np.sort(self.input_data)

        elif self.problem_domain == "Searching":

            # Generate a random array of integers
            self.input_data = np.random.randint(0, 100, size=10)

            # Generate a random target integer
            self.target = np.random.randint(0, 100)

            # Find the index of the target in the array, or -1 if not found
            self.output_data = np.where(self.input_data == self.target)[0]

        elif self.problem_domain == "Graph traversal":

            # Generate a random graph
            self.input_data = nx.gnp_random_graph(10, 0.5)

            # Generate a random source and destination node
            self.source = np.random.randint(0, 10)
            self.destination = np.random.randint(0, 10)

            # Find a path from the source to the destination, or None if not found
            self.output_data = nx.shortest_path(self.input_data, self.source, self.destination, method="dijkstra")

        elif self.problem_domain == "Shortest path":

            # Generate a random weighted graph
            self.input_data = nx.gnp_random_graph(10, 0.5)
            for u, v in self.input_data.edges():
                self.input_data.edges[u, v]["weight"] = np.random.randint(1, 10)

            # Generate a random source and destination node
            self.source = np.random.randint(0, 10)
            self.destination = np.random.randint(0, 10)

            # Find the shortest path from the source to the destination, or None if not found
            self.output_data = nx.shortest_path(self.input_data, self.source, self.destination, weight="weight")

        elif self.problem_domain == "Knapsack":

            # Generate a random number of items
            self.n = np.random.randint(1, 10)

            # Generate a random array of weights
            self.weights = np.random.randint(1, 10, size=self.n)

            # Generate a random array of values
            self.values = np.random.randint(1, 10, size=self.n)

            # Generate a random capacity
            self.capacity = np.random.randint(10, 50)

            # Store the input data as a tuple
            self.input_data = (self.weights, self.values, self.capacity)

            # Find the optimal subset of items that maximizes the value and does not exceed the capacity
            self.output_data = self.knapsack(self.input_data)

        elif self.problem_domain == "Traveling salesman":

            # Generate a random number of cities
            self.n = np.random.randint(4, 10)

            # Generate a random matrix of distances
            self.distances = np.random.randint(1, 10, size=(self.n, self.n))

            # Make the matrix symmetric and zero-diagonal
            self.distances = (self.distances + self.distances.T) / 2
            np.fill_diagonal(self.distances, 0)

            # Store the input data as the distance matrix
            self.input_data = self.distances

            # Find the shortest tour that visits all cities exactly once and returns to the origin
            self.output_data = self.traveling_salesman(self.input_data)

        elif self.problem_domain == "Other":

            # Use an external source of data, such as a CSV, JSON, XML, SQL, or API file
            self.input_data = self.get_external_data()

            # Apply the desired algorithm or data structure to the input data
            self.output_data = self.apply_algorithm_or_data_structure(self.input_data)

        # Return the input and output data
        return self.input_data, self.output_data

    # Define the knapsack function
    def knapsack(self, input_data):

        # Unpack the input data
        weights, values, capacity = input_data

        # Get the number of items
        n = len(weights)

        # Create a table of size (n + 1) x (capacity + 1)
        table = np.zeros((n + 1, capacity + 1), dtype=int)

        # Fill the table using dynamic programming
        for i in range(1, n + 1):
            for j in range(1, capacity + 1):
                if weights[i - 1] <= j:
                    table[i][j] = max(table[i - 1][j], table[i - 1][j - weights[i - 1]] + values[i - 1])
                else:
                    table[i][j] = table[i - 1][j]

        # Get the optimal value
        optimal_value = table[n][capacity]

        # Get the optimal subset
        optimal_subset = []
        i = n
        j = capacity
        while i > 0 and j > 0:
            if table[i][j] != table[i - 1][j]:
                optimal_subset.append(i - 1)
                j -= weights[i - 1]
            i -= 1
        optimal_subset.reverse()

        # Return the optimal value and subset as a tuple
        return (optimal_value, optimal_subset)

    # Define the traveling salesman function
    def traveling_salesman(self, input_data):

        # Store the distance matrix
        distances = input_data

        # Get the number of cities
        n = len(distances)

        # Create a table of size 2^n x n
        table = np.full((2**n, n), np.inf)

        # Initialize the table for the base case
        table[1][0] = 0

        # Fill the table using dynamic programming
        for s in range(2**n):
            for i in range(n):
                if s & (1 << i):
                    for j in range(n):
                        if j != i and s & (1 << j):
                            table[s][i] = min(table[s][i], table[s ^ (1 << i)][j] + distances[j][i])

        # Get the optimal cost
        optimal_cost = np.inf
        for i in range(1, n):
            optimal_cost = min(optimal_cost, table[2**n - 1][i] + distances[i][0])

        # Get the optimal tour
        optimal_tour = []
        s = 2**n - 1
        i = 0
        while s > 0:
            optimal_tour.append(i)
            min_cost = np.inf
            min_index = -1
            for j in range(n):
                if j != i and s & (1 << j):
                    cost = table[s ^ (1 << i)][j] + distances[j][i]
                    if cost < min_cost:
                        min_cost = cost
                        min_index = j
            i = min_index
            s ^= (1 << i)
        optimal_tour.append(0)
        optimal_tour.reverse()

        # Return the optimal cost and tour as a tuple
        return (optimal_cost, optimal_tour)

    # Define the get external data function
    def get_external_data(self):

        # Get the problem domain
        problem_domain = self.problem_domain

        # Check the problem domain
        if problem_domain == "Sorting":

            # Use a CSV file as the external source of data
            file_name = "sorting_data.csv"

            # Import the csv module
            import csv

            # Open and read the CSV file
            with open(file_name, "r") as file:
                reader = csv.reader(file)
                data = list(reader)

            # Convert the data to a numpy array of integers
            data = np.array(data, dtype=int)

        elif problem_domain == "Searching":

            # Use a JSON file as the external source of data
            file_name = "searching_data.json"

            # Import the json module
            import json

            # Open and read the JSON file
            with open(file_name, "r") as file:
                data = json.load(file)

            # Convert the data to a numpy array of integers
            data = np.array(data, dtype=int)

        elif problem_domain == "Graph traversal":

            # Use an XML file as the external source of data
            file_name = "graph_data.xml"

            # Import the xml module
            import xml.etree.ElementTree as ET

            # Parse the XML file
            tree = ET.parse(file_name)
            root = tree.getroot()

            # Create a networkx graph
            data = nx.Graph()

            # Add the nodes and edges from the XML file
            for node in root.findall("node"):
                data.add_node(node.get("id"))
            for edge in root.findall("edge"):
                data.add_edge(edge.get("source"), edge.get("target"), weight=int(edge.get("weight")))

        elif problem_domain == "Shortest path":

            # Use a SQL file as the external source of data
            file_name = "shortest_path_data.sql"

            # Import the sqlite3 module
            import sqlite3

            # Connect to the SQL file
            conn = sqlite3.connect(file_name)
            cursor = conn.cursor()

            # Create a networkx graph
            data = nx.Graph()

            # Add the nodes and edges from the SQL file
            cursor.execute("SELECT * FROM nodes")
            nodes = cursor.fetchall()
            for node in nodes:
                data.add_node(node[0])
            cursor.execute("SELECT * FROM edges")
            edges = cursor.fetchall()
            for edge in edges:
                data.add_edge(edge[0], edge[1], weight=edge[2])

            # Close the connection
            conn.close()

        elif problem_domain == "Knapsack":

            # Use an API as the external source of data
            url = "https://knapsack-api.herokuapp.com/"

            # Import the requests module
            import requests

            # Get the response from the API
            response = requests.get(url)

            # Check the status code
            if response.status_code == 200:

                # Parse the JSON data
                data = response.json()

                # Unpack the data
                weights = data["weights"]
                values = data["values"]
                capacity = data["capacity"]

                # Store the data as a tuple
                data = (weights, values, capacity)

            else:

                # Raise an exception
                raise Exception("API request failed")

        elif problem_domain == "Traveling salesman":

            # Use an API as the external source of data
            url = "https://traveling-salesman-api.herokuapp.com/"

            # Import the requests module
            import requests

            # Get the response from the API
            response = requests.get(url)

            # Check the status code
            if response.status_code == 200:

                # Parse the JSON data
                data = response.json()

                # Unpack the data
                distances = data["distances"]

                # Store the data as a numpy array
                data = np.array(distances, dtype=int)

            else:

                # Raise an exception
                raise Exception("API request failed")

        elif problem_domain == "Other":

            # Use an external source of data, such as a CSV, JSON, XML, SQL, or API file
            data = self.get_external_data()

        # Return the external data
        return data


    # Define the apply algorithm or data structure function
    def apply_algorithm_or_data_structure(self, input_data):

        # Get the desired algorithm or data structure
        desired = self.desired_algorithm_or_data_structure

        # Check if the desired algorithm or data structure is a sorting algorithm
        if desired in ["bubble sort", "insertion sort", "selection sort", "merge sort", "quick sort", "heap sort", "radix sort"]:

            # Import the sorting module
            import sorting

            # Get the corresponding sorting function
            sort_function = getattr(sorting, desired.replace(" ", "_"))

            # Apply the sorting function to the input data
            output_data = sort_function(input_data)

        # Check if the desired algorithm or data structure is a searching algorithm
        elif desired in ["linear search", "binary search", "interpolation search", "jump search", "exponential search"]:

            # Import the searching module
            import searching

            # Get the corresponding searching function
            search_function = getattr(searching, desired.replace(" ", "_"))

            # Generate a random target element
            target = random.choice(input_data)

            # Apply the searching function to the input data and the target element
            output_data = search_function(input_data, target)

        # Check if the desired algorithm or data structure is a graph traversal algorithm
        elif desired in ["breadth-first search", "depth-first search", "Dijkstra's algorithm", "A* algorithm"]:

            # Import the graph module
            import graph

            # Get the corresponding graph traversal function
            graph_traversal_function = getattr(graph, desired.replace(" ", "_"))

            # Generate a random source and destination node
            source = random.choice(list(input_data.nodes))
            destination = random.choice(list(input_data.nodes))

            # Apply the graph traversal function to the input data and the source and destination nodes
            output_data = graph_traversal_function(input_data, source, destination)

        # Check if the desired algorithm or data structure is a data structure
        elif desired in ["stack", "queue", "linked list", "binary tree", "hash table", "heap", "trie"]:

            # Import the data structure module
            import data_structure

            # Get the corresponding data structure class
            data_structure_class = getattr(data_structure, desired.capitalize())

            # Create an instance of the data structure class
            output_data = data_structure_class()

            # Perform some operations on the data structure using the input data
            output_data.perform_operations(input_data)

        # Check if the desired algorithm or data structure is other
        elif desired == "other":

            # Use an external source of algorithm or data structure, such as a library, a module, or an API
            output_data = self.get_external_algorithm_or_data_structure(input_data)

        # Return the output data
        return output_data

