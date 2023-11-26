# ui.py
# The file that will contain the code for the user interface module.

# Import the modules
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox

# Define the user interface class
class UserInterface:

    # Initialize the user interface
    def __init__(self):

        # Create the root window
        self.root = tk.Tk()
        self.root.title("Genetic Programming and Metaheuristics Project")
        self.root.geometry("800x600")
        self.root.resizable(False, False)

        # Create the main frame
        self.main_frame = ttk.Frame(self.root, padding=10)
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        # Create the title label
        self.title_label = ttk.Label(self.main_frame, text="Welcome to the Genetic Programming and Metaheuristics Project", font=("Arial", 16, "bold"))
        self.title_label.pack(pady=10)

        # Create the instruction label
        self.instruction_label = ttk.Label(self.main_frame, text="Please specify the parameters and options for the project below", font=("Arial", 12))
        self.instruction_label.pack(pady=10)

        # Create the problem domain label and combobox
        self.problem_domain_label = ttk.Label(self.main_frame, text="Problem domain:", font=("Arial", 12))
        self.problem_domain_label.pack(pady=10, anchor=tk.W)
        self.problem_domain = tk.StringVar()
        self.problem_domain_combobox = ttk.Combobox(self.main_frame, textvariable=self.problem_domain, state="readonly", width=20)
        self.problem_domain_combobox["values"] = ("Sorting", "Searching", "Graph traversal", "Shortest path", "Knapsack", "Traveling salesman", "Other")
        self.problem_domain_combobox.current(0)
        self.problem_domain_combobox.pack(pady=10, anchor=tk.W)

        # Create the desired algorithm or data structure label and entry
        self.desired_algorithm_or_data_structure_label = ttk.Label(self.main_frame, text="Desired algorithm or data structure:", font=("Arial", 12))
        self.desired_algorithm_or_data_structure_label.pack(pady=10, anchor=tk.W)
        self.desired_algorithm_or_data_structure = tk.StringVar()
        self.desired_algorithm_or_data_structure_entry = ttk.Entry(self.main_frame, textvariable=self.desired_algorithm_or_data_structure, width=40)
        self.desired_algorithm_or_data_structure_entry.pack(pady=10, anchor=tk.W)

        # Create the evaluation criteria label and entry
        self.evaluation_criteria_label = ttk.Label(self.main_frame, text="Evaluation criteria:", font=("Arial", 12))
        self.evaluation_criteria_label.pack(pady=10, anchor=tk.W)
        self.evaluation_criteria = tk.StringVar()
        self.evaluation_criteria_entry = ttk.Entry(self.main_frame, textvariable=self.evaluation_criteria, width=40)
        self.evaluation_criteria_entry.pack(pady=10, anchor=tk.W)

        # Create the optimization parameters label and frame
        self.optimization_parameters_label = ttk.Label(self.main_frame, text="Optimization parameters:", font=("Arial", 12))
        self.optimization_parameters_label.pack(pady=10, anchor=tk.W)
        self.optimization_parameters_frame = ttk.Frame(self.main_frame, padding=10)
        self.optimization_parameters_frame.pack(pady=10, anchor=tk.W)

        # Create the population size label and spinbox
        self.population_size_label = ttk.Label(self.optimization_parameters_frame, text="Population size:", font=("Arial", 12))
        self.population_size_label.grid(row=0, column=0, padx=10, pady=10, sticky=tk.W)
        self.population_size = tk.IntVar()
        self.population_size.set(100)
        self.population_size_spinbox = ttk.Spinbox(self.optimization_parameters_frame, from_=1, to=1000, increment=1, textvariable=self.population_size, width=10)
        self.population_size_spinbox.grid(row=0, column=1, padx=10, pady=10, sticky=tk.W)

        # Create the number of generations label and spinbox
        self.number_of_generations_label = ttk.Label(self.optimization_parameters_frame, text="Number of generations:", font=("Arial", 12))
        self.number_of_generations_label.grid(row=1, column=0, padx=10, pady=10, sticky=tk.W)
        self.number_of_generations = tk.IntVar()
        self.number_of_generations.set(100)
        self.number_of_generations_spinbox = ttk.Spinbox(self.optimization_parameters_frame, from_=1, to=1000, increment=1, textvariable=self.number_of_generations, width=10)
        self.number_of_generations_spinbox.grid(row=1, column=1, padx=10, pady=10, sticky=tk.W)

        # Create the crossover rate label and scale
        self.crossover_rate_label = ttk.Label(self.optimization_parameters_frame, text="Crossover rate:", font=("Arial", 12))
        self.crossover_rate_label.grid(row=2, column=0, padx=10, pady=10, sticky=tk.W)
        self.crossover_rate = tk.DoubleVar()
        self.crossover_rate.set(0.8)
        self.crossover_rate_scale = ttk.Scale(self.optimization_parameters_frame, from_=0.0, to=1.0, value=0.8, variable=self.crossover_rate, orient=tk.HORIZONTAL, length=200)
        self.crossover_rate_scale.grid(row=2, column=1, padx=10, pady=10, sticky=tk.W)

        # Create the mutation rate label and scale
        self.mutation_rate_label = ttk.Label(self.optimization_parameters_frame, text="Mutation rate:", font=("Arial", 12))
        self.mutation_rate_label.grid(row=3, column=0, padx=10, pady=10, sticky=tk.W)
        self.mutation_rate = tk.DoubleVar()
        self.mutation_rate.set(0.2)
        self.mutation_rate_scale = ttk.Scale(self.optimization_parameters_frame, from_=0.0, to=1.0, value=0.2, variable=self.mutation_rate, orient=tk.HORIZONTAL, length=200)
        self.mutation_rate_scale.grid(row=3, column=1, padx=10, pady=10, sticky=tk.W)

        # Create the metaheuristics technique label and combobox
        self.metaheuristics_technique_label = ttk.Label(self.optimization_parameters_frame, text="Metaheuristics technique:", font=("Arial", 12))
        self.metaheuristics_technique_label.grid(row=4, column=0, padx=10, pady=10, sticky=tk.W)
        self.metaheuristics_technique = tk.StringVar()
        self.metaheuristics_technique_combobox = ttk.Combobox(self.optimization_parameters_frame, textvariable=self.metaheuristics_technique, state="readonly", width=20)
        self.metaheuristics_technique_combobox["values"] = ("Simulated annealing", "Tabu search", "Ant colony optimization", "Particle swarm optimization", "Other")
        self.metaheuristics_technique_combobox.current(0)
        self.metaheuristics_technique_combobox.grid(row=4, column=1, padx=10, pady=10, sticky=tk.W)

        # Create the submit button
        self.submit_button = ttk.Button(self.main_frame, text="Submit", command=self.submit)
        self.submit_button.pack(pady=10)

        # Start the main loop
        self.root.mainloop()

    # Define the submit function
    def submit(self):

        # Get the user input from the widgets
        self.user_input = {
            "problem_domain": self.problem_domain.get(),
            "desired_algorithm_or_data_structure": self.desired_algorithm_or_data_structure.get(),
            "evaluation_criteria": self.evaluation_criteria.get(),
            "population_size": self.population_size.get(),
            "number_of_generations": self.number_of_generations.get(),
            "crossover_rate": self.crossover_rate.get(),
            "mutation_rate": self.mutation_rate.get(),
            "metaheuristics_technique": self.metaheuristics_technique.get()
        }

        # Validate the user input
        if self.validate_user_input():

            # Show a confirmation message
            messagebox.showinfo("Confirmation", "Your input has been submitted successfully.")

            # Quit the root window
            self.root.quit()

    # Define the validate user input function
    def validate_user_input(self):

        # Check if the desired algorithm or data structure is empty
        if not self.user_input["desired_algorithm_or_data_structure"]:

            # Show an error message
            messagebox.showerror("Error", "Please enter the desired algorithm or data structure.")

            # Return False
            return False

        # Check if the evaluation criteria is empty
        if not self.user_input["evaluation_criteria"]:

            # Show an error message
            messagebox.showerror("Error", "Please enter the evaluation criteria.")

            # Return False
            return False

        # Return True
        return True

    # Define the get user input function
    def get_user_input(self):

        # Return the user input
        return self.user_input
