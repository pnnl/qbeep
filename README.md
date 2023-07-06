# Q-BEEP - Quantum Bayesian Error Mitigation Employing Poisson Modelling
## Published : ISCA 2023,  https://dl.acm.org/doi/abs/10.1145/3579371.3589043

The following repository contains a script performing the Q-BEEP algorithm described in the above publication. 

## Dependencies

The project requires the following Python libraries:

- Qiskit
- SciPy
- NumPy
- Pandas
- Matplotlib
- IPython

You can install these dependencies using pip:

```bash
pip install qiskit scipy numpy pandas matplotlib ipython
```

## Usage

The main python file contains all functions used in implementing the Q-BEEP algorithm.
To implement, you need to provide a list of circuits, a list of results, and an IBM backend. All data types should follow qiskits formatting (dictionary for results, etc).

Tune parameters such as the number of:
- Iterations
- Poisson Calculation Methodology
- Graph Instantiation Methodology
- Learning Rate

The resultant fidelity history for each circuit/result pair passed is under the fidelities object.
