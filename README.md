# Congestion Routing Game Implementation

A comprehensive Python implementation of congestion routing games where multiple agents choose among several routes, and each agent's cost increases as more agents select the same route.

## Features

- **Flexible Game Setup**: Support for any number of agents and routes
- **Multiple Cost Functions**: Linear, quadratic, and exponential congestion cost models
- **Nash Equilibrium Computation**: 
  - Analytical solutions for 2-agent games using nashpy
  - Best response dynamics for n-agent games
  - Mixed strategy equilibrium using fictitious play
- **Social Optimum Analysis**: Calculate socially optimal allocations
- **Price of Anarchy**: Measure efficiency loss between Nash and social optimum
- **Learning Dynamics**: Reinforcement learning and regret minimization simulation
- **Evolutionary Game Theory**: Replicator dynamics and evolutionary stable strategies
- **Mechanism Design**: Pigouvian taxes and subsidization schemes
- **Network Scenarios**: Realistic routing scenarios (traffic, data center, supply chain, internet)
- **Visualization**: Bar charts showing equilibrium route distributions with save functionality
- **Analysis Tools**: Performance analysis, sensitivity testing, and cost function comparison
- **Robust Error Handling**: Graceful handling of visualization and computational issues
- **Scalability**: Efficient algorithms for both small and large games

## Files Structure

- `congestion_routing_game.py` - Main implementation with the CongestionRoutingGame class
- `simple_examples.py` - Basic examples and interactive demonstration
- `game_analysis_utils.py` - Advanced analysis and testing utilities
- `test_congestion_game.py` - Comprehensive test suite
- `final_demo.py` - Complete demonstration of all features
- `price_of_anarchy_test.py` - Specific test for price of anarchy
- `network_routing_scenarios.py` - Realistic network routing scenarios
- `advanced_analysis.py` - Learning dynamics and evolutionary game theory
- `test_mixed_strategy.py` - Test script for mixed strategy solver
- `README.md` - This documentation file

## Installation

Install required packages:
```bash
pip install nashpy numpy matplotlib scipy
```

## Quick Start

### Running Basic Examples
```python
python simple_examples.py
```

This will run three basic examples:
1. The exact scenario from the problem description (2 agents, 2 routes)
2. A 3-agent, 2-route game with quadratic costs
3. A 4-agent, 3-route game with linear costs

### Running the Main Program
```python
python congestion_routing_game.py
```

This provides:
- Predefined examples
- Interactive custom scenario creation
- Automatic visualization of results

### Running the Final Demo
```python
python final_demo.py
```

This provides a comprehensive demonstration including:
- The exact problem scenario
- Large game analysis
- Cost function comparisons
- Sensitivity analysis
- Performance scaling tests

### Network Routing Scenarios
```python
python network_routing_scenarios.py
```

Runs realistic network routing scenarios including:
- Traffic network routing
- Data center packet routing  
- Supply chain logistics
- Internet backbone routing

### Advanced Learning and Evolutionary Analysis
```python
python advanced_analysis.py
```

Provides cutting-edge analysis including:
- Reinforcement learning simulation
- Regret minimization dynamics
- Evolutionary game theory
- Mechanism design for efficiency improvement

## Usage Examples

### Basic Usage
```python
from congestion_routing_game import CongestionRoutingGame

# Create a game with 3 agents and 2 routes
game = CongestionRoutingGame(num_agents=3, num_routes=2)

# Solve using best response dynamics
results = game.best_response_dynamics()

# Print and visualize results
game.print_results(results)
game.visualize_equilibrium(results)
```

### Custom Cost Function
```python
# Define a custom cost function
def custom_cost(congestion):
    return 2 * (congestion ** 1.5)  # Custom congestion cost

# Create game with custom cost
game = CongestionRoutingGame(num_agents=4, num_routes=3, cost_function=custom_cost)
results = game.best_response_dynamics()
```

### Social Optimum and Price of Anarchy
```python
# Calculate social optimum
game = CongestionRoutingGame(num_agents=4, num_routes=3)
social_opt = game.calculate_social_optimum()

# Compare Nash vs Social Optimum
comparison = game.compare_nash_vs_social_optimum()
print(f"Price of Anarchy: {comparison['price_of_anarchy']:.3f}")
print(f"Efficiency Loss: {comparison['efficiency_loss']:.3f}")
```

### Mixed Strategy Equilibrium
```python
# Compute mixed strategy Nash equilibrium
game = CongestionRoutingGame(num_agents=3, num_routes=2)
mixed_result = game.mixed_strategy_solver()

for i, strategy in enumerate(mixed_result['mixed_strategies']):
    print(f'Agent {i+1} strategy: {strategy}')
print(f'Total expected cost: {mixed_result["total_expected_cost"]:.3f}')
```

### Network Routing Scenarios
```python
from network_routing_scenarios import NetworkRoutingScenarios

# Traffic network scenario
scenarios = NetworkRoutingScenarios()
traffic_config = scenarios.traffic_network_scenario()
results = run_scenario_analysis(traffic_config)
```

### Learning Dynamics
```python
from advanced_analysis import LearningDynamics

# Simulate reinforcement learning
learning = LearningDynamics(game)
rl_results = learning.reinforcement_learning(num_rounds=1000)
print(f"Final costs: {rl_results['final_costs']}")
```

## Problem Scenario Implementation

The exact scenario from the problem description is implemented:

> 2 agents, 2 routes: if both pick route 1, each pays a cost of 5; if one picks route 1 and one picks route 2, each pays 2; if both pick route 2, each pays 5.

```python
def problem_cost_function(congestion):
    if congestion == 1:
        return 2  # Cost when alone on a route
    elif congestion == 2:
        return 5  # Cost when both agents on same route
    else:
        return congestion ** 2  # Fallback for other cases

game = CongestionRoutingGame(num_agents=2, num_routes=2, cost_function=problem_cost_function)
results = game.solve_two_agent_game()
```

## Cost Functions

Three predefined cost functions are available:

1. **Linear**: `cost = base_cost × congestion`
2. **Quadratic**: `cost = base_cost × congestion²` (default)
3. **Exponential**: `cost = base_cost × e^(congestion-1)`

```python
from congestion_routing_game import example_cost_functions

cost_funcs = example_cost_functions()
linear_game = CongestionRoutingGame(3, 2, cost_funcs['linear'])
```

## Nash Equilibrium Methods

### For 2-Agent Games
Uses the `nashpy` library for exact analytical solutions:
- Supports mixed strategy Nash equilibria
- Enumerates all equilibria via support enumeration
- Provides exact expected payoffs

### For N-Agent Games (N > 2)
Uses best response dynamics:
- Iterative algorithm where each agent best responds to others
- Converges to pure strategy Nash equilibria
- Suitable for larger games where analytical solutions are intractable

## Output Interpretation

### 2-Agent Games
- **Strategy vectors**: Probability distributions over routes
- **Expected costs**: Expected cost for each player at equilibrium

### N-Agent Games
- **Strategy profile**: List of route choices for each agent
- **Route distribution**: Number of agents on each route
- **Individual costs**: Cost for each agent
- **Total system cost**: Sum of all agent costs

## Visualization

The program automatically generates bar charts showing:
- For 2-agent games: Mixed strategy probabilities for each player
- For n-agent games: Number of agents on each route at equilibrium

## Advanced Features

### Performance Analysis
- Benchmarks computation time vs game size
- Compares analytical vs numerical methods

### Sensitivity Analysis
- Tests how equilibrium changes with cost function parameters
- Plots cost vs equilibrium outcomes

### Batch Testing
- Runs multiple game configurations
- Summarizes results across different setups

## Mathematical Background

### Congestion Games
In a congestion game:
- Players choose from a finite set of resources (routes)
- Cost of using a resource depends on congestion (number of users)
- Players aim to minimize their individual costs

### Nash Equilibrium
A strategy profile where no player can unilaterally deviate and improve their payoff.

### Cost Functions
The congestion cost typically increases with the number of users:
- **Marginal cost pricing**: Each additional user increases cost
- **Polynomial growth**: Common in traffic and network applications
- **Exponential growth**: Models severe congestion scenarios

## Requirements

- Python 3.7+
- numpy
- matplotlib
- nashpy
- scipy

## License

This implementation is for educational and research purposes.

## Contributing

Feel free to extend the implementation with:
- Additional solution algorithms
- More sophisticated cost functions
- Alternative equilibrium concepts
- Enhanced visualization options
