"""
Congestion Routing Game Implementation

This program models a congestion routing game where multiple agents choose among
several routes, and each agent's cost increases as more agents select the same route.

The program supports:
- User-specified number of agents and routes
- Payoff matrix construction based on congestion costs
- Nash Equilibrium computation using nashpy for 2-agent games
- Numerical approaches for larger games
- Visualization of equilibrium outcomes

Author: GitHub Copilot
Date: July 31, 2025
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import nashpy as nash
from itertools import product, combinations_with_replacement
from scipy.optimize import minimize
import warnings
import os

# Set matplotlib to non-interactive backend if no display
if os.environ.get('DISPLAY') is None:
    matplotlib.use('Agg')

class CongestionRoutingGame:
    """
    A class to model and solve congestion routing games.
    
    In a congestion routing game, agents choose among routes and experience
    costs that increase with the number of agents using the same route.
    """
    
    def __init__(self, num_agents, num_routes, cost_function=None):
        """
        Initialize the congestion routing game.
        
        Args:
            num_agents (int): Number of agents in the game
            num_routes (int): Number of available routes
            cost_function (callable): Function to calculate cost based on congestion
                                    Default: quadratic cost function
        """
        self.num_agents = num_agents
        self.num_routes = num_routes
        
        # Default quadratic cost function: cost = base_cost * (num_agents_on_route)^2
        if cost_function is None:
            self.cost_function = lambda congestion, base_cost=1: base_cost * (congestion ** 2)
        else:
            self.cost_function = cost_function
            
        # Generate all possible strategy profiles (only for smaller games)
        if num_agents * num_routes < 1000:  # Limit to prevent memory issues
            self.strategy_profiles = list(product(range(num_routes), repeat=num_agents))
        else:
            self.strategy_profiles = None  # Will compute on demand for large games
            print(f"Note: Strategy enumeration disabled for large game ({num_agents} agents, {num_routes} routes)")
        
        # Create payoff matrices
        self.payoff_matrices = self._create_payoff_matrices()
    
    def _create_payoff_matrices(self):
        """
        Create payoff (cost) matrices for all agents.
        
        Returns:
            list: List of payoff matrices, one for each agent
        """
        print(f"Creating payoff matrices for {self.num_agents} agents and {self.num_routes} routes...")
        
        # For each agent, create a payoff matrix
        payoff_matrices = []
        
        for agent_idx in range(self.num_agents):
            # Create matrix dimensions based on strategy space
            if self.num_agents == 2:
                # For 2-agent games, create traditional 2D matrices
                matrix = np.zeros((self.num_routes, self.num_routes))
                
                for i in range(self.num_routes):
                    for j in range(self.num_routes):
                        # Calculate congestion for each route
                        route_counts = [0] * self.num_routes
                        route_counts[i] += 1  # Agent 0's choice
                        route_counts[j] += 1  # Agent 1's choice
                        
                        # Agent's cost is based on congestion of their chosen route
                        if agent_idx == 0:
                            chosen_route = i
                        else:
                            chosen_route = j
                            
                        cost = self.cost_function(route_counts[chosen_route])
                        matrix[i, j] = -cost  # Negative because nashpy expects utilities, not costs
                        
                payoff_matrices.append(matrix)
            else:
                # For n-agent games (n > 2), store costs for each strategy profile
                costs = {}
                for profile in self.strategy_profiles:
                    # Count agents on each route
                    route_counts = [0] * self.num_routes
                    for route_choice in profile:
                        route_counts[route_choice] += 1
                    
                    # Calculate cost for this agent given the strategy profile
                    agent_route = profile[agent_idx]
                    cost = self.cost_function(route_counts[agent_route])
                    costs[profile] = cost
                
                payoff_matrices.append(costs)
        
        return payoff_matrices
    
    def solve_two_agent_game(self):
        """
        Solve the game for exactly 2 agents using nashpy.
        
        Returns:
            tuple: Nash equilibria and expected payoffs
        """
        if self.num_agents != 2:
            raise ValueError("This method is only for 2-agent games")
        
        print("\nSolving 2-agent game using nashpy...")
        
        # Create the game
        A = self.payoff_matrices[0]  # Player 1's payoff matrix
        B = self.payoff_matrices[1]  # Player 2's payoff matrix
        
        print(f"Player 1 payoff matrix (negative costs):\n{A}")
        print(f"Player 2 payoff matrix (negative costs):\n{B}")
        
        game = nash.Game(A, B)
        
        # Find Nash equilibria
        equilibria = list(game.support_enumeration())
        
        results = []
        for i, eq in enumerate(equilibria):
            strategy1, strategy2 = eq
            expected_payoff1 = np.sum(strategy1 @ A @ strategy2.T)
            expected_payoff2 = np.sum(strategy1 @ B @ strategy2.T)
            
            results.append({
                'equilibrium_id': i + 1,
                'strategy_player1': strategy1,
                'strategy_player2': strategy2,
                'expected_cost_player1': -expected_payoff1,  # Convert back to cost
                'expected_cost_player2': -expected_payoff2   # Convert back to cost
            })
        
        return results
    
    def best_response_dynamics(self, max_iterations=1000, tolerance=1e-6):
        """
        Find Nash equilibrium using best response dynamics for n-agent games.
        
        Args:
            max_iterations (int): Maximum number of iterations
            tolerance (float): Convergence tolerance
            
        Returns:
            dict: Final strategy profile and costs
        """
        print(f"\nSolving {self.num_agents}-agent game using best response dynamics...")
        
        # Initialize with random strategy profile
        current_profile = [np.random.randint(0, self.num_routes) for _ in range(self.num_agents)]
        
        for iteration in range(max_iterations):
            old_profile = current_profile.copy()
            
            # Each agent best responds to others' strategies
            for agent_idx in range(self.num_agents):
                best_route = self._find_best_response(agent_idx, current_profile)
                current_profile[agent_idx] = best_route
            
            # Check for convergence
            if current_profile == old_profile:
                print(f"Converged after {iteration + 1} iterations")
                converged = True
                iterations = iteration + 1
                break
        else:
            print(f"Did not converge after {max_iterations} iterations")
            converged = False
            iterations = max_iterations
        
        # Calculate final costs
        route_counts = [0] * self.num_routes
        for route_choice in current_profile:
            route_counts[route_choice] += 1
        
        agent_costs = []
        for agent_idx in range(self.num_agents):
            agent_route = current_profile[agent_idx]
            cost = self.cost_function(route_counts[agent_route])
            agent_costs.append(cost)
        
        return {
            'strategy_profile': current_profile,
            'route_distribution': route_counts,
            'agent_costs': agent_costs,
            'total_cost': sum(agent_costs),
            'converged': converged,
            'iterations': iterations,
            'equilibrium': current_profile  # Add this for compatibility with UI
        }
    
    def _find_best_response(self, agent_idx, current_profile):
        """
        Find the best response for a given agent to others' strategies.
        
        Args:
            agent_idx (int): Index of the agent
            current_profile (list): Current strategy profile
            
        Returns:
            int: Best route choice for the agent
        """
        best_route = 0
        best_cost = float('inf')
        
        for route in range(self.num_routes):
            # Create temporary profile with agent's new choice
            temp_profile = current_profile.copy()
            temp_profile[agent_idx] = route
            
            # Count agents on each route
            route_counts = [0] * self.num_routes
            for route_choice in temp_profile:
                route_counts[route_choice] += 1
            
            # Calculate agent's cost for this choice
            cost = self.cost_function(route_counts[route])
            
            if cost < best_cost:
                best_cost = cost
                best_route = route
        
        return best_route
    
    def calculate_social_optimum(self):
        """
        Calculate the social optimum (minimum total cost) allocation.
        
        Returns:
            dict: Social optimum allocation and costs
        """
        print("Calculating social optimum...")
        
        if self.strategy_profiles is None:
            # For large games, use heuristic approach
            return self._heuristic_social_optimum()
        
        min_total_cost = float('inf')
        best_profile = None
        
        # Try all possible strategy profiles
        for profile in self.strategy_profiles:
            # Count agents on each route
            route_counts = [0] * self.num_routes
            for route_choice in profile:
                route_counts[route_choice] += 1
            
            # Calculate total social cost
            total_cost = 0
            for route_idx, count in enumerate(route_counts):
                if count > 0:
                    # Total cost for this route = count * cost_per_agent
                    route_cost = count * self.cost_function(count)
                    total_cost += route_cost
            
            if total_cost < min_total_cost:
                min_total_cost = total_cost
                best_profile = profile
        
        # Calculate individual costs for the optimal profile
        route_counts = [0] * self.num_routes
        for route_choice in best_profile:
            route_counts[route_choice] += 1
        
        agent_costs = []
        for agent_idx in range(self.num_agents):
            agent_route = best_profile[agent_idx]
            cost = self.cost_function(route_counts[agent_route])
            agent_costs.append(cost)
        
        return {
            'strategy_profile': best_profile,
            'route_distribution': route_counts,
            'agent_costs': agent_costs,
            'total_cost': min_total_cost,
            'average_cost': min_total_cost / self.num_agents
        }
    
    def compare_nash_vs_social_optimum(self):
        """
        Compare Nash equilibrium with social optimum.
        
        Returns:
            dict: Comparison results including price of anarchy
        """
        print("\nComparing Nash Equilibrium vs Social Optimum")
        print("=" * 50)
        
        # Get Nash equilibrium
        if self.num_agents == 2:
            nash_results = self.solve_two_agent_game()
            nash_total_cost = sum([eq['expected_cost_player1'] + eq['expected_cost_player2'] 
                                 for eq in nash_results]) / len(nash_results)
        else:
            nash_results = self.best_response_dynamics()
            nash_total_cost = nash_results['total_cost']
        
        # Get social optimum
        social_opt = self.calculate_social_optimum()
        
        # Calculate price of anarchy
        price_of_anarchy = nash_total_cost / social_opt['total_cost']
        
        comparison = {
            'nash_equilibrium': nash_results,
            'social_optimum': social_opt,
            'nash_total_cost': nash_total_cost,
            'social_optimal_cost': social_opt['total_cost'],
            'price_of_anarchy': price_of_anarchy,
            'efficiency_loss': nash_total_cost - social_opt['total_cost']
        }
        
        print(f"Nash Equilibrium total cost: {nash_total_cost:.4f}")
        print(f"Social Optimum total cost: {social_opt['total_cost']:.4f}")
        print(f"Price of Anarchy: {price_of_anarchy:.4f}")
        print(f"Efficiency Loss: {nash_total_cost - social_opt['total_cost']:.4f}")
        
        if self.num_agents > 2:
            print(f"Nash route distribution: {nash_results['route_distribution']}")
        print(f"Social optimal route distribution: {social_opt['route_distribution']}")
        
        return comparison
    
    def _heuristic_social_optimum(self):
        """
        Heuristic approach to find social optimum for large games.
        
        Returns:
            dict: Approximate social optimum
        """
        print("Using heuristic approach for large game social optimum...")
        
        # Start with equal distribution
        agents_per_route = self.num_agents // self.num_routes
        remainder = self.num_agents % self.num_routes
        
        route_counts = [agents_per_route] * self.num_routes
        
        # Distribute remainder agents
        for i in range(remainder):
            route_counts[i] += 1
        
        # Try to improve by moving agents between routes
        improved = True
        iterations = 0
        max_iterations = 100
        
        while improved and iterations < max_iterations:
            improved = False
            iterations += 1
            
            for from_route in range(self.num_routes):
                for to_route in range(self.num_routes):
                    if from_route == to_route or route_counts[from_route] <= 0:
                        continue
                    
                    # Calculate current total cost
                    current_cost = sum(count * self.cost_function(count) 
                                     for count in route_counts)
                    
                    # Try moving one agent
                    new_counts = route_counts.copy()
                    new_counts[from_route] -= 1
                    new_counts[to_route] += 1
                    
                    new_cost = sum(count * self.cost_function(count) 
                                 for count in new_counts)
                    
                    if new_cost < current_cost:
                        route_counts = new_counts
                        improved = True
                        break
                
                if improved:
                    break
        
        # Create strategy profile
        profile = []
        for route_idx, count in enumerate(route_counts):
            profile.extend([route_idx] * count)
        
        # Calculate individual costs
        agent_costs = []
        for agent_idx in range(self.num_agents):
            agent_route = profile[agent_idx]
            cost = self.cost_function(route_counts[agent_route])
            agent_costs.append(cost)
        
        total_cost = sum(agent_costs)
        
        return {
            'strategy_profile': profile,
            'route_distribution': route_counts,
            'agent_costs': agent_costs,
            'total_cost': total_cost,
            'average_cost': total_cost / self.num_agents
        }
    
    def mixed_strategy_solver(self, max_iterations=1000, learning_rate=0.1):
        """
        Solve for mixed strategy Nash equilibrium using fictitious play.
        
        Args:
            max_iterations (int): Maximum number of iterations
            learning_rate (float): Learning rate for strategy updates
            
        Returns:
            dict: Mixed strategy equilibrium and convergence info
        """
        print(f"Computing mixed strategy equilibrium using fictitious play...")
        
        # Initialize mixed strategies (uniform distribution)
        strategies = []
        for agent_idx in range(self.num_agents):
            strategy = np.ones(self.num_routes) / self.num_routes
            strategies.append(strategy)
        
        # Track strategy evolution
        strategy_history = []
        
        for iteration in range(max_iterations):
            old_strategies = [s.copy() for s in strategies]
            strategy_history.append([s.copy() for s in strategies])
            
            # Each agent updates their strategy based on expected payoffs
            for agent_idx in range(self.num_agents):
                # Calculate expected payoffs for each route
                route_payoffs = np.zeros(self.num_routes)
                
                for route in range(self.num_routes):
                    # Calculate expected cost if this agent chooses this route
                    expected_cost = 0
                    
                    # Consider all possible strategy profiles of other agents
                    other_strategies = strategies[:agent_idx] + strategies[agent_idx+1:]
                    
                    # Simplified calculation: assume others play their current mixed strategies
                    expected_route_counts = np.zeros(self.num_routes)
                    expected_route_counts[route] += 1  # This agent on the route
                    
                    # Add expected counts from other agents
                    for other_agent_idx, other_strategy in enumerate(other_strategies):
                        for other_route in range(self.num_routes):
                            expected_route_counts[other_route] += other_strategy[other_route]
                    
                    # Calculate expected cost for this agent
                    expected_cost = self.cost_function(expected_route_counts[route])
                    route_payoffs[route] = -expected_cost  # Negative because we want to minimize cost
                
                # Update strategy using softmax/logit response
                exp_payoffs = np.exp(learning_rate * route_payoffs)
                new_strategy = exp_payoffs / np.sum(exp_payoffs)
                strategies[agent_idx] = new_strategy
            
            # Check for convergence
            converged = True
            for agent_idx in range(self.num_agents):
                if np.linalg.norm(strategies[agent_idx] - old_strategies[agent_idx]) > 1e-6:
                    converged = False
                    break
            
            if converged:
                print(f"Mixed strategy equilibrium converged after {iteration + 1} iterations")
                break
        else:
            print(f"Mixed strategy solver did not converge after {max_iterations} iterations")
        
        # Calculate expected costs
        expected_costs = []
        for agent_idx in range(self.num_agents):
            agent_strategy = strategies[agent_idx]
            expected_cost = 0
            
            for route in range(self.num_routes):
                # Expected number of agents on this route
                expected_count = agent_strategy[route]
                for other_agent_idx in range(self.num_agents):
                    if other_agent_idx != agent_idx:
                        expected_count += strategies[other_agent_idx][route]
                
                route_cost = self.cost_function(expected_count)
                expected_cost += agent_strategy[route] * route_cost
            
            expected_costs.append(expected_cost)
        
        return {
            'mixed_strategies': strategies,
            'expected_costs': expected_costs,
            'total_expected_cost': sum(expected_costs),
            'convergence_iterations': iteration + 1 if converged else max_iterations,
            'converged': converged,
            'strategy_history': strategy_history
        }
    
    def analyze_convergence(self, results_history):
        """
        Analyze convergence properties of the algorithm.
        
        Args:
            results_history (list): History of strategy profiles
            
        Returns:
            dict: Convergence analysis
        """
        if not results_history:
            return {'error': 'No history provided'}
        
        # Calculate strategy variance over time
        num_iterations = len(results_history)
        strategy_variance = []
        
        for iteration in range(1, num_iterations):
            total_variance = 0
            for agent_idx in range(self.num_agents):
                current_strategy = results_history[iteration][agent_idx]
                previous_strategy = results_history[iteration-1][agent_idx]
                variance = np.sum((current_strategy - previous_strategy) ** 2)
                total_variance += variance
            strategy_variance.append(total_variance)
        
        return {
            'strategy_variance_history': strategy_variance,
            'final_variance': strategy_variance[-1] if strategy_variance else 0,
            'convergence_rate': np.mean(np.diff(strategy_variance)) if len(strategy_variance) > 1 else 0
        }
    
    def visualize_equilibrium(self, results, title="Nash Equilibrium Route Distribution", save_path=None):
        """
        Visualize the equilibrium outcome with a bar chart.
        
        Args:
            results (dict): Results from equilibrium computation
            title (str): Title for the plot
            save_path (str): Optional path to save the plot
        """
        try:
            plt.figure(figsize=(10, 6))
            
            if self.num_agents == 2 and 'strategy_player1' in results[0]:
                # For 2-agent games, show mixed strategies
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
                
                # Player 1 strategy
                routes = [f"Route {i+1}" for i in range(self.num_routes)]
                ax1.bar(routes, results[0]['strategy_player1'])
                ax1.set_title('Player 1 Strategy (Probabilities)')
                ax1.set_ylabel('Probability')
                ax1.set_ylim(0, 1)
                
                # Player 2 strategy
                ax2.bar(routes, results[0]['strategy_player2'])
                ax2.set_title('Player 2 Strategy (Probabilities)')
                ax2.set_ylabel('Probability')
                ax2.set_ylim(0, 1)
                
                plt.tight_layout()
            
            else:
                # For n-agent games, show route distribution
                routes = [f"Route {i+1}" for i in range(self.num_routes)]
                route_counts = results['route_distribution']
                
                bars = plt.bar(routes, route_counts)
                plt.title(title)
                plt.xlabel('Routes')
                plt.ylabel('Number of Agents')
                plt.ylim(0, max(route_counts) + 1)
                
                # Add value labels on bars
                for bar, count in zip(bars, route_counts):
                    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                            str(count), ha='center', va='bottom')
            
            if save_path:
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                print(f"Plot saved to: {save_path}")
            else:
                # Try to show, but handle gracefully if no display
                try:
                    plt.show()
                except Exception as e:
                    print(f"Cannot display plot (no GUI available): {e}")
                    print("Plot created successfully but cannot be displayed.")
                    
        except Exception as e:
            print(f"Error creating visualization: {e}")
            print("Continuing without visualization...")
        finally:
            plt.close('all')  # Clean up memory
    
    def print_results(self, results):
        """
        Print the results in a formatted way.
        
        Args:
            results: Results from equilibrium computation
        """
        print("\n" + "="*60)
        print("NASH EQUILIBRIUM RESULTS")
        print("="*60)
        
        if self.num_agents == 2 and isinstance(results, list):
            # 2-agent game results
            for result in results:
                print(f"\nEquilibrium {result['equilibrium_id']}:")
                print(f"Player 1 strategy: {result['strategy_player1']}")
                print(f"Player 2 strategy: {result['strategy_player2']}")
                print(f"Player 1 expected cost: {result['expected_cost_player1']:.4f}")
                print(f"Player 2 expected cost: {result['expected_cost_player2']:.4f}")
        
        else:
            # n-agent game results
            print(f"\nEquilibrium strategy profile: {results['strategy_profile']}")
            print(f"Route distribution: {results['route_distribution']}")
            print(f"Individual agent costs: {results['agent_costs']}")
            print(f"Total system cost: {results['total_cost']:.4f}")
            print(f"Average cost per agent: {results['total_cost']/self.num_agents:.4f}")


def example_cost_functions():
    """
    Define some example cost functions for different congestion models.
    """
    # Linear cost function
    linear_cost = lambda congestion, base_cost=1: base_cost * congestion
    
    # Quadratic cost function (default)
    quadratic_cost = lambda congestion, base_cost=1: base_cost * (congestion ** 2)
    
    # Exponential cost function
    exponential_cost = lambda congestion, base_cost=1: base_cost * np.exp(congestion - 1)
    
    return {
        'linear': linear_cost,
        'quadratic': quadratic_cost,
        'exponential': exponential_cost
    }


def main():
    """
    Main function to demonstrate the congestion routing game.
    """
    print("Congestion Routing Game Simulation")
    print("="*40)
    
    # Example 1: 2 agents, 2 routes (as specified in the problem)
    print("\nExample 1: 2 agents, 2 routes")
    print("-" * 30)
    
    # Custom cost function for the example scenario
    def example_cost(congestion):
        if congestion == 1:
            return 2  # Cost when alone on a route
        elif congestion == 2:
            return 5  # Cost when both agents on same route
        else:
            return congestion ** 2  # Fallback for other cases
    
    game1 = CongestionRoutingGame(num_agents=2, num_routes=2, cost_function=example_cost)
    results1 = game1.solve_two_agent_game()
    game1.print_results(results1)
    game1.visualize_equilibrium(results1)
    
    # Example 2: 4 agents, 3 routes with quadratic cost
    print("\n\nExample 2: 4 agents, 3 routes (quadratic cost)")
    print("-" * 45)
    
    game2 = CongestionRoutingGame(num_agents=4, num_routes=3)
    results2 = game2.best_response_dynamics()
    game2.print_results(results2)
    game2.visualize_equilibrium(results2, "4-Agent Game: Route Distribution at Equilibrium")
    
    # Example 3: User input scenario
    print("\n\nCustom Scenario")
    print("-" * 20)
    
    try:
        num_agents = int(input("Enter number of agents: "))
        num_routes = int(input("Enter number of routes: "))
        
        if num_agents <= 0 or num_routes <= 0:
            print("Invalid input. Using default values: 3 agents, 2 routes")
            num_agents, num_routes = 3, 2
        
        print(f"\nSetting up game with {num_agents} agents and {num_routes} routes...")
        
        # Choose cost function
        cost_functions = example_cost_functions()
        print("\nAvailable cost functions:")
        for name, func in cost_functions.items():
            print(f"- {name}")
        
        cost_choice = input("Choose cost function (linear/quadratic/exponential) [default: quadratic]: ").strip().lower()
        if cost_choice not in cost_functions:
            cost_choice = 'quadratic'
        
        chosen_cost_func = cost_functions[cost_choice]
        
        game3 = CongestionRoutingGame(num_agents, num_routes, chosen_cost_func)
        
        if num_agents == 2:
            results3 = game3.solve_two_agent_game()
        else:
            results3 = game3.best_response_dynamics()
        
        game3.print_results(results3)
        game3.visualize_equilibrium(results3, f"Custom Game: {num_agents} Agents, {num_routes} Routes")
        
    except (ValueError, KeyboardInterrupt):
        print("\nSkipping custom scenario.")
    
    print("\n" + "="*60)
    print("SIMULATION COMPLETE")
    print("="*60)


if __name__ == "__main__":
    # Suppress warnings for cleaner output
    warnings.filterwarnings('ignore')
    main()
