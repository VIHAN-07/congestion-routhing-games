"""
Advanced Analysis Module for Congestion Routing Games

This module provides advanced game-theoretic analysis including:
- Learning dynamics simulation
- Evolutionary game theory
- Stochastic stability analysis
- Mechanism design for efficiency improvement

Author: GitHub Copilot
Date: July 31, 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from congestion_routing_game import CongestionRoutingGame
import random
from collections import defaultdict


class LearningDynamics:
    """
    Simulate various learning dynamics in congestion games.
    """
    
    def __init__(self, game):
        self.game = game
        self.num_agents = game.num_agents
        self.num_routes = game.num_routes
    
    def reinforcement_learning(self, num_rounds=1000, exploration_rate=0.1, learning_rate=0.01):
        """
        Simulate reinforcement learning where agents update strategies based on experience.
        
        Args:
            num_rounds (int): Number of learning rounds
            exploration_rate (float): Probability of random exploration
            learning_rate (float): Rate of strategy adjustment
            
        Returns:
            dict: Learning simulation results
        """
        print(f"Simulating reinforcement learning over {num_rounds} rounds...")
        
        # Initialize Q-values for each agent and route
        q_values = np.random.normal(0, 0.1, (self.num_agents, self.num_routes))
        
        # Track history
        strategy_history = []
        cost_history = []
        route_usage_history = []
        
        for round_num in range(num_rounds):
            # Each agent chooses route based on epsilon-greedy policy
            chosen_routes = []
            
            for agent_idx in range(self.num_agents):
                if random.random() < exploration_rate:
                    # Explore: choose random route
                    route = random.randint(0, self.num_routes - 1)
                else:
                    # Exploit: choose route with highest Q-value
                    route = np.argmax(q_values[agent_idx])
                
                chosen_routes.append(route)
            
            # Calculate costs based on chosen routes
            route_counts = [0] * self.num_routes
            for route in chosen_routes:
                route_counts[route] += 1
            
            agent_costs = []
            for agent_idx in range(self.num_agents):
                agent_route = chosen_routes[agent_idx]
                cost = self.game.cost_function(route_counts[agent_route])
                agent_costs.append(cost)
                
                # Update Q-value with negative cost (since we want to minimize cost)
                reward = -cost
                old_q = q_values[agent_idx, agent_route]
                q_values[agent_idx, agent_route] = old_q + learning_rate * (reward - old_q)
            
            # Record history
            strategy_history.append(chosen_routes.copy())
            cost_history.append(agent_costs.copy())
            route_usage_history.append(route_counts.copy())
            
            # Decay exploration rate
            exploration_rate *= 0.9995
        
        return {
            'strategy_history': strategy_history,
            'cost_history': cost_history,
            'route_usage_history': route_usage_history,
            'final_q_values': q_values,
            'final_strategies': chosen_routes,
            'final_costs': agent_costs,
            'total_rounds': num_rounds
        }
    
    def regret_minimization(self, num_rounds=1000):
        """
        Simulate regret minimization learning.
        
        Args:
            num_rounds (int): Number of rounds to simulate
            
        Returns:
            dict: Regret minimization results
        """
        print(f"Simulating regret minimization over {num_rounds} rounds...")
        
        # Track cumulative regret for each agent and route
        cumulative_regret = np.zeros((self.num_agents, self.num_routes))
        
        # Track history
        strategy_history = []
        regret_history = []
        
        for round_num in range(num_rounds):
            # Each agent chooses route based on regret-matching
            chosen_routes = []
            
            for agent_idx in range(self.num_agents):
                # Calculate positive regrets
                positive_regrets = np.maximum(cumulative_regret[agent_idx], 0)
                
                if np.sum(positive_regrets) > 0:
                    # Choose proportional to positive regrets
                    probabilities = positive_regrets / np.sum(positive_regrets)
                else:
                    # Uniform random if no positive regrets
                    probabilities = np.ones(self.num_routes) / self.num_routes
                
                route = np.random.choice(self.num_routes, p=probabilities)
                chosen_routes.append(route)
            
            # Calculate costs and regrets
            route_counts = [0] * self.num_routes
            for route in chosen_routes:
                route_counts[route] += 1
            
            round_regrets = np.zeros((self.num_agents, self.num_routes))
            
            for agent_idx in range(self.num_agents):
                agent_route = chosen_routes[agent_idx]
                actual_cost = self.game.cost_function(route_counts[agent_route])
                
                # Calculate regret for each possible route
                for route in range(self.num_routes):
                    # Cost if agent had chosen this route instead
                    alt_route_counts = route_counts.copy()
                    alt_route_counts[agent_route] -= 1
                    alt_route_counts[route] += 1
                    
                    alt_cost = self.game.cost_function(alt_route_counts[route])
                    regret = actual_cost - alt_cost
                    round_regrets[agent_idx, route] = regret
                
                # Update cumulative regret
                cumulative_regret[agent_idx] += round_regrets[agent_idx]
            
            # Record history
            strategy_history.append(chosen_routes.copy())
            regret_history.append(round_regrets.copy())
        
        return {
            'strategy_history': strategy_history,
            'regret_history': regret_history,
            'cumulative_regret': cumulative_regret,
            'final_strategies': chosen_routes,
            'total_rounds': num_rounds
        }


class EvolutionaryGameAnalysis:
    """
    Evolutionary game theory analysis for congestion games.
    """
    
    def __init__(self, game):
        self.game = game
        self.num_agents = game.num_agents
        self.num_routes = game.num_routes
    
    def replicator_dynamics(self, initial_population=None, time_steps=1000, dt=0.01):
        """
        Simulate replicator dynamics evolution.
        
        Args:
            initial_population (array): Initial population distribution
            time_steps (int): Number of time steps
            dt (float): Time step size
            
        Returns:
            dict: Replicator dynamics simulation results
        """
        print(f"Simulating replicator dynamics over {time_steps} time steps...")
        
        if initial_population is None:
            # Start with uniform distribution
            population = np.ones(self.num_routes) / self.num_routes
        else:
            population = np.array(initial_population)
        
        # Track evolution
        population_history = [population.copy()]
        fitness_history = []
        
        for t in range(time_steps):
            # Calculate fitness for each strategy (route)
            fitness = np.zeros(self.num_routes)
            
            for route in range(self.num_routes):
                # Expected cost if an agent uses this route
                expected_congestion = population[route] * self.num_agents
                expected_cost = self.game.cost_function(expected_congestion)
                fitness[route] = -expected_cost  # Negative cost = fitness
            
            # Calculate average fitness
            avg_fitness = np.sum(population * fitness)
            
            # Update population according to replicator dynamics
            for route in range(self.num_routes):
                dpdt = population[route] * (fitness[route] - avg_fitness)
                population[route] += dt * dpdt
            
            # Ensure population stays normalized and non-negative
            population = np.maximum(population, 1e-10)
            population = population / np.sum(population)
            
            # Record history
            population_history.append(population.copy())
            fitness_history.append(fitness.copy())
        
        return {
            'population_history': population_history,
            'fitness_history': fitness_history,
            'final_population': population,
            'time_steps': time_steps
        }
    
    def evolutionary_stable_strategy(self):
        """
        Find evolutionarily stable strategies (ESS).
        
        Returns:
            dict: ESS analysis results
        """
        print("Analyzing evolutionarily stable strategies...")
        
        # Run replicator dynamics from multiple starting points
        ess_candidates = []
        
        num_trials = 20
        for trial in range(num_trials):
            # Random initial population
            initial_pop = np.random.dirichlet(np.ones(self.num_routes))
            
            result = self.replicator_dynamics(initial_population=initial_pop, time_steps=2000)
            final_pop = result['final_population']
            
            # Check if this is a new equilibrium (not already found)
            is_new = True
            for candidate in ess_candidates:
                if np.linalg.norm(final_pop - candidate) < 0.01:
                    is_new = False
                    break
            
            if is_new:
                ess_candidates.append(final_pop.copy())
        
        return {
            'ess_candidates': ess_candidates,
            'num_equilibria': len(ess_candidates)
        }


class MechanismDesign:
    """
    Mechanism design to improve efficiency in congestion games.
    """
    
    def __init__(self, game):
        self.game = game
        self.num_agents = game.num_agents
        self.num_routes = game.num_routes
    
    def pigouvian_taxes(self):
        """
        Calculate Pigouvian taxes to achieve social optimum.
        
        Returns:
            dict: Tax scheme and efficiency improvement
        """
        print("Calculating Pigouvian taxes for efficiency...")
        
        # Get social optimum
        social_opt = self.game.calculate_social_optimum()
        
        # Calculate marginal externality costs
        taxes = {}
        
        for route in range(self.num_routes):
            route_usage = social_opt['route_distribution'][route]
            
            if route_usage > 0:
                # Marginal cost imposed on others by one additional user
                current_cost = self.game.cost_function(route_usage)
                marginal_cost = self.game.cost_function(route_usage + 1)
                
                # Tax should equal the marginal externality
                marginal_externality = (route_usage + 1) * marginal_cost - route_usage * current_cost - marginal_cost
                taxes[route] = max(0, marginal_externality)
            else:
                taxes[route] = 0
        
        return {
            'social_optimum': social_opt,
            'pigouvian_taxes': taxes,
            'tax_revenue': sum(taxes[route] * social_opt['route_distribution'][route] 
                             for route in range(self.num_routes))
        }
    
    def subsidization_scheme(self):
        """
        Design subsidization scheme to improve efficiency.
        
        Returns:
            dict: Subsidy scheme results
        """
        print("Designing subsidization scheme...")
        
        # Get current Nash equilibrium
        nash_result = self.game.best_response_dynamics()
        social_opt = self.game.calculate_social_optimum()
        
        # Calculate subsidies to incentivize socially optimal routing
        subsidies = {}
        
        for route in range(self.num_routes):
            nash_usage = nash_result['route_distribution'][route]
            optimal_usage = social_opt['route_distribution'][route]
            
            if optimal_usage > nash_usage:
                # Subsidize under-used routes
                subsidy_per_user = (self.game.cost_function(nash_usage) - 
                                  self.game.cost_function(optimal_usage)) * 0.5
                subsidies[route] = max(0, subsidy_per_user)
            else:
                subsidies[route] = 0
        
        total_subsidy_cost = sum(subsidies[route] * social_opt['route_distribution'][route] 
                               for route in range(self.num_routes))
        
        efficiency_gain = nash_result['total_cost'] - social_opt['total_cost']
        
        return {
            'subsidies': subsidies,
            'total_subsidy_cost': total_subsidy_cost,
            'efficiency_gain': efficiency_gain,
            'net_benefit': efficiency_gain - total_subsidy_cost
        }


def run_advanced_analysis(game, scenario_name="Default"):
    """
    Run comprehensive advanced analysis on a congestion game.
    
    Args:
        game (CongestionRoutingGame): Game to analyze
        scenario_name (str): Name of the scenario
        
    Returns:
        dict: Complete advanced analysis results
    """
    print(f"\nADVANCED ANALYSIS: {scenario_name}")
    print("=" * 50)
    
    results = {}
    
    # Learning dynamics analysis
    print("\n1. Learning Dynamics Analysis")
    print("-" * 30)
    
    learning = LearningDynamics(game)
    
    # Reinforcement learning
    rl_results = learning.reinforcement_learning(num_rounds=500)
    results['reinforcement_learning'] = rl_results
    
    print(f"Reinforcement Learning:")
    print(f"  Final route distribution: {[rl_results['route_usage_history'][-1].count(i) for i in range(game.num_routes)]}")
    print(f"  Average final cost: {np.mean(rl_results['final_costs']):.2f}")
    
    # Regret minimization
    if game.num_agents <= 10:  # Only for smaller games due to computational complexity
        regret_results = learning.regret_minimization(num_rounds=300)
        results['regret_minimization'] = regret_results
        
        print(f"Regret Minimization:")
        print(f"  Final route distribution: {[regret_results['strategy_history'][-1].count(i) for i in range(game.num_routes)]}")
        print(f"  Cumulative regret range: [{np.min(regret_results['cumulative_regret']):.2f}, {np.max(regret_results['cumulative_regret']):.2f}]")
    
    # Evolutionary analysis
    print("\n2. Evolutionary Game Analysis")
    print("-" * 30)
    
    evolution = EvolutionaryGameAnalysis(game)
    
    # Replicator dynamics
    replicator_results = evolution.replicator_dynamics(time_steps=1000)
    results['replicator_dynamics'] = replicator_results
    
    print(f"Replicator Dynamics:")
    print(f"  Final population: {replicator_results['final_population']}")
    print(f"  Implied route distribution: {(replicator_results['final_population'] * game.num_agents).astype(int)}")
    
    # ESS analysis
    ess_results = evolution.evolutionary_stable_strategy()
    results['evolutionary_stable_strategies'] = ess_results
    
    print(f"Evolutionarily Stable Strategies:")
    print(f"  Number of ESS candidates: {ess_results['num_equilibria']}")
    for i, ess in enumerate(ess_results['ess_candidates']):
        print(f"  ESS {i+1}: {ess}")
    
    # Mechanism design
    print("\n3. Mechanism Design Analysis")
    print("-" * 30)
    
    mechanism = MechanismDesign(game)
    
    # Pigouvian taxes
    tax_results = mechanism.pigouvian_taxes()
    results['pigouvian_taxes'] = tax_results
    
    print(f"Pigouvian Taxes:")
    print(f"  Taxes by route: {tax_results['pigouvian_taxes']}")
    print(f"  Total tax revenue: {tax_results['tax_revenue']:.2f}")
    
    # Subsidization
    subsidy_results = mechanism.subsidization_scheme()
    results['subsidization'] = subsidy_results
    
    print(f"Subsidization Scheme:")
    print(f"  Subsidies by route: {subsidy_results['subsidies']}")
    print(f"  Total subsidy cost: {subsidy_results['total_subsidy_cost']:.2f}")
    print(f"  Net benefit: {subsidy_results['net_benefit']:.2f}")
    
    return results


if __name__ == "__main__":
    # Test with a sample game
    print("ADVANCED CONGESTION GAME ANALYSIS")
    print("=" * 40)
    
    # Create a test game
    test_game = CongestionRoutingGame(num_agents=6, num_routes=3)
    
    # Run advanced analysis
    results = run_advanced_analysis(test_game, "Test Scenario")
    
    print("\n" + "=" * 40)
    print("ADVANCED ANALYSIS COMPLETE")
    print("=" * 40)
