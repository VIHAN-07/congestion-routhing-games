"""
Additional utilities and testing functions for the Congestion Routing Game.

This module provides extended functionality including:
- Performance analysis tools
- Alternative solution methods
- Sensitivity analysis
- Batch testing capabilities

Author: GitHub Copilot
Date: July 31, 2025
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from congestion_routing_game import CongestionRoutingGame, example_cost_functions
import time
import os

# Set matplotlib to non-interactive backend if no display
if os.environ.get('DISPLAY') is None:
    matplotlib.use('Agg')


class GameAnalyzer:
    """
    A class to analyze and test congestion routing games.
    """
    
    def __init__(self):
        self.results_history = []
    
    def performance_analysis(self, max_agents=6, max_routes=4):
        """
        Analyze computational performance for different game sizes.
        
        Args:
            max_agents (int): Maximum number of agents to test
            max_routes (int): Maximum number of routes to test
        """
        print("Performance Analysis")
        print("="*20)
        
        results = []
        
        for num_agents in range(2, max_agents + 1):
            for num_routes in range(2, max_routes + 1):
                print(f"Testing {num_agents} agents, {num_routes} routes...", end=" ")
                
                game = CongestionRoutingGame(num_agents, num_routes)
                
                start_time = time.time()
                if num_agents == 2:
                    equilibria = game.solve_two_agent_game()
                    method = "Analytical (nashpy)"
                else:
                    equilibria = game.best_response_dynamics()
                    method = "Best Response Dynamics"
                end_time = time.time()
                
                computation_time = end_time - start_time
                print(f"{computation_time:.4f}s ({method})")
                
                results.append({
                    'agents': num_agents,
                    'routes': num_routes,
                    'time': computation_time,
                    'method': method
                })
        
        self._plot_performance_results(results)
        return results
    
    def _plot_performance_results(self, results):
        """Plot performance analysis results."""
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # Group by method
            analytical_results = [r for r in results if r['method'].startswith('Analytical')]
            numerical_results = [r for r in results if r['method'].startswith('Best Response')]
            
            # Plot 1: Computation time vs number of agents
            if analytical_results:
                agents_anal = [r['agents'] for r in analytical_results]
                times_anal = [r['time'] for r in analytical_results]
                ax1.plot(agents_anal, times_anal, 'bo-', label='Analytical (2 agents only)')
            
            if numerical_results:
                agents_num = [r['agents'] for r in numerical_results]
                times_num = [r['time'] for r in numerical_results]
                ax1.plot(agents_num, times_num, 'ro-', label='Best Response Dynamics')
            
            ax1.set_xlabel('Number of Agents')
            ax1.set_ylabel('Computation Time (seconds)')
            ax1.set_title('Performance vs Number of Agents')
            ax1.legend()
            ax1.grid(True)
            
            # Plot 2: Computation time vs number of routes
            if numerical_results:
                routes_num = [r['routes'] for r in numerical_results]
                times_num = [r['time'] for r in numerical_results]
                ax2.scatter(routes_num, times_num, c='red', alpha=0.6)
            
            ax2.set_xlabel('Number of Routes')
            ax2.set_ylabel('Computation Time (seconds)')
            ax2.set_title('Performance vs Number of Routes')
            ax2.grid(True)
            
            plt.tight_layout()
            
            # Try to show or save
            try:
                plt.show()
            except:
                save_path = "performance_analysis.png"
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                print(f"Performance plot saved to: {save_path}")
            finally:
                plt.close()
                
        except Exception as e:
            print(f"Error creating performance plots: {e}")
    
    def sensitivity_analysis(self, base_agents=3, base_routes=2, cost_function_name='quadratic'):
        """
        Perform sensitivity analysis by varying cost function parameters.
        
        Args:
            base_agents (int): Base number of agents
            base_routes (int): Base number of routes
            cost_function_name (str): Name of cost function to analyze
        """
        print(f"\nSensitivity Analysis: {cost_function_name} cost function")
        print("="*50)
        
        cost_functions = example_cost_functions()
        base_cost_func = cost_functions[cost_function_name]
        
        # Test different cost multipliers
        multipliers = [0.5, 1.0, 1.5, 2.0, 3.0]
        results = []
        
        for multiplier in multipliers:
            print(f"Testing cost multiplier: {multiplier}")
            
            # Create modified cost function
            modified_cost_func = lambda congestion: base_cost_func(congestion, base_cost=multiplier)
            
            game = CongestionRoutingGame(base_agents, base_routes, modified_cost_func)
            
            if base_agents == 2:
                equilibria = game.solve_two_agent_game()
                total_cost = sum([eq['expected_cost_player1'] + eq['expected_cost_player2'] 
                                for eq in equilibria]) / len(equilibria)
            else:
                equilibrium = game.best_response_dynamics()
                total_cost = equilibrium['total_cost']
            
            results.append({
                'multiplier': multiplier,
                'total_cost': total_cost
            })
        
        # Plot sensitivity results
        multipliers_list = [r['multiplier'] for r in results]
        costs_list = [r['total_cost'] for r in results]
        
        try:
            plt.figure(figsize=(8, 6))
            plt.plot(multipliers_list, costs_list, 'bo-', linewidth=2, markersize=8)
            plt.xlabel('Cost Function Multiplier')
            plt.ylabel('Total System Cost at Equilibrium')
            plt.title(f'Sensitivity Analysis: {cost_function_name.title()} Cost Function')
            plt.grid(True, alpha=0.3)
            
            try:
                plt.show()
            except:
                save_path = f"sensitivity_analysis_{cost_function_name}.png"
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                print(f"Sensitivity plot saved to: {save_path}")
            finally:
                plt.close()
        except Exception as e:
            print(f"Error creating sensitivity plot: {e}")
        
        return results
    
    def compare_cost_functions(self, num_agents=3, num_routes=2):
        """
        Compare different cost functions on the same game setup.
        
        Args:
            num_agents (int): Number of agents
            num_routes (int): Number of routes
        """
        print(f"\nComparing Cost Functions: {num_agents} agents, {num_routes} routes")
        print("="*60)
        
        cost_functions = example_cost_functions()
        results = {}
        
        for name, cost_func in cost_functions.items():
            print(f"Testing {name} cost function...")
            
            game = CongestionRoutingGame(num_agents, num_routes, cost_func)
            
            if num_agents == 2:
                equilibria = game.solve_two_agent_game()
                avg_cost = sum([eq['expected_cost_player1'] + eq['expected_cost_player2'] 
                              for eq in equilibria]) / len(equilibria)
                route_dist = "Mixed strategies (see detailed output)"
            else:
                equilibrium = game.best_response_dynamics()
                avg_cost = equilibrium['total_cost'] / num_agents
                route_dist = equilibrium['route_distribution']
            
            results[name] = {
                'average_cost': avg_cost,
                'route_distribution': route_dist
            }
        
        # Print comparison
        print("\nComparison Results:")
        print("-" * 30)
        for name, result in results.items():
            print(f"{name.title()} cost:")
            print(f"  Average cost per agent: {result['average_cost']:.4f}")
            print(f"  Route distribution: {result['route_distribution']}")
            print()
        
        # Visualize comparison
        names = list(results.keys())
        avg_costs = [results[name]['average_cost'] for name in names]
        
        try:
            plt.figure(figsize=(8, 6))
            bars = plt.bar(names, avg_costs, color=['blue', 'red', 'green'])
            plt.xlabel('Cost Function Type')
            plt.ylabel('Average Cost per Agent')
            plt.title(f'Cost Function Comparison: {num_agents} Agents, {num_routes} Routes')
            
            # Add value labels on bars
            for bar, cost in zip(bars, avg_costs):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{cost:.3f}', ha='center', va='bottom')
            
            try:
                plt.show()
            except:
                save_path = f"cost_comparison_{num_agents}agents_{num_routes}routes.png"
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                print(f"Cost comparison plot saved to: {save_path}")
            finally:
                plt.close()
        except Exception as e:
            print(f"Error creating cost comparison plot: {e}")
        
        return results
    
    def batch_test(self, test_configurations):
        """
        Run batch tests on multiple game configurations.
        
        Args:
            test_configurations (list): List of dicts with 'agents', 'routes', 'cost_func'
        """
        print("Batch Testing")
        print("="*15)
        
        all_results = []
        
        for i, config in enumerate(test_configurations):
            num_agents = config['agents']
            num_routes = config['routes']
            cost_func = config.get('cost_func', None)
            
            print(f"\nTest {i+1}: {num_agents} agents, {num_routes} routes")
            
            game = CongestionRoutingGame(num_agents, num_routes, cost_func)
            
            if num_agents == 2:
                results = game.solve_two_agent_game()
                test_result = {
                    'config': config,
                    'equilibria_count': len(results),
                    'avg_total_cost': sum([r['expected_cost_player1'] + r['expected_cost_player2'] 
                                         for r in results]) / len(results)
                }
            else:
                results = game.best_response_dynamics()
                test_result = {
                    'config': config,
                    'final_profile': results['strategy_profile'],
                    'total_cost': results['total_cost'],
                    'route_distribution': results['route_distribution']
                }
            
            all_results.append(test_result)
        
        return all_results


def run_comprehensive_analysis():
    """
    Run a comprehensive analysis of the congestion routing game.
    """
    analyzer = GameAnalyzer()
    
    print("Comprehensive Analysis of Congestion Routing Games")
    print("="*55)
    
    # 1. Performance analysis
    print("\n1. Performance Analysis")
    performance_results = analyzer.performance_analysis(max_agents=5, max_routes=3)
    
    # 2. Sensitivity analysis
    print("\n2. Sensitivity Analysis")
    sensitivity_results = analyzer.sensitivity_analysis()
    
    # 3. Cost function comparison
    print("\n3. Cost Function Comparison")
    comparison_results = analyzer.compare_cost_functions()
    
    # 4. Batch testing
    print("\n4. Batch Testing")
    test_configs = [
        {'agents': 2, 'routes': 2},
        {'agents': 2, 'routes': 3},
        {'agents': 3, 'routes': 2},
        {'agents': 4, 'routes': 3},
        {'agents': 5, 'routes': 2}
    ]
    batch_results = analyzer.batch_test(test_configs)
    
    print("\nBatch Test Summary:")
    for i, result in enumerate(batch_results):
        config = result['config']
        print(f"Test {i+1} ({config['agents']} agents, {config['routes']} routes):")
        if 'avg_total_cost' in result:
            print(f"  Average total cost: {result['avg_total_cost']:.4f}")
        else:
            print(f"  Total cost: {result['total_cost']:.4f}")
            print(f"  Route distribution: {result['route_distribution']}")
    
    print("\n" + "="*55)
    print("COMPREHENSIVE ANALYSIS COMPLETE")
    print("="*55)


if __name__ == "__main__":
    run_comprehensive_analysis()
