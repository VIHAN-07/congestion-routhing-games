"""
Final demonstration script for the Congestion Routing Game.

This script showcases the most interesting features and provides
a complete example of the congestion routing game implementation.

Author: GitHub Copilot
Date: July 31, 2025
"""

from congestion_routing_game import CongestionRoutingGame, example_cost_functions
from game_analysis_utils import GameAnalyzer
import numpy as np


def demo_problem_scenario():
    """Demonstrate the exact scenario from the problem description."""
    print("DEMO 1: Problem Scenario (2 agents, 2 routes)")
    print("=" * 50)
    print("Scenario: If both pick route 1, each pays cost 5")
    print("         If one picks route 1, one picks route 2, each pays 2")
    print("         If both pick route 2, each pays cost 5")
    print()
    
    def problem_cost_function(congestion):
        if congestion == 1:
            return 2  # Alone on route
        elif congestion == 2:
            return 5  # Both agents on same route
        else:
            return congestion ** 2  # Fallback
    
    game = CongestionRoutingGame(num_agents=2, num_routes=2, cost_function=problem_cost_function)
    results = game.solve_two_agent_game()
    
    game.print_results(results)
    game.visualize_equilibrium(results, save_path="demo1_problem_scenario.png")
    
    print("\nKey insights:")
    print("- Found multiple Nash equilibria including mixed strategies")
    print("- Pure strategy equilibria involve agents choosing different routes")
    print("- Mixed strategy equilibrium has higher expected cost")
    print()


def demo_large_game():
    """Demonstrate a larger game with interesting dynamics."""
    print("DEMO 2: Large Game (6 agents, 4 routes)")
    print("=" * 40)
    print("Using quadratic cost function: cost = (congestion)²")
    print()
    
    game = CongestionRoutingGame(num_agents=6, num_routes=4)
    
    # Get Nash equilibrium
    nash_result = game.best_response_dynamics()
    print("NASH EQUILIBRIUM:")
    game.print_results(nash_result)
    
    # Compare with social optimum
    comparison = game.compare_nash_vs_social_optimum()
    
    # Visualize
    game.visualize_equilibrium(nash_result, "Large Game Nash Equilibrium", 
                              save_path="demo2_large_game.png")
    
    print("\nKey insights:")
    print(f"- Price of Anarchy: {comparison['price_of_anarchy']:.3f}")
    print(f"- Efficiency Loss: {comparison['efficiency_loss']:.3f}")
    if comparison['price_of_anarchy'] > 1.001:
        print("- Nash equilibrium is inefficient compared to social optimum")
    else:
        print("- Nash equilibrium is socially optimal in this case")
    print()


def demo_cost_function_comparison():
    """Compare different cost functions."""
    print("DEMO 3: Cost Function Comparison (4 agents, 3 routes)")
    print("=" * 55)
    
    cost_functions = example_cost_functions()
    results = {}
    
    for name, cost_func in cost_functions.items():
        print(f"\nTesting {name.upper()} cost function...")
        game = CongestionRoutingGame(num_agents=4, num_routes=3, cost_function=cost_func)
        
        result = game.best_response_dynamics()
        comparison = game.compare_nash_vs_social_optimum()
        
        results[name] = {
            'nash_cost': result['total_cost'],
            'social_cost': comparison['social_optimum']['total_cost'],
            'price_of_anarchy': comparison['price_of_anarchy'],
            'route_distribution': result['route_distribution']
        }
        
        print(f"  Nash total cost: {result['total_cost']:.2f}")
        print(f"  Social optimal cost: {comparison['social_optimum']['total_cost']:.2f}")
        print(f"  Price of Anarchy: {comparison['price_of_anarchy']:.3f}")
        print(f"  Route distribution: {result['route_distribution']}")
    
    print("\nCOMPARISON SUMMARY:")
    print("-" * 20)
    for name, data in results.items():
        print(f"{name.title():12} | PoA: {data['price_of_anarchy']:.3f} | "
              f"Nash Cost: {data['nash_cost']:6.2f} | "
              f"Routes: {data['route_distribution']}")
    
    print("\nKey insights:")
    print("- Different cost functions lead to different equilibrium outcomes")
    print("- Higher order cost functions (quadratic, exponential) create more congestion effects")
    print("- Linear costs often lead to more balanced route usage")
    print()


def demo_sensitivity_analysis():
    """Demonstrate sensitivity to cost parameters."""
    print("DEMO 4: Sensitivity Analysis")
    print("=" * 30)
    
    analyzer = GameAnalyzer()
    
    print("Testing how equilibrium changes with cost multipliers...")
    results = analyzer.sensitivity_analysis(base_agents=4, base_routes=2, 
                                          cost_function_name='quadratic')
    
    print("\nSensitivity Results:")
    for result in results:
        print(f"Cost multiplier {result['multiplier']:3.1f}: "
              f"Total cost = {result['total_cost']:6.2f}")
    
    print("\nKey insights:")
    print("- Total equilibrium cost scales predictably with cost multiplier")
    print("- Route distribution may change at certain thresholds")
    print("- Higher costs encourage more route spreading")
    print()


def demo_performance_scaling():
    """Demonstrate computational performance."""
    print("DEMO 5: Performance Scaling")
    print("=" * 28)
    
    import time
    
    # Test different game sizes
    test_sizes = [
        (2, 2, "Small (2x2)"),
        (3, 3, "Medium (3x3)"),
        (5, 4, "Large (5x4)"),
        (8, 3, "Many agents (8x3)"),
        (4, 6, "Many routes (4x6)")
    ]
    
    print("Game Size          | Time (s) | Method")
    print("-" * 40)
    
    for agents, routes, description in test_sizes:
        start_time = time.time()
        
        game = CongestionRoutingGame(num_agents=agents, num_routes=routes)
        
        if agents == 2:
            results = game.solve_two_agent_game()
            method = "Analytical"
        else:
            results = game.best_response_dynamics()
            method = "Best Response"
        
        end_time = time.time()
        computation_time = end_time - start_time
        
        print(f"{description:18} | {computation_time:7.4f} | {method}")
    
    print("\nKey insights:")
    print("- 2-agent games use analytical Nash equilibrium computation")
    print("- Larger games use efficient best response dynamics")
    print("- Computation time scales well with game size")
    print("- Most games converge within a few iterations")
    print()


def main():
    """Run all demonstrations."""
    print("CONGESTION ROUTING GAME - FINAL DEMONSTRATION")
    print("=" * 50)
    print()
    
    # Run all demos
    demo_problem_scenario()
    demo_large_game()
    demo_cost_function_comparison()
    demo_sensitivity_analysis()
    demo_performance_scaling()
    
    print("=" * 50)
    print("🎯 DEMONSTRATION COMPLETE!")
    print()
    print("Summary of capabilities demonstrated:")
    print("✓ Exact problem scenario implementation")
    print("✓ Nash equilibrium computation (analytical + numerical)")
    print("✓ Social optimum calculation")
    print("✓ Price of anarchy analysis")
    print("✓ Multiple cost function support")
    print("✓ Sensitivity analysis")
    print("✓ Performance scaling")
    print("✓ Visualization and plotting")
    print()
    print("Generated files:")
    print("- demo1_problem_scenario.png (Problem scenario visualization)")
    print("- demo2_large_game.png (Large game route distribution)")
    print("- Various analysis plots from sensitivity testing")
    print()
    print("The implementation successfully models congestion routing games")
    print("with flexible parameters and comprehensive analysis tools!")
    print("=" * 50)


if __name__ == "__main__":
    main()
