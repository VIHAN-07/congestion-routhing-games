"""
Simple Example Script for Congestion Routing Game

This script demonstrates the basic functionality of the congestion routing game
with clear, easy-to-understand examples.

Author: GitHub Copilot
Date: July 31, 2025
"""

from congestion_routing_game import CongestionRoutingGame, example_cost_functions


def run_basic_examples():
    """
    Run basic examples to demonstrate the congestion routing game.
    """
    print("CONGESTION ROUTING GAME - BASIC EXAMPLES")
    print("="*45)
    
    # Example 1: The scenario from the problem description
    print("\nExample 1: 2 agents, 2 routes (Problem Scenario)")
    print("-" * 50)
    print("Scenario: If both pick route 1, each pays cost 5")
    print("         If one picks route 1, one picks route 2, each pays cost 2")
    print("         If both pick route 2, each pays cost 5")
    
    def problem_cost_function(congestion):
        """Cost function matching the problem description."""
        if congestion == 1:
            return 2  # Alone on route
        elif congestion == 2:
            return 5  # Both agents on same route
        else:
            return congestion ** 2  # Fallback
    
    game1 = CongestionRoutingGame(num_agents=2, num_routes=2, cost_function=problem_cost_function)
    results1 = game1.solve_two_agent_game()
    
    print("\nPayoff matrices (negative costs for nashpy):")
    print("Player 1 matrix:")
    print(game1.payoff_matrices[0])
    print("Player 2 matrix:")
    print(game1.payoff_matrices[1])
    
    game1.print_results(results1)
    
    # Example 2: Simple quadratic cost
    print("\n\nExample 2: 3 agents, 2 routes (Quadratic Cost)")
    print("-" * 48)
    print("Cost function: cost = (number of agents on route)^2")
    
    game2 = CongestionRoutingGame(num_agents=3, num_routes=2)
    results2 = game2.best_response_dynamics()
    game2.print_results(results2)
    
    # Example 3: Linear cost function
    print("\n\nExample 3: 4 agents, 3 routes (Linear Cost)")
    print("-" * 45)
    print("Cost function: cost = number of agents on route")
    
    cost_functions = example_cost_functions()
    linear_cost = cost_functions['linear']
    
    game3 = CongestionRoutingGame(num_agents=4, num_routes=3, cost_function=linear_cost)
    results3 = game3.best_response_dynamics()
    game3.print_results(results3)
    
    print("\n" + "="*45)
    print("BASIC EXAMPLES COMPLETE")
    print("="*45)
    
    return results1, results2, results3


def demonstrate_cost_functions():
    """
    Demonstrate different cost functions and their effects.
    """
    print("\n\nCOST FUNCTION DEMONSTRATION")
    print("="*30)
    
    cost_functions = example_cost_functions()
    
    print("Testing different cost functions with 3 agents, 2 routes:")
    print("-" * 55)
    
    for name, cost_func in cost_functions.items():
        print(f"\n{name.upper()} COST FUNCTION:")
        print(f"Cost for 1 agent on route: {cost_func(1)}")
        print(f"Cost for 2 agents on route: {cost_func(2)}")
        print(f"Cost for 3 agents on route: {cost_func(3)}")
        
        game = CongestionRoutingGame(num_agents=3, num_routes=2, cost_function=cost_func)
        result = game.best_response_dynamics()
        
        print(f"Equilibrium route distribution: {result['route_distribution']}")
        print(f"Total system cost: {result['total_cost']:.2f}")


def interactive_demo():
    """
    Interactive demonstration allowing user input.
    """
    print("\n\nINTERACTIVE DEMONSTRATION")
    print("="*27)
    
    try:
        # Get user input
        num_agents = int(input("Enter number of agents (2-10): "))
        num_routes = int(input("Enter number of routes (2-5): "))
        
        # Validate input
        if not (2 <= num_agents <= 10) or not (2 <= num_routes <= 5):
            print("Invalid input. Using default: 3 agents, 2 routes")
            num_agents, num_routes = 3, 2
        
        # Choose cost function
        print("\nAvailable cost functions:")
        print("1. Linear (cost = congestion)")
        print("2. Quadratic (cost = congestion^2) [default]")
        print("3. Exponential (cost = e^(congestion-1))")
        
        choice = input("Choose cost function (1-3) [default: 2]: ").strip()
        
        cost_functions = example_cost_functions()
        if choice == "1":
            cost_func = cost_functions['linear']
            cost_name = "linear"
        elif choice == "3":
            cost_func = cost_functions['exponential']
            cost_name = "exponential"
        else:
            cost_func = cost_functions['quadratic']
            cost_name = "quadratic"
        
        print(f"\nSetting up game: {num_agents} agents, {num_routes} routes, {cost_name} cost")
        print("-" * 60)
        
        # Create and solve game
        game = CongestionRoutingGame(num_agents, num_routes, cost_func)
        
        if num_agents == 2:
            results = game.solve_two_agent_game()
            game.print_results(results)
            
            # Ask if user wants to see visualization
            show_viz = input("\nShow visualization? (y/n) [default: y]: ").strip().lower()
            if show_viz != 'n':
                game.visualize_equilibrium(results)
        else:
            results = game.best_response_dynamics()
            game.print_results(results)
            
            # Ask if user wants to see visualization
            show_viz = input("\nShow visualization? (y/n) [default: y]: ").strip().lower()
            if show_viz != 'n':
                game.visualize_equilibrium(results)
        
        return results
        
    except (ValueError, KeyboardInterrupt):
        print("\nInvalid input or cancelled. Running default example instead.")
        
        # Run default example
        default_game = CongestionRoutingGame(num_agents=3, num_routes=2)
        default_results = default_game.best_response_dynamics()
        default_game.print_results(default_results)
        return default_results


if __name__ == "__main__":
    # Run all demonstrations
    basic_results = run_basic_examples()
    demonstrate_cost_functions()
    interactive_demo()
    
    print("\n" + "="*60)
    print("ALL DEMONSTRATIONS COMPLETE")
    print("Thank you for exploring the Congestion Routing Game!")
    print("="*60)
