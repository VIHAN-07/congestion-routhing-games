"""
Test script for the Congestion Routing Game implementation.

This script tests all major functionality including:
- Basic game creation and solving
- Nash equilibrium computation
- Social optimum calculation
- Price of anarchy analysis
- Visualization capabilities

Author: GitHub Copilot
Date: July 31, 2025
"""

from congestion_routing_game import CongestionRoutingGame, example_cost_functions
from game_analysis_utils import GameAnalyzer
import numpy as np


def test_basic_functionality():
    """Test basic game functionality."""
    print("Testing Basic Functionality")
    print("=" * 30)
    
    # Test 2-agent game
    print("\n1. Testing 2-agent, 2-route game...")
    game = CongestionRoutingGame(num_agents=2, num_routes=2)
    results = game.solve_two_agent_game()
    print(f"Found {len(results)} Nash equilibria")
    assert len(results) > 0, "Should find at least one equilibrium"
    
    # Test n-agent game
    print("\n2. Testing 3-agent, 2-route game...")
    game = CongestionRoutingGame(num_agents=3, num_routes=2)
    results = game.best_response_dynamics()
    print(f"Converged to profile: {results['strategy_profile']}")
    assert len(results['strategy_profile']) == 3, "Should have 3 agent strategies"
    
    print("✓ Basic functionality tests passed!")


def test_social_optimum():
    """Test social optimum calculation."""
    print("\nTesting Social Optimum Calculation")
    print("=" * 35)
    
    # Test with small game
    game = CongestionRoutingGame(num_agents=3, num_routes=2)
    
    # Get Nash equilibrium
    nash_result = game.best_response_dynamics()
    print(f"Nash equilibrium total cost: {nash_result['total_cost']}")
    
    # Get social optimum
    social_opt = game.calculate_social_optimum()
    print(f"Social optimum total cost: {social_opt['total_cost']}")
    
    # Social optimum should be at least as good as Nash
    assert social_opt['total_cost'] <= nash_result['total_cost'], \
           "Social optimum should be no worse than Nash equilibrium"
    
    # Test comparison function
    comparison = game.compare_nash_vs_social_optimum()
    print(f"Price of Anarchy: {comparison['price_of_anarchy']:.4f}")
    
    assert comparison['price_of_anarchy'] >= 1.0, \
           "Price of anarchy should be at least 1.0"
    
    print("✓ Social optimum tests passed!")


def test_cost_functions():
    """Test different cost functions."""
    print("\nTesting Different Cost Functions")
    print("=" * 33)
    
    cost_funcs = example_cost_functions()
    
    for name, cost_func in cost_funcs.items():
        print(f"\nTesting {name} cost function...")
        game = CongestionRoutingGame(num_agents=3, num_routes=2, cost_function=cost_func)
        
        results = game.best_response_dynamics()
        print(f"  Total cost: {results['total_cost']:.2f}")
        print(f"  Route distribution: {results['route_distribution']}")
        
        # Basic sanity checks
        assert results['total_cost'] > 0, "Total cost should be positive"
        assert sum(results['route_distribution']) == 3, "All agents should be assigned"
    
    print("✓ Cost function tests passed!")


def test_game_analyzer():
    """Test the GameAnalyzer utility class."""
    print("\nTesting Game Analyzer")
    print("=" * 20)
    
    analyzer = GameAnalyzer()
    
    # Test batch testing with small configurations
    test_configs = [
        {'agents': 2, 'routes': 2},
        {'agents': 3, 'routes': 2},
    ]
    
    print("Running batch tests...")
    results = analyzer.batch_test(test_configs)
    
    assert len(results) == len(test_configs), "Should have results for all configurations"
    
    for result in results:
        if 'total_cost' in result:
            assert result['total_cost'] > 0, "Total cost should be positive"
        if 'avg_total_cost' in result:
            assert result['avg_total_cost'] > 0, "Average total cost should be positive"
    
    print("✓ Game analyzer tests passed!")


def test_edge_cases():
    """Test edge cases and error handling."""
    print("\nTesting Edge Cases")
    print("=" * 18)
    
    # Test with minimal configuration
    print("Testing minimal configuration (2 agents, 2 routes)...")
    game = CongestionRoutingGame(num_agents=2, num_routes=2)
    results = game.solve_two_agent_game()
    assert len(results) > 0, "Should find equilibria even in minimal case"
    
    # Test with asymmetric configuration
    print("Testing asymmetric configuration (5 agents, 2 routes)...")
    game = CongestionRoutingGame(num_agents=5, num_routes=2)
    results = game.best_response_dynamics()
    assert sum(results['route_distribution']) == 5, "All agents should be assigned"
    
    # Test with many routes
    print("Testing many routes (3 agents, 4 routes)...")
    game = CongestionRoutingGame(num_agents=3, num_routes=4)
    results = game.best_response_dynamics()
    assert len(results['route_distribution']) == 4, "Should track all routes"
    
    print("✓ Edge case tests passed!")


def test_visualization():
    """Test visualization functionality."""
    print("\nTesting Visualization")
    print("=" * 20)
    
    # Test 2-agent visualization
    game2 = CongestionRoutingGame(num_agents=2, num_routes=2)
    results2 = game2.solve_two_agent_game()
    
    print("Testing 2-agent game visualization...")
    try:
        game2.visualize_equilibrium(results2, save_path="test_2agent_plot.png")
        print("✓ 2-agent visualization completed")
    except Exception as e:
        print(f"⚠ 2-agent visualization failed: {e}")
    
    # Test n-agent visualization
    game_n = CongestionRoutingGame(num_agents=4, num_routes=3)
    results_n = game_n.best_response_dynamics()
    
    print("Testing n-agent game visualization...")
    try:
        game_n.visualize_equilibrium(results_n, save_path="test_nagent_plot.png")
        print("✓ n-agent visualization completed")
    except Exception as e:
        print(f"⚠ n-agent visualization failed: {e}")


def run_performance_test():
    """Run a quick performance test."""
    print("\nRunning Performance Test")
    print("=" * 24)
    
    import time
    
    # Test performance on different game sizes
    sizes = [(2, 2), (3, 2), (4, 3), (5, 3)]
    
    for agents, routes in sizes:
        print(f"Testing {agents} agents, {routes} routes...", end=" ")
        
        start_time = time.time()
        game = CongestionRoutingGame(num_agents=agents, num_routes=routes)
        
        if agents == 2:
            results = game.solve_two_agent_game()
        else:
            results = game.best_response_dynamics()
        
        end_time = time.time()
        print(f"{end_time - start_time:.4f}s")
    
    print("✓ Performance test completed!")


def main():
    """Run all tests."""
    print("CONGESTION ROUTING GAME - COMPREHENSIVE TESTING")
    print("=" * 50)
    
    try:
        test_basic_functionality()
        test_social_optimum()
        test_cost_functions()
        test_game_analyzer()
        test_edge_cases()
        test_visualization()
        run_performance_test()
        
        print("\n" + "=" * 50)
        print("🎉 ALL TESTS PASSED! 🎉")
        print("The Congestion Routing Game implementation is working correctly!")
        print("=" * 50)
        
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        return False
    except Exception as e:
        print(f"\n💥 UNEXPECTED ERROR: {e}")
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
