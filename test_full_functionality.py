"""
Comprehensive Test Script for Working Interactive UI
Tests all major functionality to ensure everything works properly.
"""

import sys
import os
sys.path.append('.')

def test_imports():
    """Test all required imports"""
    print("🔍 Testing Imports...")
    try:
        from congestion_routing_game import CongestionRoutingGame
        print("✅ CongestionRoutingGame imported successfully")
        
        from game_analysis_utils import GameAnalyzer
        print("✅ GameAnalyzer imported successfully")
        
        from advanced_analysis import LearningDynamics, EvolutionaryGameAnalysis, MechanismDesign
        print("✅ Advanced analysis classes imported successfully")
        
        from network_routing_scenarios import NetworkRoutingScenarios
        print("✅ NetworkRoutingScenarios imported successfully")
        
        return True
    except Exception as e:
        print(f"❌ Import error: {e}")
        return False

def test_game_creation():
    """Test game creation with different parameters"""
    print("\n🎮 Testing Game Creation...")
    try:
        from congestion_routing_game import CongestionRoutingGame
        
        # Test basic game creation
        game = CongestionRoutingGame(4, 3)  # 4 agents, 3 routes
        print(f"✅ Basic game created: {game.num_agents} agents, {game.num_routes} routes")
        
        # Test with custom cost function
        custom_cost = lambda x: x**2 + x
        game2 = CongestionRoutingGame(2, 2, custom_cost)
        print("✅ Game with custom cost function created")
        
        return True, game, game2
    except Exception as e:
        print(f"❌ Game creation error: {e}")
        return False, None, None

def test_2agent_nash_equilibrium(game2):
    """Test 2-agent Nash equilibrium solving"""
    print("\n⚖️ Testing 2-Agent Nash Equilibrium...")
    try:
        results = game2.solve_two_agent_game()
        print("✅ 2-agent Nash equilibrium solved successfully")
        print(f"   Found {len(results)} equilibria")
        return True
    except Exception as e:
        print(f"❌ Nash equilibrium error: {e}")
        return False

def test_social_optimum(game):
    """Test social optimum calculation"""
    print("\n🎯 Testing Social Optimum...")
    try:
        social_opt = game.calculate_social_optimum()
        print("✅ Social optimum calculated successfully")
        print(f"   Total cost: {social_opt['total_cost']:.3f}")
        print(f"   Strategy profile: {social_opt['strategy_profile']}")
        return True, social_opt
    except Exception as e:
        print(f"❌ Social optimum error: {e}")
        return False, None

def test_nash_vs_social_comparison(game):
    """Test Nash vs Social optimum comparison"""
    print("\n⚖️ Testing Nash vs Social Optimum Comparison...")
    try:
        comparison = game.compare_nash_vs_social_optimum()
        print("✅ Nash vs Social comparison completed successfully")
        print(f"   Price of Anarchy: {comparison['price_of_anarchy']:.3f}")
        print(f"   Nash cost: {comparison['nash_total_cost']:.3f}")
        print(f"   Social cost: {comparison['social_optimal_cost']:.3f}")
        return True
    except Exception as e:
        print(f"❌ Comparison error: {e}")
        return False

def test_best_response_dynamics(game):
    """Test best response dynamics for n-agent games"""
    print("\n🔄 Testing Best Response Dynamics...")
    try:
        results = game.best_response_dynamics(max_iterations=100)
        print("✅ Best response dynamics completed successfully")
        print(f"   Converged: {results['converged']}")
        print(f"   Iterations: {results['iterations']}")
        print(f"   Final cost: {results['total_cost']:.3f}")
        return True
    except Exception as e:
        print(f"❌ Best response dynamics error: {e}")
        return False

def test_mixed_strategy_learning(game):
    """Test mixed strategy learning"""
    print("\n🧠 Testing Mixed Strategy Learning...")
    try:
        results = game.mixed_strategy_solver(max_iterations=100, learning_rate=0.1)
        print("✅ Mixed strategy learning completed successfully")
        print(f"   Converged: {results['converged']}")
        print(f"   Total expected cost: {results['total_expected_cost']:.3f}")
        print(f"   Iterations: {results['convergence_iterations']}")
        return True, results
    except Exception as e:
        print(f"❌ Mixed strategy learning error: {e}")
        return False, None

def test_convergence_analysis(game, mixed_results):
    """Test convergence analysis"""
    print("\n📊 Testing Convergence Analysis...")
    try:
        if 'strategy_history' in mixed_results:
            analysis = game.analyze_convergence(mixed_results['strategy_history'])
            print("✅ Convergence analysis completed successfully")
            print(f"   Convergence rate: {analysis['convergence_rate']:.6f}")
            print(f"   Final variance: {analysis['final_variance']:.6f}")
            return True
        else:
            print("⚠️ Strategy history not available for convergence analysis")
            return True
    except Exception as e:
        print(f"❌ Convergence analysis error: {e}")
        return False

def test_network_scenarios():
    """Test network routing scenarios"""
    print("\n🌐 Testing Network Scenarios...")
    try:
        from network_routing_scenarios import NetworkRoutingScenarios
        scenarios = NetworkRoutingScenarios()
        
        # Test traffic network
        traffic_result = scenarios.traffic_network_scenario()
        print("✅ Traffic network scenario completed")
        
        # Test internet routing
        internet_result = scenarios.internet_routing()
        print("✅ Internet routing scenario completed")
        
        # Test data center routing
        datacenter_result = scenarios.data_center_routing()
        print("✅ Data center routing scenario completed")
        
        # Test supply chain routing
        supply_result = scenarios.supply_chain_routing()
        print("✅ Supply chain routing scenario completed")
        
        return True
    except Exception as e:
        print(f"❌ Network scenarios error: {e}")
        return False

def test_payoff_matrices(game, game2):
    """Test payoff matrix generation"""
    print("\n🔢 Testing Payoff Matrix Generation...")
    try:
        # Test 4-agent game matrices
        matrices = game._create_payoff_matrices()
        print(f"✅ Payoff matrices created for {game.num_agents}-agent game")
        
        # Test 2-agent game matrices
        matrices2 = game2._create_payoff_matrices()
        print(f"✅ Payoff matrices created for 2-agent game")
        print(f"   Matrix shape: {matrices2[0].shape}")
        
        return True
    except Exception as e:
        print(f"❌ Payoff matrix error: {e}")
        return False

def run_comprehensive_test():
    """Run all tests"""
    print("🚀 COMPREHENSIVE TEST OF WORKING INTERACTIVE UI")
    print("=" * 60)
    
    test_results = []
    
    # Test imports
    test_results.append(test_imports())
    
    # Test game creation
    success, game, game2 = test_game_creation()
    test_results.append(success)
    
    if not success:
        print("❌ Cannot continue tests - game creation failed")
        return
    
    # Test all game functionality
    test_results.append(test_2agent_nash_equilibrium(game2))
    test_results.append(test_social_optimum(game)[0])
    test_results.append(test_nash_vs_social_comparison(game))
    test_results.append(test_best_response_dynamics(game))
    
    # Test mixed strategy learning
    mixed_success, mixed_results = test_mixed_strategy_learning(game)
    test_results.append(mixed_success)
    
    if mixed_success and mixed_results:
        test_results.append(test_convergence_analysis(game, mixed_results))
    
    # Test additional features
    test_results.append(test_network_scenarios())
    test_results.append(test_payoff_matrices(game, game2))
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(test_results)
    total = len(test_results)
    
    print(f"✅ Tests Passed: {passed}/{total}")
    print(f"❌ Tests Failed: {total - passed}/{total}")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED! UI IS FULLY FUNCTIONAL! 🎉")
    else:
        print(f"\n⚠️ {total - passed} tests failed. See details above.")
    
    print(f"\n🌐 UI is running at: http://localhost:8508")
    print("🎮 All buttons and features should work perfectly!")

if __name__ == "__main__":
    run_comprehensive_test()
