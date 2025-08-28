"""
2-Iteration Convergence Example
Demonstrates how mixed strategy learning works over 2 iterations
"""

import numpy as np

def demonstrate_2_iteration_convergence():
    print("🔄 2-ITERATION CONVERGENCE EXAMPLE")
    print("=" * 60)
    
    print("📋 GAME SETUP:")
    print("- Agents: 3")
    print("- Routes: 2")
    print("- Cost Function: cost = 1.0 + 4.0 × congestion")
    print("- Learning Rate: 0.5 (higher for more dramatic changes)")
    print()
    
    # Simulation parameters
    num_agents = 3
    num_routes = 2
    learning_rate = 0.5
    base_cost = 1.0
    slope = 4.0
    
    def cost_function(congestion):
        return base_cost + slope * congestion
    
    print("🎯 ITERATION 0 - INITIALIZATION:")
    print("-" * 40)
    
    # Initialize with slightly unbalanced strategies
    strategies = [
        np.array([0.7, 0.3]),  # Agent 1: prefers Route 0
        np.array([0.6, 0.4]),  # Agent 2: prefers Route 0  
        np.array([0.4, 0.6])   # Agent 3: prefers Route 1
    ]
    
    for i, strategy in enumerate(strategies):
        print(f"Agent {i+1}: [{strategy[0]:.3f}, {strategy[1]:.3f}]")
    
    print(f"\nExpected Route Distribution:")
    route_0_expected = sum(s[0] for s in strategies)
    route_1_expected = sum(s[1] for s in strategies)
    print(f"- Route 0: {route_0_expected:.3f} agents")
    print(f"- Route 1: {route_1_expected:.3f} agents")
    
    print(f"\nExpected Costs:")
    cost_0 = cost_function(route_0_expected)
    cost_1 = cost_function(route_1_expected)
    print(f"- Route 0 cost: {base_cost} + {slope} × {route_0_expected:.3f} = {cost_0:.3f}")
    print(f"- Route 1 cost: {base_cost} + {slope} × {route_1_expected:.3f} = {cost_1:.3f}")
    print()
    
    print("🔄 ITERATION 1:")
    print("-" * 40)
    
    old_strategies = [s.copy() for s in strategies]
    
    # Update each agent's strategy
    for agent_idx in range(num_agents):
        print(f"\nAgent {agent_idx + 1} Decision Making:")
        
        # Calculate expected payoffs for each route
        route_payoffs = np.zeros(num_routes)
        
        for route in range(num_routes):
            # Calculate expected congestion if this agent chooses this route
            expected_congestion = 1  # This agent
            
            # Add expected contribution from other agents
            for other_idx in range(num_agents):
                if other_idx != agent_idx:
                    expected_congestion += strategies[other_idx][route]
            
            # Calculate cost (negative because we minimize cost)
            cost = cost_function(expected_congestion)
            route_payoffs[route] = -cost
            
            print(f"  Route {route}: expected congestion = {expected_congestion:.3f}")
            print(f"             expected cost = {cost:.3f}")
            print(f"             payoff = {route_payoffs[route]:.3f}")
        
        # Update strategy using softmax
        exp_payoffs = np.exp(learning_rate * route_payoffs)
        new_strategy = exp_payoffs / np.sum(exp_payoffs)
        
        print(f"  Old strategy: [{old_strategies[agent_idx][0]:.3f}, {old_strategies[agent_idx][1]:.3f}]")
        print(f"  New strategy: [{new_strategy[0]:.3f}, {new_strategy[1]:.3f}]")
        
        # Calculate change
        change = np.linalg.norm(new_strategy - old_strategies[agent_idx])
        print(f"  Strategy change: {change:.6f}")
        
        strategies[agent_idx] = new_strategy
    
    # Check convergence after iteration 1
    max_change = max(np.linalg.norm(strategies[i] - old_strategies[i]) 
                     for i in range(num_agents))
    
    print(f"\n📊 AFTER ITERATION 1:")
    print(f"Maximum strategy change: {max_change:.6f}")
    print(f"Convergence threshold: 0.000001")
    
    if max_change < 1e-6:
        print("✅ CONVERGED after 1 iteration!")
        return
    else:
        print("❌ NOT CONVERGED - Continue to iteration 2")
    
    print(f"\nUpdated Route Distribution:")
    route_0_expected = sum(s[0] for s in strategies)
    route_1_expected = sum(s[1] for s in strategies)
    print(f"- Route 0: {route_0_expected:.3f} agents")
    print(f"- Route 1: {route_1_expected:.3f} agents")
    print()
    
    print("🔄 ITERATION 2:")
    print("-" * 40)
    
    old_strategies_2 = [s.copy() for s in strategies]
    
    # Update each agent's strategy again
    for agent_idx in range(num_agents):
        print(f"\nAgent {agent_idx + 1} Decision Making:")
        
        # Calculate expected payoffs for each route
        route_payoffs = np.zeros(num_routes)
        
        for route in range(num_routes):
            # Calculate expected congestion if this agent chooses this route
            expected_congestion = 1  # This agent
            
            # Add expected contribution from other agents
            for other_idx in range(num_agents):
                if other_idx != agent_idx:
                    expected_congestion += strategies[other_idx][route]
            
            # Calculate cost (negative because we minimize cost)
            cost = cost_function(expected_congestion)
            route_payoffs[route] = -cost
            
            print(f"  Route {route}: expected congestion = {expected_congestion:.3f}")
            print(f"             expected cost = {cost:.3f}")
        
        # Update strategy using softmax
        exp_payoffs = np.exp(learning_rate * route_payoffs)
        new_strategy = exp_payoffs / np.sum(exp_payoffs)
        
        print(f"  Old strategy: [{old_strategies_2[agent_idx][0]:.3f}, {old_strategies_2[agent_idx][1]:.3f}]")
        print(f"  New strategy: [{new_strategy[0]:.3f}, {new_strategy[1]:.3f}]")
        
        # Calculate change
        change = np.linalg.norm(new_strategy - old_strategies_2[agent_idx])
        print(f"  Strategy change: {change:.6f}")
        
        strategies[agent_idx] = new_strategy
    
    # Check convergence after iteration 2
    max_change = max(np.linalg.norm(strategies[i] - old_strategies_2[i]) 
                     for i in range(num_agents))
    
    print(f"\n📊 AFTER ITERATION 2:")
    print(f"Maximum strategy change: {max_change:.6f}")
    print(f"Convergence threshold: 0.000001")
    
    if max_change < 1e-6:
        print("✅ CONVERGED after 2 iterations!")
    else:
        print("❌ NOT CONVERGED - Would continue to iteration 3")
    
    print(f"\n🎯 FINAL EQUILIBRIUM:")
    print("-" * 30)
    for i, strategy in enumerate(strategies):
        print(f"Agent {i+1}: [{strategy[0]:.3f}, {strategy[1]:.3f}]")
    
    # Calculate final expected costs
    final_route_0 = sum(s[0] for s in strategies)
    final_route_1 = sum(s[1] for s in strategies)
    final_cost_0 = cost_function(final_route_0)
    final_cost_1 = cost_function(final_route_1)
    
    print(f"\nFinal Route Distribution:")
    print(f"- Route 0: {final_route_0:.3f} agents, cost = {final_cost_0:.3f}")
    print(f"- Route 1: {final_route_1:.3f} agents, cost = {final_cost_1:.3f}")
    
    # Calculate total expected cost
    total_expected = 0
    for i, strategy in enumerate(strategies):
        agent_expected_cost = strategy[0] * final_cost_0 + strategy[1] * final_cost_1
        total_expected += agent_expected_cost
        print(f"Agent {i+1} expected cost: {agent_expected_cost:.3f}")
    
    print(f"\nTotal Expected Cost: {total_expected:.3f}")
    
    print(f"\n✨ KEY INSIGHTS:")
    print("- Started with unbalanced strategies")
    print("- Agents gradually learned to balance the routes")
    print("- Each iteration brought strategies closer to equilibrium")
    print("- Convergence occurs when strategy changes become negligible")

if __name__ == "__main__":
    demonstrate_2_iteration_convergence()
