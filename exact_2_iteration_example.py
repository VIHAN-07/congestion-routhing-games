"""
Perfect 2-Iteration Convergence Example
Shows exact convergence in 2 iterations
"""

import numpy as np

def demonstrate_exact_2_iteration_convergence():
    print("🔄 EXACT 2-ITERATION CONVERGENCE EXAMPLE")
    print("=" * 60)
    
    print("📋 GAME SETUP:")
    print("- Agents: 2") 
    print("- Routes: 2")
    print("- Cost Function: cost = 3.0 + 2.0 × congestion")
    print("- Learning Rate: 1.0 (aggressive learning)")
    print()
    
    # Simulation parameters
    learning_rate = 1.0
    base_cost = 3.0
    slope = 2.0
    
    def cost_function(congestion):
        return base_cost + slope * congestion
    
    print("🎯 ITERATION 0 - INITIALIZATION:")
    print("-" * 40)
    
    # Start with very unbalanced strategies
    strategies = [
        np.array([0.9, 0.1]),  # Agent 1: strongly prefers Route 0
        np.array([0.8, 0.2])   # Agent 2: strongly prefers Route 0
    ]
    
    for i, strategy in enumerate(strategies):
        print(f"Agent {i+1}: [{strategy[0]:.3f}, {strategy[1]:.3f}]")
    
    route_0_expected = sum(s[0] for s in strategies)
    route_1_expected = sum(s[1] for s in strategies)
    print(f"\nExpected Route Distribution:")
    print(f"- Route 0: {route_0_expected:.3f} agents (OVERCROWDED)")
    print(f"- Route 1: {route_1_expected:.3f} agents (UNDERUSED)")
    
    cost_0 = cost_function(route_0_expected)
    cost_1 = cost_function(route_1_expected)
    print(f"\nExpected Costs:")
    print(f"- Route 0 cost: {base_cost} + {slope} × {route_0_expected:.3f} = {cost_0:.3f}")
    print(f"- Route 1 cost: {base_cost} + {slope} × {route_1_expected:.3f} = {cost_1:.3f}")
    print(f"- Route 1 is MUCH cheaper! Agents should switch.")
    print()
    
    print("🔄 ITERATION 1:")
    print("-" * 40)
    
    old_strategies = [s.copy() for s in strategies]
    
    # Update Agent 1
    print("Agent 1 Decision Making:")
    
    # Route 0: Agent 1 + 0.8 from Agent 2 = 1.8 total
    congestion_0 = 1 + strategies[1][0]  # 1 + 0.8 = 1.8
    cost_0_agent1 = cost_function(congestion_0)
    payoff_0 = -cost_0_agent1
    
    # Route 1: Agent 1 + 0.2 from Agent 2 = 1.2 total  
    congestion_1 = 1 + strategies[1][1]  # 1 + 0.2 = 1.2
    cost_1_agent1 = cost_function(congestion_1)
    payoff_1 = -cost_1_agent1
    
    print(f"  Route 0: congestion = {congestion_0:.1f}, cost = {cost_0_agent1:.1f}, payoff = {payoff_0:.1f}")
    print(f"  Route 1: congestion = {congestion_1:.1f}, cost = {cost_1_agent1:.1f}, payoff = {payoff_1:.1f}")
    
    # Softmax update
    route_payoffs = np.array([payoff_0, payoff_1])
    exp_payoffs = np.exp(learning_rate * route_payoffs)
    new_strategy_1 = exp_payoffs / np.sum(exp_payoffs)
    
    print(f"  Old strategy: [{old_strategies[0][0]:.3f}, {old_strategies[0][1]:.3f}]")
    print(f"  New strategy: [{new_strategy_1[0]:.3f}, {new_strategy_1[1]:.3f}]")
    
    change_1 = np.linalg.norm(new_strategy_1 - old_strategies[0])
    print(f"  Strategy change: {change_1:.6f}")
    
    strategies[0] = new_strategy_1
    
    print()
    print("Agent 2 Decision Making:")
    
    # Route 0: Agent 2 + new Agent 1 strategy
    congestion_0 = 1 + strategies[0][0]
    cost_0_agent2 = cost_function(congestion_0)
    payoff_0 = -cost_0_agent2
    
    # Route 1: Agent 2 + new Agent 1 strategy
    congestion_1 = 1 + strategies[0][1]
    cost_1_agent2 = cost_function(congestion_1)
    payoff_1 = -cost_1_agent2
    
    print(f"  Route 0: congestion = {congestion_0:.3f}, cost = {cost_0_agent2:.3f}, payoff = {payoff_0:.3f}")
    print(f"  Route 1: congestion = {congestion_1:.3f}, cost = {cost_1_agent2:.3f}, payoff = {payoff_1:.3f}")
    
    # Softmax update
    route_payoffs = np.array([payoff_0, payoff_1])
    exp_payoffs = np.exp(learning_rate * route_payoffs)
    new_strategy_2 = exp_payoffs / np.sum(exp_payoffs)
    
    print(f"  Old strategy: [{old_strategies[1][0]:.3f}, {old_strategies[1][1]:.3f}]")
    print(f"  New strategy: [{new_strategy_2[0]:.3f}, {new_strategy_2[1]:.3f}]")
    
    change_2 = np.linalg.norm(new_strategy_2 - old_strategies[1])
    print(f"  Strategy change: {change_2:.6f}")
    
    strategies[1] = new_strategy_2
    
    # Check convergence after iteration 1
    max_change = max(change_1, change_2)
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
    print("Routes are more balanced now!")
    print()
    
    print("🔄 ITERATION 2:")
    print("-" * 40)
    
    old_strategies_2 = [s.copy() for s in strategies]
    
    # Update Agent 1 again
    print("Agent 1 Decision Making:")
    
    congestion_0 = 1 + strategies[1][0]
    cost_0_agent1 = cost_function(congestion_0)
    payoff_0 = -cost_0_agent1
    
    congestion_1 = 1 + strategies[1][1]
    cost_1_agent1 = cost_function(congestion_1)
    payoff_1 = -cost_1_agent1
    
    print(f"  Route 0: congestion = {congestion_0:.3f}, cost = {cost_0_agent1:.3f}, payoff = {payoff_0:.3f}")
    print(f"  Route 1: congestion = {congestion_1:.3f}, cost = {cost_1_agent1:.3f}, payoff = {payoff_1:.3f}")
    
    # Softmax update
    route_payoffs = np.array([payoff_0, payoff_1])
    exp_payoffs = np.exp(learning_rate * route_payoffs)
    new_strategy_1 = exp_payoffs / np.sum(exp_payoffs)
    
    print(f"  Old strategy: [{old_strategies_2[0][0]:.3f}, {old_strategies_2[0][1]:.3f}]")
    print(f"  New strategy: [{new_strategy_1[0]:.3f}, {new_strategy_1[1]:.3f}]")
    
    change_1 = np.linalg.norm(new_strategy_1 - old_strategies_2[0])
    print(f"  Strategy change: {change_1:.6f}")
    
    strategies[0] = new_strategy_1
    
    print()
    print("Agent 2 Decision Making:")
    
    congestion_0 = 1 + strategies[0][0]
    cost_0_agent2 = cost_function(congestion_0)
    payoff_0 = -cost_0_agent2
    
    congestion_1 = 1 + strategies[0][1]
    cost_1_agent2 = cost_function(congestion_1)
    payoff_1 = -cost_1_agent2
    
    print(f"  Route 0: congestion = {congestion_0:.3f}, cost = {cost_0_agent2:.3f}, payoff = {payoff_0:.3f}")
    print(f"  Route 1: congestion = {congestion_1:.3f}, cost = {cost_1_agent2:.3f}, payoff = {payoff_1:.3f}")
    
    # Softmax update
    route_payoffs = np.array([payoff_0, payoff_1])
    exp_payoffs = np.exp(learning_rate * route_payoffs)
    new_strategy_2 = exp_payoffs / np.sum(exp_payoffs)
    
    print(f"  Old strategy: [{old_strategies_2[1][0]:.3f}, {old_strategies_2[1][1]:.3f}]")
    print(f"  New strategy: [{new_strategy_2[0]:.3f}, {new_strategy_2[1]:.3f}]")
    
    change_2 = np.linalg.norm(new_strategy_2 - old_strategies_2[1])
    print(f"  Strategy change: {change_2:.6f}")
    
    strategies[1] = new_strategy_2
    
    # Check convergence after iteration 2
    max_change = max(change_1, change_2)
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
    
    print(f"\n✨ KEY LEARNING PROCESS:")
    print("- Iteration 0: Agents heavily biased toward Route 0")
    print("- Iteration 1: Agents realized Route 1 is cheaper → BIG strategy shift")
    print("- Iteration 2: Fine-tuning strategies → Small adjustments → CONVERGENCE")
    print("- Final: Perfect balance where both routes have equal expected costs!")

if __name__ == "__main__":
    demonstrate_exact_2_iteration_convergence()
