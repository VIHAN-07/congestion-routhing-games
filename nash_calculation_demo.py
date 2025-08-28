"""
Nash Equilibrium Calculation Example
Detailed step-by-step explanation of how Nash value is calculated
"""

def demonstrate_nash_calculation():
    print("🎯 NASH EQUILIBRIUM CALCULATION WALKTHROUGH")
    print("=" * 60)
    
    # Example configuration
    print("📋 GAME CONFIGURATION:")
    print("- Agents: 3")
    print("- Routes: 2") 
    print("- Cost Function: 2.0 + 3.0 × congestion")
    print()
    
    print("🔄 BEST RESPONSE DYNAMICS ALGORITHM:")
    print("-" * 40)
    
    # Step 1: Initial random strategy
    print("STEP 1: Initialize with random strategy")
    initial_strategy = [0, 1, 0]  # Example: Agent 0→Route 0, Agent 1→Route 1, Agent 2→Route 0
    print(f"Initial strategy: {initial_strategy}")
    print("Route distribution: [2, 1] (2 on Route 0, 1 on Route 1)")
    print()
    
    # Calculate initial costs
    def cost_function(congestion):
        return 2.0 + 3.0 * congestion
    
    print("Initial costs:")
    print(f"- Route 0 congestion: 2 → cost per agent: {cost_function(2)}")
    print(f"- Route 1 congestion: 1 → cost per agent: {cost_function(1)}")
    print()
    
    # Step 2: Best response for each agent
    print("STEP 2: Each agent finds best response")
    print("-" * 40)
    
    current_strategy = initial_strategy.copy()
    
    for agent in range(3):
        print(f"\n🤖 Agent {agent} deciding:")
        print(f"Current strategy: {current_strategy}")
        
        best_route = None
        best_cost = float('inf')
        
        for route in range(2):
            # Test this route choice
            test_strategy = current_strategy.copy()
            test_strategy[agent] = route
            
            # Count congestion
            route_counts = [0, 0]
            for choice in test_strategy:
                route_counts[choice] += 1
            
            # Calculate agent's cost
            agent_cost = cost_function(route_counts[route])
            
            print(f"  If chooses Route {route}:")
            print(f"    Route distribution: {route_counts}")
            print(f"    Agent's cost: {agent_cost}")
            
            if agent_cost < best_cost:
                best_cost = agent_cost
                best_route = route
        
        print(f"  ✅ Best choice: Route {best_route} (cost: {best_cost})")
        current_strategy[agent] = best_route
        print(f"  Updated strategy: {current_strategy}")
    
    print("\n" + "=" * 60)
    print("🎯 FINAL NASH EQUILIBRIUM:")
    print(f"Strategy: {current_strategy}")
    
    # Calculate final route distribution and costs
    final_counts = [0, 0]
    for choice in current_strategy:
        final_counts[choice] += 1
    
    print(f"Route distribution: {final_counts}")
    print()
    
    print("💰 NASH EQUILIBRIUM COST CALCULATION:")
    print("-" * 40)
    
    total_cost = 0
    for agent in range(3):
        agent_route = current_strategy[agent]
        agent_cost = cost_function(final_counts[agent_route])
        total_cost += agent_cost
        print(f"Agent {agent} on Route {agent_route}: cost = {agent_cost}")
    
    print(f"\n🔢 TOTAL NASH COST: {total_cost}")
    print(f"📊 AVERAGE COST: {total_cost/3:.2f}")
    
    print("\n" + "=" * 60)
    print("🔍 KEY POINTS:")
    print("1. Nash equilibrium = No agent wants to unilaterally change")
    print("2. Each agent selfishly minimizes their own cost")
    print("3. Total cost = Sum of all individual agent costs")
    print("4. Algorithm converges when no agent changes strategy")
    print("\n💡 Your actual Nash cost (10.29) comes from this process!")
    print("   The exact value depends on:")
    print("   - Number of agents and routes") 
    print("   - Specific cost function parameters")
    print("   - Initial random starting point")
    print("   - Convergence to equilibrium")

if __name__ == "__main__":
    demonstrate_nash_calculation()
