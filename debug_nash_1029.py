"""
Debug script to find the exact parameters that produce Nash cost 10.29
"""
from congestion_routing_game import CongestionRoutingGame

def find_nash_1029():
    print("🔍 SEARCHING FOR PARAMETERS THAT GIVE NASH COST 10.29")
    print("=" * 55)
    
    # Test different cost function parameters
    cost_params = [
        (1, 4),   # a=1, b=4
        (2, 3),   # a=2, b=3  
        (0, 5),   # a=0, b=5
        (1, 3),   # a=1, b=3
        (3, 2),   # a=3, b=2
    ]
    
    for a, b in cost_params:
        print(f"\n🧪 TESTING: Cost = {a} + {b} × congestion")
        print("-" * 40)
        
        try:
            # Create game with custom cost function
            game = CongestionRoutingGame(num_agents=2, num_routes=3, cost_params=(a, b))
            result = game.best_response_dynamics()
            
            if result['converged']:
                total_cost = result['total_cost']
                print(f"✅ Converged! Total cost: {total_cost:.4f}")
                
                if abs(total_cost - 10.29) < 0.01:
                    print("🎯 FOUND IT! This matches 10.29!")
                    
                    strategy = result['strategy_profile']
                    route_counts = result['route_distribution']
                    
                    print(f"Strategy: {strategy}")
                    print(f"Route distribution: {route_counts}")
                    
                    for i, agent_route in enumerate(strategy):
                        agent_cost = a + b * route_counts[agent_route]
                        print(f"Agent {i}: Route {agent_route}, cost = {a} + {b}×{route_counts[agent_route]} = {agent_cost}")
                    
                    return
                    
            else:
                print("❌ Did not converge")
                
        except Exception as e:
            print(f"❌ Error: {e}")
    
    # Try with more agents
    print(f"\n🧪 TESTING WITH MORE AGENTS...")
    print("-" * 40)
    
    for num_agents in [3, 4, 5]:
        for a, b in [(1, 3), (2, 3)]:
            try:
                game = CongestionRoutingGame(num_agents=num_agents, num_routes=3, cost_params=(a, b))
                result = game.best_response_dynamics()
                
                if result['converged']:
                    total_cost = result['total_cost']
                    avg_cost = total_cost / num_agents
                    
                    print(f"Agents: {num_agents}, Cost: {a}+{b}x → Total: {total_cost:.4f}, Avg: {avg_cost:.4f}")
                    
                    if abs(avg_cost - 10.29/2) < 0.01 or abs(total_cost - 10.29) < 0.01:
                        print("🎯 POSSIBLE MATCH!")
                        strategy = result['strategy_profile']
                        print(f"Strategy: {strategy}")
                        
            except Exception as e:
                continue

if __name__ == "__main__":
    find_nash_1029()
