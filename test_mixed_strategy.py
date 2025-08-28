from congestion_routing_game import CongestionRoutingGame

# Test mixed strategy solver
game = CongestionRoutingGame(3, 2)
mixed_result = game.mixed_strategy_solver(max_iterations=100)

print('Mixed strategy equilibrium:')
for i, strategy in enumerate(mixed_result['mixed_strategies']):
    print(f'Agent {i+1}: {strategy}')
print(f'Converged: {mixed_result["converged"]}')
print(f'Total expected cost: {mixed_result["total_expected_cost"]:.3f}')
