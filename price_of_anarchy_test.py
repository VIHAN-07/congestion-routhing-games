from congestion_routing_game import CongestionRoutingGame

def high_congestion_cost(congestion):
    return congestion ** 3

print('Testing with cubic cost function...')
game = CongestionRoutingGame(6, 3, high_congestion_cost)
comparison = game.compare_nash_vs_social_optimum()
print()
print('Results summary:')
print(f'Price of Anarchy: {comparison["price_of_anarchy"]:.2f}')
print(f'Efficiency Loss: {comparison["efficiency_loss"]:.2f}')
