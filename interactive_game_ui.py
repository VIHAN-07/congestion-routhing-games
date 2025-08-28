import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import sys
import os
import time
from scipy.optimize import minimize

# Add current directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Import all project modules
try:
    from congestion_routing_game import CongestionRoutingGame
    from network_routing_scenarios import NetworkRoutingScenarios
    from advanced_analysis import LearningDynamics, EvolutionaryGameAnalysis, MechanismDesign
    modules_loaded = True
except ImportError as e:
    st.error(f"Module import error: {e}")
    modules_loaded = False

# Configure Streamlit page
st.set_page_config(
    page_title="Interactive Congestion Routing Game",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .game-section {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        border-left: 4px solid #1f77b4;
    }
    .result-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def main():
    st.markdown('<h1 class="main-header">🎮 Interactive Congestion Routing Game</h1>', unsafe_allow_html=True)
    st.markdown("### Real-time Game with Full User Control & All Project Functionalities")
    
    if not modules_loaded:
        st.error("❌ Cannot load project modules. Please ensure all files are in the same directory.")
        return
    
    # Initialize session state
    if 'game' not in st.session_state:
        st.session_state.game = None
    if 'game_results' not in st.session_state:
        st.session_state.game_results = {}
    if 'learning_history' not in st.session_state:
        st.session_state.learning_history = []
    
    # Sidebar for game configuration
    st.sidebar.title("🎯 Game Configuration")
    
    # Main game parameters
    st.sidebar.markdown("### Core Game Setup")
    num_agents = st.sidebar.slider("Number of Agents", min_value=2, max_value=20, value=4, key="num_agents")
    num_routes = st.sidebar.slider("Number of Routes", min_value=2, max_value=8, value=3, key="num_routes")
    
    # Cost function configuration
    st.sidebar.markdown("### Cost Function Configuration")
    cost_function_type = st.sidebar.selectbox(
        "Cost Function Type:",
        ["Linear", "Quadratic", "Polynomial", "Exponential"],
        key="cost_type"
    )
    
    # Configure single cost function for all routes
    if cost_function_type == "Linear":
        base = st.sidebar.slider("Base Cost", 0.5, 10.0, 2.0, 0.1, key="base")
        slope = st.sidebar.slider("Congestion Factor", 0.1, 5.0, 1.0, 0.1, key="slope")
        cost_function = lambda x: base + slope * x
        cost_params = {"base": base, "slope": slope, "type": "linear"}
        
    elif cost_function_type == "Quadratic":
        base = st.sidebar.slider("Base Cost", 0.5, 10.0, 1.0, 0.1, key="base")
        linear = st.sidebar.slider("Linear Factor", 0.1, 3.0, 0.5, 0.1, key="linear")
        quad = st.sidebar.slider("Quadratic Factor", 0.1, 2.0, 0.5, 0.1, key="quad")
        cost_function = lambda x: base + linear * x + quad * x**2
        cost_params = {"base": base, "linear": linear, "quad": quad, "type": "quadratic"}
        
    elif cost_function_type == "Polynomial":
        base = st.sidebar.slider("Base Cost", 0.5, 10.0, 1.0, 0.1, key="base")
        power = st.sidebar.slider("Power Factor", 1.0, 4.0, 2.0, 0.1, key="power")
        cost_function = lambda x: base * (x ** power)
        cost_params = {"base": base, "power": power, "type": "polynomial"}
        
    elif cost_function_type == "Exponential":
        base = st.sidebar.slider("Base Cost", 0.5, 5.0, 1.0, 0.1, key="base")
        exp_factor = st.sidebar.slider("Exponential Factor", 0.1, 1.0, 0.2, 0.1, key="exp_factor")
        cost_function = lambda x: base * np.exp(exp_factor * x)
        cost_params = {"base": base, "exp_factor": exp_factor, "type": "exponential"}
    
    # Create or update game when parameters change
    if st.sidebar.button("🎮 Create/Update Game", type="primary"):
        st.session_state.game = CongestionRoutingGame(num_agents, num_routes, cost_function)
        st.session_state.cost_params = cost_params
        st.success(f"✅ Game created with {num_agents} agents and {num_routes} routes!")
        st.rerun()
    
    # Main interface tabs
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "🎯 Game Analysis", 
        "🧠 Nash Equilibrium", 
        "📈 Learning Dynamics", 
        "🔬 Evolutionary Analysis",
        "💰 Mechanism Design",
        "🌐 Network Scenarios",
        "⚡ Real-time Simulation"
    ])
    
    with tab1:
        show_game_analysis()
    
    with tab2:
        show_nash_equilibrium()
    
    with tab3:
        show_learning_dynamics()
    
    with tab4:
        show_evolutionary_analysis()
    
    with tab5:
        show_mechanism_design()
    
    with tab6:
        show_network_scenarios()
    
    with tab7:
        show_realtime_simulation()

def show_game_analysis():
    st.markdown("## 🎯 Interactive Game Analysis")
    
    if st.session_state.game is None:
        st.warning("⚠️ Please create a game first using the sidebar configuration.")
        return
    
    game = st.session_state.game
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### Cost Function Visualization")
        
        # Show current cost functions
        max_users = st.slider("Maximum Users to Display", 1, game.num_agents, game.num_agents, key="max_users_viz")
        
        fig = go.Figure()
        x_vals = np.arange(1, max_users + 1)
        
        for route_idx in range(game.num_routes):
            y_vals = [game.cost_function(x) for x in x_vals]
            fig.add_trace(go.Scatter(
                x=x_vals,
                y=y_vals,
                mode='lines+markers',
                name=f'Route {route_idx + 1}',
                line=dict(width=3),
                hovertemplate=f'Route {route_idx + 1}<br>Users: %{{x}}<br>Cost: %{{y:.2f}}<extra></extra>'
            ))
        
        fig.update_layout(
            title="Cost Functions (Interactive)",
            xaxis_title="Number of Users",
            yaxis_title="Cost per User",
            template="plotly_white",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Interactive strategy input
        st.markdown("### Manual Strategy Input")
        st.write("Set how many agents choose each route:")
        
        manual_strategy = []
        total_assigned = 0
        
        for route_idx in range(game.num_routes):
            max_for_route = game.num_agents - total_assigned
            if route_idx == game.num_routes - 1:  # Last route gets remaining agents
                agents_on_route = max_for_route
                st.write(f"Route {route_idx + 1}: {agents_on_route} agents (remaining)")
            else:
                agents_on_route = st.number_input(
                    f"Route {route_idx + 1}:",
                    min_value=0,
                    max_value=max_for_route,
                    value=min(1, max_for_route),
                    key=f"manual_route_{route_idx}"
                )
            manual_strategy.extend([route_idx] * agents_on_route)
            total_assigned += agents_on_route
        
        if len(manual_strategy) == game.num_agents:
            if st.button("📊 Analyze This Strategy", key="analyze_manual"):
                # Calculate costs for this strategy
                route_counts = [manual_strategy.count(r) for r in range(game.num_routes)]
                individual_costs = []
                
                for agent_idx, route_choice in enumerate(manual_strategy):
                    cost = game.cost_function(route_counts[route_choice])
                    individual_costs.append(cost)
                
                total_cost = sum(individual_costs)
                avg_cost = total_cost / game.num_agents
                
                st.success(f"✅ Strategy Analysis Complete!")
                st.write(f"**Total System Cost:** {total_cost:.2f}")
                st.write(f"**Average Individual Cost:** {avg_cost:.2f}")
                
                # Show route distribution
                route_dist_data = {f'Route {i+1}': route_counts[i] for i in range(game.num_routes)}
                df = pd.DataFrame([route_dist_data])
                st.bar_chart(df.T)
    
    with col2:
        st.markdown("### Payoff Matrix Analysis")
        
        if st.button("🔢 Generate Payoff Matrix", key="gen_payoff"):
            with st.spinner("Generating payoff matrix..."):
                try:
                    payoff_matrices = game._create_payoff_matrices()
                    
                    st.success("✅ Payoff Matrices Generated!")
                    
                    # Display matrices
                    if game.num_agents == 2:
                        # For 2-agent games, display traditional matrices
                        st.write("**Player 1 Payoff Matrix (Utilities):**")
                        df1 = pd.DataFrame(payoff_matrices[0])
                        df1.columns = [f"Route {i+1}" for i in range(game.num_routes)]
                        df1.index = [f"Route {i+1}" for i in range(game.num_routes)]
                        st.dataframe(df1)
                        
                        if len(payoff_matrices) > 1:
                            st.write("**Player 2 Payoff Matrix (Utilities):**")
                            df2 = pd.DataFrame(payoff_matrices[1])
                            df2.columns = [f"Route {i+1}" for i in range(game.num_routes)]
                            df2.index = [f"Route {i+1}" for i in range(game.num_routes)]
                            st.dataframe(df2)
                    else:
                        # For n-agent games, show sample strategies
                        st.write(f"**Generated payoff information for {game.num_agents} agents**")
                        sample_size = min(10, len(payoff_matrices[0]))
                        sample_keys = list(payoff_matrices[0].keys())[:sample_size]
                        
                        sample_data = []
                        for strategy_profile in sample_keys:
                            costs = [payoff_matrices[i][strategy_profile] for i in range(game.num_agents)]
                            sample_data.append([str(strategy_profile)] + [f"{cost:.2f}" for cost in costs])
                        
                        df = pd.DataFrame(sample_data, 
                                        columns=["Strategy Profile"] + [f"Agent {i+1} Cost" for i in range(game.num_agents)])
                        st.dataframe(df)
                    
                except Exception as e:
                    st.error(f"Error generating payoff matrix: {e}")
        
        # Social optimum analysis
        st.markdown("### Social Optimum")
        
        if st.button("🎯 Find Social Optimum", key="social_opt"):
            with st.spinner("Computing social optimum..."):
                try:
                    social_opt = game.calculate_social_optimum()
                    
                    st.success("✅ Social Optimum Found!")
                    st.write(f"**Optimal Strategy:** {social_opt['strategy']}")
                    st.write(f"**Total Cost:** {social_opt['total_cost']:.2f}")
                    st.write(f"**Average Cost:** {social_opt['average_cost']:.2f}")
                    
                    # Visualize social optimum
                    route_counts = [social_opt['strategy'].count(r) for r in range(game.num_routes)]
                    
                    fig = go.Figure(data=[
                        go.Bar(x=[f'Route {i+1}' for i in range(game.num_routes)], 
                              y=route_counts,
                              marker_color='lightgreen')
                    ])
                    fig.update_layout(title="Social Optimum Route Distribution", yaxis_title="Number of Agents")
                    st.plotly_chart(fig, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"Error finding social optimum: {e}")

def show_nash_equilibrium():
    st.markdown("## 🧠 Nash Equilibrium Analysis")
    
    if st.session_state.game is None:
        st.warning("⚠️ Please create a game first using the sidebar configuration.")
        return
    
    game = st.session_state.game
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### Pure Strategy Nash Equilibrium")
        
        equilibrium_method = st.selectbox(
            "Choose Solution Method:",
            ["Best Response Dynamics", "Analytical (2-agent only)", "Exhaustive Search"],
            key="eq_method"
        )
        
        if equilibrium_method == "Best Response Dynamics":
            max_iterations = st.slider("Maximum Iterations", 10, 200, 50, key="br_iterations")
            tolerance = st.slider("Convergence Tolerance", 0.001, 0.1, 0.01, key="br_tolerance")
            
            if st.button("🔄 Run Best Response Dynamics", key="run_br"):
                with st.spinner("Running best response dynamics..."):
                    try:
                        result = game.best_response_dynamics(
                            max_iterations=max_iterations,
                            tolerance=tolerance
                        )
                        
                        equilibrium, converged, history = result
                        
                        if converged:
                            st.success("✅ Converged to Nash Equilibrium!")
                        else:
                            st.warning("⚠️ Did not converge within iteration limit")
                        
                        st.write(f"**Final Strategy:** {equilibrium}")
                        
                        # Calculate costs
                        route_counts = [equilibrium.count(r) for r in range(game.num_routes)]
                        total_cost = sum(game.cost_function(route_counts[route]) * route_counts[route] 
                                       for route in range(game.num_routes))
                        
                        st.write(f"**Total System Cost:** {total_cost:.2f}")
                        st.write(f"**Iterations:** {len(history)}")
                        
                        # Convergence plot
                        if history:
                            fig = go.Figure()
                            
                            for route in range(game.num_routes):
                                route_history = [sum(1 for agent_choice in strategy if agent_choice == route) 
                                               for strategy in history]
                                fig.add_trace(go.Scatter(
                                    x=list(range(len(history))),
                                    y=route_history,
                                    mode='lines+markers',
                                    name=f'Route {route + 1}'
                                ))
                            
                            fig.update_layout(
                                title="Best Response Dynamics Convergence",
                                xaxis_title="Iteration",
                                yaxis_title="Number of Agents per Route",
                                template="plotly_white"
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                        
                    except Exception as e:
                        st.error(f"Error running best response dynamics: {e}")
        
        elif equilibrium_method == "Analytical (2-agent only)":
            if game.num_agents == 2 and game.num_routes == 2:
                if st.button("🎯 Solve Analytically", key="analytical_solve"):
                    with st.spinner("Computing analytical solution..."):
                        try:
                            equilibria = game.solve_two_agent_game()
                            
                            if equilibria:
                                st.success(f"✅ Found {len(equilibria)} Nash Equilibria!")
                                for i, eq in enumerate(equilibria):
                                    st.write(f"**Equilibrium {i+1}:** Agent 1 → Route {eq[0]+1}, Agent 2 → Route {eq[1]+1}")
                            else:
                                st.warning("No pure strategy Nash equilibrium found.")
                                
                        except Exception as e:
                            st.error(f"Error in analytical solution: {e}")
            else:
                st.info("Analytical solution only available for 2-agent, 2-route games.")
    
    with col2:
        st.markdown("### Mixed Strategy Analysis")
        
        if st.button("🎲 Compute Mixed Strategy Equilibrium", key="mixed_strategy"):
            with st.spinner("Computing mixed strategy equilibrium..."):
                try:
                    rounds = st.slider("Fictitious Play Rounds", 100, 2000, 500, key="fp_rounds")
                    
                    mixed_eq = game.mixed_strategy_solver(rounds=rounds)
                    
                    st.success("✅ Mixed Strategy Analysis Complete!")
                    
                    # Display mixed strategies for each agent
                    for agent_idx, strategy in enumerate(mixed_eq):
                        st.write(f"**Agent {agent_idx + 1} Strategy:**")
                        for route_idx, prob in enumerate(strategy):
                            st.write(f"  Route {route_idx + 1}: {prob:.3f}")
                    
                    # Visualize mixed strategies
                    fig = go.Figure()
                    
                    for agent_idx, strategy in enumerate(mixed_eq):
                        fig.add_trace(go.Bar(
                            name=f'Agent {agent_idx + 1}',
                            x=[f'Route {i+1}' for i in range(game.num_routes)],
                            y=strategy
                        ))
                    
                    fig.update_layout(
                        title="Mixed Strategy Nash Equilibrium",
                        xaxis_title="Routes",
                        yaxis_title="Probability",
                        barmode='group',
                        template="plotly_white"
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"Error computing mixed strategies: {e}")
        
        # Price of Anarchy analysis
        st.markdown("### Efficiency Analysis")
        
        if st.button("⚖️ Compare Nash vs Social Optimum", key="poa_analysis"):
            with st.spinner("Analyzing efficiency..."):
                try:
                    comparison = game.compare_nash_vs_social_optimum()
                    
                    st.success("✅ Efficiency Analysis Complete!")
                    
                    nash_cost = comparison.get('nash_cost', 0)
                    social_cost = comparison.get('social_cost', 0)
                    poa = comparison.get('price_of_anarchy', 1)
                    
                    # Display metrics
                    col_metric1, col_metric2, col_metric3 = st.columns(3)
                    
                    with col_metric1:
                        st.metric("Nash Equilibrium Cost", f"{nash_cost:.2f}")
                    with col_metric2:
                        st.metric("Social Optimum Cost", f"{social_cost:.2f}")
                    with col_metric3:
                        st.metric("Price of Anarchy", f"{poa:.2f}")
                    
                    # Efficiency visualization
                    fig = go.Figure(data=[
                        go.Bar(name='Nash Equilibrium', x=['System Cost'], y=[nash_cost], marker_color='orange'),
                        go.Bar(name='Social Optimum', x=['System Cost'], y=[social_cost], marker_color='green')
                    ])
                    
                    fig.update_layout(
                        title="Nash Equilibrium vs Social Optimum",
                        yaxis_title="Total System Cost",
                        template="plotly_white"
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    if poa > 1.2:
                        st.warning(f"⚠️ High inefficiency detected (PoA = {poa:.2f}). Consider mechanism design interventions.")
                    else:
                        st.success(f"✅ Reasonable efficiency (PoA = {poa:.2f})")
                        
                except Exception as e:
                    st.error(f"Error in efficiency analysis: {e}")

def show_learning_dynamics():
    st.markdown("## 📈 Interactive Learning Dynamics")
    
    if st.session_state.game is None:
        st.warning("⚠️ Please create a game first using the sidebar configuration.")
        return
    
    game = st.session_state.game
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### Q-Learning Simulation")
        
        # Q-Learning parameters
        episodes = st.slider("Training Episodes", 100, 5000, 1000, key="q_episodes")
        learning_rate = st.slider("Learning Rate (α)", 0.01, 0.5, 0.1, key="q_alpha")
        exploration_rate = st.slider("Exploration Rate (ε)", 0.01, 0.5, 0.2, key="q_epsilon")
        discount_factor = st.slider("Discount Factor (γ)", 0.8, 0.99, 0.95, key="q_gamma")
        
        if st.button("🧠 Run Q-Learning", key="run_qlearning"):
            with st.spinner("Training Q-Learning agents..."):
                try:
                    # Initialize learning dynamics
                    learning_dynamics = LearningDynamics(game)
                    
                    # Run Q-learning
                    results = learning_dynamics.q_learning(
                        episodes=episodes,
                        learning_rate=learning_rate,
                        exploration_rate=exploration_rate,
                        discount_factor=discount_factor
                    )
                    
                    st.success("✅ Q-Learning Training Complete!")
                    
                    # Store results
                    st.session_state.learning_history = results['episode_rewards']
                    
                    # Display final strategy
                    final_strategy = results.get('final_strategy', [])
                    if final_strategy:
                        route_counts = [final_strategy.count(r) for r in range(game.num_routes)]
                        st.write(f"**Learned Strategy:** {final_strategy}")
                        st.write(f"**Final Reward:** {results['episode_rewards'][-1]:.2f}")
                    
                    # Plot learning curve
                    fig = go.Figure()
                    
                    episodes_list = list(range(len(results['episode_rewards'])))
                    fig.add_trace(go.Scatter(
                        x=episodes_list,
                        y=results['episode_rewards'],
                        mode='lines',
                        name='Episode Reward',
                        line=dict(color='#1f77b4')
                    ))
                    
                    # Add moving average
                    window_size = min(50, len(results['episode_rewards']) // 10)
                    if len(results['episode_rewards']) > window_size:
                        moving_avg = np.convolve(results['episode_rewards'], 
                                               np.ones(window_size)/window_size, mode='valid')
                        fig.add_trace(go.Scatter(
                            x=list(range(window_size-1, len(results['episode_rewards']))),
                            y=moving_avg,
                            mode='lines',
                            name='Moving Average',
                            line=dict(color='#ff7f0e', width=3)
                        ))
                    
                    fig.update_layout(
                        title="Q-Learning Training Progress",
                        xaxis_title="Episode",
                        yaxis_title="Reward",
                        template="plotly_white"
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"Error in Q-Learning: {e}")
    
    with col2:
        st.markdown("### Regret Minimization")
        
        regret_rounds = st.slider("Regret Minimization Rounds", 100, 2000, 500, key="regret_rounds")
        
        if st.button("📉 Run Regret Minimization", key="run_regret"):
            with st.spinner("Running regret minimization..."):
                try:
                    learning_dynamics = LearningDynamics(game)
                    
                    results = learning_dynamics.regret_minimization(rounds=regret_rounds)
                    
                    st.success("✅ Regret Minimization Complete!")
                    
                    # Display results
                    final_regrets = results.get('final_regrets', [])
                    cumulative_regret = results.get('cumulative_regret', [])
                    
                    st.write(f"**Final Average Regret:** {np.mean(final_regrets):.4f}")
                    
                    # Plot regret evolution
                    fig = go.Figure()
                    
                    if cumulative_regret:
                        fig.add_trace(go.Scatter(
                            x=list(range(len(cumulative_regret))),
                            y=cumulative_regret,
                            mode='lines',
                            name='Cumulative Regret',
                            line=dict(color='#d62728', width=2)
                        ))
                    
                    fig.update_layout(
                        title="Regret Minimization Progress",
                        xaxis_title="Round",
                        yaxis_title="Cumulative Regret",
                        template="plotly_white"
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"Error in regret minimization: {e}")
        
        # Multi-agent learning
        st.markdown("### Multi-Agent Learning")
        
        if st.button("👥 Simulate Multi-Agent Learning", key="multiagent_learning"):
            with st.spinner("Simulating multi-agent learning..."):
                try:
                    learning_dynamics = LearningDynamics(game)
                    
                    # Run multi-agent learning simulation
                    results = learning_dynamics.multi_agent_learning(rounds=1000)
                    
                    st.success("✅ Multi-Agent Learning Complete!")
                    
                    # Show convergence results
                    if 'strategy_evolution' in results:
                        strategy_evolution = results['strategy_evolution']
                        
                        fig = go.Figure()
                        
                        for route in range(game.num_routes):
                            route_usage = [sum(1 for agent_strategy in round_strategies 
                                             for agent_choice in agent_strategy 
                                             if agent_choice == route) 
                                         for round_strategies in strategy_evolution]
                            
                            fig.add_trace(go.Scatter(
                                x=list(range(len(strategy_evolution))),
                                y=route_usage,
                                mode='lines',
                                name=f'Route {route + 1}',
                                line=dict(width=2)
                            ))
                        
                        fig.update_layout(
                            title="Multi-Agent Strategy Evolution",
                            xaxis_title="Learning Round",
                            yaxis_title="Total Route Usage",
                            template="plotly_white"
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                except Exception as e:
                    st.error(f"Error in multi-agent learning: {e}")

def show_evolutionary_analysis():
    st.markdown("## 🔬 Evolutionary Game Theory")
    
    if st.session_state.game is None:
        st.warning("⚠️ Please create a game first using the sidebar configuration.")
        return
    
    game = st.session_state.game
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### Replicator Dynamics")
        
        population_size = st.slider("Population Size", 100, 5000, 1000, key="pop_size")
        time_steps = st.slider("Time Steps", 50, 1000, 300, key="time_steps")
        mutation_rate = st.slider("Mutation Rate", 0.0, 0.1, 0.01, key="mutation_rate")
        
        # Initial population distribution
        st.markdown("**Initial Population Distribution:**")
        initial_dist = []
        remaining_pop = population_size
        
        for route in range(game.num_routes):
            if route == game.num_routes - 1:
                # Last route gets remaining population
                route_pop = remaining_pop
                st.write(f"Route {route + 1}: {route_pop} agents")
            else:
                route_pop = st.slider(
                    f"Route {route + 1}:",
                    0, remaining_pop, 
                    remaining_pop // game.num_routes,
                    key=f"init_pop_{route}"
                )
                remaining_pop -= route_pop
            
            initial_dist.extend([route] * route_pop)
        
        if st.button("🧬 Run Replicator Dynamics", key="run_replicator"):
            with st.spinner("Simulating population evolution..."):
                try:
                    evolutionary_analysis = EvolutionaryGameAnalysis(game)
                    
                    results = evolutionary_analysis.replicator_dynamics(
                        initial_population=initial_dist,
                        time_steps=time_steps,
                        mutation_rate=mutation_rate
                    )
                    
                    st.success("✅ Evolutionary Simulation Complete!")
                    
                    # Plot population evolution
                    population_history = results.get('population_history', [])
                    
                    if population_history:
                        fig = go.Figure()
                        
                        for route in range(game.num_routes):
                            route_frequencies = []
                            for pop_state in population_history:
                                route_count = sum(1 for agent in pop_state if agent == route)
                                frequency = route_count / len(pop_state)
                                route_frequencies.append(frequency)
                            
                            fig.add_trace(go.Scatter(
                                x=list(range(len(population_history))),
                                y=route_frequencies,
                                mode='lines',
                                name=f'Route {route + 1}',
                                line=dict(width=3)
                            ))
                        
                        fig.update_layout(
                            title="Population Strategy Evolution",
                            xaxis_title="Time Steps",
                            yaxis_title="Strategy Frequency",
                            template="plotly_white"
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Final state analysis
                        final_state = population_history[-1]
                        final_counts = [final_state.count(r) for r in range(game.num_routes)]
                        
                        st.markdown("**Final Population Distribution:**")
                        for route, count in enumerate(final_counts):
                            percentage = (count / len(final_state)) * 100
                            st.write(f"Route {route + 1}: {count} agents ({percentage:.1f}%)")
                    
                except Exception as e:
                    st.error(f"Error in replicator dynamics: {e}")
    
    with col2:
        st.markdown("### Evolutionarily Stable Strategy (ESS)")
        
        if st.button("🎯 Find ESS", key="find_ess"):
            with st.spinner("Computing ESS..."):
                try:
                    evolutionary_analysis = EvolutionaryGameAnalysis(game)
                    
                    ess_result = evolutionary_analysis.find_ess()
                    
                    if ess_result['is_ess']:
                        st.success("✅ Evolutionarily Stable Strategy Found!")
                        ess_strategy = ess_result['ess_strategy']
                        
                        # Display ESS
                        route_distribution = [ess_strategy.count(r) for r in range(game.num_routes)]
                        
                        fig = go.Figure(data=[
                            go.Bar(x=[f'Route {i+1}' for i in range(game.num_routes)], 
                                  y=route_distribution,
                                  marker_color='lightblue')
                        ])
                        fig.update_layout(
                            title="Evolutionarily Stable Strategy",
                            yaxis_title="Number of Agents",
                            template="plotly_white"
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        st.write(f"**ESS Strategy:** {ess_strategy}")
                        st.write(f"**Stability Score:** {ess_result.get('stability_score', 'N/A')}")
                        
                    else:
                        st.warning("⚠️ No pure ESS found. Mixed strategies may exist.")
                        
                except Exception as e:
                    st.error(f"Error finding ESS: {e}")
        
        # Population dynamics with perturbations
        st.markdown("### Stability Analysis")
        
        perturbation_size = st.slider("Perturbation Size", 0.01, 0.2, 0.05, key="perturbation")
        
        if st.button("🔀 Test Strategy Stability", key="test_stability"):
            with st.spinner("Testing strategy stability..."):
                try:
                    evolutionary_analysis = EvolutionaryGameAnalysis(game)
                    
                    stability_result = evolutionary_analysis.stability_analysis(
                        perturbation_size=perturbation_size
                    )
                    
                    st.success("✅ Stability Analysis Complete!")
                    
                    is_stable = stability_result.get('is_stable', False)
                    return_time = stability_result.get('return_time', None)
                    
                    if is_stable:
                        st.success(f"✅ Strategy is stable! Returns to equilibrium in {return_time} steps.")
                    else:
                        st.warning("⚠️ Strategy is unstable under perturbations.")
                    
                    # Plot stability test
                    if 'perturbation_history' in stability_result:
                        history = stability_result['perturbation_history']
                        
                        fig = go.Figure()
                        
                        for route in range(game.num_routes):
                            route_freq = []
                            for state in history:
                                freq = sum(1 for agent in state if agent == route) / len(state)
                                route_freq.append(freq)
                            
                            fig.add_trace(go.Scatter(
                                x=list(range(len(history))),
                                y=route_freq,
                                mode='lines',
                                name=f'Route {route + 1}'
                            ))
                        
                        fig.update_layout(
                            title="Strategy Recovery After Perturbation",
                            xaxis_title="Time Steps",
                            yaxis_title="Strategy Frequency",
                            template="plotly_white"
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"Error in stability analysis: {e}")

def show_mechanism_design():
    st.markdown("## 💰 Interactive Mechanism Design")
    
    if st.session_state.game is None:
        st.warning("⚠️ Please create a game first using the sidebar configuration.")
        return
    
    game = st.session_state.game
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### Pigouvian Tax Design")
        
        # Tax parameters
        tax_type = st.selectbox("Tax Structure:", ["Flat Rate", "Congestion-Based", "Route-Specific"], key="tax_type")
        
        if tax_type == "Flat Rate":
            tax_rate = st.slider("Tax Rate per Agent", 0.0, 10.0, 2.0, 0.1, key="flat_tax")
            tax_schedule = [tax_rate] * game.num_routes
            
        elif tax_type == "Congestion-Based":
            base_tax = st.slider("Base Tax", 0.0, 5.0, 1.0, 0.1, key="base_tax")
            congestion_multiplier = st.slider("Congestion Multiplier", 0.1, 3.0, 0.5, 0.1, key="cong_mult")
            tax_schedule = [base_tax + congestion_multiplier] * game.num_routes
            
        elif tax_type == "Route-Specific":
            tax_schedule = []
            for route in range(game.num_routes):
                route_tax = st.slider(f"Tax for Route {route + 1}", 0.0, 10.0, 1.0, 0.1, key=f"route_tax_{route}")
                tax_schedule.append(route_tax)
        
        if st.button("💸 Apply Pigouvian Tax", key="apply_tax"):
            with st.spinner("Analyzing tax impact..."):
                try:
                    mechanism_design = MechanismDesign(game)
                    
                    tax_result = mechanism_design.pigouvian_tax(tax_rates=tax_schedule)
                    
                    st.success("✅ Tax Analysis Complete!")
                    
                    # Display results
                    pre_tax_cost = tax_result.get('pre_tax_cost', 0)
                    post_tax_cost = tax_result.get('post_tax_cost', 0)
                    tax_revenue = tax_result.get('tax_revenue', 0)
                    efficiency_gain = tax_result.get('efficiency_gain', 0)
                    
                    # Metrics
                    col_metric1, col_metric2, col_metric3 = st.columns(3)
                    
                    with col_metric1:
                        st.metric("Pre-tax Cost", f"{pre_tax_cost:.2f}")
                    with col_metric2:
                        st.metric("Post-tax Cost", f"{post_tax_cost:.2f}")
                    with col_metric3:
                        st.metric("Tax Revenue", f"{tax_revenue:.2f}")
                    
                    # Efficiency improvement
                    if efficiency_gain > 0:
                        st.success(f"✅ Efficiency improved by {efficiency_gain:.2%}")
                    else:
                        st.warning(f"⚠️ Efficiency decreased by {abs(efficiency_gain):.2%}")
                    
                    # Visualization
                    pre_post_data = {
                        'Scenario': ['Pre-tax', 'Post-tax'],
                        'System Cost': [pre_tax_cost, post_tax_cost],
                        'Tax Revenue': [0, tax_revenue]
                    }
                    
                    fig = go.Figure()
                    fig.add_trace(go.Bar(name='System Cost', x=pre_post_data['Scenario'], 
                                        y=pre_post_data['System Cost'], marker_color='orange'))
                    fig.add_trace(go.Bar(name='Tax Revenue', x=pre_post_data['Scenario'], 
                                        y=pre_post_data['Tax Revenue'], marker_color='green'))
                    
                    fig.update_layout(
                        title="Tax Impact Analysis",
                        yaxis_title="Cost/Revenue",
                        barmode='group',
                        template="plotly_white"
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"Error in tax analysis: {e}")
    
    with col2:
        st.markdown("### Subsidization Schemes")
        
        subsidy_type = st.selectbox("Subsidy Type:", ["Route Subsidies", "Usage-Based", "Welfare-Maximizing"], key="subsidy_type")
        
        if subsidy_type == "Route Subsidies":
            subsidy_schedule = []
            for route in range(game.num_routes):
                route_subsidy = st.slider(f"Subsidy for Route {route + 1}", 0.0, 5.0, 0.5, 0.1, key=f"route_subsidy_{route}")
                subsidy_schedule.append(route_subsidy)
        
        if st.button("🎁 Apply Subsidies", key="apply_subsidy"):
            with st.spinner("Analyzing subsidy impact..."):
                try:
                    mechanism_design = MechanismDesign(game)
                    
                    subsidy_result = mechanism_design.subsidization_scheme(subsidy_rates=subsidy_schedule)
                    
                    st.success("✅ Subsidy Analysis Complete!")
                    
                    # Display results
                    pre_subsidy_cost = subsidy_result.get('pre_subsidy_cost', 0)
                    post_subsidy_cost = subsidy_result.get('post_subsidy_cost', 0)
                    total_subsidy = subsidy_result.get('total_subsidy', 0)
                    welfare_change = subsidy_result.get('welfare_change', 0)
                    
                    # Metrics
                    col_metric1, col_metric2, col_metric3 = st.columns(3)
                    
                    with col_metric1:
                        st.metric("Pre-subsidy Cost", f"{pre_subsidy_cost:.2f}")
                    with col_metric2:
                        st.metric("Post-subsidy Cost", f"{post_subsidy_cost:.2f}")
                    with col_metric3:
                        st.metric("Total Subsidy", f"{total_subsidy:.2f}")
                    
                    # Welfare analysis
                    if welfare_change > 0:
                        st.success(f"✅ Social welfare improved by {welfare_change:.2f}")
                    else:
                        st.warning(f"⚠️ Social welfare decreased by {abs(welfare_change):.2f}")
                    
                except Exception as e:
                    st.error(f"Error in subsidy analysis: {e}")
        
        # Welfare optimization
        st.markdown("### Welfare Optimization")
        
        if st.button("⚖️ Find Optimal Mechanism", key="optimal_mechanism"):
            with st.spinner("Computing optimal mechanism..."):
                try:
                    mechanism_design = MechanismDesign(game)
                    
                    optimal_result = mechanism_design.welfare_optimization()
                    
                    st.success("✅ Optimal Mechanism Found!")
                    
                    optimal_mechanism = optimal_result.get('optimal_mechanism', {})
                    optimal_welfare = optimal_result.get('optimal_welfare', 0)
                    
                    st.write(f"**Optimal Social Welfare:** {optimal_welfare:.2f}")
                    
                    if 'optimal_taxes' in optimal_mechanism:
                        st.write("**Optimal Tax Schedule:**")
                        for route, tax in enumerate(optimal_mechanism['optimal_taxes']):
                            st.write(f"  Route {route + 1}: {tax:.2f}")
                    
                    if 'optimal_subsidies' in optimal_mechanism:
                        st.write("**Optimal Subsidy Schedule:**")
                        for route, subsidy in enumerate(optimal_mechanism['optimal_subsidies']):
                            st.write(f"  Route {route + 1}: {subsidy:.2f}")
                    
                except Exception as e:
                    st.error(f"Error in welfare optimization: {e}")

def show_network_scenarios():
    st.markdown("## 🌐 Real-World Network Scenarios")
    
    scenario_type = st.selectbox(
        "Choose Network Scenario:",
        ["Traffic Network", "Data Center Routing", "Supply Chain Network", "Internet Routing"],
        key="network_scenario"
    )
    
    try:
        network_scenarios = NetworkRoutingScenarios()
        
        if scenario_type == "Traffic Network":
            st.markdown("### Traffic Network Analysis")
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                # Traffic scenario parameters
                num_commuters = st.slider("Number of Commuters", 100, 2000, 500, key="traffic_commuters")
                rush_hour = st.checkbox("Rush Hour Conditions", key="rush_hour")
                highway_capacity = st.slider("Highway Capacity", 200, 1000, 400, key="highway_cap")
                city_capacity = st.slider("City Route Capacity", 300, 1200, 600, key="city_cap")
                
                if st.button("🚗 Analyze Traffic Scenario", key="analyze_traffic"):
                    with st.spinner("Analyzing traffic network..."):
                        try:
                            traffic_result = network_scenarios.traffic_network_scenario(
                                num_commuters=num_commuters,
                                rush_hour=rush_hour,
                                highway_capacity=highway_capacity,
                                city_capacity=city_capacity
                            )
                            
                            st.success("✅ Traffic Analysis Complete!")
                            
                            # Display results
                            equilibrium = traffic_result.get('equilibrium_strategy', [])
                            total_time = traffic_result.get('total_travel_time', 0)
                            avg_time = traffic_result.get('average_travel_time', 0)
                            
                            # Route distribution
                            highway_users = equilibrium.count(0) if equilibrium else 0
                            city_users = equilibrium.count(1) if equilibrium else 0
                            
                            st.write(f"**Highway Users:** {highway_users}")
                            st.write(f"**City Route Users:** {city_users}")
                            st.write(f"**Average Travel Time:** {avg_time:.2f} minutes")
                            
                        except Exception as e:
                            st.error(f"Error in traffic analysis: {e}")
            
            with col2:
                # Traffic visualization would go here
                st.markdown("#### Traffic Flow Visualization")
                
                # Create sample traffic visualization
                routes = ['Highway', 'City Route']
                capacities = [highway_capacity if 'highway_capacity' in locals() else 400, 
                             city_capacity if 'city_capacity' in locals() else 600]
                
                fig = go.Figure(data=[
                    go.Bar(x=routes, y=capacities, name='Capacity', marker_color='lightblue'),
                ])
                
                fig.update_layout(
                    title="Route Capacities",
                    yaxis_title="Capacity (vehicles/hour)",
                    template="plotly_white"
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        elif scenario_type == "Data Center Routing":
            st.markdown("### Data Center Load Balancing")
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                packets_per_second = st.slider("Packets per Second", 1000, 50000, 10000, key="packets_sec")
                num_servers = st.slider("Number of Servers", 2, 10, 4, key="num_servers")
                server_capacity = st.slider("Server Capacity", 2000, 20000, 8000, key="server_cap")
                
                if st.button("💻 Analyze Data Center", key="analyze_datacenter"):
                    with st.spinner("Analyzing data center routing..."):
                        try:
                            dc_result = network_scenarios.data_center_routing(
                                packets_per_second=packets_per_second,
                                num_servers=num_servers,
                                server_capacity=server_capacity
                            )
                            
                            st.success("✅ Data Center Analysis Complete!")
                            
                            # Display load balancing results
                            server_loads = dc_result.get('server_loads', [])
                            response_time = dc_result.get('avg_response_time', 0)
                            
                            st.write(f"**Average Response Time:** {response_time:.2f} ms")
                            
                            if server_loads:
                                load_df = pd.DataFrame({
                                    'Server': [f'Server {i+1}' for i in range(len(server_loads))],
                                    'Load': server_loads,
                                    'Utilization': [load/server_capacity*100 for load in server_loads]
                                })
                                
                                st.dataframe(load_df)
                            
                        except Exception as e:
                            st.error(f"Error in data center analysis: {e}")
            
            with col2:
                # Server utilization visualization
                if 'server_loads' in locals():
                    fig = go.Figure(data=[
                        go.Bar(x=[f'Server {i+1}' for i in range(num_servers)], 
                              y=[100/num_servers]*num_servers,  # Sample utilization
                              marker_color='lightcoral')
                    ])
                    
                    fig.update_layout(
                        title="Server Utilization",
                        yaxis_title="Utilization (%)",
                        template="plotly_white"
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error loading network scenarios: {e}")

def show_realtime_simulation():
    st.markdown("## ⚡ Real-Time Game Simulation")
    
    if st.session_state.game is None:
        st.warning("⚠️ Please create a game first using the sidebar configuration.")
        return
    
    game = st.session_state.game
    
    st.markdown("### Live Interactive Simulation")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("#### Simulation Controls")
        
        simulation_type = st.selectbox(
            "Simulation Type:",
            ["Manual Agent Control", "Auto Best Response", "Learning Agents", "Evolutionary Dynamics"],
            key="sim_type"
        )
        
        if simulation_type == "Manual Agent Control":
            st.markdown("**Control Each Agent:**")
            
            agent_choices = []
            for agent in range(game.num_agents):
                choice = st.selectbox(
                    f"Agent {agent + 1} Route:",
                    [f"Route {r+1}" for r in range(game.num_routes)],
                    key=f"agent_{agent}_choice"
                )
                agent_choices.append(int(choice.split()[-1]) - 1)
            
            if st.button("📊 Update Game State", key="update_manual"):
                # Calculate current costs
                route_counts = [agent_choices.count(r) for r in range(game.num_routes)]
                individual_costs = []
                
                for agent_idx, route_choice in enumerate(agent_choices):
                    cost = game.cost_function(route_counts[route_choice])
                    individual_costs.append(cost)
                
                total_cost = sum(individual_costs)
                
                # Store results
                st.session_state.game_results = {
                    'strategy': agent_choices,
                    'route_counts': route_counts,
                    'individual_costs': individual_costs,
                    'total_cost': total_cost,
                    'timestamp': time.time()
                }
        
        elif simulation_type == "Auto Best Response":
            auto_iterations = st.slider("Auto Iterations", 1, 50, 10, key="auto_iter")
            iteration_delay = st.slider("Delay (seconds)", 0.1, 2.0, 0.5, key="iter_delay")
            
            if st.button("🔄 Start Auto Simulation", key="start_auto"):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                current_strategy = [0] * game.num_agents  # Start with all on route 0
                history = []
                
                for iteration in range(auto_iterations):
                    # Best response step
                    new_strategy = current_strategy.copy()
                    
                    for agent in range(game.num_agents):
                        best_route = 0
                        best_cost = float('inf')
                        
                        for route in range(game.num_routes):
                            # Test this route
                            test_strategy = new_strategy.copy()
                            test_strategy[agent] = route
                            
                            route_counts = [test_strategy.count(r) for r in range(game.num_routes)]
                            cost = game.cost_function(route_counts[route])
                            
                            if cost < best_cost:
                                best_cost = cost
                                best_route = route
                        
                        new_strategy[agent] = best_route
                    
                    current_strategy = new_strategy
                    history.append(current_strategy.copy())
                    
                    # Update progress
                    progress_bar.progress((iteration + 1) / auto_iterations)
                    status_text.text(f"Iteration {iteration + 1}/{auto_iterations}")
                    
                    time.sleep(iteration_delay)
                
                # Final results
                route_counts = [current_strategy.count(r) for r in range(game.num_routes)]
                total_cost = sum(game.cost_function(route_counts[route]) * route_counts[route] 
                               for route in range(game.num_routes))
                
                st.session_state.game_results = {
                    'strategy': current_strategy,
                    'route_counts': route_counts,
                    'total_cost': total_cost,
                    'history': history,
                    'converged': True
                }
                
                st.success(f"✅ Auto simulation complete! Final cost: {total_cost:.2f}")
    
    with col2:
        st.markdown("#### Real-Time Results")
        
        if st.session_state.game_results:
            results = st.session_state.game_results
            
            # Current state metrics
            st.markdown("**Current Game State:**")
            
            col_metric1, col_metric2 = st.columns(2)
            with col_metric1:
                st.metric("Total System Cost", f"{results.get('total_cost', 0):.2f}")
            with col_metric2:
                avg_cost = results.get('total_cost', 0) / game.num_agents
                st.metric("Average Cost", f"{avg_cost:.2f}")
            
            # Route distribution
            route_counts = results.get('route_counts', [])
            if route_counts:
                fig = go.Figure(data=[
                    go.Bar(x=[f'Route {i+1}' for i in range(game.num_routes)], 
                          y=route_counts,
                          marker_color='lightblue',
                          text=route_counts,
                          textposition='auto')
                ])
                
                fig.update_layout(
                    title="Current Route Distribution",
                    yaxis_title="Number of Agents",
                    template="plotly_white",
                    height=300
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Individual costs
            individual_costs = results.get('individual_costs', [])
            if individual_costs:
                st.markdown("**Individual Agent Costs:**")
                cost_df = pd.DataFrame({
                    'Agent': [f'Agent {i+1}' for i in range(len(individual_costs))],
                    'Route': [f'Route {results["strategy"][i]+1}' for i in range(len(individual_costs))],
                    'Cost': individual_costs
                })
                st.dataframe(cost_df)
            
            # Convergence history (if available)
            if 'history' in results:
                history = results['history']
                if len(history) > 1:
                    st.markdown("**Convergence History:**")
                    
                    fig = go.Figure()
                    
                    for route in range(game.num_routes):
                        route_usage = [h.count(route) for h in history]
                        fig.add_trace(go.Scatter(
                            x=list(range(len(history))),
                            y=route_usage,
                            mode='lines+markers',
                            name=f'Route {route + 1}'
                        ))
                    
                    fig.update_layout(
                        title="Route Usage Over Time",
                        xaxis_title="Iteration",
                        yaxis_title="Number of Agents",
                        template="plotly_white",
                        height=300
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
        
        # Real-time controls
        st.markdown("#### Quick Actions")
        
        if st.button("🔄 Reset Game", key="reset_game"):
            st.session_state.game_results = {}
            st.success("Game reset!")
            st.rerun()
        
        if st.button("📈 Analyze Current State", key="analyze_current"):
            if st.session_state.game_results:
                results = st.session_state.game_results
                strategy = results.get('strategy', [])
                
                # Check if current state is Nash equilibrium
                is_equilibrium = True
                for agent in range(game.num_agents):
                    current_route = strategy[agent]
                    current_cost = results['individual_costs'][agent]
                    
                    # Check if agent wants to deviate
                    for alt_route in range(game.num_routes):
                        if alt_route != current_route:
                            # Calculate cost if agent switches
                            test_strategy = strategy.copy()
                            test_strategy[agent] = alt_route
                            test_counts = [test_strategy.count(r) for r in range(game.num_routes)]
                            alt_cost = game.cost_function(test_counts[alt_route])
                            
                            if alt_cost < current_cost - 0.01:  # Small tolerance
                                is_equilibrium = False
                                break
                    
                    if not is_equilibrium:
                        break
                
                if is_equilibrium:
                    st.success("✅ Current state is a Nash Equilibrium!")
                else:
                    st.warning("⚠️ Current state is NOT a Nash Equilibrium. Some agents want to deviate.")
            else:
                st.info("No current game state to analyze.")

if __name__ == "__main__":
    main()
