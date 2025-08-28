"""
Working Interactive Game Theory UI
Only includes buttons and features that work properly with the original CongestionRoutingGame class.
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import time
import warnings
warnings.filterwarnings('ignore')

# Import the original modules
try:
    from congestion_routing_game import CongestionRoutingGame
    from game_analysis_utils import GameAnalyzer
    from advanced_analysis import LearningDynamics, EvolutionaryGameAnalysis, MechanismDesign
    from network_routing_scenarios import NetworkRoutingScenarios
except ImportError as e:
    st.error(f"Import error: {e}")
    st.stop()

def main():
    st.set_page_config(
        page_title="Game Theory Interactive Platform",
        page_icon="🎯",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.title("🎯 Interactive Game Theory Platform")
    st.markdown("*Real-time interactive game with full project functionality*")

    # Initialize session state
    if 'game' not in st.session_state:
        st.session_state.game = None
    if 'results' not in st.session_state:
        st.session_state.results = None

    # Sidebar for game configuration
    st.sidebar.title("🎮 Game Configuration")
    
    # Basic parameters
    num_agents = st.sidebar.slider("Number of Agents", min_value=2, max_value=10, value=4, key="num_agents")
    num_routes = st.sidebar.slider("Number of Routes", min_value=2, max_value=6, value=3, key="num_routes")
    
    # Cost function configuration
    st.sidebar.markdown("### Cost Function Configuration")
    cost_function_type = st.sidebar.selectbox(
        "Cost Function Type:",
        ["Quadratic", "Linear", "Cubic", "Exponential"],
        key="cost_type"
    )
    
    # Configure single cost function (matching original design)
    if cost_function_type == "Quadratic":
        base = st.sidebar.slider("Base Cost", 0.5, 5.0, 1.0, 0.1, key="base")
        factor = st.sidebar.slider("Quadratic Factor", 0.1, 2.0, 1.0, 0.1, key="factor")
        cost_function = lambda x: base * (x ** 2) if x > 0 else 0
        st.sidebar.write(f"Cost function: {base} × congestion²")
        
    elif cost_function_type == "Linear":
        base = st.sidebar.slider("Base Cost", 0.5, 5.0, 2.0, 0.1, key="base")
        slope = st.sidebar.slider("Slope", 0.1, 3.0, 1.0, 0.1, key="slope")
        cost_function = lambda x: base + slope * x
        st.sidebar.write(f"Cost function: {base} + {slope} × congestion")
        
    elif cost_function_type == "Cubic":
        base = st.sidebar.slider("Base Cost", 0.5, 3.0, 1.0, 0.1, key="base")
        factor = st.sidebar.slider("Cubic Factor", 0.1, 1.0, 0.5, 0.1, key="factor")
        cost_function = lambda x: base * (x ** 3) if x > 0 else 0
        st.sidebar.write(f"Cost function: {base} × congestion³")
        
    elif cost_function_type == "Exponential":
        base = st.sidebar.slider("Base Cost", 0.5, 2.0, 1.0, 0.1, key="base")
        exp_factor = st.sidebar.slider("Exponential Factor", 0.1, 0.8, 0.3, 0.1, key="exp_factor")
        cost_function = lambda x: base * np.exp(exp_factor * x)
        st.sidebar.write(f"Cost function: {base} × e^({exp_factor} × congestion)")
    
    # Create or update game
    if st.sidebar.button("🎮 Create/Update Game", type="primary"):
        with st.spinner("Creating game..."):
            st.session_state.game = CongestionRoutingGame(num_agents, num_routes, cost_function)
            st.session_state.results = None
        st.success(f"✅ Game created with {num_agents} agents and {num_routes} routes!")
        st.rerun()

    # Main content
    if st.session_state.game is None:
        st.info("👆 Please create a game using the sidebar to get started!")
        return

    game = st.session_state.game

    # Create tabs for different analyses
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🎯 Game Analysis", 
        "⚖️ Nash Equilibrium", 
        "📈 Learning Dynamics", 
        "🌐 Network Scenarios",
        "📊 Visualizations"
    ])

    # Tab 1: Game Analysis
    with tab1:
        st.header("🎯 Game Analysis")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("### Cost Function Visualization")
            
            # Show current cost function
            max_users = st.slider("Maximum Users to Display", 1, game.num_agents, game.num_agents, key="max_users_viz")
            
            fig = go.Figure()
            x_vals = np.arange(1, max_users + 1)
            y_vals = [game.cost_function(x) for x in x_vals]
            
            fig.add_trace(go.Scatter(
                x=x_vals,
                y=y_vals,
                mode='lines+markers',
                name='Cost Function',
                line=dict(width=3, color='blue'),
                marker=dict(size=8)
            ))
            
            fig.update_layout(
                title="Cost vs Congestion",
                xaxis_title="Number of Users on Route",
                yaxis_title="Cost per User",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("### Game Information")
            
            # Basic info
            st.write(f"**Agents:** {game.num_agents}")
            st.write(f"**Routes:** {game.num_routes}")
            st.write(f"**Cost Function:** {cost_function_type}")
            
            # Payoff Matrix Display (only for 2-agent games)
            if st.button("🔢 Show Payoff Matrices", key="show_payoff"):
                if game.num_agents == 2:
                    with st.spinner("Generating payoff matrices..."):
                        try:
                            payoff_matrices = game._create_payoff_matrices()
                            
                            st.success("✅ Payoff Matrices Generated!")
                            
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
                                
                        except Exception as e:
                            st.error(f"Error generating payoff matrices: {e}")
                else:
                    st.info("Payoff matrix display is only available for 2-agent games.")
    
    # Tab 2: Nash Equilibrium Analysis
    with tab2:
        st.header("⚖️ Nash Equilibrium Analysis")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("### Nash Equilibrium Solver")
            
            if game.num_agents == 2:
                if st.button("🎯 Solve 2-Agent Nash Equilibrium", key="nash_2agent"):
                    with st.spinner("Computing Nash equilibrium..."):
                        try:
                            equilibria = game.solve_two_agent_game()
                            
                            st.success("✅ Nash Equilibrium Found!")
                            
                            # Display results
                            if 'pure_strategies' in equilibria and equilibria['pure_strategies']:
                                st.write("**Pure Strategy Nash Equilibria:**")
                                for i, eq in enumerate(equilibria['pure_strategies']):
                                    st.write(f"Equilibrium {i+1}: {eq}")
                            
                            if 'mixed_strategies' in equilibria and equilibria['mixed_strategies']:
                                st.write("**Mixed Strategy Nash Equilibria:**")
                                for i, eq in enumerate(equilibria['mixed_strategies']):
                                    st.write(f"Mixed Equilibrium {i+1}:")
                                    st.write(f"  Player 1: {eq[0]}")
                                    st.write(f"  Player 2: {eq[1]}")
                            
                            st.session_state.results = equilibria
                            
                        except Exception as e:
                            st.error(f"Error computing Nash equilibrium: {e}")
            else:
                st.info("Exact Nash equilibrium solver is only available for 2-agent games.")
                
                # Alternative: Best response dynamics for n-agent games
                if st.button("🔄 Run Best Response Dynamics", key="best_response"):
                    with st.spinner("Running best response dynamics..."):
                        try:
                            max_iter = st.selectbox("Max Iterations:", [100, 500, 1000], index=1, key="max_iter_br")
                            
                            results = game.best_response_dynamics(max_iterations=max_iter)
                            
                            if results['converged']:
                                st.success("✅ Converged to Nash Equilibrium!")
                            else:
                                st.warning("⚠️ Did not converge within iteration limit")
                            
                            st.write(f"**Final Strategy:** {results['equilibrium']}")
                            st.write(f"**Iterations:** {results['iterations']}")
                            st.write(f"**Total Cost:** {results['total_cost']:.2f}")
                            
                            # Show route distribution
                            route_counts = [results['equilibrium'].count(r) for r in range(game.num_routes)]
                            
                            fig = go.Figure(data=[
                                go.Bar(x=[f'Route {i+1}' for i in range(game.num_routes)], 
                                      y=route_counts,
                                      marker_color='lightblue')
                            ])
                            fig.update_layout(
                                title="Route Distribution at Equilibrium",
                                xaxis_title="Routes",
                                yaxis_title="Number of Agents"
                            )
                            st.plotly_chart(fig, use_container_width=True)
                            
                            st.session_state.results = results
                            
                        except Exception as e:
                            st.error(f"Error in best response dynamics: {e}")

        with col2:
            st.markdown("### Social Optimum Analysis")
            
            if st.button("🎯 Calculate Social Optimum", key="social_opt"):
                with st.spinner("Computing social optimum..."):
                    try:
                        social_opt = game.calculate_social_optimum()
                        
                        st.success("✅ Social Optimum Found!")
                        st.write(f"**Optimal Strategy:** {social_opt['strategy_profile']}")
                        st.write(f"**Total Cost:** {social_opt['total_cost']:.2f}")
                        st.write(f"**Average Cost:** {social_opt['average_cost']:.2f}")
                        
                        # Visualize social optimum
                        route_counts = social_opt['route_distribution']
                        
                        fig = go.Figure(data=[
                            go.Bar(x=[f'Route {i+1}' for i in range(game.num_routes)], 
                                  y=route_counts,
                                  marker_color='lightgreen')
                        ])
                        fig.update_layout(
                            title="Socially Optimal Route Distribution",
                            xaxis_title="Routes",
                            yaxis_title="Number of Agents",
                            height=300
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                    except Exception as e:
                        st.error(f"Error calculating social optimum: {e}")
            
            # Compare Nash vs Social Optimum
            st.markdown("### Nash vs Social Optimum Comparison")
            
            if st.button("⚖️ Compare Nash vs Social Optimum", key="compare_nash_social"):
                with st.spinner("Comparing equilibria..."):
                    try:
                        comparison = game.compare_nash_vs_social_optimum()
                        
                        st.success("✅ Comparison Complete!")
                        
                        # Display comparison results
                        nash_cost = comparison['nash_total_cost']
                        social_cost = comparison['social_optimal_cost']
                        poa = comparison['price_of_anarchy']
                        
                        st.write(f"**Nash Equilibrium Cost:** {nash_cost:.2f}")
                        st.write(f"**Social Optimum Cost:** {social_cost:.2f}")
                        st.write(f"**Price of Anarchy:** {poa:.2f}")
                        st.write(f"**Efficiency Loss:** {comparison['efficiency_loss']:.2f}")
                        
                        # Visualization
                        fig = go.Figure()
                        fig.add_trace(go.Bar(
                            name='Nash Equilibrium',
                            x=['Total Cost'],
                            y=[nash_cost],
                            marker_color='coral'
                        ))
                        fig.add_trace(go.Bar(
                            name='Social Optimum',
                            x=['Total Cost'],
                            y=[social_cost],
                            marker_color='lightgreen'
                        ))
                        
                        fig.update_layout(
                            title="Nash Equilibrium vs Social Optimum",
                            yaxis_title="Total Cost",
                            height=300
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                    except Exception as e:
                        st.error(f"Error in comparison: {e}")

    # Tab 3: Learning Dynamics
    with tab3:
        st.header("📈 Learning Dynamics")
        
        # Mixed Strategy Solver
        st.markdown("### Mixed Strategy Learning")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            max_iterations = st.slider("Max Iterations", 100, 2000, 1000, key="mixed_iter")
            learning_rate = st.slider("Learning Rate", 0.01, 0.5, 0.1, 0.01, key="learning_rate")
            
            if st.button("🧠 Run Mixed Strategy Solver", key="mixed_solver"):
                with st.spinner("Running mixed strategy learning..."):
                    try:
                        results = game.mixed_strategy_solver(
                            max_iterations=max_iterations,
                            learning_rate=learning_rate
                        )
                        
                        st.success("✅ Learning Complete!")
                        
                        # Display final mixed strategies
                        st.write("**Final Mixed Strategies:**")
                        for agent_idx, strategy in enumerate(results['mixed_strategies']):
                            st.write(f"Agent {agent_idx + 1}: {[f'{prob:.3f}' for prob in strategy]}")
                        
                        st.write(f"**Converged:** {'Yes' if results['converged'] else 'No'}")
                        st.write(f"**Total Expected Cost:** {results['total_expected_cost']:.3f}")
                        st.write(f"**Iterations:** {results['convergence_iterations']}")
                        
                        # Store results for visualization
                        st.session_state.mixed_results = results
                        
                    except Exception as e:
                        st.error(f"Error in mixed strategy learning: {e}")

        with col2:
            # Convergence Analysis
            if 'mixed_results' in st.session_state:
                st.markdown("### Convergence Analysis")
                
                if st.button("📊 Analyze Convergence", key="analyze_conv"):
                    try:
                        # Use strategy history if available
                        if 'strategy_history' in st.session_state.mixed_results:
                            convergence_analysis = game.analyze_convergence(st.session_state.mixed_results['strategy_history'])
                            
                            # Plot convergence using expected costs
                            expected_costs = st.session_state.mixed_results['expected_costs']
                            
                            fig = go.Figure()
                            fig.add_trace(go.Scatter(
                                x=list(range(len(expected_costs))),
                                y=expected_costs,
                                mode='lines+markers',
                                name='Expected Cost per Agent',
                                line=dict(color='blue', width=2)
                            ))
                            
                            fig.update_layout(
                                title="Learning Convergence - Expected Costs",
                                xaxis_title="Agent",
                                yaxis_title="Expected Cost",
                                height=300
                            )
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Display convergence metrics
                            st.write(f"**Convergence Rate:** {convergence_analysis['convergence_rate']:.4f}")
                            if 'final_variance' in convergence_analysis:
                                st.write(f"**Final Variance:** {convergence_analysis['final_variance']:.4f}")
                            
                            # Show convergence over time if available
                            if 'strategy_variance_history' in convergence_analysis and convergence_analysis['strategy_variance_history']:
                                variance_history = convergence_analysis['strategy_variance_history']
                                
                                fig2 = go.Figure()
                                fig2.add_trace(go.Scatter(
                                    x=list(range(len(variance_history))),
                                    y=variance_history,
                                    mode='lines',
                                    name='Strategy Variance',
                                    line=dict(color='red', width=2)
                                ))
                                
                                fig2.update_layout(
                                    title="Strategy Variance Over Time",
                                    xaxis_title="Iteration",
                                    yaxis_title="Total Strategy Variance",
                                    height=300
                                )
                                st.plotly_chart(fig2, use_container_width=True)
                                
                        else:
                            st.info("Strategy history not available for detailed convergence analysis.")
                        
                    except Exception as e:
                        st.error(f"Error in convergence analysis: {e}")
                        st.info("Convergence analysis may not be available for this game configuration.")

    # Tab 4: Network Scenarios
    with tab4:
        st.header("🌐 Interactive Network Scenarios")
        
        st.markdown("""
        **Create and analyze realistic network routing scenarios with full control over parameters.**
        """)
        
        # Scenario Configuration Section
        st.subheader("🎛️ Scenario Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            scenario_type = st.selectbox(
                "Select Scenario Type:",
                ["Traffic Network", "Internet Routing", "Data Center", "Supply Chain", "Custom"],
                key="scenario_type"
            )
            
            # Dynamic parameter adjustment based on scenario
            if scenario_type == "Traffic Network":
                default_agents = 10
                default_routes = 3
                route_names = ["Highway", "City Route", "Scenic Route"]
                cost_desc = "Travel time increases with congestion"
            elif scenario_type == "Internet Routing":
                default_agents = 10
                default_routes = 5
                route_names = ["Tier-1 A", "Tier-1 B", "Regional-1", "Regional-2", "Backup"]
                cost_desc = "Peering costs + latency + bandwidth costs"
            elif scenario_type == "Data Center":
                default_agents = 8
                default_routes = 4
                route_names = ["Path A (Fast)", "Path B (Balanced)", "Path C (Backup)", "Path D (Emergency)"]
                cost_desc = "Network latency increases exponentially"
            elif scenario_type == "Supply Chain":
                default_agents = 8
                default_routes = 3
                route_names = ["Express Route", "Standard Route", "Economy Route"]
                cost_desc = "Shipping cost + congestion penalties"
            else:  # Custom
                default_agents = 5
                default_routes = 3
                route_names = [f"Route {i+1}" for i in range(3)]
                cost_desc = "User-defined cost function"
        
        with col2:
            num_agents_scenario = st.slider(
                "Number of Agents/Players:", 
                min_value=2, max_value=20, value=default_agents, 
                key="agents_scenario",
                help="Number of decision makers in the scenario"
            )
            
            num_routes_scenario = st.slider(
                "Number of Routes/Paths:", 
                min_value=2, max_value=8, value=default_routes,
                key="routes_scenario", 
                help="Number of alternative routes available"
            )
        
        # Cost Function Configuration
        st.subheader("📈 Cost Function Configuration")
        
        cost_function_type = st.selectbox(
            "Cost Function Type:",
            ["Scenario Default", "Linear", "Quadratic", "Exponential", "Custom Formula"],
            key="cost_func_scenario"
        )
        
        col3, col4, col5 = st.columns(3)
        
        if cost_function_type != "Scenario Default":
            with col3:
                base_cost_scenario = st.number_input(
                    "Base Cost:", value=10.0, min_value=0.0, 
                    key="base_scenario",
                    help="Fixed cost component"
                )
            
            with col4:
                if cost_function_type == "Linear":
                    slope_scenario = st.number_input(
                        "Congestion Multiplier:", value=5.0, min_value=0.1,
                        key="slope_scenario"
                    )
                elif cost_function_type == "Quadratic":
                    quad_coef = st.number_input(
                        "Quadratic Coefficient:", value=2.0, min_value=0.1,
                        key="quad_scenario"
                    )
                elif cost_function_type == "Exponential":
                    exp_rate = st.number_input(
                        "Exponential Rate:", value=0.3, min_value=0.1,
                        key="exp_scenario"
                    )
            
            with col5:
                if cost_function_type == "Custom Formula":
                    custom_formula = st.text_input(
                        "Formula (use 'x' for congestion):",
                        value="10 + 3*x**2",
                        key="custom_scenario",
                        help="Example: 10 + 3*x**2 or 5*exp(0.2*x)"
                    )
        
        # Route Names Configuration
        st.subheader("🛣️ Route Configuration")
        
        if num_routes_scenario != len(route_names):
            route_names = [f"Route {i+1}" for i in range(num_routes_scenario)]
        
        route_cols = st.columns(min(num_routes_scenario, 4))
        updated_route_names = []
        
        for i in range(num_routes_scenario):
            col_idx = i % len(route_cols)
            with route_cols[col_idx]:
                route_name = st.text_input(
                    f"Route {i+1} Name:",
                    value=route_names[i] if i < len(route_names) else f"Route {i+1}",
                    key=f"route_name_{i}"
                )
                updated_route_names.append(route_name)
        
        # Real-time Preview
        st.subheader("👀 Configuration Preview")
        
        preview_col1, preview_col2 = st.columns(2)
        
        with preview_col1:
            st.markdown("**Scenario Summary:**")
            st.write(f"- Type: {scenario_type}")
            st.write(f"- Agents: {num_agents_scenario}")
            st.write(f"- Routes: {num_routes_scenario}")
            st.write(f"- Cost Function: {cost_function_type}")
            st.write(f"- Description: {cost_desc}")
        
        with preview_col2:
            st.markdown("**Routes:**")
            for i, name in enumerate(updated_route_names):
                st.write(f"Route {i}: {name}")
        
        # Run Scenario Button
        if st.button(f"🚀 Run Interactive {scenario_type} Scenario", key="run_interactive_scenario"):
            with st.spinner(f"Analyzing {scenario_type} scenario..."):
                try:
                    # Create cost function based on user selection
                    if cost_function_type == "Scenario Default":
                        # Use predefined scenario cost functions
                        if scenario_type == "Traffic Network":
                            cost_func = lambda x: 20 + 3 * (x ** 1.5)
                        elif scenario_type == "Internet Routing":
                            cost_func = lambda x: 50 + 10*x + 5*(x**2)
                        elif scenario_type == "Data Center":
                            cost_func = lambda x: 5 * np.exp(0.3 * (x - 1))
                        elif scenario_type == "Supply Chain":
                            cost_func = lambda x: 1000 + 200*(x**2)
                        else:
                            cost_func = lambda x: 2 + 3*x
                    elif cost_function_type == "Linear":
                        cost_func = lambda x: base_cost_scenario + slope_scenario * x
                    elif cost_function_type == "Quadratic":
                        cost_func = lambda x: base_cost_scenario + quad_coef * (x ** 2)
                    elif cost_function_type == "Exponential":
                        cost_func = lambda x: base_cost_scenario * np.exp(exp_rate * (x - 1))
                    elif cost_function_type == "Custom Formula":
                        try:
                            cost_func = lambda x: eval(custom_formula.replace('x', str(x)))
                            # Test the function
                            test_cost = cost_func(1)
                        except:
                            st.error("Invalid custom formula! Using default linear function.")
                            cost_func = lambda x: 10 + 5*x
                    
                    # Create the game
                    scenario_game = CongestionRoutingGame(
                        num_agents=num_agents_scenario, 
                        num_routes=num_routes_scenario, 
                        cost_function=cost_func
                    )
                    
                    # Run analysis
                    nash_result = scenario_game.best_response_dynamics()
                    social_opt = scenario_game.calculate_social_optimum()
                    comparison = scenario_game.compare_nash_vs_social_optimum()
                    
                    # Store results
                    st.session_state.scenario_results = {
                        'scenario_type': scenario_type,
                        'nash': nash_result,
                        'social': social_opt,
                        'comparison': comparison,
                        'route_names': updated_route_names,
                        'game': scenario_game
                    }
                    
                    st.success(f"✅ {scenario_type} scenario analysis completed!")
                    
                except Exception as e:
                    st.error(f"Error running scenario: {e}")
        
        # Results Display
        if 'scenario_results' in st.session_state and st.session_state.scenario_results:
            st.subheader("📊 Scenario Results")
            
            results = st.session_state.scenario_results
            
            # Create tabs for different result views
            result_tab1, result_tab2, result_tab3, result_tab4 = st.tabs([
                "📈 Nash Equilibrium", "🎯 Social Optimum", "⚖️ Comparison", "🎮 Interactive Analysis"
            ])
            
            with result_tab1:
                st.markdown("### Nash Equilibrium Results")
                
                if results['nash']['converged']:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric("Total Nash Cost", f"{results['nash']['total_cost']:.2f}")
                        st.metric("Average Cost per Agent", f"{results['nash']['total_cost']/num_agents_scenario:.2f}")
                        st.metric("Convergence", f"{results['nash']['iterations']} iterations")
                    
                    with col2:
                        st.markdown("**Strategy Profile:**")
                        strategy_df = pd.DataFrame({
                            'Agent': [f"Agent {i}" for i in range(len(results['nash']['strategy_profile']))],
                            'Chosen Route': [updated_route_names[route] for route in results['nash']['strategy_profile']],
                            'Individual Cost': [f"{cost:.2f}" for cost in results['nash']['agent_costs']]
                        })
                        st.dataframe(strategy_df, width='stretch')
                    
                    # Route distribution visualization
                    route_dist = results['nash']['route_distribution']
                    fig_nash = px.bar(
                        x=updated_route_names[:len(route_dist)], 
                        y=route_dist,
                        title="Nash Equilibrium Route Distribution",
                        labels={'x': 'Routes', 'y': 'Number of Agents'}
                    )
                    st.plotly_chart(fig_nash, width='stretch')
                
                else:
                    st.warning("Nash equilibrium did not converge. Try adjusting parameters.")
            
            with result_tab2:
                st.markdown("### Social Optimum Results")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Total Social Cost", f"{results['social']['total_cost']:.2f}")
                    st.metric("Average Cost per Agent", f"{results['social']['total_cost']/num_agents_scenario:.2f}")
                
                with col2:
                    st.markdown("**Optimal Route Distribution:**")
                    social_dist = results['social']['route_distribution']
                    
                    # Calculate cost per agent for each route
                    route_costs = []
                    for i, count in enumerate(social_dist):
                        if count > 0:
                            # Get the cost function from the game
                            cost_per_agent = results['game'].cost_function(count)
                            route_costs.append(f"{cost_per_agent:.2f}")
                        else:
                            route_costs.append("0.00")
                    
                    social_df = pd.DataFrame({
                        'Route': updated_route_names[:len(social_dist)],
                        'Agents': social_dist,
                        'Cost per Agent': route_costs
                    })
                    st.dataframe(social_df, width='stretch')
                
                # Social optimum visualization
                fig_social = px.bar(
                    x=updated_route_names[:len(social_dist)], 
                    y=social_dist,
                    title="Social Optimum Route Distribution",
                    labels={'x': 'Routes', 'y': 'Number of Agents'},
                    color_discrete_sequence=['#2E8B57']
                )
                st.plotly_chart(fig_social, width='stretch')
            
            with result_tab3:
                st.markdown("### Nash vs Social Optimum Comparison")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        "Price of Anarchy", 
                        f"{results['comparison']['price_of_anarchy']:.4f}",
                        help="Ratio of Nash cost to Social Optimum cost"
                    )
                
                with col2:
                    st.metric(
                        "Efficiency Loss", 
                        f"{results['comparison']['efficiency_loss']:.2f}",
                        help="Additional cost due to selfish behavior"
                    )
                
                with col3:
                    improvement = ((results['nash']['total_cost'] - results['social']['total_cost']) / results['nash']['total_cost']) * 100
                    st.metric(
                        "Potential Improvement", 
                        f"{improvement:.1f}%",
                        help="Percentage cost reduction if agents cooperate"
                    )
                
                # Comparison visualization
                comparison_data = pd.DataFrame({
                    'Route': updated_route_names[:max(len(results['nash']['route_distribution']), len(results['social']['route_distribution']))],
                    'Nash Equilibrium': results['nash']['route_distribution'] + [0] * (len(updated_route_names) - len(results['nash']['route_distribution'])),
                    'Social Optimum': results['social']['route_distribution'] + [0] * (len(updated_route_names) - len(results['social']['route_distribution']))
                })
                
                fig_comparison = px.bar(
                    comparison_data, 
                    x='Route', 
                    y=['Nash Equilibrium', 'Social Optimum'],
                    title="Route Distribution: Nash vs Social Optimum",
                    barmode='group'
                )
                st.plotly_chart(fig_comparison, width='stretch')
            
            with result_tab4:
                st.markdown("### Interactive Analysis Tools")
                
                # What-if analysis
                st.markdown("#### 🔄 What-If Analysis")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Modify Parameters:**")
                    
                    # Agent manipulation
                    if st.button("➕ Add Agent", key="add_agent_scenario"):
                        # Rerun with +1 agent
                        pass
                    
                    if st.button("➖ Remove Agent", key="remove_agent_scenario") and num_agents_scenario > 2:
                        # Rerun with -1 agent
                        pass
                    
                    # Cost sensitivity analysis
                    sensitivity_factor = st.slider(
                        "Cost Sensitivity Multiplier:",
                        min_value=0.5, max_value=3.0, value=1.0, step=0.1,
                        key="sensitivity_scenario",
                        help="Multiply all costs by this factor"
                    )
                    
                    if st.button("🔄 Rerun with New Sensitivity", key="rerun_sensitivity"):
                        st.info("Feature coming soon!")
                
                with col2:
                    st.markdown("**Export Results:**")
                    
                    if st.button("💾 Save Scenario Configuration", key="save_scenario"):
                        config = {
                            'scenario_type': scenario_type,
                            'num_agents': num_agents_scenario,
                            'num_routes': num_routes_scenario,
                            'route_names': updated_route_names,
                            'cost_function_type': cost_function_type
                        }
                        st.json(config)
                        st.success("Configuration displayed above!")
                    
                    if st.button("📊 Generate Report", key="generate_report"):
                        st.markdown(f"""
                        ## {scenario_type} Analysis Report
                        
                        **Configuration:**
                        - Agents: {num_agents_scenario}
                        - Routes: {num_routes_scenario}
                        - Cost Function: {cost_function_type}
                        
                        **Key Results:**
                        - Nash Cost: {results['nash']['total_cost']:.2f}
                        - Social Cost: {results['social']['total_cost']:.2f}
                        - Price of Anarchy: {results['comparison']['price_of_anarchy']:.4f}
                        - Efficiency Loss: {results['comparison']['efficiency_loss']:.2f}
                        
                        **Conclusion:**
                        The scenario shows {improvement:.1f}% efficiency loss due to selfish routing decisions.
                        """)
                        st.success("Report generated!")
        
        # Help section
        with st.expander("ℹ️ How to Use Interactive Network Scenarios"):
            st.markdown("""
            **Steps to analyze network scenarios:**
            
            1. **Choose Scenario Type**: Select from predefined scenarios or create custom
            2. **Configure Parameters**: Adjust number of agents, routes, and cost functions
            3. **Customize Routes**: Name your routes to match your scenario
            4. **Run Analysis**: Click the run button to compute equilibria
            5. **Explore Results**: Use the tabs to examine different aspects
            6. **What-If Analysis**: Modify parameters to see how results change
            
            **Scenario Types:**
            - **Traffic Network**: Commuters choosing routes with travel time costs
            - **Internet Routing**: ISPs routing through backbone connections
            - **Data Center**: Data flows choosing server paths with latency costs
            - **Supply Chain**: Companies choosing distribution routes with shipping costs
            - **Custom**: Define your own scenario with custom parameters
            
            **Cost Functions:**
            - **Linear**: Cost increases proportionally with congestion
            - **Quadratic**: Cost increases quadratically (rapid growth)
            - **Exponential**: Cost grows exponentially (severe congestion effects)
            - **Custom**: Define your own mathematical formula
            """)
        
        try:
            # Legacy support for old scenario runner
            if st.button("🔧 Run Original Scenarios", key="run_original"):
                network_scenarios = NetworkRoutingScenarios()
                
                if scenario_type == "Internet Routing":
                    result = network_scenarios.internet_routing()
                elif scenario_type == "Traffic Network":
                    result = network_scenarios.traffic_network_scenario()
                elif scenario_type == "Data Center":
                    result = network_scenarios.data_center_routing()
                elif scenario_type == "Supply Chain":
                    result = network_scenarios.supply_chain_routing()
                
                st.write("**Original Scenario Results:**")
                st.json(result)
                        
        except Exception as e:
            st.error(f"Network scenarios not available: {e}")

    # Tab 5: Visualizations
    with tab5:
        st.header("📊 Advanced Visualizations")
        
        if st.session_state.results:
            st.markdown("### Current Results Visualization")
            
            if st.button("📈 Visualize Current Results", key="viz_results"):
                try:
                    # Use the game's built-in visualization
                    game.visualize_equilibrium(st.session_state.results, 
                                             title="Current Game Results")
                    st.success("✅ Visualization saved as PNG file!")
                    
                except Exception as e:
                    st.error(f"Error creating visualization: {e}")
        
        # Strategy comparison visualization
        st.markdown("### Strategy Comparison")
        
        if st.button("🔄 Manual Strategy Analysis", key="manual_analysis"):
            st.markdown("#### Configure Manual Strategy")
            
            manual_strategy = []
            cols = st.columns(game.num_routes)
            total_assigned = 0
            
            for route_idx in range(game.num_routes):
                with cols[route_idx]:
                    max_for_route = game.num_agents - total_assigned
                    if route_idx == game.num_routes - 1:  # Last route gets remaining agents
                        agents_on_route = max_for_route
                        st.write(f"Route {route_idx + 1}: {agents_on_route} agents (auto)")
                    else:
                        agents_on_route = st.number_input(
                            f"Route {route_idx + 1}",
                            min_value=0,
                            max_value=max_for_route,
                            value=min(1, max_for_route),
                            key=f"manual_route_{route_idx}"
                        )
                    
                manual_strategy.extend([route_idx] * agents_on_route)
                total_assigned += agents_on_route
            
            if len(manual_strategy) == game.num_agents:
                # Calculate costs for this strategy
                route_counts = [manual_strategy.count(r) for r in range(game.num_routes)]
                individual_costs = []
                
                for agent_idx, route_choice in enumerate(manual_strategy):
                    cost = game.cost_function(route_counts[route_choice])
                    individual_costs.append(cost)
                
                total_cost = sum(individual_costs)
                avg_cost = total_cost / game.num_agents
                
                st.write(f"**Total Cost:** {total_cost:.2f}")
                st.write(f"**Average Cost:** {avg_cost:.2f}")
                
                # Visualize strategy
                fig = go.Figure(data=[
                    go.Bar(x=[f'Route {i+1}' for i in range(game.num_routes)], 
                          y=route_counts,
                          marker_color='lightcoral')
                ])
                fig.update_layout(
                    title="Manual Strategy Distribution",
                    xaxis_title="Routes",
                    yaxis_title="Number of Agents"
                )
                st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
