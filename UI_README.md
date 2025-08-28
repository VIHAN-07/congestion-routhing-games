# 🚗 Nash Equilibrium for Congestion Routing Games - Web Interface

## 🎯 Interactive Web Dashboard

This Streamlit web application provides an interactive interface to explore and analyze congestion routing games, Nash equilibria, and various game theory algorithms.

## 🚀 Quick Start

### Option 1: Run with Python
```bash
python launch_ui.py
```

### Option 2: Run with Streamlit directly
```bash
streamlit run streamlit_app.py
```

### Option 3: Windows Batch File
```bash
run_ui.bat
```

The web interface will open at: **http://localhost:8501**

## 📱 Features & Navigation

### 🏠 **Home & Overview**
- Project introduction and statistics
- Quick demo of 2-agent game
- Key achievements summary

### 🎮 **Basic Game Setup**
- Interactive game configuration
- Adjustable parameters (agents, routes, cost functions)
- Real-time payoff matrix generation
- Cost function visualization

### 📊 **Nash Equilibrium Analysis**
- 2-agent analytical solutions using nashpy
- Multi-agent numerical computation
- Best response dynamics visualization
- Mixed strategy equilibrium computation

### 🧠 **Advanced Algorithms**
- Best Response Dynamics with convergence plots
- Mixed Strategy Nash Equilibrium finder
- Fictitious Play simulation
- Algorithm comparison and analysis

### 🌐 **Network Scenarios**
- **Traffic Networks**: Highway vs city route analysis
- **Data Centers**: Server load balancing optimization
- **Supply Chains**: Distribution route selection
- **Internet Routing**: BGP and ISP routing scenarios

### 📈 **Learning Dynamics**
- **Q-Learning**: Reinforcement learning for route selection
- **Regret Minimization**: Online learning algorithms
- **Multi-Agent Learning**: Adaptive behavior simulation
- Training progress visualization with rewards and exploration

### 🔬 **Evolutionary Analysis**
- **Replicator Dynamics**: Population strategy evolution
- **Evolutionarily Stable Strategies (ESS)**: Stability analysis
- **Population Dynamics**: Mutation and selection effects
- Real-time evolution visualization

### 💰 **Mechanism Design**
- **Pigouvian Taxes**: Externality correction mechanisms
- **Subsidization Schemes**: Incentive alignment tools
- **Congestion Pricing**: Dynamic pricing strategies
- **Welfare Analysis**: Social optimum vs Nash equilibrium

### 📋 **Performance Benchmarks**
- Algorithm computation time analysis
- Memory usage comparison
- Scalability testing (up to 100+ agents)
- Convergence rate analysis

## 🎨 **Interactive Visualizations**

The web interface includes:

- 📊 **Real-time Charts**: Plotly-powered interactive graphs
- 🎯 **Parameter Sliders**: Adjust game parameters on-the-fly
- 📈 **Convergence Plots**: Visualize algorithm convergence
- 🌍 **Network Diagrams**: Route selection visualizations
- 📉 **Performance Metrics**: Algorithm comparison charts
- 🎲 **Strategy Distributions**: Nash equilibrium visualizations

## 💻 **Technical Features**

### **Interactive Controls**
- Slider widgets for parameter adjustment
- Dropdown menus for algorithm selection
- Checkboxes for feature toggles
- Button triggers for computations

### **Real-time Computation**
- Instant results for parameter changes
- Progressive algorithm execution
- Live plotting and updates
- Error handling with user feedback

### **Educational Tools**
- Step-by-step algorithm explanations
- Mathematical formula displays
- Concept definitions and theory
- Best practice recommendations

## 🔧 **Customization**

### **Adding New Scenarios**
1. Edit `streamlit_app.py`
2. Add new scenario in `show_network_scenarios()`
3. Include visualization and analysis code

### **Adding New Algorithms**
1. Implement algorithm in core modules
2. Add UI controls in appropriate section
3. Create visualization functions
4. Update documentation

### **Styling Customization**
- Modify CSS in the `st.markdown()` sections
- Adjust color schemes in Plotly charts
- Update layout and spacing
- Add custom icons and emojis

## 📊 **Sample Screenshots**

The interface includes:
- **Dashboard Overview**: Project statistics and quick demo
- **Game Configuration**: Interactive parameter setup
- **Algorithm Results**: Nash equilibrium computation results
- **Convergence Plots**: Best response dynamics visualization  
- **Network Analysis**: Real-world scenario simulations
- **Performance Charts**: Algorithm comparison benchmarks

## 🛠️ **Troubleshooting**

### Common Issues:

**Module Import Errors:**
```bash
# Ensure all project files are in the same directory
# Check that core modules are present:
ls *.py
```

**Package Missing:**
```bash
pip install -r requirements.txt
```

**Port Already in Use:**
```bash
streamlit run streamlit_app.py --server.port=8502
```

**Browser Not Opening:**
- Manually navigate to http://localhost:8501
- Check firewall settings
- Try different port number

## 🌐 **Deployment Options**

### **Local Development**
- Run locally for development and testing
- Perfect for presentations and demos

### **Streamlit Cloud**
1. Push to GitHub repository
2. Connect at https://share.streamlit.io
3. Deploy with one click

### **Docker Deployment**
```dockerfile
FROM python:3.11
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
EXPOSE 8501
CMD ["streamlit", "run", "streamlit_app.py"]
```

### **Cloud Platforms**
- Deploy on Heroku, AWS, GCP, or Azure
- Use provided requirements.txt for dependencies
- Set PORT environment variable for cloud deployment

## 📚 **Educational Use**

Perfect for:
- **University Courses**: Game theory and algorithmic economics
- **Research Demonstrations**: Interactive algorithm showcases
- **Student Projects**: Hands-on learning with game theory
- **Conference Presentations**: Live algorithm demonstrations
- **Industry Training**: Network optimization concepts

## 🎓 **Learning Objectives**

Students/users will learn:
- Nash equilibrium computation techniques
- Convergence properties of iterative algorithms
- Real-world applications of game theory
- Performance trade-offs between algorithms
- Mechanism design principles
- Evolutionary game theory concepts

## 🏆 **Conclusion**

This interactive web interface transforms complex game theory concepts into accessible, visual, and hands-on learning experiences. It's perfect for education, research, and professional demonstrations of congestion routing games and Nash equilibrium analysis.

---

**🚀 Ready to explore? Run `python launch_ui.py` and start your game theory journey!**
