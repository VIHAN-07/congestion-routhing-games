#!/usr/bin/env python3
"""
Launch script for the Nash Equilibrium Congestion Routing Games Web Interface
"""
import subprocess
import webbrowser
import time
import sys
import os

def main():
    print("🚗 Nash Equilibrium for Congestion Routing Games")
    print("=" * 50)
    print("Starting Streamlit web interface...")
    print()
    print("📱 Web interface will open at: http://localhost:8501")
    print("🛑 Press Ctrl+C to stop the server")
    print()
    
    # Change to the script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    try:
        # Start streamlit
        subprocess.run([sys.executable, "-m", "streamlit", "run", "streamlit_app.py"], check=True)
    except KeyboardInterrupt:
        print("\n\n👋 Shutting down the web interface...")
    except FileNotFoundError:
        print("❌ Error: Streamlit not installed. Please install requirements:")
        print("   pip install -r requirements.txt")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error starting web interface: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
