"""
Streamlit Cloud Entry Point for ICS Intrusion Detection System
This file serves as the main entry point for Streamlit Cloud deployment
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import and run the main demo application
from demo.app import main

if __name__ == "__main__":
    main()
