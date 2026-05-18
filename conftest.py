import sys
import os

# Add src/ to sys.path so tests can import nio_mcp without installing the package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
