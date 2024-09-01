import argparse

# Create the parser
parser = argparse.ArgumentParser(description='Command line help for app.py')

# Add expected arguments
parser.add_argument('--host', type=str, help='Host where the app will run')
parser.add_argument('--port', type=int, help='Port to bind to')
parser.add_argument('--debug', action='store_true', help='Run the app in debug mode')

# Parse the arguments
args = parser.parse_args()

print("Script completed successfully, no errors.")
