# main.py
from flask import Flask

# Initialize the Flask application
app = Flask(__name__)

# Define a route for the home page
@app.route('/')
def home():
    return "Welcome to Baby Crying Detection System!"

# Define another route for the /about page
@app.route('/about')
def about():
    return "This system detects baby crying reasons based on sensor data."

# Start the Flask app
if __name__ == "__main__":
    app.run(debug=True)
