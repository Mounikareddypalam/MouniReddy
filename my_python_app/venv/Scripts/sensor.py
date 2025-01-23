# sensor.py
import random

def read_sensor_data():
    """
    Simulates sensor data for testing purposes.
    In a real-world scenario, this function would read data from a hardware sensor.
    Returns a random integer between 0 and 4, corresponding to the categories.
    """
    return random.randint(0, 4)
