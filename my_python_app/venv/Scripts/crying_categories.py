# crying_categories.py
categories = ["belly_pain", "burping", "discomfort", "hungry", "tired"]

def get_crying_reason(sensor_value):
    """
    Maps sensor values to crying categories.
    The sensor_value is a number corresponding to the category index.
    """
    try:
        return categories[sensor_value]
    except IndexError:
        return "Unknown reason"
