"""
SmartHousePricePredictor — Input Schema & Validation Module
Defines required feature ordering, validation boundaries, and defensive checks
for California residential property valuation requests.
"""

import math

# Strict expected feature ordering
FEATURE_NAMES = [
    "MedInc",
    "HouseAge",
    "AveRooms",
    "AveBedrms",
    "Population",
    "AveOccup",
    "Latitude",
    "Longitude"
]

# Physical and geographic constraints for California census blocks
FEATURE_BOUNDS = {
    "MedInc": {"min": 0.1, "max": 25.0, "label": "Median Income ($10k units, range: 0.1 - 25.0)"},
    "HouseAge": {"min": 1.0, "max": 100.0, "label": "House Age in years (range: 1 - 100)"},
    "AveRooms": {"min": 1.0, "max": 50.0, "label": "Average Rooms (range: 1.0 - 50.0)"},
    "AveBedrms": {"min": 0.5, "max": 20.0, "label": "Average Bedrooms (range: 0.5 - 20.0)"},
    "Population": {"min": 1.0, "max": 50000.0, "label": "Block Population (range: 1 - 50,000)"},
    "AveOccup": {"min": 0.5, "max": 50.0, "label": "Average Occupants per household (range: 0.5 - 50.0)"},
    "Latitude": {"min": 32.5, "max": 42.0, "label": "California Latitude (range: 32.5° to 42.0° N)"},
    "Longitude": {"min": -124.5, "max": -114.0, "label": "California Longitude (range: -124.5° to -114.0° W)"}
}


def validate_input_payload(data):
    """
    Validates user input against schema, numeric validity, bounds, and physical sanity.

    Args:
        data (dict): Request payload dictionary.

    Returns:
        tuple: (parsed_values_list, error_message_or_none)
    """
    if not isinstance(data, dict):
        return None, "Invalid request format. Expected a JSON object with feature keys."

    parsed_values = []
    missing_fields = []
    invalid_fields = []

    for feature in FEATURE_NAMES:
        if feature not in data or data[feature] is None or str(data[feature]).strip() == "":
            missing_fields.append(feature)
            continue

        raw_val = data[feature]
        try:
            val = float(raw_val)
        except (ValueError, TypeError):
            invalid_fields.append(f"Field '{feature}' must be a valid numeric number (received: '{raw_val}').")
            continue

        if not math.isfinite(val):
            invalid_fields.append(f"Field '{feature}' must be a finite number (received: '{val}').")
            continue

        bounds = FEATURE_BOUNDS[feature]
        if val < bounds["min"] or val > bounds["max"]:
            invalid_fields.append(
                f"Field '{feature}' value {val} is out of realistic bounds [{bounds['min']}, {bounds['max']}] for {bounds['label']}."
            )
            continue

        parsed_values.append(val)

    if missing_fields:
        return None, f"Missing required fields: {', '.join(missing_fields)}"

    if invalid_fields:
        return None, " | ".join(invalid_fields)

    # Physical constraint check: Bedrooms cannot exceed total rooms
    ave_rooms = parsed_values[FEATURE_NAMES.index("AveRooms")]
    ave_bedrms = parsed_values[FEATURE_NAMES.index("AveBedrms")]
    if ave_bedrms > ave_rooms:
        return None, f"Average Bedrooms ({ave_bedrms}) cannot exceed Average Total Rooms ({ave_rooms})."

    return parsed_values, None
