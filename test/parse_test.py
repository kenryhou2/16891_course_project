import numpy as np
import re

def parse_config_file(filepath):
    data_dict = {}
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or ":" not in line:
                continue  # skip empty or malformed lines

            # Split only on the first colon to preserve colons in values if any
            key, value = line.split(":", 1)
            key = key.strip()
            value = value.strip()

            # Handle integer or float value (Coverage)
            if re.match(r"^-?\d+(\.\d+)?$", value):
                data_dict[key] = float(value) if '.' in value else int(value)
            else:
                try:
                    # Attempt to convert string list to numpy array
                    array_val = np.fromstring(value.strip("[]"), sep=',')
                    data_dict[key] = array_val
                except ValueError:
                    print(f"Could not parse line: {line}")
    return data_dict

# Example usage
if __name__ == "__main__":
    filepath = "./test_2arms_35p_coverage.txt"  # Replace with your file path
    parsed_data = parse_config_file(filepath)

    for k, v in parsed_data.items():
        print(f"{k}: {v}")
