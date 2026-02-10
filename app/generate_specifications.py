import sys
from pathlib import Path

import yaml
from main import app

sys.path.append(".")


FOLDER_PATH = Path(__file__).parent

if __name__ == "__main__":
    data = app.openapi()
    data["info"]["title"] = "Ecommerce-Recommendation"

    with open(f"{FOLDER_PATH}/api-specifications/openapi.yaml", "wt") as fy:
        fy.write(yaml.dump(data, sort_keys=False))
