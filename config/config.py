import os
from dotenv import load_dotenv

# Force loading from project root
dotenv_path = os.path.join(os.path.dirname(__file__), "..", ".env")
load_dotenv(dotenv_path)

def get_env_variable(var_name, default=None):
    return os.getenv(var_name, default)
