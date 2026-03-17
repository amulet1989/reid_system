import yaml
import os

def load_config(config_path="/app/config.yml"):
    """
    Lee el archivo YAML maestro y devuelve un diccionario con toda la configuración.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"No se encontró el archivo de configuración en: {config_path}")
        
    with open(config_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
        
    return config

# Instanciamos la configuración al importar este módulo para que esté lista para usar
cfg = load_config()