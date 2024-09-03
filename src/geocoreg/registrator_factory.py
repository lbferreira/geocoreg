from typing import Dict, List, Tuple
from . import registrators

# Registrator factory provides registrator classes and kwargs
registrator_factory: Dict[str, Tuple[registrators.Registrator, dict]] = {
    "pcc": (registrators.PCCRegistrator, {}),
    "konia": (registrators.KoniaRegistrator, {}),
}


def get_available_registrators() -> List[str]:
    """Get the available registrators.

    Returns:
        List[str]: List of available registrators.
    """
    return list(registrator_factory.keys())


def build_registrator(registrator_name: str) -> registrators.Registrator:
    """Create a registrator object.
    To see the available registrators, call the function get_available_registrators from this module.

    Args:
        registrator_name (str): Name of the registrator to create.

    Returns:
        registrators.Registrator: Registrator object.
    """
    if registrator_name not in registrator_factory:
        raise ValueError(
            f"Registrator {registrator_name} not found. Available registrators: {list(registrator_factory.keys())}"
        )

    registrator_class, kwargs = registrator_factory[registrator_name]
    return registrator_class(**kwargs)
