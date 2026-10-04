"""Semantic class mapping for the TruckScenes to OmniHD adapter."""


OMNIHD_CLASSES = ("car", "pedestrian", "rider", "large_vehicle")


TRUCKSCENES_TO_OMNIHD = {
    "vehicle.car": "car",
    "vehicle.truck": "large_vehicle",
    "vehicle.bus.bendy": "large_vehicle",
    "vehicle.bus.rigid": "large_vehicle",
    "vehicle.construction": "large_vehicle",
    "vehicle.motorcycle": "rider",
    "static_object.bicycle_rack": None,
    "vehicle.bicycle": "rider",
    "vehicle.trailer": "large_vehicle",
    "vehicle.emergency.police": None,
    "vehicle.emergency.ambulance": None,
    "human.pedestrian.adult": "pedestrian",
    "human.pedestrian.child": "pedestrian",
    "human.pedestrian.construction_worker": "pedestrian",
    "human.pedestrian.stroller": None,
    "human.pedestrian.wheelchair": None,
    "human.pedestrian.personal_mobility": "rider",
    "human.pedestrian.police_officer": "pedestrian",
    "animal": None,
    "movable_object.trafficcone": None,
    "movable_object.barrier": None,
    "movable_object.pushable_pullable": None,
    "movable_object.debris": None,
    "vehicle.train": None,
    "static_object.traffic_sign": None,
    "vehicle.other": None,
    "vehicle.ego_trailer": None,
}


def map_truckscenes_category(category_name: str) -> str | None:
    """Map a TruckScenes category to an OmniHD class or ``None``."""
    if category_name not in TRUCKSCENES_TO_OMNIHD:
        raise KeyError(f"Unknown TruckScenes category: {category_name!r}")
    return TRUCKSCENES_TO_OMNIHD[category_name]


def is_supported_category(category_name: str) -> bool:
    """Return whether a TruckScenes category maps to an OmniHD class."""
    return map_truckscenes_category(category_name) is not None
