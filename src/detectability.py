def filter_objects(
    object_df,
    weather=None,
    area=None,
    visibility=None,
    category=None,
    distance_range=None,
):
    """Filter Ground Truth objects by scene and object conditions."""

    result = object_df.copy()

    if weather is not None:
        result = result[result["weather"] == weather]

    if area is not None:
        result = result[result["area"] == area]

    if visibility is not None:
        result = result[result["visibility_level"] == visibility]

    if category is not None:
        result = result[result["category"] == category]

    if distance_range is not None:
        min_distance, max_distance = distance_range

        if min_distance is not None:
            result = result[result["distance_m"] >= min_distance]

        if max_distance is not None:
            result = result[result["distance_m"] < max_distance]

    sample_tokens = result["sample_token"].drop_duplicates().tolist()

    return result.reset_index(drop=True), sample_tokens