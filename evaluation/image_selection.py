import os


def get_final_image_path(prompt_dir, refinement_iterations=3, last_restart_step=0, selection_mode="first_early_stop"):
    """
    Determines which image file represents the final answer for a given prompt folder,
    following main.py's output naming convention (final_{N}_refined_{restart_step}_.png
    for the last refinement round, ES{n}_final_image.png for early-stop markers).

    Args:
        prompt_dir (str): Path to the prompt's output folder (e.g. images_dir/tag/prompt_001).
        refinement_iterations (int): Number of refinement rounds the pipeline ran (main.py's
            len(restart_steps)).
        last_restart_step (int): The restart step value used for the final round (the last
            entry of main.py's --restart_steps).
        selection_mode (str): "first_early_stop" checks ES1 -> ES2 -> ... -> ES{refinement_iterations}
            in order and returns the first one found, falling back to the final round's image if
            no ES marker exists. "final_only" always returns the final round's image, ignoring any
            ES markers.

    Returns:
        str or None: Path to the selected image, or None if nothing matching was found.
    """
    final_round_image = os.path.join(
        prompt_dir, f"final_{refinement_iterations}_refined_{last_restart_step}_.png"
    )

    if selection_mode == "final_only":
        return final_round_image if os.path.exists(final_round_image) else None

    if selection_mode != "first_early_stop":
        raise ValueError(f"Unknown selection_mode: {selection_mode!r}")

    for i in range(1, refinement_iterations + 1):
        es_path = os.path.join(prompt_dir, f"ES{i}_final_image.png")
        if os.path.exists(es_path):
            return es_path

    if os.path.exists(final_round_image):
        return final_round_image

    return None