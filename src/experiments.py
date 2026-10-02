from itertools import combinations


def generate_experiments(reductions, combination_sizes):

    experiments = []

    enabled = [name for name, active in reductions.items() if active]

    for size in sorted(combination_sizes):

        # validation
        if size < 0 or size > len(enabled):
            raise ValueError(
                f"Invalid combination size {size}. "
                f"Must be between 0 and {len(enabled)}."
            )

        # baseline
        if size == 0:
            experiments.append(["baseline"])
            continue

        # reduction combinations
        for combo in combinations(enabled, size):
            experiments.append(list(combo))

    return experiments