from typing import Tuple

import numpy as np


weights = np.array([3, 5, 9, 7])
values = np.array([2, 4, 7, 6])
capacity = 15
max_values_table = np.zeros((weights.size, capacity))
selection_flags = np.zeros(max_values_table.shape)


def compute_max_value() -> float:
    for item_index in range(max_values_table.shape[0]):
        for current_capacity in range(max_values_table.shape[1]):
            if item_index == 0:
                if weights[item_index] <= current_capacity + 1:
                    max_values_table[item_index, current_capacity] = values[
                        item_index
                    ]
                    selection_flags[item_index, current_capacity] = 1
                else:
                    max_values_table[item_index, current_capacity] = 0
                    selection_flags[item_index, current_capacity] = 0
            else:
                previous_max = max_values_table[
                    item_index - 1, current_capacity
                ]
                if weights[item_index] <= current_capacity + 1:
                    new_value = values[item_index]
                    if current_capacity >= weights[item_index]:
                        new_value += max_values_table[
                            item_index - 1,
                            current_capacity - weights[item_index],
                        ]
                    if new_value > previous_max:
                        max_values_table[item_index, current_capacity] = (
                            new_value
                        )
                        selection_flags[item_index, current_capacity] = 1
                        continue
                max_values_table[item_index, current_capacity] = previous_max
                selection_flags[item_index, current_capacity] = 0

    return max_values_table[-1, -1]


def retrieve_optimal_selection() -> Tuple[np.ndarray, int]:
    selection_result = np.zeros(weights.size)
    remaining_capacity = capacity - 1

    for item_index in reversed(range(weights.size)):
        selection_result[item_index] = selection_flags[
            item_index, remaining_capacity
        ]
        remaining_capacity -= round(
            selection_result[item_index] * weights[item_index]
        )

    return selection_result, capacity - remaining_capacity - 1


def main():
    a = compute_max_value()
    b = retrieve_optimal_selection()
    print(f"Вместимость рюкзака: {capacity}")
    print(f"Вес помещённых в рюкзак предметов: {b[1]}")
    selected_idxs = np.nonzero(b[0])[0]
    print(f"Индексы выбранных предметов: {selected_idxs + 1}")
    print(f"Общая ценность выбранных предметов: {a}")


if __name__ == "__main__":
    main()
