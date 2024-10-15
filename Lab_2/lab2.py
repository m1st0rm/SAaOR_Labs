import numpy as np


profit_matrix = np.array([[0, 1, 2, 3], [0, 0, 1, 2], [0, 2, 2, 3]])
optimal_profit = np.zeros(profit_matrix.shape)
allocation_matrix = np.zeros(profit_matrix.shape)


def calculate_optimal_profit():
    for agent in range(profit_matrix.shape[0]):
        for resource in range(profit_matrix.shape[1]):
            if agent == 0:
                optimal_profit[agent, resource] = profit_matrix[
                    agent, resource
                ]
                allocation_matrix[agent, resource] = resource
            else:
                temp_values = profit_matrix[agent, : resource + 1] + np.flip(
                    optimal_profit[agent - 1, : resource + 1]
                )
                allocation_matrix[agent, resource] = np.argmax(temp_values)
                optimal_profit[agent, resource] = temp_values[
                    int(allocation_matrix[agent, resource])
                ]

    return optimal_profit[-1, -1]


def retrieve_resource_allocation():
    final_allocation = np.zeros(profit_matrix.shape[0])
    remaining_resource = profit_matrix.shape[1] - 1

    for i, allocation_row in enumerate(np.flip(allocation_matrix, 0)):
        final_allocation[-i - 1] = allocation_row[int(remaining_resource)]
        remaining_resource -= allocation_row[int(remaining_resource)]

    return final_allocation


def main():
    max_profit = calculate_optimal_profit()
    optimal_allocation = retrieve_resource_allocation()
    print(f"Max Profit: {max_profit}")
    print(f"Optimal Resource Allocation: {optimal_allocation}")


if __name__ == "__main__":
    main()
