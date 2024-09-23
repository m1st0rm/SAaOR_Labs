from typing import Optional, Tuple

import numpy as np
from numpy.linalg import inv


A_matrix = np.array([[3, 2, 1, 0], [-3, 2, 0, 1]])
cost_vector = np.array([0, 1, 0, 0])
b_vector = np.array([6, 0])


def invert_matrix(
    matrix_inv: np.ndarray, vector: np.ndarray, pivot_row: int
) -> np.ndarray:
    multiplier = np.dot(matrix_inv, vector)

    if multiplier[pivot_row] == 0:
        raise Exception("Матрица не является инвертируемой.")

    pivot_value = multiplier[pivot_row]
    multiplier[pivot_row] = -1
    multiplier *= -1 / pivot_value

    result_matrix = np.empty(matrix_inv.shape)
    for row_index, row in enumerate(matrix_inv):
        for col_index, element in enumerate(row):
            result_matrix[row_index, col_index] = (
                matrix_inv[pivot_row, col_index] * multiplier[row_index]
            )
            if row_index != pivot_row:
                result_matrix[row_index, col_index] += element

    return result_matrix


def simplex_iteration(
    A: np.ndarray, c: np.ndarray, x: np.ndarray, basis: np.ndarray
) -> Tuple[Optional[np.ndarray], np.ndarray]:
    A_basis = A[:, basis]

    try:
        A_basis_inv = inv(A_basis)
    except np.linalg.LinAlgError:
        raise Exception("Задача нерешаема.")

    while True:
        cost_basis = c[basis]

        dual_variables = np.dot(cost_basis, A_basis_inv)

        reduced_costs = np.dot(dual_variables, A) - c

        if (reduced_costs >= 0).all():
            return x, basis

        entering_variable_index = np.argwhere(reduced_costs < 0)[0, 0]

        z_column = np.dot(A_basis_inv, A[:, entering_variable_index])

        theta_values = x[basis] / z_column
        theta_values[z_column <= 0] = np.inf

        leaving_variable_index = np.argmin(theta_values)
        min_theta = theta_values[leaving_variable_index]

        if min_theta == np.inf:
            return None

        for i, var in enumerate(basis):
            x[var] -= min_theta * z_column[i]

        basis[leaving_variable_index] = entering_variable_index
        x[entering_variable_index] = min_theta

        new_column = A[:, entering_variable_index]
        A_basis[:, leaving_variable_index] = new_column
        A_basis_inv = invert_matrix(
            A_basis_inv, new_column, leaving_variable_index
        )


def phase_one(
    A: np.ndarray, b: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    num_columns = A.shape[1]
    num_rows = A.shape[0]

    A[b < 0] = -A[b < 0]

    artificial_cost_vector = np.concatenate(
        (np.zeros(num_columns), np.full(num_rows, -1))
    )
    extended_matrix = np.concatenate((A, np.identity(num_rows)), axis=1)
    extended_solution = np.concatenate((np.zeros(num_columns), b))
    basis_indices = np.array(range(num_columns, num_columns + num_rows))

    extended_solution, basis_indices = simplex_iteration(
        extended_matrix,
        artificial_cost_vector,
        extended_solution,
        basis_indices,
    )

    if (extended_solution[num_columns:] != 0).any():
        raise Exception("Задача нерешаема.")

    final_solution = extended_solution[:num_columns]

    A_basis_inv = inv(extended_matrix[:, basis_indices])
    new_A = A
    new_b = b

    while True:
        max_basis_index = np.argmax(basis_indices)
        basis_var = basis_indices[max_basis_index]
        row_index = basis_var - num_columns

        if basis_var < num_columns:
            return final_solution, basis_indices, new_A, new_b

        row_in_inverse_basis = np.dot(A_basis_inv, new_A)
        non_zero_indices = np.nonzero(
            row_in_inverse_basis[max_basis_index, :]
        )[0]
        non_zero_indices = np.setdiff1d(
            non_zero_indices[non_zero_indices < num_columns], basis_indices
        )

        if non_zero_indices.size == 0:
            new_A = np.delete(new_A, row_index, axis=0)
            new_b = np.delete(new_b, row_index, axis=0)
            basis_indices = np.delete(basis_indices, max_basis_index, axis=0)
            extended_matrix = np.delete(extended_matrix, row_index, axis=0)
            A_basis_inv = inv(extended_matrix[:, basis_indices])
        else:
            entering_variable = non_zero_indices[0]
            basis_indices[max_basis_index] = entering_variable
            invert_matrix(
                A_basis_inv, new_A[:, entering_variable], max_basis_index
            )


def solve_simplex(
    A: np.ndarray, c: np.ndarray, b: np.ndarray
) -> Tuple[Optional[np.ndarray], np.ndarray]:
    solution, basis, A, b = phase_one(A, b)
    return simplex_iteration(A, c, solution, basis)


def fractional_part(value: float) -> float:
    return value - int(value)


def generate_gomory_cut(
    A: np.ndarray, x: np.ndarray, basis: np.ndarray
) -> Optional[Tuple[np.ndarray, float]]:
    for index, component in enumerate(x):
        if int(component) != component:
            row_index = np.where(basis == index)[0]

            A_basis_inv = inv(A[:, basis])
            non_basis_indices = np.array(
                [i for i in range(x.size) if i not in basis]
            )
            Q_matrix = np.dot(A_basis_inv, A[:, non_basis_indices])
            fractional_row = Q_matrix[row_index]

            gomory_cut = np.zeros(x.size + 1)
            gomory_cut[non_basis_indices] = np.array(
                [fractional_part(value) for value in fractional_row[0]]
            )
            gomory_cut[-1] = -1

            return gomory_cut, fractional_part(component)

    return None


def main() -> None:
    simplex_result = solve_simplex(A_matrix, cost_vector, b_vector)

    if simplex_result is None:
        print(
            "Задача несовместна или её целевая функция неограничена сверху на множестве допустимых планов."
        )
        return

    gomory_cut = generate_gomory_cut(A_matrix, *simplex_result)

    if gomory_cut is None:
        print(f"Оптимальный план задачи: {simplex_result[0]}")
        return

    print(
        f"Вектор коэффициентов при переменных в отскекающем ограничении Гомори : {gomory_cut[0]}"
    )
    print(f"Свободный член в отскекающем ограничении Гомори : {gomory_cut[1]}")


if __name__ == "__main__":
    main()
