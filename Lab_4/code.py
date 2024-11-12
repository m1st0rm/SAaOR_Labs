from typing import List, Tuple


START_VERTEX = 0
END_VERTEX = 3
GRAPH = [  # each row represents a vertex and contains a list of pairs (neighbor_vertex, edge_weight)
    [(1, 4), (4, 2)],
    [(2, 10), (4, 5)],
    [(3, 11)],
    [],
    [(5, 3)],
    [(2, 4)],
]


def _topological_sort(
    graph: List[List[Tuple[int, int]]], current_vertex: int, state: List[int]
) -> List[int]:
    if state[current_vertex] == 1:
        raise Exception("Graph contains a cycle!")

    state[current_vertex] = 1
    sorted_vertices = []

    for neighbor, _ in graph[current_vertex]:
        if state[neighbor] != 2:
            sorted_vertices = (
                _topological_sort(graph, neighbor, state) + sorted_vertices
            )

    state[current_vertex] = 2

    return [current_vertex] + sorted_vertices


def topological_sort(
    graph: List[List[Tuple[int, int]]], start_vertex: int
) -> List[int]:
    state = [0] * len(graph)
    return _topological_sort(graph, start_vertex, state)


def _find_longest_path(
    graph: List[List[Tuple[int, int]]],
    sorted_vertices: List[int],
    start_index: int,
    end_index: int,
) -> Tuple[int, List[int]]:
    vertex_count = end_index - start_index

    if vertex_count == 0:
        return 0, [sorted_vertices[start_index]]

    available_vertices = {
        sorted_vertices[end_index]: (sorted_vertices[end_index], 0)
    }  # maps vertex to (next_vertex, distance)

    while end_index > start_index:
        end_index -= 1

        current_vertex = sorted_vertices[end_index]
        max_distance = 0

        for neighbor, weight in graph[sorted_vertices[end_index]]:
            if neighbor in available_vertices:
                possible_distance = available_vertices[neighbor][1] + weight
                if possible_distance > max_distance:
                    current_vertex = neighbor
                    max_distance = possible_distance

        if max_distance > 0:
            available_vertices[sorted_vertices[end_index]] = (
                current_vertex,
                max_distance,
            )

    starting_vertex = sorted_vertices[start_index]

    if starting_vertex not in available_vertices:
        raise Exception("No path to the destination")

    path = [starting_vertex]

    while available_vertices[starting_vertex][0] != starting_vertex:
        starting_vertex = available_vertices[starting_vertex][0]
        path.append(starting_vertex)

    return max_distance, path


def longest_path(
    graph: List[List[Tuple[int, int]]],
    sorted_vertices: List[int],
    start_vertex: int,
    end_vertex: int,
) -> Tuple[int, List[int]]:
    for i, vertex in enumerate(sorted_vertices):
        if vertex == start_vertex:
            start_index = i
            break
        if vertex == end_vertex:
            raise Exception("No path to the destination")

    for i, vertex in enumerate(sorted_vertices, start_index):
        if vertex == end_vertex:
            end_index = i
            break

    try:
        return _find_longest_path(
            graph, sorted_vertices, start_index, end_index
        )
    except UnboundLocalError:
        raise Exception("No path to the destination")


def main() -> None:
    sorted_vertices = topological_sort(GRAPH, START_VERTEX)
    print(longest_path(GRAPH, sorted_vertices, START_VERTEX, END_VERTEX))


if __name__ == "__main__":
    main()
