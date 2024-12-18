from typing import List, Optional, Tuple


GRAPH: List[List[int]] = [
    [3],
    [3, 4],
    [3, 4, 5],
]

source_nodes: List[int] = []
sink_nodes: List[int] = []


def initialize_graph() -> None:
    """
    Инициализирует граф: добавляет дополнительные вершины s и t
    """
    for index, node in enumerate(GRAPH[:]):
        source_nodes.append(index)
        for neighbor in node:
            while neighbor >= len(GRAPH):
                GRAPH.append([])
            if neighbor not in sink_nodes:
                sink_nodes.append(neighbor)

    GRAPH.append(source_nodes[:])
    GRAPH.append([])

    for node in sink_nodes:
        GRAPH[node].append(len(GRAPH) - 1)


def find_path(start_node: int, end_node: int) -> Optional[List[int]]:
    """
    Ищет путь в графе от вершины start_node к узлу end_node.

    :param start_node: Начальная вершина.
    :param end_node: Конечная вершина.
    :return: Список вершин пути или None, если пути нет.
    """
    if start_node == end_node:
        return [start_node]

    for neighbor in GRAPH[start_node]:
        sub_path = find_path(neighbor, end_node)
        if sub_path is not None:
            return [start_node] + sub_path

    return None


def execute_matching_algorithm() -> List[Tuple[int, int]]:
    """
    Выполняет основной алгоритм обработки графа, находя пути от s к t
    и модифицируя граф.

    :return: Список оставшихся ребер в графе.
    """
    source_node = len(GRAPH) - 2
    sink_node = len(GRAPH) - 1

    path = find_path(source_node, sink_node)

    while path is not None:
        GRAPH[path[0]].remove(path[1])
        GRAPH[path[-2]].remove(path[-1])

        for node_index in range(1, len(path) - 2):
            GRAPH[path[node_index]].remove(path[node_index + 1])
            GRAPH[path[node_index + 1]].append(path[node_index])

        path = find_path(source_node, sink_node)

    remaining_edges = []
    for node in sink_nodes:
        for neighbor in GRAPH[node]:
            if neighbor != sink_node:
                remaining_edges.append((node, neighbor))

    return remaining_edges


def main() -> None:
    """
    Основная функция программы. Инициализирует граф и запускает алгоритм нахождения наибольшего паросочетания.
    """
    initialize_graph()
    print(execute_matching_algorithm())


if __name__ == "__main__":
    main()
