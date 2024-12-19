import pprint
from collections import deque
from copy import deepcopy

import networkx as nx


# Создание графа с направленными рёбрами и их ёмкостями
graph = nx.DiGraph()
graph.add_edge("s", "a", capacity=3)
graph.add_edge("s", "b", capacity=2)
graph.add_edge("a", "b", capacity=2)
graph.add_edge("a", "t", capacity=1)
graph.add_edge("b", "t", capacity=2)


def labeling(flow_graph: nx.DiGraph, source: str):
    """
    Функция для пометки всех достижимых вершин в остаточном графе от истока
    с учётом остаточной ёмкости рёбер.

    Аргументы:
    flow_graph -- остаточный граф с рёбрами и их ёмкостями
    source -- исток (начальная вершина)

    Возвращает:
    visited -- множество всех посещённых вершин
    labels -- словарь меток, где ключ — вершина, значение — пара (родитель, текущая вершина)
    """
    queue = deque([source])
    visited = set([source])
    labels = {source: None}

    while queue:
        u = queue.popleft()
        for v in flow_graph.neighbors(u):
            if v not in visited and flow_graph[u][v]["capacity"] > 0:
                labels[v] = (u, v)
                visited.add(v)
                queue.append(v)

    return visited, labels


def ff_algo(graph: nx.DiGraph, source: str, sink: str) -> int:
    """
    Алгоритм Форда-Фалкерсона для нахождения максимального потока в поточном графе.

    Аргументы:
    graph -- ориентированный граф с рёбрами и ёмкостями
    source -- исток (начальная вершина)
    sink -- сток (конечная вершина)

    Возвращает:
    max_flow -- величина максимального потока
    flow -- словарь с потоком для каждого рёбра
    """
    # Добавление обратных рёбер с ёмкостью 0, если их ещё нет
    for edge in graph.edges():
        if (edge[1], edge[0]) not in graph.edges():
            graph.add_edge(edge[1], edge[0], capacity=0)
    max_flow = 0

    # Инициализация потока для всех рёбер
    flow = {edge: 0 for edge in graph.edges()}

    # Создание остаточного графа
    residual_graph = deepcopy(graph)
    for u, v in residual_graph.edges():
        residual_graph[u][v]["capacity"] = (
            graph[u][v]["capacity"] - flow[(u, v)] + flow[(v, u)]
        )

    while True:
        # Поиск увеличивающегося пути
        visited, labels = labeling(residual_graph, source)
        if sink not in visited:
            break

        # Восстановление пути от истока до стока
        path, v = [], sink
        while True:
            u, v = labels[v]
            path.append((u, v))
            v = u
            if v == source:
                break
        path.reverse()

        # Нахождение минимальной ёмкости вдоль пути
        path_capacity = min(graph[u][v]["capacity"] for u, v in path)

        # Обновление потока для рёбер вдоль пути
        flow_updates = {
            edge: (path_capacity if edge in path else 0)
            for edge in graph.edges()
        }

        # Обновление остаточного графа и потока
        for u, v in path:
            flow[(u, v)] += flow_updates[(u, v)]
            residual_graph[u][v]["capacity"] -= path_capacity
            residual_graph[v][u]["capacity"] += path_capacity
        max_flow += path_capacity

    return max_flow, flow


result = ff_algo(graph, "s", "t")
print(f"Мощность максимального потока: {result[0]}")
print("Максимальный поток:")
pprint.pprint(result[1])
