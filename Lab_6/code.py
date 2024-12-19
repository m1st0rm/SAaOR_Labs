class HungarianAlgorithm:
    """
    Реализация алгоритма Венгерского (Куна-Мункреса) для решения задачи назначения.
    Алгоритм находит оптимальное паросочетание работников на задачи с минимальной общей стоимостью.

    Атрибуты:
        costs (list[list[int]]): 2D список, представляющий матрицу стоимости.
        size (int): Размер матрицы стоимости (предполагается, что матрица квадратная).
        a_values (list[int]): Временные значения для строк.
        b_values (list[int]): Временные значения для столбцов.
        a_temp (list[int]): Временные изменения для строк.
        b_temp (list[int]): Временные изменения для столбцов.
        a_change (list[int]): Изменения в значениях строк.
        b_change (list[int]): Изменения в значениях столбцов.
        equal_pairs (list[tuple[int, int]]): Список равных пар (строка, столбец), где a + b равно стоимости.
        lesser_pairs (list[tuple[int, int]]): Список меньших пар (строка, столбец), где a + b < стоимость.
        connections (list[list[int]]): Список смежности для двудольного графа.
        matching (list[tuple[int, int]]): Итоговые пары паросочетания.
        reachable_nodes (set[int]): Множество достижимых узлов в двудольном графе.
        row_marked (list[int]): Метки для строк в графе.
        col_marked (list[int]): Метки для столбцов в графе.
        min_difference (float): Минимальное различие, вычисленное во время корректировок.
        graph_network (list[list[int]]): Сеть двудольного графа.
        input_nodes (list[int]): Список индексов входных узлов.
        output_nodes (list[int]): Список индексов выходных узлов.
    """

    def __init__(self, costs: list[list[int]]):
        """
        Инициализация алгоритма Венгерского с заданной матрицей стоимости.

        Аргументы:
            costs (list[list[int]]): 2D список, представляющий матрицу стоимости.
        """
        self.costs = costs
        self.size = len(costs)
        self.a_values = [0] * self.size
        self.b_values = [
            min(row[col] for row in costs) for col in range(self.size)
        ]
        self.a_temp = []
        self.b_temp = []
        self.a_change = []
        self.b_change = []
        self.equal_pairs = []
        self.lesser_pairs = []
        self.connections = []
        self.matching = []
        self.reachable_nodes = set()
        self.row_marked = []
        self.col_marked = []
        self.min_difference = 0
        self.graph_network = []
        self.input_nodes = []
        self.output_nodes = []

    def setup_graph(self, data: list[list[int]], count: int) -> None:
        """
        Настройка двудольного графа для задачи назначения.

        Аргументы:
            data (list[list[int]]): Список смежности, представляющий граф.
            count (int): Количество узлов с каждой стороны двудольного графа.
        """
        self.graph_network = data
        self.input_nodes = [idx for idx in range(count)]
        self.output_nodes = [idx + count for idx in range(count)]

        self.graph_network.append(self.input_nodes[:])
        self.graph_network.append([])

        for output_node in self.output_nodes:
            self.graph_network[output_node].append(len(self.graph_network) - 1)

    def find_path(self, start: int, end: int) -> list[int] | None:
        """
        Находит путь в графе от начального узла к конечному узлу.

        Аргументы:
            start (int): Индекс начального узла.
            end (int): Индекс конечного узла.

        Возвращает:
            list[int] | None: Путь от start до end в виде списка узлов или None, если путь не найден.
        """
        if start == end:
            return [start]

        for connected in self.graph_network[start]:
            path = self.find_path(connected, end)
            if path is not None:
                return [start] + path

        return None

    def collect_all_reachable(self, start: int) -> set[int]:
        """
        Собирает все узлы, достижимые от начального узла.

        Аргументы:
            start (int): Индекс начального узла.

        Возвращает:
            set[int]: Множество всех достижимых узлов от начального узла.
        """
        reachable = set()

        for connected in self.graph_network[start]:
            sub_nodes = self.collect_all_reachable(connected)
            if sub_nodes is not None:
                reachable |= {connected} | sub_nodes

        return reachable

    def modify_graph(self) -> list[tuple[int, int]]:
        """
        Модифицирует граф, находя увеличивающие пути и корректируя паросочетание.

        Возвращает:
            list[tuple[int, int]]: Список кортежей, представляющих новое паросочетание.
        """
        start = len(self.graph_network) - 2
        end = len(self.graph_network) - 1

        path = self.find_path(start, end)

        while path is not None:
            self.graph_network[path[0]].remove(path[1])
            self.graph_network[path[-2]].remove(path[-1])

            for idx in range(1, len(path) - 2):
                self.graph_network[path[idx]].remove(path[idx + 1])
                self.graph_network[path[idx + 1]].append(path[idx])

            path = self.find_path(start, end)

        matches = []
        for output_node in self.output_nodes:
            for connected in self.graph_network[output_node]:
                if connected != len(self.graph_network) - 1:
                    matches.append((output_node, connected))

        return matches

    def init_values(self) -> list[tuple[int, int]] | None:
        """
        Инициализирует значения для алгоритма, оценивает пары и обрабатывает паросочетание.

        Возвращает:
            list[tuple[int, int]] | None: Список назначенных пар или None, если решение не найдено.
        """
        self.evaluate_pairs()
        return self.process_matching()

    def evaluate_pairs(self) -> None:
        """
        Оценивает пары (строка, столбец) на основе матрицы стоимости и значений a и b.
        Пары классифицируются как равные или меньшие в зависимости от условия a + b = стоимость или a + b < стоимость.
        """
        self.equal_pairs = []
        self.lesser_pairs = []

        for row in range(self.size):
            for col in range(self.size):
                cost = self.costs[row][col]
                if self.a_values[row] + self.b_values[col] == cost:
                    self.equal_pairs.append((row, col))
                elif self.a_values[row] + self.b_values[col] < cost:
                    self.lesser_pairs.append((row, col))
        self.build_connections()

    def build_connections(self) -> None:
        """
        Строит соединения между равными парами для двудольного графа.
        """
        self.connections = [[] for _ in range(self.size * 2)]

        for row, col in self.equal_pairs:
            self.connections[row].append(self.size + col)
        self.perform_matching()

    def perform_matching(self) -> None:
        """
        Выполняет процесс паросочетания, настраивая граф и модифицируя его.
        """
        self.setup_graph(self.connections, self.size)
        self.matching = self.modify_graph()
        self.process_matching()

    def process_matching(self) -> list[tuple[int, int]] | None:
        """
        Обрабатывает текущее паросочетание и пытается найти полное паросочетание.

        Возвращает:
            list[tuple[int, int]] | None: Результирующее паросочетание или None, если полное паросочетание не найдено.
        """
        if len(self.matching) == self.size:
            return self.produce_result()
        self.mark_nodes()
        return None

    def produce_result(self) -> list[tuple[int, int]]:
        """
        Производит финальный результат, преобразуя паросочетание в пары (источник, сток).

        Возвращает:
            list[tuple[int, int]]: Итоговые пары паросочетания.
        """
        result = []
        for sink, source in self.matching:
            result.append((source, sink - self.size))
        return result

    def mark_nodes(self) -> None:
        """
        Помечает узлы, которые достижимы, и классифицирует их в помеченные строки и столбцы.
        """
        self.reachable_nodes = self.collect_all_reachable(
            len(self.connections) - 2
        )
        self.classify_nodes()

    def classify_nodes(self) -> None:
        """
        Классифицирует достижимые узлы в помеченные строки и столбцы и корректирует значения a и b.
        """
        self.row_marked = [
            node for node in self.reachable_nodes if node < self.size
        ]
        self.col_marked = [
            node - self.size
            for node in self.reachable_nodes
            if node >= self.size
        ]
        self.adjust_values()

    def adjust_values(self) -> None:
        """
        Корректирует значения a и b на основе помеченных узлов и вычисляет минимальное различие.
        """
        self.a_change = [
            1 if i in self.row_marked else -1 for i in range(self.size)
        ]
        self.b_change = [
            -1 if j in self.col_marked else 1 for j in range(self.size)
        ]
        self.calculate_min_difference()

    def calculate_min_difference(self) -> None:
        """
        Вычисляет минимальное различие, необходимое для корректировки значений a и b.
        """
        differences = [
            (self.costs[row][col] - self.a_values[row] - self.b_values[col])
            / 2
            for row in range(self.size)
            if row in self.row_marked
            for col in range(self.size)
            if col not in self.col_marked
        ]
        self.min_difference = min(differences)
        self.apply_adjustments()

    def apply_adjustments(self) -> None:
        """
        Применяет вычисленные корректировки к значениям a и b и повторно оценивает пары.
        """
        self.a_temp = [
            self.a_values[i] + self.min_difference * self.a_change[i]
            for i in range(self.size)
        ]
        self.b_temp = [
            self.b_values[j] + self.min_difference * self.b_change[j]
            for j in range(self.size)
        ]
        self.a_values, self.b_values = self.a_temp, self.b_temp
        self.evaluate_pairs()

    def run(self) -> None:
        """
        Запускает алгоритм Венгерского и выводит результат и общую стоимость.
        """
        result = self.init_values()
        total_cost = sum(self.costs[row][col] for row, col in result)

        print(f"Сумма весов: {total_cost} \n")
        print(
            f"Искомое паросочетание, сумма весов которых минимальна: {', '.join(map(str, result))} \n"
        )
        print("Оптимальное решение задачи о назначениях:")
        for i in range(len(self.costs)):
            row = ""
            for j in range(len(self.costs[i])):
                if (i, j) in result:
                    row += f"\t{self.costs[i][j]}*"
                else:
                    row += f"\t{self.costs[i][j]}"
            print(row.strip())


if __name__ == "__main__":
    costs = [
        [7, 2, 1, 9, 4],
        [9, 6, 9, 5, 5],
        [3, 8, 3, 1, 8],
        [7, 9, 4, 2, 2],
        [8, 4, 7, 4, 8],
    ]
    hungarian = HungarianAlgorithm(costs)
    hungarian.run()
