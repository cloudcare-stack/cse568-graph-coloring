"""CSE 568 Project 01: binary-genome graph coloring.

Requires DEAP (or the instructor's DEAP-compatible package).
run_ga returns the best normalized fitness as a float.
"""
import random
from deap import base, creator, tools


class AdjNode:
    """One neighbor in an adjacency linked list."""
    def __init__(self, vertex):
        self.vertex = vertex
        self.next = None


class Graph:
    def __init__(self, num_nodes, num_colors):
        self.n = num_nodes
        self.k = num_colors
        self.num_nodes = num_nodes
        self.num_colors = num_colors
        self.bits_per_color = max(1, (num_colors - 1).bit_length())
        self.graph = [None for _ in range(num_nodes)]
        self.adj = {node: [] for node in range(num_nodes)}
        self.adj_list = self.adj
        self.edges = []
        self._edge_set = set()

    def add_edge(self, u, v):
        if not (0 <= u < self.num_nodes and 0 <= v < self.num_nodes):
            raise ValueError('Edge endpoint outside graph')
        edge = (min(u, v), max(u, v))
        if edge not in self._edge_set:
            self._edge_set.add(edge)
            self.edges.append(edge)
            self.adj[u].append(v)
            node = AdjNode(v)
            node.next = self.graph[u]
            self.graph[u] = node
            if u != v:
                self.adj[v].append(u)
                node = AdjNode(u)
                node.next = self.graph[v]
                self.graph[v] = node


def init_graph(file_path):
    """Read node count, color count, then undirected edge pairs."""
    with open(file_path, encoding='utf-8') as stream:
        lines = [line.strip() for line in stream if line.strip()]
    if len(lines) < 2:
        raise ValueError('Missing graph header')
    n, k = int(lines[0]), int(lines[1])
    if n < 0 or k < 1:
        raise ValueError('Need nonnegative node count and positive color count')
    graph = Graph(n, k)
    for line in lines[2:]:
        u, v = map(int, line.split())
        graph.add_edge(u, v)
    return graph


def decode_individual(graph, individual):
    """Decode fixed-width, most-significant-bit-first colors; invalid => None."""
    width = graph.bits_per_color
    if len(individual) != graph.num_nodes * width:
        return None
    colors = []
    for offset in range(0, len(individual), width):
        color = 0
        for bit in individual[offset:offset + width]:
            if bit not in (0, 1):
                return None
            color = (color << 1) | int(bit)
        if color >= graph.num_colors:
            return None
        colors.append(color)
    return colors


def eval_graph(graph, individual):
    """Reward each satisfied undirected edge exactly once."""
    colors = decode_individual(graph, individual)
    if colors is None:
        return (0,)
    satisfied = sum(colors[u] != colors[v] for u, v in graph.edges)
    return (satisfied / len(graph.edges) if graph.edges else 1.0,)


def _evolve(file_path, population_size=300, ngen=200, cxpb=0.8, mutpb=0.3):
    """Evolve binary colorings with tournament selection and elitism."""
    if population_size < 1 or ngen < 0:
        raise ValueError('Invalid population size or generation count')
    if not (0 <= cxpb <= 1 and 0 <= mutpb <= 1):
        raise ValueError('Probabilities must be between zero and one')
    graph = init_graph(file_path)
    if not hasattr(creator, 'GraphColorFitness'):
        creator.create('GraphColorFitness', base.Fitness, weights=(1.0,))
    if not hasattr(creator, 'GraphColorIndividual'):
        creator.create('GraphColorIndividual', list, fitness=creator.GraphColorFitness)
    width = graph.bits_per_color
    size = graph.num_nodes * width

    def make_individual():
        # Start from valid colors, avoiding unused binary encodings.
        bits = []
        for _ in range(graph.num_nodes):
            color = random.randrange(graph.num_colors)
            bits.extend((color >> shift) & 1 for shift in reversed(range(width)))
        return creator.GraphColorIndividual(bits)

    def repair(individual):
        # Crossover and bit mutation can produce unused color codes.
        for offset in range(0, size, width):
            value = 0
            for bit in individual[offset:offset + width]:
                value = (value << 1) | bit
            if value >= graph.num_colors:
                value = random.randrange(graph.num_colors)
                individual[offset:offset + width] = [
                    (value >> shift) & 1 for shift in reversed(range(width))]

    toolbox = base.Toolbox()
    toolbox.register('evaluate', eval_graph, graph)
    toolbox.register('select', tools.selTournament, tournsize=3)
    toolbox.register('mate', tools.cxTwoPoint)
    toolbox.register('mutate', tools.mutFlipBit, indpb=1.0 / max(1, size))
    population = [make_individual() for _ in range(population_size)]
    hall = tools.HallOfFame(1)
    for individual in population:
        individual.fitness.values = toolbox.evaluate(individual)
    hall.update(population)
    for _ in range(ngen):
        if hall[0].fitness.values[0] == 1.0:
            break
        offspring = [toolbox.clone(ind) for ind in
                     toolbox.select(population, population_size)]
        for first, second in zip(offspring[::2], offspring[1::2]):
            if size >= 2 and random.random() < cxpb:
                toolbox.mate(first, second)
        for individual in offspring:
            if random.random() < mutpb:
                toolbox.mutate(individual)
            repair(individual)
            individual.fitness.values = toolbox.evaluate(individual)
        offspring[0] = toolbox.clone(hall[0])
        population = offspring
        hall.update(population)
    best = toolbox.clone(hall[0])
    return best, best.fitness.values


def run_ga(file_path, population_size=300, ngen=200, cxpb=0.8, mutpb=0.3):
    """Return the best normalized fitness as a numeric value."""
    _, fitness = _evolve(file_path, population_size, ngen, cxpb, mutpb)
    return float(fitness[0])


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('graph_file')
    parser.add_argument('--seed', type=int, default=None)
    args = parser.parse_args()
    random.seed(args.seed)
    best, fitness = _evolve(args.graph_file)
    graph = init_graph(args.graph_file)
    print('Colors:', decode_individual(graph, best))
    print(f'Satisfied edges: {round(fitness[0] * len(graph.edges))}/{len(graph.edges)}')
    print(f'Fitness: {fitness[0]:.6f}')
