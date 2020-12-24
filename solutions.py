#! /usr/bin/env python3

import sys
import re
import itertools
import numpy
import time
from functools import lru_cache
from collections import deque, defaultdict, Counter

inputs = dict((f"day{i+1}", f"inputs/{i+1}") for i in range(25))


# https://adventofcode.com/2020/day/1
def day1(input):
    numbers = {int(i) for i in input.split()}
    n = sorted(numbers) # speeds up quadratic solution

    print(next(x * (2020-x) for x in numbers if 2020-x in numbers))
    print(next(x * y * (2020-x-y) for x in n for y in n if 2020-x-y in numbers))


# https://adventofcode.com/2020/day/2
def day2(input):
    profiles = input.split('\n')
    r = re.compile(r'(\d+)-(\d+) (\S): (\S+)')

    count_part1 = 0
    count_part2 = 0
    for password_profile in profiles:
        lo, hi, c, pw = r.match(password_profile).groups()
        count_part1 += (int(lo) <= pw.count(c) <= int(hi))
        count_part2 += (pw[int(lo)-1] == c) ^ (pw[int(hi)-1] == c)

    print(count_part1)
    print(count_part2)


# https://adventofcode.com/2020/day/3
def day3(input):
    lines = input.split()
    width = len(lines[0])

    def count_trees(xstep, ystep):
        return sum(1 for i, l in enumerate(lines[::ystep]) if (l[(xstep*i)%width] == '#'))

    print(count_trees(3, 1))
    print(numpy.prod([count_trees(x,y) for x,y in ((1,1), (3,1), (5,1), (7,1), (1,2))]))


# https://adventofcode.com/2020/day/4
def day4(input):
    passports = input.split('\n\n')

    required_fields = {'byr', 'iyr', 'eyr', 'hgt', 'hcl', 'ecl', 'pid'}

    def hgt(v):
        r = re.compile(r'^(\d+)(in|cm)?$')
        height, unit = r.match(v).groups()
        return {
            'cm': 150 <= int(height) <= 193,
            'in':  59 <= int(height) <=  76,
        }.get(unit, False)

    validators = {
        'byr': lambda v: 1920 <= int(v) <= 2002,
        'iyr': lambda v: 2010 <= int(v) <= 2020,
        'eyr': lambda v: 2020 <= int(v) <= 2030,
        'hgt': hgt,
        'hcl': lambda v: re.compile(r'^#[\da-z]{6}$').match(v) is not None,
        'ecl': lambda v: v in {'amb', 'blu', 'brn', 'gry', 'grn', 'hzl', 'oth'},
        'pid': lambda v: re.compile(r'^\d{9}$').match(v) is not None,
        'cid': lambda v: True,
    }

    count_part1 = 0
    count_part2 = 0
    for pp in passports:
        passport_fields = dict(pair.split(':') for pair in pp.split())
        missing_fields = required_fields - passport_fields.keys()
        count_part1 += (not missing_fields)
        count_part2 += (not missing_fields and all(validators[k](v) for k, v in passport_fields.items()))

        # for k, v in passport_fields.items():
        #     print(f'{k}\t{validators[k](v)}\t{v}')
        # print()

    print(count_part1)
    print(count_part2)


# https://adventofcode.com/2020/day/5
def day5(input):
    boarding_passes = input.split()

    ids = {
        int(re.sub(r'[BR]', '1', re.sub(r'[FL]', '0', boarding_pass)), 2)
        for boarding_pass in boarding_passes
    }

    print(max(ids))
    print(min(set(range(min(ids), max(ids))) - ids))


# https://adventofcode.com/2020/day/6
def day6(input):
    answer_sets = input.split('\n\n')

    print(sum(len(set(answers.replace('\n', ''))) for answers in answer_sets))
    print(sum(len(set.intersection(*[set(a) for a in answers.split()])) for answers in answer_sets))


# https://adventofcode.com/2020/day/7
def day7(input):
    bag_regulations = input.strip().split('\n')

    r = re.compile(r'(\d+) (.+?) bags?')
    bags = {}
    for regulation in bag_regulations:
        parent, children = regulation.split(' bags contain ')
        bags[parent] = {child: int(n) for n, child in r.findall(children)}

    @lru_cache
    def contains(parent, bag):
        return bag in bags[parent] or any(contains(k, bag) for k in bags[parent])

    @lru_cache
    def count_children(bag):
        return sum(v * (count_children(k) + 1) for k, v in bags[bag].items())

    print(sum(contains(k, 'shiny gold') for k in bags))
    print(count_children('shiny gold'))


# https://adventofcode.com/2020/day/8
def day8(input):
    instructions = input.strip().split('\n')

    r = re.compile(r'(?P<op>\S+) (?P<arg>[+-]\d+)')

    def make_instruction(match):
        return (
            match['op'],
            int(match['arg'])
        )

    program = [make_instruction(r.match(i)) for i in instructions]

    def acc(program_state, x):
        program_state['accumulator'] += x
        program_state['instruction'] += 1

    def nop(program_state, _):
        program_state['instruction'] += 1

    def jmp(program_state, x):
        program_state['instruction'] += x

    def run_until_loop(program):
        program_state = {
            'instruction': 0,
            'accumulator': 0,
            'done': False,
        }

        visited = set()
        while program_state['instruction'] not in visited and not program_state['done']:
            instruction = program_state['instruction']
            visited.add(instruction)

            op, arg = program[instruction]
            {
                'acc': acc,
                'nop': nop,
                'jmp': jmp,
            }[op](program_state, arg)
            program_state['done'] = program_state['instruction'] >= len(program)

        return program_state

    def alternate_programs(program):
        swaps = {'jmp': 'nop', 'nop': 'jmp'}
        for i, instruction in enumerate(program):
            op, arg = instruction
            if op in swaps:
                new_program = program.copy()
                new_program[i] = (swaps[op], arg)
                yield new_program

    print(run_until_loop(program)['accumulator'])
    print(next(
        program_state['accumulator']
            for program_state in (
                run_until_loop(p) for p in alternate_programs(program)
            ) if program_state['done']
        ))


# https://adventofcode.com/2020/day/9
def day9(input):
    data = [int(i) for i in input.split()]

    def invalid_number():
        xmas = deque(data[:25], maxlen=25)
        for i in data[25:]:
            sums = {sum(x) for x in itertools.product(xmas, repeat=2)}
            if i not in sums:
                return i
            xmas.append(i)

    def encryption_weakness(invalid_number):
        xmas = deque()
        for i in data:
            xmas.append(i)
            while sum(xmas) > invalid_number:
                xmas.popleft()
            if len(xmas) >= 2 and sum(xmas) == invalid_number:
                return min(xmas) + max(xmas)

    print(invalid_number())
    print(encryption_weakness(invalid_number()))


# https://adventofcode.com/2020/day/10
def day10(input):
    adapters = sorted([int(x) for x in input.split()])
    joltages = adapters + [max(adapters) + 3]

    def diffs():
        d = [j1 - j0 for j0, j1 in zip([0] + joltages, joltages)]
        return d.count(1) * d.count(3)

    def possible_combinations():
        dp = deque([(0, 1)], maxlen=3)
        for joltage in joltages:
            ways = sum(w for j, w in dp if (joltage - j <= 3))
            dp.append((joltage, ways))
        return dp[2][1]

    print(diffs())
    print(possible_combinations())


# https://adventofcode.com/2020/day/11
def day11(input):
    seatmap = input.split()

    height = len(seatmap)
    width = len(seatmap[0])

    def adjacent(seatmap, x, y, i, j):
        if (0 <= x+i < width) and (0 <= y+j < height):
            return seatmap[y+j][x+i] == '#'
        return False

    def visible(seatmap, x, y, i, j):
        while True:
            x += i
            y += j
            if not ((0 <= x < width) and (0 <= y < height)):
                break
            if seatmap[y][x] == '.':
                continue
            return seatmap[y][x] == '#'
        return False

    def new_state(seatmap, x, y, check, max_neighbours):
        seat = seatmap[y][x]
        neighbours = sum(
            check(seatmap, x, y, i, j)
            for i, j in itertools.product([-1, 0, 1], repeat=2)
            if (i or j)
        )
        if seat == 'L' and neighbours == 0:
            return '#'
        if seat == '#' and neighbours >= max_neighbours:
            return 'L'
        return seat

    def run(seatmap, check, max_neighbours):
        seatmap = seatmap.copy()
        while True:
            for row in seatmap:
                print(row, file=sys.stderr)
            time.sleep(1/120)
            new = [
                ''.join(
                    new_state(seatmap, x, y, check, max_neighbours)
                    for x in range(width)
                ) for y in range(height)
            ]
            if (seatmap == new):
                break
            seatmap = new

        return sum(l.count('#') for l in seatmap)

    part1 = run(seatmap, check=adjacent, max_neighbours=4)
    part2 = run(seatmap, check=visible, max_neighbours=5)
    print(part1)
    print(part2)


# https://adventofcode.com/2020/day/12
def day12(input):
    instructions = input.strip().split('\n')

    r = re.compile(r'(?P<action>[A-Z])(?P<value>\d+)')

    class Point:
        def __init__(self, x, y):
            self.p = complex(x, y)
        def move(self, direction):
            self.p += direction
        def turn(self, rotation):
            self.p *= rotation
        def manhattan(self):
            return int(abs(self.p.real) + abs(self.p.imag))

    def part1():
        ship = Point(0, 0)
        direction = Point(1, 0)

        actions = {
            'N': lambda y: ship.move(complex(0,  y)),
            'S': lambda y: ship.move(complex(0, -y)),
            'E': lambda x: ship.move(complex( x, 0)),
            'W': lambda x: ship.move(complex(-x, 0)),
            'L': lambda v: direction.turn(complex(0,  1) ** (v // 90)),
            'R': lambda v: direction.turn(complex(0, -1) ** (v // 90)),
            'F': lambda v: ship.move(v * direction.p),
        }

        for instruction in instructions:
            match = r.match(instruction)
            actions[match['action']](int(match['value']))

        return ship.manhattan()

    def part2():
        ship = Point(0, 1)
        waypoint = Point(10, 1)

        part2_actions = {
            'N': lambda y: waypoint.move(complex(0,  y)),
            'S': lambda y: waypoint.move(complex(0, -y)),
            'E': lambda x: waypoint.move(complex( x, 0)),
            'W': lambda x: waypoint.move(complex(-x, 0)),
            'L': lambda v: waypoint.turn(complex(0,  1) ** (v // 90)),
            'R': lambda v: waypoint.turn(complex(0, -1) ** (v // 90)),
            'F': lambda v: ship.move(v * waypoint.p)
        }

        for instruction in instructions:
            match = r.match(instruction)
            part2_actions[match['action']](int(match['value']))

        return ship.manhattan()

    print(part1())
    print(part2())


# https://adventofcode.com/2020/day/13
def day13(input):
    notes = input.split()
    buses = [(i, int(bus)) for i, bus in enumerate(notes[1].split(',')) if bus != 'x']

    def part1():
        t0 = int(notes[0])
        wait_times = [(bus, (bus - t0) % bus) for _, bus in buses]
        return numpy.prod(min(wait_times, key=lambda x: x[1]))

    def part2():
        t0, lcm = 0, 1
        for i, bus in buses:
            k = (-(t0 + i) * pow(lcm, -1, bus)) % bus
            t0 += k * lcm
            lcm *= bus
        return t0

    print(part1())
    print(part2())


# https://adventofcode.com/2020/day/14
def day14(input):
    initialization = input.strip().split('\n')

    r = re.compile(r'(?P<op>mask|mem)(?:\[(?P<address>\d+)\])? = (?P<value>\S+)')

    part1_memory = {}
    part2_memory = {}

    for l in initialization:
        op, address, value = r.match(l).groups()
        if op == 'mask':
            x0 = int(value.replace('X', '0'), 2)
            x1 = int(value.replace('X', '1'), 2)

            unset = ~(x0^x1)
            bits = [1 << i for i, x in enumerate(reversed(value)) if x == "X"]
            masks = [0]
            for bit in bits:
                masks += [(m | bit) for m in masks]

        else:
            address = int(address)
            value = int(value)

            part1_memory[address] = (value | x0) & x1

            address = (address & unset) | x0
            for m in masks:
                part2_memory[address | m] = value

    print(sum(part1_memory.values()))
    print(sum(part2_memory.values()))


# https://adventofcode.com/2020/day/15
def day15(input):
    numbers = [int(i) for i in input.split(',')]

    def nth_number(turns):
        age = [-1] * turns
        for t, n in enumerate(numbers):
            age[n] = t

        for t in range(len(numbers)-1, turns-1):
            seen = (0 if age[n] < 0 else t - age[n])
            age[n] = t
            n = seen
        return n

    print(nth_number(2020))
    print(nth_number(30000000))


# https://adventofcode.com/2020/day/16
def day16(input):
    ticket_data = input.strip().split('\n\n')

    r = re.compile(r'(\d+)-(\d+)')
    def parse_rule(rule):
        field, ranges = rule.split(': ')
        return field, [tuple(map(int, m)) for m in r.findall(ranges)]

    rules = dict(parse_rule(rule) for rule in ticket_data[0].split('\n'))
    our_ticket = list(map(int, ticket_data[1].split('your ticket:\n')[1].split(',')))
    other_tickets = [list(map(int, t.split(','))) for t in ticket_data[2].split('nearby tickets:\n')[1].split()]

    @lru_cache
    def valid_fields(value):
        return {
            field for field, ranges in rules.items()
            if any(mi <= value <= ma for mi, ma in ranges)
        }

    def part1():
        return sum([v for values in other_tickets for v in values if not valid_fields(v)])

    def part2():
        options = [[valid_fields(v) for v in values] for values in other_tickets]

        possible = {
            i: set.intersection(*(fields[i] for fields in options if all(fields)))
            for i in range(len(rules))
        }

        solved_fields = {}
        for i in sorted(possible, key=lambda i: len(possible[i])):
            solved_fields[i] = min(possible[i] - set(solved_fields.values()))

        departure_values = [our_ticket[k] for k, v in solved_fields.items() if v.startswith('departure')]
        return numpy.prod(departure_values)

    print(part1())
    print(part2())


# https://adventofcode.com/2020/day/17
def day17(input):
    initial_slice = input.split()

    def neighbour_coordinates(p):
        return [tuple(a+b for a, b in zip(p, t)) for t in itertools.product([-1, 0, 1], repeat=len(p)) if any(t)]

    def simulate(dimensions):
        active_coordinates = {(x,y)+(0,)*(dimensions-2) for y, l in enumerate(initial_slice) for x, c in enumerate(l) if c == '#'}
        for _ in range(6):
            total_neighbours = Counter(p for coordinate in active_coordinates for p in neighbour_coordinates(coordinate))
            active_coordinates = {p for p, n in total_neighbours.items() if (p in active_coordinates and n == 2) or n == 3}
        return len(active_coordinates)

    print(simulate(3))
    print(simulate(4))


# https://adventofcode.com/2020/day/18
def day18(input):
    equations = input.strip().split('\n')

    def calc(*args):
        a, op, b = args
        return str(int(a) + int(b) if op == '+' else int(a) * int(b))

    def regex_replace(pattern, fn, s):
        r = re.compile(pattern)
        while m := r.search(s):
            b, e = m.span()
            s = s[:b] + fn(*m.groups()) + s[e:]
        return s

    def part1(eq):
        return regex_replace(r'(\d+) ([*+]) (\d+)', calc, eq)

    def part2(eq):
        eq = regex_replace(r'(\d+) ([+]) (\d+)', calc, eq)
        eq = regex_replace(r'(\d+) ([*]) (\d+)', calc, eq)
        return eq

    def solve(eq, fn):
        return int(fn(regex_replace(r'\(([^()]*)\)', fn, eq)))

    print(sum(solve(equation, part1) for equation in equations))
    print(sum(solve(equation, part2) for equation in equations))


# https://adventofcode.com/2020/day/19
def day19(input):
    raw_rules, messages = input.strip().split('\n\n')

    rules = dict(tuple(r.split(': ')) for r in raw_rules.split('\n'))
    part1_rules = rules.copy()
    part2_rules = rules.copy()
    part2_rules.update({'8': '42 | 42 8', '11': '42 31 | 42 11 31'})

    def count_matches(rules, messages):
        def recursive_regex(rule, d=0):
            if d > 20:
                return ""
            if rule.startswith('"'):
                return rule.strip('"')
            return f"(?:{'|'.join(''.join(recursive_regex(rules[k], d+1) for k in r.split()) for r in rule.split('|'))})"

        r = re.compile(f"^{recursive_regex(rules['0'])}$")
        return sum(1 for message in messages.split('\n') if r.match(message))

    print(count_matches(part1_rules, messages))
    print(count_matches(part2_rules, messages))


# https://adventofcode.com/2020/day/20
def day20(input):
    raw_tile_data = input.strip().split('\n\n')

    class Tile:
        def parser(raw):
            r = re.compile(r'Tile (?P<id>\d+):')
            return Tile(int(r.match(raw[0])['id']), raw[1:])
        def __init__(self, tileid, data):
            self.id = tileid
            self.data = data
            self.top = data[0]
            self.bottom = data[-1]
            self.right = ''.join(row[-1] for row in data)
            self.left = ''.join(row[0] for row in data)
            self.edges = (self.top, self.bottom, self.right, self.left)
            self.center = [row[1:-1] for row in data[1:-1]]
        def edge_variations(self):
            return [edge for e in self.edges for edge in (e, e[::-1])]
        def __str__(self):
            return '\n'.join(' '.join(row) for row in self.data)
        def rotated(self):
            return Tile(self.id, [''.join(row) for row in zip(*reversed(self.data))])
        def flipped(self):
            return Tile(self.id, [row[::-1] for row in self.data])
        def all_variations(self):
            tile = self
            for i in range(4):
                yield tile
                yield tile.flipped()
                tile = tile.rotated()
        def __hash__(self):
            return hash(self.id)
        def __eq__(self, other):
            return self.id == other.id

    tiles = [Tile.parser(raw.split('\n')) for raw in raw_tile_data]

    def corner_tiles():
        edge_counts = Counter(edge for tile in tiles for edge in tile.edge_variations())
        return [tile for tile in tiles if sum(edge_counts[edge] for edge in tile.edges) == 6]

    tiles_with_edge = {}
    for tile in tiles:
        for edge in tile.edge_variations():
            tiles_with_edge.setdefault(edge, set()).add(tile)

    def get_neighbour(tile, direction):
        adjacent= {'left': 'right', 'right': 'left', 'top': 'bottom', 'bottom': 'top'}
        edge = getattr(tile, direction)
        if neighbour := tiles_with_edge[edge] - {tile}:
            return next(other for other in next(iter(neighbour)).all_variations() if edge == getattr(other, adjacent[direction]))

    def dfs(tile, p, visited={}):
        if (p in visited):
            return visited
        visited[p] = tile
        neigbours = [(offset, get_neighbour(tile, direction)) for offset, direction in ((-1j, 'top'), (1, 'right'), (1j, 'bottom'), (-1, 'left'))]
        for offset, neighbour in neigbours:
            if neighbour:
                dfs(neighbour, p+offset, visited)
        return visited

    corner = next(tile for tile in corner_tiles()[0].all_variations() if not ((tiles_with_edge[tile.top] | tiles_with_edge[tile.left]) - {tile}))
    visited = dfs(corner, 0)
    x_max = max(int(v.real) for v in visited.keys())
    y_max = max(int(v.imag) for v in visited.keys())

    world = []
    for y in range(y_max+1):
        for x in zip(*[visited[x + y*1j].center for x in range(x_max+1)]):
            print(x, file=sys.stderr)
            world.append(''.join(x))
        print(file=sys.stderr)

    def monster_at(world, x, y, sea_monster):
        coordinates = set()
        if (len(sea_monster) + y > len(world)) or (len(sea_monster[0]) + x > len(world[0])):
            return set()
        for j, row in enumerate(sea_monster):
            for i, c in enumerate(row):
                if c == '#':
                    coordinates.add((x+i, y+j))
                if not (c == ' ' or world[y+j][x+i] == '#'):
                    return set()
        return coordinates

    sea_monsters = [monster.data for monster in Tile('sea monster', [
            '                  # ',
            '#    ##    ##    ###',
            ' #  #  #  #  #  #   ',
        ]).all_variations()]

    monster_coordinates = set.union(*[
        monster_at(world, x, y, sea_monster)
        for sea_monster in sea_monsters
        for y in range(len(world))
        for x in range(len(world[0]))])

    for x, y in monster_coordinates:
        world[y] = world[y][:x] + 'O' + world[y][x+1:]

    for y in world:
        print(y, file=sys.stderr)

    print(numpy.prod([tile.id for tile in corner_tiles()]))
    print(sum(y.count('#') for y in world))


# https://adventofcode.com/2020/day/21
def day21(input):
    ingredients_list = input.strip().split('\n')

    r = re.compile(r'(?P<ingredients>.*) \(contains (?P<allergens>.*)\)')

    ingredient_counts = Counter()
    possible = {}
    for l in ingredients_list:
        m = r.match(l)
        ingredients = set(m['ingredients'].split())
        ingredient_counts.update(ingredients)
        for allergen in m['allergens'].split(', '):
            if allergen in possible:
                possible[allergen] &= ingredients
            else:
                possible[allergen] = ingredients.copy()

    def part1():
        non_allergenic = ingredient_counts.keys() - set.union(*possible.values())
        return sum(ingredient_counts[ingredient] for ingredient in non_allergenic)

    def part2():
        solved = {}
        for k in sorted(possible, key=lambda k: len(possible[k])):
            solved[min(possible[k] - set(solved.keys()))] = k
        return ','.join(k for k in sorted(solved, key=solved.get))

    print(part1())
    print(part2())


# https://adventofcode.com/2020/day/22
def day22(input):
    decks = input.strip().split('\n\n')
    deck1 = [int(x) for x in decks[0].split('\n')[1:]]
    deck2 = [int(x) for x in decks[1].split('\n')[1:]]

    def score(deck):
        return sum((i+1) * x for i, x in enumerate(reversed(deck)))

    def part1(deck1, deck2):
        while deck1 and deck2:
            x1 = deck1.pop(0)
            x2 = deck2.pop(0)
            if x1 > x2:
                deck1 += (x1, x2)
            else:
                deck2 += (x2, x1)
        return max(score(deck1), score(deck2))

    def recursive_combat(deck1, deck2):
        cache = set()
        while deck1 and deck2:
            state = (tuple(deck1), tuple(deck2))
            if state in cache:
                break
            cache.add(state)

            x1 = deck1.pop(0)
            x2 = deck2.pop(0)

            if x1 > len(deck1) or x2 > len(deck2):
                winner = 1 if x1 > x2 else 2
            else:
                s1, _ = recursive_combat(deck1[:x1], deck2[:x2])
                winner = 1 if s1 else 2

            if winner == 1:
                deck1 += (x1, x2)
            else:
                deck2 += (x2, x1)

        return deck1, deck2

    def part2(deck1, deck2):
        deck1, deck2 = recursive_combat(deck1, deck2)
        return max(score(deck1), score(deck2))

    print(part1(deck1.copy(), deck2.copy()))
    print(part2(deck1.copy(), deck2.copy()))


# https://adventofcode.com/2020/day/23
def day23(input):
    initial_cups = [int(x) for x in input.strip()]

    def cupgame(cups, moves):
        last = max(cups)
        prev = lambda cup: cup - 1 if cup > 1 else last

        nex = [i+1 for i in range(len(cups) + 1)]
        cup = cups[0]
        for label in reversed(cups):
            nex[label] = cup
            cup = label

        for _ in range(moves):
            three = nex[cup], nex[nex[cup]], nex[nex[nex[cup]]]
            nex[cup] = nex[three[-1]]

            dst = prev(cup)
            while dst in three:
                dst = prev(dst)

            nex[dst], nex[three[-1]] = three[0], nex[dst]
            cup = nex[cup]

        cup = nex[1]
        while cup != 1:
            yield cup
            cup = nex[cup]

    def part1():
        cup_iter = cupgame(initial_cups, 100)
        return ''.join(str(x) for x in cup_iter)

    def part2():
        initial_cups.extend(range(max(initial_cups) + 1, 1_000_000 + 1))
        cup_iter = cupgame(initial_cups, 10_000_000)
        return next(cup_iter) * next(cup_iter)

    print(part1())
    print(part2())


# https://adventofcode.com/2020/day/24
def day24(input):
    tiles = input.strip().split()

    direction = {
        'e':   1+0j,
        'ne':  0+1j,
        'nw': -1+1j,
        'w':  -1+0j,
        'sw':  0-1j,
        'se':  1-1j,
    }
    r = re.compile(r'([ns]?[ew])')
    flips = Counter(sum(direction[d] for d in r.findall(tile)) for tile in tiles)
    blacks = {k for k, v in flips.items() if v % 2}

    def neighbour_coordinates(p):
        return [p + t for t in direction.values()]

    def simulate(blacks, days):
        for _ in range(days):
            total_neighbours = Counter(p for coordinate in blacks for p in neighbour_coordinates(coordinate))
            blacks = {p for p, n in total_neighbours.items() if (p in blacks and n == 1) or n == 2}
        return blacks

    print(len(blacks))
    print(len(simulate(blacks, 100)))


def profiler(method):
    def wrapper(*arg, **kw):
        t0 = time.time()
        ret = method(*arg, **kw)
        print(f'[{method.__name__}] {time.time()-t0:2.5f} sec', file=sys.stderr)
        return ret
    return wrapper

@profiler
def solver(day):
    with open(inputs[day], "r") as f:
        globals()[day](f.read())

globals()[sys.argv[1]](*sys.argv[2:])
