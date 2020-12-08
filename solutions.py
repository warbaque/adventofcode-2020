#! /usr/bin/env python3

import sys
import re
import itertools
import numpy
from functools import lru_cache

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


def solver(day):
    with open(inputs[day], "r") as f:
        globals()[day](f.read())

globals()[sys.argv[1]](*sys.argv[2:])
