#! /usr/bin/env python3

import sys
import re
import itertools
import numpy

inputs = dict((f"day{i+1}", f"inputs/{i+1}") for i in range(25))

def day1(input):
    numbers = {int(i) for i in input.split()}

    print(numpy.prod(next(x for x in itertools.product(numbers, repeat=2) if sum(x) == 2020)))
    print(numpy.prod(next(x for x in itertools.product(numbers, repeat=3) if sum(x) == 2020)))

    # O(n) and O(n²) solutions
    # print(next(x * (2020-x) for x in numbers if 2020-x in numbers))
    # print(next(x * y * (2020-x-y) for x in numbers for y in numbers if 2020-x-y in numbers))


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


def day3(input):
    lines = input.split()
    width = len(lines[0])

    def count_trees(xstep, ystep):
        return sum(1 for i, l in enumerate(lines[::ystep]) if (l[(xstep*i)%width] == '#'))

    print(count_trees(3, 1))
    print(numpy.prod([count_trees(x,y) for x,y in ((1,1), (3,1), (5,1), (7,1), (1,2))]))


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

        for k, v in passport_fields.items():
            print(f'{k}\t{validators[k](v)}\t{v}')
        print()

    print(count_part1)
    print(count_part2)


def day5(input):
    boarding_passes = input.split()

    ids = {
        int(re.sub(r'[BR]', '1', re.sub(r'[FL]', '0', boarding_pass)), 2)
        for boarding_pass in boarding_passes
    }

    print(max(ids))
    print(min(set(range(min(ids), max(ids))) - ids))


def solver(day):
    with open(inputs[day], "r") as f:
        globals()[day](f.read())

globals()[sys.argv[1]](*sys.argv[2:])
