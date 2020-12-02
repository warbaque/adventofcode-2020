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

def solver(day):
    with open(inputs[day], "r") as f:
        globals()[day](f.read())

globals()[sys.argv[1]](*sys.argv[2:])
