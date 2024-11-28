
def even() -> list:
    return [2 * i for i in range(1, 51)]
    # if all(item % 2 == 0 for item in set):
    #     return [2 * i for i in range(1, 51)]
    # return []

def odd() -> list:
    return [2 * i + 1 for i in range(0, 50)]
    # if all(item % 2 == 1 for item in set):
    #     return [2 * i + 1 for i in range(0, 50)]
    # return []


def btw(set = {}, start = None, end = None) -> list:
    if start and end:
        maxi = end
        mini = start
    else:
        maxi = max(set)
        mini = min(set)
    return list(range(mini, maxi + 1))


def primes() -> list:
    prime = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97}
    return list(prime)
    # if set <= prime:
    #     return list(prime)
    # return []

def nonprimes() -> list:
    return [1, 4, 6, 8, 9, 10, 12, 14, 15, 16, 18, 20, 21, 22, 24, 25, 26, 27, 28, 30, 32, 33, 34, 35, 36, 38, 39, 40, 42, 44, 45, 46, 48, 49, 50, 51, 52, 54, 55, 56, 57, 58, 60, 62, 63, 64, 65, 66, 68, 69, 70, 72, 74, 75, 76, 77, 78, 80, 81, 82, 84, 85, 86, 87, 88, 90, 91, 92, 93, 94, 95, 96, 98, 99, 100]
    non_prime_set = {1, 4, 6, 8, 9, 10, 12, 14, 15, 16, 18, 20, 21, 22, 24, 25, 26, 27, 28, 30, 32, 33, 34, 35, 36, 38, 39, 40, 42, 44, 45, 46, 48, 49, 50, 51, 52, 54, 55, 56, 57, 58, 60, 62, 63, 64, 65, 66, 68, 69, 70, 72, 74, 75, 76, 77, 78, 80, 81, 82, 84, 85, 86, 87, 88, 90, 91, 92, 93, 94, 95, 96, 98, 99, 100}
    if set <= non_prime_set:
        return list(non_prime_set)
    return []

def same(set: set) -> list:
    return list(set)

def twodigit() -> list:
    return list(range(10, 100))
    # two_digit_numbers = set(range(10, 100))
    # if set <= two_digit_numbers:
    #     return list(two_digit_numbers)
    # return []

def onedigit() -> list:
    return list(range(1, 10))
    # one_digit_numbers = set(range(1, 10))
    # if set <= one_digit_numbers:
    #     return list(one_digit_numbers)
    # return []

def multiples(num: int) -> list:
    return list(range(num, 101, num))
    # possible = set(range(1, 101, num))
    # if set <= possible:
    #     return list(possible)
    # return []

def not_multiples(n: int) -> list:
    return [num for num in range(1,101) if num % n != 0]

def contains(digit: int):
    digit_str = str(digit)
    result_set = [num for num in range(1, 101) if digit_str in str(num)]
    return result_set

def contains_even_digit(num):
    even_digits = {'2', '4', '6', '8'} 
    return any(digit in even_digits for digit in str(num))


def contains_even():
    result_set = [num for num in range(1, 101) if contains_even_digit(num)]
    return result_set

def starts(digit: int):
    digit_str = str(digit)
    results_set = [num for num in range(1, 101) if str(num).startswith(digit_str)]
    return results_set

def starts_even():
    return [num for num in range(1, 101) if int(str(num)[0]) % 2 == 0]

def starts_odd():
    return [num for num in range(1, 101) if int(str(num)[0]) % 2 != 0]

def ends(digit: int):
    return [num for num in range(1, 101) if num % 10 == digit]

def ends_even():
    return [num for num in range(1, 101) if int(str(num)[-1]) % 2 == 0]

def ends_odd():
    return [num for num in range(1, 101) if int(str(num)[-1]) % 2 != 0]

#might want to add this to multiples
def start_end():
    output = list(range(1,10))
    output.append(num for num in range(11, 101, 11))
    return output

def sum_digit(num: int):
    return sum(int(digit) for digit in str(num))

def odd_sum():
    return [num for num in range(1, 101) if sum_digit(num) % 2 == 1]

def math(exponent: int, multiplier: int):
    return [num for num in range(1, 101) if (num/multiplier ** 1/exponent).is_integer()]

def sum_digit_eq(n: int):
    return [num for num in range(1,101) if sum_digit(num) % n == 0]