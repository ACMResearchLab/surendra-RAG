
test_functions = [
    '''def _vK2mD5(s):
    vowels = 'aeiou'
    consonants = 'bcdfghjklmnpqrstvwxyz'
    vowel_count = sum(1 for char in s if char.lower() in vowels)
    consonant_count = sum(1 for char in s if char.lower() in consonants)
    return vowel_count, consonant_count''',

    '''def _gT4nE7(matrix):
    flattened = [item for sublist in matrix for item in sublist]
    unique_numbers = set(flattened)
    return len(unique_numbers)''',

    '''def _pR8sN3(s):
    word_list = s.split()
    longest_word = max(word_list, key=len)
    return longest_word''',

    '''def _kL6sP2(lst):
    return sum(x * x for x in lst if x % 2 == 0)''',

    '''def _mN9dE1(nums):
    if len(nums) < 2:
        return None
    max_product = nums[0] * nums[1]
    for i in range(len(nums)):
        for j in range(i + 1, len(nums)):
            max_product = max(max_product, nums[i] * nums[j])
    return max_product''',

    '''def _cJ5tR6(s):
    char_count = {}
    for char in s:
        if char.isalpha():
            char = char.lower()
            char_count[char] = char_count.get(char, 0) + 1
    return char_count''',

    '''def _tH7gF3(lst):
    sorted_list = sorted(lst)
    if len(sorted_list) % 2 == 0:
        return (sorted_list[len(sorted_list) // 2 - 1] + sorted_list[len(sorted_list) // 2]) / 2
    else:
        return sorted_list[len(sorted_list) // 2]''',

    '''def _uE5rQ9(num):
    if num < 2:
        return False
    for i in range(2, int(num ** 0.5) + 1):
        if num % i == 0:
            return False
    return True''',

    '''def _xR2jH5(lst):
    return [i for i in range(len(lst)) if all(lst[i] % j != 0 for j in range(2, lst[i]))]''',

    '''def _sG8dM2(s):
    return sum(1 for char in s if char.isupper()), sum(1 for char in s if char.islower())''',

    '''def _aG6tF8(matrix):
    return [[matrix[j][i] for j in range(len(matrix))] for i in range(len(matrix[0]))]''',

    '''def _dR9sG4(lst):
    return [x for x in lst if all(x % i != 0 for i in range(2, int(x ** 0.5) + 1))]''',

    '''def _fE3jR5(s):
    return any(char.isdigit() for char in s), any(char.isalpha() for char in s)''',

    '''def _pR7sK2(n):
    return n * (n + 1) * (2 * n + 1) // 6''',

    '''def _qL2tM9(matrix):
    return [max(row) for row in matrix]''',

    '''def _yR8hN5(s):
    return [char for char in s if char.isalnum()]''',

    '''def _jM3tS8(s):
    return s[::-1]''',

    '''def _hN4fD3(lst):
    return [x for x in lst if x == sum(int(digit) ** 3 for digit in str(x))]''',

    '''def _eG7tJ4(s):
    return ''.join(char for char in s if char.isdigit())''',

    '''def _rF6jH9(matrix):
    return [[matrix[j][i] for j in range(len(matrix))] for i in range(len(matrix[0]))]''',

    '''def _tH9sE5(n):
    return n * (n + 1) // 2''',

    '''def _pR8jN6(matrix):
    return sum(sum(row) for row in matrix)''',

    '''def _kL9sD2(matrix):
    return [[row[i] for row in matrix] for i in range(len(matrix[0]))]''',

    '''def _bN6jD8(lst):
    return max(lst) - min(lst)''',

    '''def _tR9fJ5(lst):
    return [x for x in lst if lst.count(x) == 1]''',

    '''def _wE2fR5(s):
    return sum(1 for char in s if char in 'aeiouAEIOU')''',

    '''def _zF3tS7(lst):
    return sum(x ** 2 for x in lst)''',

    '''def _rE4hT9(n):
    return sum(1 for i in range(1, n + 1) if n % i == 0)''',

    '''def _uH5tF9(matrix):
    return [[row[i] for row in matrix] for i in range(len(matrix[0]))]''',

    '''def _jT3sF7(matrix):
    return sum(row.count(1) for row in matrix)''',

    '''def _bH6dK4(lst):
    return [x for x in lst if x % 2 == 0]''',

    '''def _fE2jR5(lst):
    return [x for x in lst if x % 2 != 0]''',

    '''def _wE5rT4(matrix):
    return [max(row) for row in matrix]''',

    '''def _jT4dH8(matrix):
    return [[row[i] for row in matrix] for i in range(len(matrix[0]))]''',

    '''def _vT2gE4(s):
    return ''.join(sorted(s))''',

    '''def _dH6fJ5(lst):
    return sum(1 for x in lst if x > 0)''',

    '''def _wQ2fJ7(s):
    return sum(1 for char in s if char.islower())'''
]
