from cpp.ga.util import abs_pmx_insert


def test_pmx_insert():
    a = abs_pmx_insert([8, 4, 7, 3, 6, 2, 5, 1, 9, 0], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 3, 8)
    assert(a == [0, 7, 4, 3, 6, 2, 5, 1, 8, 9])