from numba import njit
import numpy as np

# import collections
# collections.Callable = collections.abc.Callable

@njit
def get_clipping_sections(d: np.ndarray):
    section_start = -1
    res = []
    for i in range(d.shape[0]):
        if section_start == -1:  # Not in a section
            if d[i]:
                section_start = i
        else:  # In a section
            if not d[i]:
                res.append((section_start, i))
                section_start = -1
    # Handle case where the clipping section goes till the end of the array
    if section_start != -1:
        res.append((section_start, d.shape[0]))
    return res


@njit
def join_segments(segments: np.ndarray, join_close):
    res = []
    start, end = segments[0][0], segments[0][1]
    for i in range(1, len(segments)):
        if segments[i][0] - end <= join_close:  # new start is close enough to current end -> continue current segment
            end = max(segments[i][1], end)
        else:  # end prev segment, create new
            res.append((start, end))
            start, end = segments[i][0], segments[i][1]
    res.append((start, end))
    return res

@njit
def get_valid_sections(data: np.ndarray, join_close=250, min_length=0):
    """From 1d bool array returs list of index pairs of chunks without True.
    Second index in pair is not included, as in python indexing behavior
    If 2 sequences of Trues are apart by less then join_close, they are joined
    Also sequences of length less then min_length are discarded"""
    segments = get_clipping_sections(data)
    if len(segments) == 0:
        return [(0,  data.shape[0]), ]
    joined_segments = join_segments(segments, join_close)

    good_parts = []
    if joined_segments[0][0] >= min_length:
        good_parts.append((0, joined_segments[0][0]))
    for i in range(1, len(joined_segments)):
        if joined_segments[i][0] - joined_segments[i-1][1] >= min_length:
            good_parts.append((joined_segments[i-1][1], joined_segments[i][0]))
    if data.shape[0] - joined_segments[-1][1] >= min_length:
        good_parts.append((joined_segments[-1][1], data.shape[0]))
    return good_parts


# Test cases
def test_get_valid_sections():
    # Test case 1: No clipping sections
    data1 = np.zeros(100, dtype=np.bool_)
    assert get_valid_sections(data1, join_close=5) == [(0, 100)]

    # Test case 2: One clipping section
    data2 = np.zeros(100, dtype=np.bool_)
    data2[10:20] = True
    assert get_valid_sections(data2, join_close=5) == [(0, 10), (20, 100)]

    # Test case 3: Two close clipping sections joined
    data3 = np.zeros(100, dtype=np.bool_)
    data3[10:20] = True
    data3[22:30] = True
    assert get_valid_sections(data3, join_close=5) == [(0, 10), (30, 100)]

    # Test case 4: Two far clipping sections not joined
    data4 = np.zeros(100, dtype=np.bool_)
    data4[10:20] = True
    data4[30:40] = True
    assert get_valid_sections(data4, join_close=5) == [(0, 10), (20, 30), (40, 100)]

    # Test case 5: Minimum length requirement
    data5 = np.zeros(100, dtype=np.bool_)
    data5[10:20] = True
    data5[24:30] = True
    data5[70:80] = True
    assert get_valid_sections(data5, join_close=5, min_length=15) == [(30, 70), (80, 100)]

    print("All tests passed!")


if __name__ == "__main__":
    data = np.random.randn(4, 100)
    is_clipping = np.any(np.abs(data) > 2, axis=0)

    test_get_valid_sections()