# coding: utf-8
# Author: Marc Weber

"""
=========================================================================================
General tools
=========================================================================================
"""

import numpy as np
from collections import deque
from pathlib import Path
import os.path
import gzip
import bz2
from glob import glob
from operator import itemgetter
from itertools import groupby
from itertools import islice
import re





def sliding_window_array(array, windowSize):
    """Rolling or sliding window for 1D numpy array
    High performance solution.
    See http://stackoverflow.com/questions/27852343/split-python-sequence-time-series-array-into-subsequences-with-overlap
    
    >>> sliding_window_array(np.array([0,1,2,3,4,5,6,7,8,9]), 3)
    array([[0, 1, 2],
           [1, 2, 3],
           [2, 3, 4],
           [3, 4, 5],
           [4, 5, 6],
           [5, 6, 7],
           [6, 7, 8],
           [7, 8, 9]])
    """
    shape = (array.size - windowSize + 1, windowSize)
    strides = array.strides * 2
    return np.lib.stride_tricks.as_strided(array, shape=shape, strides=strides)



def sliding_window_string(string, n=2):
    """
    Rolling or sliding window iterator for a string
    
    >>> list((x for x in sliding_window_string('abcdefghijkl', 2)))
    ['ab', 'bc', 'cd', 'de', 'ef', 'fg', 'gh', 'hi', 'ij', 'jk', 'kl']
    """
    it = iter(string)
    win = deque((next(it, None) for _ in range(n)), maxlen=n)
    yield ''.join(win)
    append = win.append
    for e in it:
        append(e)
        yield ''.join(win)


def sliding_window_string_slices(string, n=2, step=1, include_rolling_edges=False):
    """
    Returns a generator that will iterate through
    the defined chunks of input sequence. Input sequence
    must be sliceable.
    """

    # Verify the inputs
    if not ((type(n) == type(0)) and (type(step) == type(0))):
        raise Exception("**ERROR** type(winSize) and type(step) must be int.")
    if n > len(string):
        raise Exception("**ERROR** winSize must not be larger than sequence length.")

    # Pre-compute number of chunks to emit
    # numOfChunks = (len(string) - n) // step

    # Do the work
    for i in range(0, len(string) - n, step):
        yield string[i:i+n]


def rolling_window_deque(iterable, size=2, step=1, yield_position=False):
    """
    Adapted from different solutions at https://stackoverflow.com/questions/6822725/rolling-or-sliding-window-iterator-in-python
    """
    if size < 0 or step < 1:
        raise ValueError
    it = iter(iterable)
    initialSize = min(step, size)
    q = deque((next(it, None) for _ in range(initialSize)), maxlen=size)
    posLeft = 0
    posRight = initialSize

    reachedLastElement = False
    stepConsumed = 0
    while not reachedLastElement:
        if yield_position:
            yield (q, posLeft, posRight)
        else:
            yield q
        stepConsumed = 0
        for _ in range(step):
            try:
                q.append(next(it))
                stepConsumed += 1
                posRight += 1
                if len(q) == size and posRight > size:
                    posLeft += 1
            except StopIteration:
                reachedLastElement = True

    # Complete the step advance
    for _ in range(step - stepConsumed):
        try:
            q.popleft()
            posLeft += 1
        except IndexError:
            # The step was larger than the window size and there is no partial window to yield
            # at the end.
            return

    # Finish rolling the window at the end
    while len(q) > 0:
        if yield_position:
            yield (q, posLeft, posRight)
        else:
            yield q

        for _ in range(step):
            try:
                q.popleft()
                posLeft += 1
            except IndexError:
                return


# Detect compressed file by magic bytes at start of the file, and return
# gzip, bz2, or normal file handles.
# Note: *does not work*

# magic_dict = {
#     u"\x1f\x8b\x08": gzip.open,
#     u"\x42\x5a\x68": bz2.BZ2File,
# }
# max_len = max(len(x) for x in magic_dict)

# def open_by_magic(filename):
#     with open(filename, 'r', encoding='utf-8') as f:
#         file_start = f.read(max_len)
#         print(file_start)
#     for magic, fn in magic_dict.items():
#         if file_start.startswith(magic):
#             print("detected magic ", magic)
#             return fn(filename, 'r')
#     print("magic not detected")
#     return open(filename)


def open_by_suffix(filename):
    """Detect compressed file by filename suffix, and return gzip, bz2, or normal file handles."""
    if Path(filename).suffix == '.gz':
        return gzip.open(filename, 'rt')
    elif Path(filename).suffix == '.bz2':
        return bz2.BZ2file(filename, 'rt')
    else:
        return open(filename, 'r')

# with open_by_suffix('/users/lserrano/mweber/Translation_model/Ribosome_profiling_data/Mpn/RNA-seq/TEST/MPN_RP10_P2_UD_15116_AGTTCC_read1.fastq.gz') as f:
#     f.read()


def estimateNbLines(filename, learn_size_bytes=1024*1024):
    """ Estimate the number of lines in the given file without reading the whole file."""

    file_size = os.path.getsize(filename)
    learn_size_bytes = min(learn_size_bytes, file_size)    # Not necessary?

    with open(filename, 'rb') as file:
        buf = file.read(learn_size_bytes)
        numLines = file_size / (len(buf) // buf.count(b'\n'))

    return numLines


def glob_file_list(fileList):
    "Apply glob on the list of file pattern, filter for files only, remove duplicated files."
    if fileList is None or fileList == '':
        fileList = None
    else:
        fileList = [fn for pattern in fileList for fn in glob(pattern) if Path(fn).is_file()]
        # Remove duplicated files from list (can happen if multiple patterns match same files)
        fileList = sorted(list(set(fileList)))
    return fileList


def find_groups_consecutive_values(data):

    ranges = []
    for key, group in groupby(enumerate(data), lambda x: x[0] - x[1]):
        group = list(map(itemgetter(1), group))
        ranges.append((group[0], group[-1]))
        
    return ranges


def piecewise(xArr, intervalList, outputList):
    """
    Wrapper around the numpy piecewise function for intervals. Accepts both numpy arrays and scalars.

                  | outputList[0] for x <= interval[0]
    piecewise() = | outputList[1] for interval[0] < x <= interval[1]
                  | ...
                  | outputList[-1] for interval[n-1] < x <= interval[n]
                  | outputList[-1] for interval[n] < x   (optional)
    """
    if len(intervalList) < 2:
        raise ValueError("The interval list should be of length > 1.")
    conditions = [xArr <= intervalList[0]]
    conditions = conditions + [np.logical_and(intervalList[i] < xArr, xArr <= intervalList[i + 1])
                               for i in range(0, len(intervalList)-1)]
    out = np.piecewise(xArr, conditions, outputList)
    if type(xArr) is not np.array:
        out = float(out)
    return out


def replace_multiple_strings_one_pass(s, repDict):
    """
    Replace multiple strings in **one pass**, taking as argument a dictionary of {'old_text':'new_text'}.

    Note that the replacement is made in one pass, such that we avoid problems of unwanted
    replacements due to iterative modification of the string, such as:
    "spamham sha".replace("spam", "eggs").replace("sha","md5") being "eggmd5m md5" instead of "eggsham md5"

    See https://stackoverflow.com/questions/6116978/how-to-replace-multiple-substrings-of-a-string
    """
    pattern = re.compile("|".join([re.escape(k) for k in repDict.keys()]))
    return pattern.sub(lambda match: repDict[match.group(0)], s)


def group_overlapping_intervals(intervals, ends_overlap=True, return_indices=False):
    """Groups intervals (1D segments) together into groups of overlapping intervals.
    Intervals will be in different groups only if they are not overlapping.
    
    Overlapping is defined as:
    if `ends_overlaps` is True:
        [a1, a2] overlaps with [b1, b2] when a1 <= b1 <= a2 and b2 > a1.
    if `ends_overlaps` is False:
        (a1, a2) overlaps with (b1, b2) when a1 <= b1 < a2 and b2 > a1.

    Example:
```
intervals = [
    (1, 3),
    (2, 4),
    (4, 7),
    (8, 10),
    (8, 9),
    (8, 12),
    (8, 10),
    (9, 10),
    (13, 18)
]

for y, interval in enumerate(intervals):
    plt.plot(interval, [y, y], c='black', lw=2)
plt.gca().set_xticks(np.arange(0, 20))

for ends_overlap in [True, False]:
    
    groups = group_overlapping_intervals(intervals, ends_overlap=ends_overlap)
    print("groups:", groups)
    for group in groups:
        start = min(min(group))
        end = max(max(group))
        y += 1
        plt.plot([start, end], [y, y], lw=3, c='red')
        
groups_indices = group_overlapping_intervals(intervals, ends_overlap=ends_overlap, return_indices=True)
print("indices:", groups_indices)
```
    """

    intervals = [(x1, x2, i) for i, (x1, x2) in enumerate(intervals)]
    intervals = sorted(intervals, key=lambda x: x[0])

    stack = list(intervals[0])
    group = [intervals[0]]
    groups = []

    for interval in intervals[1:]:
        start = interval[0]
        end = interval[1]
        end_stack = stack[1]
        if (start < end_stack) or (start <= end_stack and ends_overlap):
            group.append(interval)
            stack[1] = end
        else:
            # push the group of intervals
            groups.append(group)
            # set the stack to the new interval
            stack = list(interval)
            group = [interval]
    groups.append(group)

    if return_indices:
        groups_indices = [[i for (x1, x2, i) in group1] for group1 in groups]
        return groups_indices
    else:
        groups_x = [[(x1, x2) for (x1, x2, i) in group1] for group1 in groups]
        return groups_x
