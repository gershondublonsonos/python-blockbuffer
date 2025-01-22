import numpy as np
import pytest

from blockbuffer import (BlockBuffer, BlockBufferFullException, BlockBufferValueException)


def test_blockbuffer_basic():
    bb = BlockBuffer(block_size=4)
    assert bb.get() is None
    bb.extend([1, 2, 3, 4, 5, 6, 7, 8])
    assert np.array_equal(bb.get(), [1, 2, 3, 4])
    assert np.array_equal(bb.get(), [5, 6, 7, 8])
    assert bb.get() is None


def test_blockbuffer_capacity():
    bb = BlockBuffer(block_size=4, capacity=8, auto_resize=False)
    bb.extend([1, 2, 3, 4])
    bb.extend([5, 6, 7, 8])
    with pytest.raises(BlockBufferFullException):
        bb.extend([9])


def test_blockbuffer_auto_resize():
    bb = BlockBuffer(block_size=4, capacity=8)
    bb.extend([1, 2, 3, 4])
    assert bb.capacity == 8
    bb.extend([5, 6, 7, 8])
    assert bb.capacity == 8
    bb.extend([9, 10, 11, 12])
    assert bb.capacity == 12
    assert np.array_equal(bb.get(), [1, 2, 3, 4])
    assert np.array_equal(bb.get(), [5, 6, 7, 8])
    assert np.array_equal(bb.get(), [9, 10, 11, 12])
    assert bb.get() is None


def test_blockbuffer_auto_resize_column_major():
    bb = BlockBuffer(block_size=4, num_channels=4, capacity=8, auto_resize=True, row_major=False)
    bb.extend(np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]))
    assert bb.capacity == 8
    bb.extend(np.array([[9, 10, 11, 12], [13, 14, 15, 16], [17, 18, 19, 20], [21, 22, 23, 24]]))
    assert bb.capacity == 8
    bb.extend(np.array([[17, 18, 19, 20], [21, 22, 23, 24], [25, 26, 27, 28], [29, 30, 31, 32]]))
    assert bb.capacity == 12
    assert np.array_equal(bb.get(), np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]))
    assert np.array_equal(bb.get(), np.array([[9, 10, 11, 12], [13, 14, 15, 16], [17, 18, 19, 20], [21, 22, 23, 24]]))
    assert np.array_equal(bb.get(), np.array([[17, 18, 19, 20], [21, 22, 23, 24], [25, 26, 27, 28], [29, 30, 31, 32]]))
    assert bb.get() is None
    assert bb.capacity == 12


def test_blockbuffer_auto_resize_column_major_list_input():
    bb = BlockBuffer(block_size=4, num_channels=4, capacity=8, auto_resize=True, row_major=False)
    bb.extend([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]])
    assert bb.capacity == 8
    bb.extend([[9, 10, 11, 12], [13, 14, 15, 16], [17, 18, 19, 20], [21, 22, 23, 24]])
    assert bb.capacity == 8
    bb.extend([[17, 18, 19, 20], [21, 22, 23, 24], [25, 26, 27, 28], [29, 30, 31, 32]])
    assert bb.capacity == 12
    assert np.array_equal(bb.get(), np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]))


def test_blockbuffer_hop():
    bb = BlockBuffer(block_size=8, hop_size=2)
    bb.extend([1, 2, 3, 4])
    assert bb.get() is None
    bb.extend([5, 6, 7, 8])
    assert np.array_equal(bb.get(), [1, 2, 3, 4, 5, 6, 7, 8])
    bb.extend([9, 10, 11, 12])
    assert np.array_equal(bb.get(), [3, 4, 5, 6, 7, 8, 9, 10])
    assert np.array_equal(bb.get(), [5, 6, 7, 8, 9, 10, 11, 12])
    assert bb.get() is None


def test_blockbuffer_hop_odd_capacity():
    bb = BlockBuffer(block_size=4, hop_size=2, capacity=6)
    assert bb.get() is None
    bb.extend([1, 2, 3, 4])
    assert np.array_equal(bb.get(), [1, 2, 3, 4])
    assert bb.get() is None
    bb.extend([5, 6, 7, 8])
    assert np.array_equal(bb.get(), [3, 4, 5, 6])
    assert np.array_equal(bb.get(), [5, 6, 7, 8])
    assert bb.get() is None


def test_blockbuffer_iterator():
    bb = BlockBuffer(block_size=4, hop_size=2)
    bb.extend([1, 2, 3, 4, 5, 6, 7, 8])
    blocks = list(bb)
    assert len(blocks) == 3
    assert np.array_equal(blocks[0], [1, 2, 3, 4])
    assert np.array_equal(blocks[1], [3, 4, 5, 6])
    assert np.array_equal(blocks[2], [5, 6, 7, 8])


def test_blockbuffer_bad_values():
    # invalid hop size
    with pytest.raises(BlockBufferValueException):
        bb = BlockBuffer(block_size=4, hop_size=0)

    # invalid block size
    with pytest.raises(BlockBufferValueException):
        bb = BlockBuffer(block_size=0, hop_size=4)

    # invalid hop size vs block size
    with pytest.raises(BlockBufferValueException):
        bb = BlockBuffer(block_size=4, hop_size=5)

    # invalid capacity (must be >= block_size + hop_size)
    with pytest.raises(BlockBufferValueException):
        bb = BlockBuffer(block_size=4, hop_size=4, capacity=4)

    # invalid data format (must be 1D)
    bb = BlockBuffer(block_size=4)
    with pytest.raises(BlockBufferValueException):
        bb.extend(np.array([[1, 2], [3, 4]]))


def test_blockbuffer_pass_2D_mono_array():
    bb = BlockBuffer(block_size=4)
    assert bb.get() is None
    bb.extend(np.array([[1, 2, 3, 4, 5, 6, 7, 8]]).T)
    assert np.array_equal(bb.get(), [1, 2, 3, 4])
    assert np.array_equal(bb.get(), [5, 6, 7, 8])
    assert bb.get() is None


def test_blockbuffer_always_2d():
    bb = BlockBuffer(block_size=4, always_2d=True)
    assert bb.get() is None
    bb.extend(np.array([[1, 2, 3, 4, 5, 6, 7, 8]]).T)
    assert np.array_equal(bb.get(), np.array([[1, 2, 3, 4]]).T)
    assert np.array_equal(bb.get(), np.array([[5, 6, 7, 8]]).T)
    assert bb.get() is None


def test_blockbuffer_2D():
    bb = BlockBuffer(block_size=2, num_channels=2)
    bb.extend(np.array([[1, 2, 3, 4], [5, 6, 7, 8]]).T)
    assert np.array_equal(bb.get(), np.array([[1, 5], [2, 6]]))
    assert np.array_equal(bb.get(), np.array([[3, 7], [4, 8]]))
    assert bb.get() is None

    bb = BlockBuffer(block_size=2, num_channels=2)
    bb.extend(np.array([[1, 2], [5, 6]]).T)
    assert np.array_equal(bb.get(), np.array([[1, 5], [2, 6]]))
    assert bb.get() is None
    bb.extend(np.array([[3, 4, 5, 6], [7, 8, 9, 10]]).T)
    assert np.array_equal(bb.get(), np.array([[3, 7], [4, 8]]))
    assert np.array_equal(bb.get(), np.array([[5, 9], [6, 10]]))
    assert bb.get() is None

    bb = BlockBuffer(block_size=2, num_channels=2)
    bb.extend(np.array([[1, 2], [5, 6]]).T)
    assert np.array_equal(bb.get(), np.array([[1, 5], [2, 6]]))
    assert bb.get() is None
    bb.extend(np.array([[3, 4, 5, 6], [7, 8, 9, 10]]).T)
    assert np.array_equal(bb.get(), np.array([[3, 7], [4, 8]]))
    assert np.array_equal(bb.get(), np.array([[5, 9], [6, 10]]))
    assert bb.get() is None


def test_blockbuffer_2D_column_major():
    bb = BlockBuffer(block_size=2, num_channels=2, row_major=False)
    bb.extend(np.array([[1, 2, 3, 4], [5, 6, 7, 8]]))
    assert np.array_equal(bb.get(), np.array([[1, 2], [5, 6]]))
    assert np.array_equal(bb.get(), np.array([[3, 4], [7, 8]]))
    assert bb.get() is None
    bb.extend(np.array([[3, 4, 5, 6], [7, 8, 9, 10]]))
    assert np.array_equal(bb.get(), np.array([[3, 4], [7, 8]]))
    assert np.array_equal(bb.get(), np.array([[5, 6], [9, 10]]))
    assert bb.get() is None


def test_blockbuffer_dtypes():
    for dtype in (np.int16, np.int32, np.float16, np.float32, np.float64):
        bb = BlockBuffer(2, dtype=dtype)
        bb.extend([1, 2, 3, 4, 5, 6])
        for block in bb:
            assert block.dtype == dtype
