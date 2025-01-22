from typing import Any, Iterator, Optional, Union

import numpy as np
from numpy.typing import NDArray

from blockbuffer.exceptions import BlockBufferFullException, BlockBufferValueException

BLOCK_BUFFER_DEFAULT_CAPACITY_BLOCKS = 64


class BlockBuffer(object):

    def __init__(self,
                 block_size: int,
                 hop_size: Optional[int] = None,
                 *,
                 num_channels: int = 1,
                 capacity: Optional[int] = None,
                 auto_resize: bool = True,
                 dtype: Union[np.dtype, type] = np.dtype(np.float64),
                 always_2d: bool = False,
                 row_major: bool = True) -> None:
        """
        Args:
            block_size (int): The number of samples to return per block.
            hop_size (int): The amount the read head should be moved forward per block.
                If None, defaults to `block_size`.
            num_channels (int): The number of channels to allocate.
            capacity (int): The maximum number of samples that the buffer can hold.
                If None, defaults to `block_size * BLOCK_BUFFER_DEFAULT_CAPACITY_BLOCKS`.
            auto_resize (bool): If True, the buffer will automatically resize if it overflows.
                Does memory allocation, so should be disabled to ensure in real-time threads.
            dtype (np.dtype): The data type of the samples.
            always_2d (bool): If True, always return 2D arrays, even if num_channels == 1.
                This is useful for compatibility with soundfile's always_2d argument.
            row_major (bool): If True, store samples in row-major order (samples-by-channels).
                If False, store samples in column-major order (channels-by-samples).
        """
        self.block_size = block_size
        self.hop_size = hop_size if hop_size is not None else block_size
        self.num_channels = num_channels
        self.write_position = 0
        self.read_position = 0
        self.length = 0
        self.auto_resize = auto_resize
        self.always_2d = always_2d
        self.row_major = row_major

        if capacity:
            self.capacity = capacity
        else:
            self.capacity = self.block_size * BLOCK_BUFFER_DEFAULT_CAPACITY_BLOCKS

        if self.hop_size == 0:
            raise BlockBufferValueException("Hop size must be >0")
        if self.block_size == 0:
            raise BlockBufferValueException("Block size must be >0")
        if self.hop_size > self.block_size:
            raise BlockBufferValueException("Hop size must be <= block_size")
        if self.capacity < self.block_size + self.hop_size:
            raise BlockBufferValueException("Capacity must be >= block_size + hop_size")

        # -------------------------------------------------------------------------------
        # Ringbuffer and buffer to store and return data, initialized according to memory layout.
        # -------------------------------------------------------------------------------
        if self.row_major:
            # Row-major layout (samples-by-channels)
            self.queue = np.zeros((self.capacity, self.num_channels), dtype=dtype, order="C")
            self.return_buffer = np.zeros((self.block_size, self.num_channels), dtype=dtype, order="C")
        else:
            # Column-major layout (channels-by-samples)
            self.queue = np.zeros((self.num_channels, self.capacity), dtype=dtype, order="F")
            self.return_buffer = np.zeros((self.num_channels, self.block_size), dtype=dtype, order="F")

    def __iter__(self) -> NDArray:
        return self.blocks

    def extend(self, frames: NDArray) -> None:
        """
        Append frames to the buffer.
        Safe for usage in real-time audio applications, as no memory allocation or system I/O is done

        Args:
            frames: An array of frames to process. Can be a 1D or 2D numpy.ndarray.
        """

        # -------------------------------------------------------------------------------
        # Type checking and array validation.
        # -------------------------------------------------------------------------------
        if not isinstance(frames, np.ndarray):
            frames = np.array(frames, dtype=self.queue.dtype)

        axis = 1 if self.row_major else 0

        if frames.ndim > 2:
            raise BlockBufferValueException("Invalid number of dimensions in frames")
        else:
            if frames.ndim == 2 and frames.shape[axis] != self.num_channels or \
                    (frames.ndim == 1 and self.num_channels > 1):
                raise BlockBufferValueException(
                    f"Invalid number of channels in frames (expected {self.num_channels}, got {frames.shape[axis]})")

        num_frames = frames.shape[1 - axis] if frames.ndim == 2 else frames.shape[0]

        # -------------------------------------------------------------------------------
        # Resize the buffer (if enabled)
        # -------------------------------------------------------------------------------
        if self.length + num_frames > self.capacity:
            if self.auto_resize:
                size_increase = self.length + num_frames - self.capacity
                if self.row_major:
                    self.queue = np.pad(self.queue, ((0, size_increase), (0, 0)))
                else:
                    self.queue = np.pad(self.queue, ((0, 0), (0, size_increase)))
                self.capacity += size_increase
            else:
                raise BlockBufferFullException("Block buffer overflowed")

        # -------------------------------------------------------------------------------
        # Write the samples.
        # Logic is complex due to having to jump through hoops to avoid memory allocations.
        # -------------------------------------------------------------------------------
        self.write_position = self.write_position % self.capacity

        if self.write_position + num_frames <= self.capacity:
            # -------------------------------------------------------------------------------
            # There is enough space remaining the buffer to write the frames without wrapping.
            # -------------------------------------------------------------------------------
            if frames.ndim == 1:
                self.queue[self.write_position:self.write_position + num_frames, 0] = frames
            else:
                if self.row_major:
                    self.queue[self.write_position:self.write_position + num_frames] = frames
                else:
                    self.queue[:, self.write_position:self.write_position + num_frames] = frames
        else:
            remaining_frames = self.capacity - self.write_position
            if frames.ndim == 1:
                self.queue[self.write_position:, 0] = frames[:remaining_frames]
                self.queue[:num_frames - remaining_frames, 0] = frames[remaining_frames:]
            else:
                if self.row_major:
                    self.queue[self.write_position:] = frames[:remaining_frames]
                    self.queue[:num_frames - remaining_frames] = frames[remaining_frames:]
                else:
                    self.queue[:, self.write_position:] = frames[:, :remaining_frames]
                    self.queue[:, :num_frames - remaining_frames] = frames[:, remaining_frames:]

        # -------------------------------------------------------------------------------
        # Update length and write position
        # -------------------------------------------------------------------------------
        self.length += num_frames
        self.write_position = self.write_position + num_frames

    def get(self) -> Optional[NDArray]:
        """
        Returns a block of samples from the buffer, if any are available.

        Returns:
            An array of exactly `block_size` samples, or None if no more blocks remain
            to be read.
        """
        if self.length >= self.block_size:
            if self.read_position + self.block_size <= self.capacity:
                if self.row_major:
                    rv = self.queue[self.read_position:self.read_position + self.block_size]
                else:
                    rv = self.queue[:, self.read_position:self.read_position + self.block_size]
            else:
                remaining_frames = self.capacity - self.read_position
                if self.row_major:
                    self.return_buffer[:remaining_frames] = self.queue[self.read_position:]
                    self.return_buffer[remaining_frames:] = self.queue[:self.block_size - remaining_frames]
                else:
                    self.return_buffer[:, :remaining_frames] = self.queue[:, self.read_position:]
                    self.return_buffer[:, remaining_frames:] = self.queue[:, :self.block_size - remaining_frames]
                rv = self.return_buffer
            self.length -= self.hop_size
            self.read_position = (self.read_position + self.hop_size) % self.capacity
            if self.num_channels == 1 and not self.always_2d:
                return rv[:, 0]
            else:
                return rv
        return None

    @property
    def blocks(self) -> Iterator[NDArray[Any]]:
        """
        Returns:
            A generator which yields remaining blocks of samples.
        """
        while True:
            rv = self.get()
            if rv is not None:
                yield rv
            else:
                return
