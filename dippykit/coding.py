"""Module of various image coding functions

This module contains an assortment of functions that encode or decode images
in popular or useful manners.

"""

# This library was developed for the Georgia Tech graduate course ECE 6258:
# Digital Image Processing with Professor Ghassan AlRegib.
# For comments and feedback, please email dippykit[at]gmail.com

# Functional imports
import numpy as np
from heapq import heappush, heappop, heapify

# General imports
from typing import Dict, List, Tuple

__author__ = 'Brighton Ancelin, Motaz Alfarraj, Ghassan AlRegib'

__all__ = ['JPEG_Q_table_luminance', 'huffman_encode', 'huffman_decode',
           'huffman_dict', 'huffman_tree']


"""The quantization table used by JPEG for encoding luminance values"""
JPEG_Q_table_luminance = np.array([
    [ 16,  11,  10,  16,  24,  40,  51,  61],
    [ 12,  12,  14,  19,  26,  58,  60,  55],
    [ 14,  13,  16,  24,  40,  57,  69,  56],
    [ 14,  17,  22,  29,  51,  87,  80,  62],
    [ 18,  22,  37,  56,  68, 109, 103,  77],
    [ 24,  35,  55,  64,  81, 104, 113,  92],
    [ 49,  64,  78,  87, 103, 121, 120, 101],
    [ 72,  92,  95,  98, 112, 100, 103,  99],
])


def huffman_encode(
        im_vec: np.ndarray,
        symbol_code_dict: Dict[int, np.ndarray]=None,
        ) -> Tuple[np.ndarray, int, Dict[int, np.ndarray], Dict[int, float]]:
    """Encodes an integer vector using huffman encoding
    
    Given a vector (one-dimensional ndarray) with either signed or unsigned 
    integer dtype, this function huffman encodes it. The process for this is 
    the following:
    
    #. If no symbol-code dictionary is provided, create one. The symbol 
       code dictionary (symbol_code_dict) correlates the values in the input 
       vector to the variable-bit-length values in the output vector. In the 
       case that no symbol-code dictionary was provided, a symbol-probability 
       dictionary (symbol_probability_dict) is generated. This symbol 
       probability dictionary has a key set of all unique symbols in im_vec 
       and a value set containing each unique symbol's frequency in im_vec. 
       The symbol-code dictionary is then generated using 
       :func:`huffman_dict` with the symbol-probability dictionary as its 
       argument. This symbol-code dictionary now contains the statistically 
       optimal compressive encoding scheme.
        
    #. Translate the im_vec vector into its huffman encoded version by 
       replacing each integer value with its appropriate bit sequence in the 
       the symbol-code dictionary. 
    
    #. Pack the bit sequence into a byte sequence (represented as a 
       one-dimensional ndarray of dtype ``uint8``). The last byte is 
       right-padded with 0s if the number of bits in the bit sequence was not 
       a perfect multiple of 8. 
    
    #. Return the following data (in order) as a tuple:
    
        * byte sequence
        * the number of bits in the previous step's bit stream
        * the symbol-code dictionary (symbol_code_dict)
        * the symbol-probability dictionary (symbol_probability_dict)
        
    To reverse the huffman encoding, you can pass the first, third, 
    and second returned values to the :func:`huffman_decode` function in 
    that order. Optionally, (for optimal speed) pass the length of the 
    original image vector (im_vec) to :func:`huffman_decode` as the 
    init_arr_size keyword argument.
    
    :type im_vec: ``numpy.ndarray``
    :param im_vec: A vector with either signed integer or unsigned integer 
        dtype to be huffman encoded. With images, one can trivially make them 
        vectors by using the numpy method **im.reshape(-1)**.
    :type symbol_code_dict: ``Dict[int, numpy.ndarray]``
    :param symbol_code_dict: (Optional) A dictionary with either a signed 
        integer or unsigned integer key set and a value set of unique bit 
        sequences (represented as one-dimensional ndarrays with boolean dtype).
    :rtype: (``numpy.ndarray``, ``int``, ``Dict[int, np.ndarray]``,
        ``Dict[int, float]``)
    :return: A tuple containing the following, in order:
    
        * The huffman encoded byte sequence of the input im_vec
        * The number of bits needed to represent the encoded im_vec
        * The symbol-code dictionary which translates integer symbols to 
          binary codes
        * The symbol-probability dictionary which translates integer symbols to
          probability weights
    
    Examples:
    
    >>> import numpy as np
    >>> im = np.array([[  0, 255, 255,   0],
    ...                [255,   0,  64, 128]])
    >>> im_vec = im.reshape(-1)
    >>> im_vec
    array([  0, 255, 255,   0, 255,   0,  64, 128])
    >>> im_encoded, stream_length, symbol_code_dict, symbol_prob_dict = 
    ...         huffman_encode(im_vec)
    >>> im_encoded
    array([205, 202], dtype=uint8)
    >>> np.unpackbits(im_encoded)
    array([1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0], dtype=uint8)
    >>> stream_length
    15
    >>> symbol_code_dict
    {255: array([False]), 
     0: array([ True,  True]), 
     64: array([ True, False, False]), 
     128: array([ True, False,  True])}
    >>> symbol_prob_dict
    {0: 3, 
     64: 1, 
     128: 1, 
     255: 3}
    
    """
    assert 'i' == im_vec.dtype.kind or 'u' == im_vec.dtype.kind, \
            "Image vector must be an integer dtype for huffman encoding."
    assert 1 == len(im_vec.shape), \
            "Image vector must be a 1-dimensional ndarray."
    if symbol_code_dict is None:
        symbols, values = np.unique(im_vec, return_counts=True)
        symbol_probability_dict = dict(zip(symbols, values))
        symbol_code_dict = huffman_dict(symbol_probability_dict)
    im_enco = [symbol_code_dict[val] for val in im_vec]
    im_enco = np.concatenate(im_enco)
    return np.packbits(im_enco), im_enco.size, symbol_code_dict, \
           symbol_probability_dict


def huffman_decode(
        im_encoded: np.ndarray,
        symbol_code_dict: Dict[int, np.ndarray],
        stream_length: int=-1,
        init_arr_size: int=-1,
        ) -> np.ndarray:
    """Decodes a huffman encoded byte stream
    
    Given a huffman encoded byte stream and a symbol-code dictionary, 
    this function will decode the byte stream into its original form.
    
    The huffman encoded byte stream (im_encoded) must be represented as a 
    one-dimensional ndarray with dtype ``uint8``. The symbol-code dictionary 
    must be represented as a dictionary that maps either signed integer or 
    unsigned integer symbols to ndarrays with boolean dtype (these ndarrays 
    are the binary huffman codes that compose im_encoded). 
    
    The stream_length argument (optional) is an integer that represents the 
    number of bits in im_encoded that actually represent the image vector. 
    Often times, huffman encoding a vector yields a bit stream whose length 
    is not a perfect multiple of 8, therefore the byte stream will be padded 
    at the end with 0s. Specifying stream_length prevents against extra 
    additional values that may arise from these padded 0s.
    
    The init_arr_size argument (optional) is an integer that represents the 
    number of elements that the user anticipates the decoded image vector to 
    have. **This value has no effect on the output of the function, its only 
    purpose is to speed up computation. This value can be a rough estimate.** 
    
    To huffman encode an image vector, see :func:`huffman_encode`.
    
    :type im_encoded: ``numpy.ndarray``
    :param im_encoded: A one-dimensional ndarray of dtype ``uint8`` composed 
        of huffman encoded values.
    :type symbol_code_dict: ``Dict[int, np.ndarray]``
    :param symbol_code_dict: A dictionary that maps either signed integer or 
        unsigned integer symbols for the decoded image to binary huffman 
        codes present in the im_encoded vector.
    :type stream_length: ``int``
    :param stream_length: (Optional) The number of bits from the im_encoded 
        vector to decode.
    :type init_arr_size: ``int``
    :param init_arr_size: (Optional) The anticipated number of elements in 
        the decoded image. **This parameter has no effect on the return value 
        of this function. It only reduces the computation time when it is 
        close to the actual number of elements in the decoded image.**
    :rtype: ``numpy.ndarray``
    :return: The decoded image vector, represented as a one-dimensional 
        ndarray with dtype ``int64``.
        
    Examples:
    
    >>> import numpy as np
    >>> im = np.array([[  0, 255, 255,   0],
    ...                [255,   0,  64, 128]])
    >>> im_vec = im.reshape(-1)
    >>> im_vec
    array([  0, 255, 255,   0, 255,   0,  64, 128])
    >>> im_encoded, stream_length, symbol_code_dict, symbol_prob_dict = 
    ...         huffman_encode(im_vec)
    >>> huffman_decode(im_encoded, symbol_code_dict)
    array([  0, 255, 255,   0, 255,   0,  64, 128, 255], dtype=int64)
    >>> huffman_decode(im_encoded, symbol_code_dict, stream_length)
    array([  0, 255, 255,   0, 255,   0,  64, 128], dtype=int64)
    >>> huffman_decode(im_encoded, symbol_code_dict, stream_length, 
    ...         im_vec.size)
    array([  0, 255, 255,   0, 255,   0,  64, 128], dtype=int64)
    >>> huffman_decode(im_encoded, symbol_code_dict, stream_length, 1)
    array([  0, 255, 255,   0, 255,   0,  64, 128], dtype=int64)
    >>> huffman_decode(im_encoded, symbol_code_dict, stream_length, 10000)
    array([  0, 255, 255,   0, 255,   0,  64, 128], dtype=int64)
        
    """
    assert 'uint8' == im_encoded.dtype.name and 1 == len(im_encoded.shape), \
            "Encoded image must be a 1-dimensional ndarray of dtype uint8."
    huff_root = huffman_tree(symbol_code_dict)
    bit_stream = np.unpackbits(im_encoded).astype(bool)
    if -1 != stream_length:
        bit_stream = bit_stream[:stream_length]
    ret_dtype = np.int64
    if -1 == init_arr_size:
        # Arbitrarily chosen size
        im_vec = np.zeros(int(bit_stream.size / 4), dtype=ret_dtype)
    else:
        im_vec = np.zeros(init_arr_size, dtype=ret_dtype)
    index = 0
    cur_huff_node = huff_root
    for bit in bit_stream:
        if not bit:
            cur_huff_node = cur_huff_node[0]
        else:
            cur_huff_node = cur_huff_node[1]
        if not isinstance(cur_huff_node, list):
            if index >= im_vec.size:
                # Arbitrarily chosen size increase
                im_vec = np.append(im_vec, np.zeros(int(1.05*im_vec.size),
                                                    dtype=ret_dtype))
            im_vec[index] = cur_huff_node
            index += 1
            cur_huff_node = huff_root
    im_vec = im_vec[:index]
    return im_vec


def huffman_dict(
        symbol_probability_dict: Dict[int, float],
        ) -> Dict[int, np.ndarray]:
    """Generates a symbol-code dictionary using huffman encoding

    Given a dictionary where each key is a symbol in some integer vector and
    the corresponding values are floating point weights of the frequency of
    occurrence of said symbol, this function returns a dictionary with an
    identical key set as the input but with corresponding values
    representing the huffman codes of each symbol. These huffman codes are
    represented as vectors (one-dimensional ndarray) with boolean dtype.

    The algorithm used in this implementation of huffman dictionary
    generation is based on that of `RosettaCode
    <https://rosettacode.org/wiki/Huffman_coding#Python>`_ .

    :type symbol_probability_dict: ``Dict[int, float]``
    :param symbol_probability_dict: A dictionary with integer symbol keys
        and frequency weighting values
    :rtype: ``Dict[int, numpy.ndarray]``
    :return: A dictionary with the same integer symbol keys as the
        symbol_probability_dict argument but with huffman code values.

    Examples:

    >>> import numpy as np
    >>> rand_vec = np.array([0, 0, 1, 2, 2, 2, 2, 3, 3, 3])
    >>> symbols, values = np.unique(rand_vec, return_counts=True)
    >>> symbols
    array([0, 1, 2, 3])
    >>> values
    array([2, 1, 4, 3], dtype=int64)
    >>> symbol_probability_dict = dict(zip(symbols, values))
    >>> symbol_probability_dict
    {0: 2,
     1: 1,
     2: 4,
     3: 3}
    >>> huffman_dict(symbol_probability_dict)
    {2: array([False]),
     3: array([ True,  True]),
     0: array([ True, False,  True]),
     1: array([ True, False, False])}

    """
    heap = [[prob, [sym, np.empty(0, dtype=bool)]] for sym, prob in
            symbol_probability_dict.items()]
    heapify(heap)
    while len(heap) > 1:
        lo = heappop(heap)
        hi = heappop(heap)
        for pair in lo[1:]:
            pair[1] = np.insert(pair[1], 0, False)
        for pair in hi[1:]:
            pair[1] = np.insert(pair[1], 0, True)
        heappush(heap, [lo[0] + hi[0]] + lo[1:] + hi[1:])
    return dict(sorted(heappop(heap)[1:], key=lambda p: (p[1].size, p[0])))


def huffman_tree(
        symbol_code_dict: Dict[int, np.ndarray],
        ) -> List:
    """Generates a huffman tree from a symbol-code dictionary

    Given a symbol-code dictionary, this function generates a corresponding
    huffman tree. This huffman tree is a binary tree, and is represented as a
    length 2 list of nested length 2 lists. The first element in these
    length 2 lists represents a 0 in the huffman code, and the second
    element represents a 1 in the huffman code.

    :type symbol_code_dict: ``Dict[int, numpy.ndarray]``
    :param symbol_code_dict: A symbol-code dictionary that maps either
        signed integer or unsigned integer symbols to binary huffman codes.
        These binary huffman codes are represented as ndarrays with boolean
        dtype.
    :rtype: ``List``
    :return: The huffman tree corresponding to the symbol-code dictionary,
        represented as a list of nested lists.

    Examples:

    >>> symbol_code_dict = {0: np.array([ True,  True]),
    ...                     3: np.array([ True, False]),
    ...                     4: np.array([False, False]),
    ...                     1: np.array([False, True,  True]),
    ...                     2: np.array([False,  True, False])}
    >>> huffman_tree(symbol_code_dict)
    [[4, [2, 1]], [3, 0]]

    """
    root = []
    for symbol in symbol_code_dict:
        huff_code = symbol_code_dict[symbol].tolist()
        cur_node = root
        for bit in huff_code[:-1]:
            if 0 == len(cur_node):
                cur_node.insert(0, [])
                cur_node.insert(0, [])
            if not bit:
                cur_node = cur_node[0]
            else:
                cur_node = cur_node[1]
        if 0 == len(cur_node):
            cur_node.insert(0, [])
            cur_node.insert(0, [])
        if not huff_code[-1]:
            cur_node[0] = symbol
        else:
            cur_node[1] = symbol
    return root

