import math
import mmh3
import hashlib
from cmath import log
import numpy as np
import copy

from bitarray import bitarray
from struct import unpack, pack
element_num1 = 3000


IN_SEP = b'&&&&'
OUT_SEP = b'####'
FMT = '4sfiii'
_FMT = 'fiii'

KIND = {
    1: 'BloomFilter',
    2: 'CountingBloomFilter',
    3: 'ScalableBloomFilter',
    4: 'SCBloomFilter'
}


def primes(max_num):
    """
    get prime numbers up to max_num
    """
    if max_num <= 1:
        raise ValueError('max_num should bigger than 1.')
    _ = []
    for i in range(2, max_num):
        flag = True
        for j in range(2, i):
            if i % j == 0:
                flag = False
                break
        if flag:
            _.append(i)
    return _


def get_filter_fromfile(path, check=None):
    with open(path, 'rb') as f:
        _bytes = f.read()
    if OUT_SEP in _bytes:
        _all_bytes = [i for i in _bytes.split(OUT_SEP) if i]
        _f_bytes = _all_bytes[:-1]
        _k_bytes = _all_bytes[-1]
        *args, kind = unpack(_FMT, _k_bytes)
        if check and check != kind:
            raise TypeError(f'Not a {KIND[check]} to get.'
                            f'It\'s a {KIND[kind]}')
        filters = [_get_filter_frombytes(i)
                   for i in _f_bytes]
        filter = eval(KIND[kind])(*args)
        filter.filters = filters
        return filter
    else:
        return _get_filter_frombytes(_bytes, check)


def _get_filter_frombytes(_bytes, check=None):
    _bits = _bytes.split(IN_SEP)[0]
    _args = _bytes.split(IN_SEP)[-1]
    *args, count, kind = unpack(_FMT, _args)
    if check and check != kind:
        raise TypeError(f'Not a {KIND[check]} to get.'
                        f'It\'s a {KIND[kind]}')
    _bitarray = bitarray()
    _bitarray.frombytes(_bits)
    filter = eval(KIND[kind])(*args)
    filter.bit_array = _bitarray
    filter.count = count
    return filter


class BloomFilter(object):

    def __init__(self, error_rate, element_num=None, bit_num=None):
        self.count = 0
        self._c = element_num
        self.error_rate = error_rate
        self._b = bit_num
        self._install(error_rate, element_num, bit_num)

    def _install(self, error_rate, element_num, bit_num):
        if not error_rate or error_rate < 0:
            raise ValueError('error rate should be a positive value.')
        if bit_num and element_num:
            raise ValueError('Multi arguments are given,'
                             'needs 2 args,but got 3.')
        elif bit_num:
            element_num = -1 * bit_num * (log(2.0)*log(2.0)) \
                          / log(error_rate)
        elif element_num:
            bit_num = -1 * element_num * log(error_rate) \
                      / (log(2.0)*log(2.0))
        else:
            raise ValueError('Function arguments missing.'
                             '2 arguments should be given at least.')

        self.bit_num = self.__align_4bytes(bit_num.real)
        self.element_num = int(math.ceil(element_num.real))
        self.capacity = self.element_num if not self._c else self._c
        # self.hash_num = int(math.ceil((log(2.0) * self.bit_num
        #                                / element_num).real))
        self.hash_num = 10
        self.bit_array = bitarray(self.bit_num)
        self.bit_array.setall(False)
        self.seeds = primes(200)

    def add(self, element):
        if self._at_half_fill():
            raise IndexError('BloomFilter is at capacity.')
        i = 0
        element = self._to_str(element)
        for _ in range(self.hash_num):
            hashed_value = mmh3.hash(element,
                                     self.seeds[_]) % self.bit_num
            i += self.bit_array[hashed_value]
            self.bit_array[hashed_value] = 1
        if i == self.hash_num:
            return True
        else:
            self.count += 1
            return False

    def copy(self):
        new_filter = BloomFilter(self.error_rate,
                                 self.capacity, self._b)
        new_filter.bit_array = self.bit_array.copy()
        return new_filter

    def exists(self, element):
        element = self._to_str(element)
        for _ in range(self.hash_num):
            hashed_value = mmh3.hash(element,
                                     self.seeds[_]) % self.bit_num
            if self.bit_array[hashed_value] == 0:
                return False
        return True

    def to_pack(self):
        b_bitarray = self.bit_array.tobytes()
        b_args = pack(FMT, IN_SEP, self.error_rate,
                      self.capacity, self.count, 1)
        return b_bitarray+b_args

    def tofile(self, path, mode='wb'):
        with open(path, mode) as f:
            f.write(self.to_pack())

    @classmethod
    def fromfile(cls, path):
        return get_filter_fromfile(path, 1)

    def _option(self, other, opt):
        _ = {
            'or'	:	'__or__',
            'and' 	:	'__and__'
        }
        if self.capacity != other.capacity or \
                self.error_rate != other.error_rate or \
                self.bit_num != other.bit_num:
            raise ValueError(
                "Both filters must have equal initial arguments:"
                "element_num、bit_num、error_rate")
        new_bloom = self.copy()
        new_bloom.bit_array = getattr(new_bloom.bit_array,
                                      _[opt])(other.bit_array)
        return new_bloom

    def _to_str(self, element):
        _e_class = element.__class__.__name__
        _str = str(element)+_e_class
        bytes_like = bytes(_str, encoding='utf-8') if \
            isinstance(_str, str) else _str
        b_md5 = hashlib.md5(bytes_like).hexdigest()
        return b_md5

    def _at_half_fill(self):
        return self.bit_array.count()/self.bit_array.length()*1.0 >= 0.5

    def __align_4bytes(self, bit_num):
        return int(math.ceil(bit_num/32))*32

    def __or__(self, other):
        return self._option(other, 'or')

    def __and__(self, other):
        return self._option(other, 'and')

    def __contains__(self, item):
        return self.exists(item)

    def __len__(self):
        return self.count


class CountingBloomFilter(BloomFilter):

    def __init__(self,error_rate=0.001, element_num=element_num1, bit_num=None):
        super(CountingBloomFilter,self).__init__(error_rate, element_num, bit_num)
        self._bit_array = self.bit_array
        self.each_bit = np.zeros(self.bit_num, dtype=int)

    def trans(self):
        for b in range(len(self.each_bit)):
            start = b * 8
            end = (b + 1) * 8
            self.each_bit[b] = int(self.bit_array[start:end].to01(), 2)

    def add(self, element):
        """
        query the element status in the filter and add it into the filter
        """
        _element = self._to_str(element)
        for _ in range(self.hash_num):
            hashed_value = mmh3.hash(_element, self.seeds[_]) % self.bit_num
            raw_value = self._get_bit_value(hashed_value)
            self._set_bit_value(hashed_value, raw_value+1)
        self.count += 1


    def delete(self, element):
        """
        query the element status in the filter and delete it from the filter
        """
        if self.exists(element):
            _element = self._to_str(element)
            # print(_element)
            for _ in range(self.hash_num):
                hashed_value = mmh3.hash(_element,
                                         self.seeds[_]) % self.bit_num
                raw_value = self._get_bit_value(hashed_value)
                self._set_bit_value(hashed_value, raw_value - 1)
            self.count -= 1
            return True
        return False

    def exists(self, element):
        _element = self._to_str(element)
        for _ in range(self.hash_num):
            hashed_value = mmh3.hash(_element,
                                     self.seeds[_]) % self.bit_num
            if self._get_bit_value(hashed_value) <= 0:
                return False
        return True

    def copy(self):
        new_filter = CountingBloomFilter(self.error_rate,
                                         self.capacity, self._b)
        new_filter.each_bit= self.each_bit.copy()
        new_filter.count = self.count
        return new_filter

    def to_pack(self):
        b_bitarray = self.bit_array.tobytes()
        b_args = pack(FMT, IN_SEP, self.error_rate,
                      self.capacity, self.count, 2)
        return b_bitarray+b_args

    @classmethod
    def fromfile(cls, path):
        return get_filter_fromfile(path, 2)

    @property
    def _overflow(self):
        """if we set 4 bits to represent a abstract bit
         of the standard BloomFilter.The max overflow probability
         is (asume that binary 1111 (int:16) equals j16)
         p(j16)<=bit_num*(e*ln2/16)**16 .
         the value is pretty small so we can use 4 bits to
          represent one bit for common usage."""
        max_overflow_probability = (math.exp(1)*math.log(2)/16) ** 16 * self.bit_num
        return max_overflow_probability

    def _set_bit_value(self, index: int, value: int):
        # _bin = [int(i) for i in self._to_bin(value)]
        # start = index*8
        # end = 8*(index+1)
        # for i in range(start, end):
        #     self.bit_array[i] = _bin[i % 8]
        self.each_bit[index] = value

    def _get_bit_value(self, index: int) -> int:
        # start = index*8
        # end = (index+1)*8
        # _bits = self.bit_array[start:end]
        # _v = int(_bits.to01(), 2)
        return self.each_bit[index]

    def _to_bin(self, value: int) -> str:
        _bin = bin(value)[2:]
        if len(_bin) < 8:
            _bin = '0'*(8-len(_bin))+_bin
        return _bin


if __name__ == '__main__':
    cbf1 = CountingBloomFilter(error_rate=0.001, element_num=100)
    cbf1.add(2)
    cbf2 = CountingBloomFilter(error_rate=0.001, element_num=100)
    cbf2.add(1)
    sum_bits_of_cbf = []
    sum_bits_of_cbf.append(copy.deepcopy(cbf1.each_bit))
    sum_bits_of_cbf.append(copy.deepcopy(cbf2.each_bit))
    sum_cbf = CountingBloomFilter(error_rate=0.001, element_num=100)
    sum_cbf.each_bit =  copy.deepcopy(np.sum(np.array(sum_bits_of_cbf), axis=0))
    print(sum_cbf.exists(1))
    print(sum_cbf.exists(2))
    sum_cbf.delete(1)
    sum_cbf.delete(2)
    print(sum_cbf.exists(1))
    print(sum_cbf.exists(2))
