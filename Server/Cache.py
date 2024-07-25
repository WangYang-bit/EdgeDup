from sortedcontainers import SortedDict


class Cache:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache = {}

    def search(self, key):
        if key in self.cache:
            return self.cache[key]
        return None
    def get(self, key):
        return self.cache.get(key)

    def put(self, key, value):
        if key in self.cache:
            return
        if len(self.cache) >= self.capacity:
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
        self.cache[key] = value

    def items(self):
        return self.cache.items()

    def keys(self):
        return self.cache.keys()

    def __getitem__(self, item):
        return self.cache[item]

    def __setitem__(self, key, value):
        self.put(key, value)
        return

    def __delitem__(self, key):
        if key in self.cache:
            del self.cache[key]
        return

    def __len__(self):
        return len(self.cache)

    def __contains__(self, item):
        return item in self.cache

    def __iter__(self):
        return iter(self.cache)


    def __str__(self):
        return str(self.cache)



class LRUCache(Cache):
    def __init__(self, capacity: int):
        super().__init__(capacity)
        self.queue = []

    def search(self, key: str):
        if key in self.cache:
            self.queue.remove(key)
            self.queue.append(key)
            return self.cache[key]
        return None

    def put(self, key: str, value):
        evitObject = None
        if key in self.cache:
            self.queue.remove(key)
        elif len(self.cache) >= self.capacity:
            oldest_key = self.queue.pop(0)
            evitObject = self.cache[oldest_key]
            del self.cache[oldest_key]
        self.cache[key] = value
        self.queue.append(key)
        return evitObject

    def __delitem__(self, key):
        if key in self.cache:
            self.queue.remove(key)
            del self.cache[key]
        return
class LFUCache(Cache):
    def __init__(self, capacity: int):
        super().__init__(capacity)
        self.freq = {}
        self.freq_to_keys = {}

    def search(self, key: str):
        if key in self.cache:
            self.freq_to_keys[self.freq[key]].remove(key)
            if not self.freq_to_keys[self.freq[key]]:
                del self.freq_to_keys[self.freq[key]]
            self.freq[key] += 1
            if self.freq[key] in self.freq_to_keys:
                self.freq_to_keys[self.freq[key]].append(key)
            else:
                self.freq_to_keys[self.freq[key]] = [key]
            return self.cache[key]
        return None

    def put(self, key: str, value):
        evitObject = None
        if key in self.cache:
            self.cache[key] = value
            self.freq[key] += 1
            self.freq_to_keys[self.freq[key]].remove(key)
            if not self.freq_to_keys[self.freq[key]]:
                del self.freq_to_keys[self.freq[key]]
            if self.freq[key] in self.freq_to_keys:
                self.freq_to_keys[self.freq[key]].append(key)
            else:
                self.freq_to_keys[self.freq[key]] = [key]
            return
        if len(self.cache) >= self.capacity:
            min_freq = min(self.freq_to_keys)
            oldest_key = self.freq_to_keys[min_freq].pop(0)
            if not self.freq_to_keys[min_freq]:
                del self.freq_to_keys[min_freq]
            evitObject = self.cache[oldest_key]
            del self.cache[oldest_key]
            del self.freq[oldest_key]
        self.cache[key] = value
        self.freq[key] = 1
        if 1 in self.freq_to_keys:
            self.freq_to_keys[1].append(key)
        else:
            self.freq_to_keys[1] = [key]
        return evitObject

    def __delitem__(self, key):
        if key in self.cache:
            self.freq_to_keys[self.freq[key]].remove(key)
            if not self.freq_to_keys[self.freq[key]]:
                del self.freq_to_keys[self.freq[key]]
            del self.cache[key]
            del self.freq[key]
        return

class FIFOCache(Cache):
    def __init__(self, capacity: int):
        super().__init__(capacity)
        self.queue = []

    def search(self, key: str):
        if key in self.cache:
            return self.cache[key]
        return None

    def put(self, key: str, value):
        evitObject = None
        if key in self.cache:
            return
        if len(self.cache) >= self.capacity:
            oldest_key = self.queue.pop(0)
            evitObject = self.cache[oldest_key]
            del self.cache[oldest_key]
        self.cache[key] = value
        self.queue.append(key)
        return evitObject

class CacheFactory:
    @staticmethod
    def create_cache(cache_type: str, capacity: int):
        if cache_type == "LRU":
            return LRUCache(capacity)
        elif cache_type == "LFU":
            return LFUCache(capacity)
        elif cache_type == "FIFO":
            return FIFOCache(capacity)
        else:
            return Cache(capacity)


if __name__ == "__main__":
    # cache = CacheFactory.create_cache("LFU", 3)
    # cache.put("1", 1)
    # cache.put("2", 2)
    # cache.put("3", 3)
    # cache.search("1")
    # cache.search("1")
    # cache.search("2")
    # cache.put('4',4)
    # cache.search('4')
    # cache.put('5',5)
    # if '1' in cache:
    #     print('1 in cache')
    # for key, item in cache.items():
    #     print(item)
    sdict = SortedDict()
    for key, value in sdict.items():
        print(key, value)

