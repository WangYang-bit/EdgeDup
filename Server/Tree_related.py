import pybloom2
import copy
import numpy as np

element_num = 3000


class Tree(object):
    def __init__(self, hop, near_nodes, cbf):
        self.hop = hop
        self.near_nodes = near_nodes
        self.cbf = cbf
        self.newTree = TreeNode(name='root', BF=pybloom2.CountingBloomFilter(error_rate=0.001, element_num=element_num))

    def _to_bin(self, value: int) -> str:
        _bin = bin(value)[2:]
        if len(_bin) < 8:
            _bin = '0' * (8 - len(_bin)) + _bin
        if len(_bin) > 8:
            _bin = '11111111'
        return _bin

    def generate_hcbf_tree(self, second_level_filters):
        sum_bits_of_cbf, second_level_sum_bits = [], []
        root_filter = pybloom2.CountingBloomFilter(error_rate=0.001, element_num=element_num)
        for j in range(self.hop):
            near_nodes = self.near_nodes[j]
            sum_bits_of_cbf.clear()
            if len(near_nodes) > 1:
                second_level_filters[j + 1] = pybloom2.CountingBloomFilter(error_rate=0.001, element_num=element_num)
                for k in range(len(near_nodes)):
                    sum_bits_of_cbf.append(copy.deepcopy(self.cbf[near_nodes[k]].each_bit))
                second_level_sum_bits.append(copy.deepcopy(np.sum(np.array(sum_bits_of_cbf), axis=0)))
                second_level_filters[j + 1].each_bit = copy.deepcopy(np.sum(np.array(sum_bits_of_cbf), axis=0))
            elif len(near_nodes) == 1:
                second_level_filters[j + 1] = copy.deepcopy(self.cbf[near_nodes[0]])
                second_level_sum_bits.append(copy.deepcopy(self.cbf[near_nodes[0]].each_bit))
        root_filter.each_bit = copy.deepcopy(np.sum(second_level_sum_bits, axis=0))
        return root_filter, second_level_filters
        # root_filter is the set of BF in the first layer, and second_level_filters is the set of BF in the second layer


class TreeNode(object):
    """The basic node of tree structure"""

    def __init__(self, name, parent=None, BF=None, latency=0):
        super(TreeNode, self).__init__()
        self.name = name
        self.parent = parent
        self.child = {}
        self.BF = BF
        self.latency = latency

    def __repr__(self):
        return 'TreeNode(%s)' % self.name

    def __contains__(self, item):
        return item in self.child

    def __len__(self):
        """return number of children node"""
        return len(self.child)

    def __bool__(self, item):
        """always return True for exist node"""
        return True

    @property
    def path(self):
        """return path string (from root to current node)"""
        if self.parent is not None:
            return '%s %s' % (self.parent.path.strip(), self.name)
        else:
            return self.name

    def get_child(self, name, defval=None):
        """get a child node of current node"""
        return self.child.get(name, defval)

    def add_child(self, name, obj=None, BF=None, latency=0):
        """add a child node to current node"""
        if obj and not isinstance(obj, TreeNode):
            raise ValueError('TreeNode only add another TreeNode obj as child')
        if obj is None:
            obj = TreeNode(name)
        obj.parent = self
        self.child[name] = obj
        obj.BF = BF
        obj.latency = latency
        return obj

    def del_child(self, name):
        """remove a child node from current node"""
        if name in self.child:
            del self.child[name]

    def find_child(self, path, create=False):
        """find child node by path/name, return None if not found"""
        # convert path to a list if input is a string
        path = path if isinstance(path, list) else path.split()
        cur = self
        for sub in path:
            # search
            obj = cur.get_child(sub)
            if obj is None and create:
                # creelement_numate new node if need
                obj = cur.add_child(sub)
            # check if search done
            if obj is None:
                break
            cur = obj
        return obj

    def items(self):
        return self.child.items()

    def dump(self, indent=0):
        """dump tree to string"""
        # tab = '    '*(indent-1) + ' |- ' if indent > 0 else ''
        # print('%s%s' % (tab, self.name))
        for name, obj in self.items():
            obj.dump(indent + 1)

    def getChildren(self):
        return self.child

    def find_data_cbf(self, search_data, search_cost):
        saved_id = 1000
        result_search_cbf, saved_id, search_cost = breadthFirstByRecursion(parent=self.getChildren(),
                                                                           search_data=search_data,
                                                                           result_search_cbf=False,
                                                                           search_cost=search_cost,
                                                                           saved_id=saved_id, next_search=dict())
        return result_search_cbf, saved_id, search_cost

    def find_data_hcbf(self, search_data, search_cost):
        result_search_hcbf = False
        saved_id = 0
        search_cost += 1
        if self.BF.exists(search_data):
            result_search_hcbf, saved_id, search_cost = depthFirstByRecursion(parent=self, search_data=search_data,
                                                                              search_cost=search_cost,
                                                                              result_search_cbf=result_search_hcbf,
                                                                              saved_id=saved_id)
        return result_search_hcbf, saved_id, search_cost


def depthFirstByRecursion(parent, search_data, search_cost, result_search_cbf, saved_id):
    children = parent.child
    for c in children:
        search_cost += 1
        if children[c].BF.exists(search_data):
            if children[c].child == {}:
                saved_id = children[c].name
                result_search_cbf = True
                return result_search_cbf, saved_id, search_cost
            else:
                result_search_cbf, saved_id, search_cost = depthFirstByRecursion(children[c], search_data,
                                                                                 copy.deepcopy(search_cost),
                                                                                 result_search_cbf, saved_id)
                if result_search_cbf:
                    return result_search_cbf, saved_id, search_cost
    return result_search_cbf, saved_id, search_cost


def breadthFirstByRecursion(parent, search_data, result_search_cbf, saved_id, search_cost, next_search, index=0):
    if result_search_cbf:
        return result_search_cbf, saved_id, search_cost
    else:
        for p in parent:
            search_cost += 1
            if parent[p].BF.exists(search_data):
                result_search_cbf = True
                saved_id = parent[p].name
                return result_search_cbf, saved_id, search_cost
            else:
                if len(parent[p].getChildren()) != 0:
                    for x in parent[p].getChildren():
                        next_search[x] = parent[p].getChildren()[x]
            if index == len(parent) - 1:
                parent = next_search
                next_search = dict()
                index = 0
            else:
                index += 1
        if len(parent) != 0:
            result_search_cbf, saved_id, search_cost = breadthFirstByRecursion(parent, search_data, result_search_cbf,
                                                                               saved_id, search_cost, next_search,
                                                                               index)
        return result_search_cbf, saved_id, search_cost
