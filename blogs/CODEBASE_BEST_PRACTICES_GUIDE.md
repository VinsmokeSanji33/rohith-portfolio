# Python Mastery Guide: Google Interview Preparation

## Table of Contents
1. [Python Fundamentals & Core Concepts](#python-fundamentals--core-concepts)
2. [Advanced OOP & Design Patterns](#advanced-oop--design-patterns)
3. [Data Structures & Algorithms](#data-structures--algorithms)
4. [Concurrency & Parallelism](#concurrency--parallelism)
5. [Memory Management & Performance](#memory-management--performance)
6. [Metaclasses & Descriptors](#metaclasses--descriptors)
7. [Decorators & Context Managers](#decorators--context-managers)
8. [Type System & Annotations](#type-system--annotations)
9. [Error Handling & Exceptions](#error-handling--exceptions)
10. [System Design Patterns](#system-design-patterns)
11. [Interview Problem Patterns](#interview-problem-patterns)
12. [Google-Specific Topics](#google-specific-topics)

---

## Python Fundamentals & Core Concepts

### 1. Python's Execution Model

**Understanding Python's Internals**:

```python
# Python is interpreted, but uses bytecode compilation
def demonstrate_bytecode():
    import dis
    def add(a, b):
        return a + b
    
    # View bytecode
    dis.dis(add)
    # Output:
    #   2           0 LOAD_FAST                0 (a)
    #               2 LOAD_FAST                1 (b)
    #               4 BINARY_ADD
    #               6 RETURN_VALUE

# Python's name binding (not variables!)
x = [1, 2, 3]
y = x  # Both names point to same object
y.append(4)
print(x)  # [1, 2, 3, 4] - same object!

# Mutable vs Immutable
def demonstrate_mutability():
    # Immutable: int, str, tuple, frozenset
    a = 5
    b = a
    a = 10  # Creates new object, b still points to 5
    print(b)  # 5
    
    # Mutable: list, dict, set
    lst1 = [1, 2, 3]
    lst2 = lst1
    lst1.append(4)  # Modifies same object
    print(lst2)  # [1, 2, 3, 4]
```

**Key Interview Points**:
- Python uses reference counting + cycle detection for garbage collection
- Everything is an object (even functions and classes)
- `is` vs `==`: identity vs equality
- Small integers (-5 to 256) are interned

### 2. Advanced Data Structures

**Custom Data Structures**:

```python
from collections import deque, defaultdict, Counter, OrderedDict
from dataclasses import dataclass, field
from typing import Generic, TypeVar, Optional
import heapq

# Implementing a LRU Cache (Google interview favorite!)
class LRUCache:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache: OrderedDict[int, int] = OrderedDict()
    
    def get(self, key: int) -> int:
        if key not in self.cache:
            return -1
        # Move to end (most recently used)
        self.cache.move_to_end(key)
        return self.cache[key]
    
    def put(self, key: int, value: int) -> None:
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        if len(self.cache) > self.capacity:
            # Remove least recently used (first item)
            self.cache.popitem(last=False)

# Implementing a Trie (Prefix Tree)
class TrieNode:
    def __init__(self):
        self.children: dict[str, 'TrieNode'] = {}
        self.is_end = False

class Trie:
    def __init__(self):
        self.root = TrieNode()
    
    def insert(self, word: str) -> None:
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_end = True
    
    def search(self, word: str) -> bool:
        node = self.root
        for char in word:
            if char not in node.children:
                return False
            node = node.children[char]
        return node.is_end
    
    def startsWith(self, prefix: str) -> bool:
        node = self.root
        for char in prefix:
            if char not in node.children:
                return False
            node = node.children[char]
        return True

# Priority Queue with custom comparator
class Task:
    def __init__(self, priority: int, name: str):
        self.priority = priority
        self.name = name
    
    def __lt__(self, other):
        return self.priority < other.priority

# Usage
tasks = []
heapq.heappush(tasks, Task(3, "Low"))
heapq.heappush(tasks, Task(1, "High"))
heapq.heappush(tasks, Task(2, "Medium"))
while tasks:
    print(heapq.heappop(tasks).name)  # High, Medium, Low
```

### 3. Generators & Iterators

**Advanced Generator Patterns**:

```python
# Generator for infinite sequences
def fibonacci():
    """Infinite Fibonacci generator."""
    a, b = 0, 1
    while True:
        yield a
        a, b = b, a + b

# Usage
fib = fibonacci()
print([next(fib) for _ in range(10)])  # [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]

# Generator pipeline (functional style)
def numbers():
    yield from range(10)

def square(seq):
    for x in seq:
        yield x ** 2

def filter_even(seq):
    for x in seq:
        if x % 2 == 0:
            yield x

# Chain generators
result = list(filter_even(square(numbers())))
print(result)  # [0, 4, 16, 36, 64]

# Coroutine-based generator (two-way communication)
def coroutine_generator():
    """Generator that can receive values."""
    value = yield
    while True:
        value = yield value * 2

gen = coroutine_generator()
next(gen)  # Prime the generator
print(gen.send(5))   # 10
print(gen.send(10))   # 20

# Custom iterator class
class Countdown:
    def __init__(self, start: int):
        self.start = start
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.start <= 0:
            raise StopIteration
        self.start -= 1
        return self.start + 1

for num in Countdown(5):
    print(num)  # 5, 4, 3, 2, 1
```

---

## Advanced OOP & Design Patterns

### 1. Abstract Base Classes & Protocols

```python
from abc import ABC, abstractmethod
from typing import Protocol, runtime_checkable

# Traditional ABC approach
class Shape(ABC):
    @abstractmethod
    def area(self) -> float:
        pass
    
    @abstractmethod
    def perimeter(self) -> float:
        pass
    
    def describe(self) -> str:
        return f"Area: {self.area()}, Perimeter: {self.perimeter()}"

class Rectangle(Shape):
    def __init__(self, width: float, height: float):
        self.width = width
        self.height = height
    
    def area(self) -> float:
        return self.width * self.height
    
    def perimeter(self) -> float:
        return 2 * (self.width + self.height)

# Modern Protocol approach (structural typing)
@runtime_checkable
class Drawable(Protocol):
    def draw(self) -> None: ...
    def get_color(self) -> str: ...

class Circle:
    def draw(self) -> None:
        print("Drawing circle")
    
    def get_color(self) -> str:
        return "red"

# Works with Protocol even though Circle doesn't inherit
def render(obj: Drawable) -> None:
    obj.draw()

render(Circle())  # Works!
```

### 2. Design Patterns

**Singleton Pattern (Thread-Safe)**:

```python
import threading
from typing import Optional

class DatabaseConnection:
    _instance: Optional['DatabaseConnection'] = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not hasattr(self, 'initialized'):
            self.connection_string = "db://localhost"
            self.initialized = True
```

**Factory Pattern with Registry**:

```python
class AnimalFactory:
    _registry: dict[str, type] = {}
    
    @classmethod
    def register(cls, name: str):
        def decorator(animal_class):
            cls._registry[name] = animal_class
            return animal_class
        return decorator
    
    @classmethod
    def create(cls, name: str, *args, **kwargs):
        if name not in cls._registry:
            raise ValueError(f"Unknown animal: {name}")
        return cls._registry[name](*args, **kwargs)

@AnimalFactory.register("dog")
class Dog:
    def speak(self):
        return "Woof!"

@AnimalFactory.register("cat")
class Cat:
    def speak(self):
        return "Meow!"

# Usage
dog = AnimalFactory.create("dog")
print(dog.speak())  # Woof!
```

**Observer Pattern**:

```python
from typing import List, Callable

class EventEmitter:
    def __init__(self):
        self._listeners: dict[str, List[Callable]] = {}
    
    def on(self, event: str, callback: Callable):
        if event not in self._listeners:
            self._listeners[event] = []
        self._listeners[event].append(callback)
    
    def emit(self, event: str, *args, **kwargs):
        if event in self._listeners:
            for callback in self._listeners[event]:
                callback(*args, **kwargs)

# Usage
emitter = EventEmitter()
emitter.on("click", lambda x: print(f"Clicked: {x}"))
emitter.on("click", lambda x: print(f"Also clicked: {x}"))
emitter.emit("click", "button1")
```

**Strategy Pattern**:

```python
from abc import ABC, abstractmethod

class SortingStrategy(ABC):
    @abstractmethod
    def sort(self, data: list) -> list:
        pass

class QuickSort(SortingStrategy):
    def sort(self, data: list) -> list:
        if len(data) <= 1:
            return data
        pivot = data[len(data) // 2]
        left = [x for x in data if x < pivot]
        middle = [x for x in data if x == pivot]
        right = [x for x in data if x > pivot]
        return self.sort(left) + middle + self.sort(right)

class MergeSort(SortingStrategy):
    def sort(self, data: list) -> list:
        if len(data) <= 1:
            return data
        mid = len(data) // 2
        left = self.sort(data[:mid])
        right = self.sort(data[mid:])
        return self._merge(left, right)
    
    def _merge(self, left: list, right: list) -> list:
        result = []
        i = j = 0
        while i < len(left) and j < len(right):
            if left[i] <= right[j]:
                result.append(left[i])
                i += 1
            else:
                result.append(right[j])
                j += 1
        result.extend(left[i:])
        result.extend(right[j:])
        return result

class Sorter:
    def __init__(self, strategy: SortingStrategy):
        self.strategy = strategy
    
    def sort(self, data: list) -> list:
        return self.strategy.sort(data)

# Usage
sorter = Sorter(QuickSort())
print(sorter.sort([3, 1, 4, 1, 5, 9, 2, 6]))
```

---

## Data Structures & Algorithms

### 1. Advanced Tree Structures

```python
# Binary Search Tree with all operations
class TreeNode:
    def __init__(self, val: int):
        self.val = val
        self.left: Optional['TreeNode'] = None
        self.right: Optional['TreeNode'] = None

class BST:
    def __init__(self):
        self.root: Optional[TreeNode] = None
    
    def insert(self, val: int) -> None:
        self.root = self._insert(self.root, val)
    
    def _insert(self, node: Optional[TreeNode], val: int) -> TreeNode:
        if node is None:
            return TreeNode(val)
        if val < node.val:
            node.left = self._insert(node.left, val)
        elif val > node.val:
            node.right = self._insert(node.right, val)
        return node
    
    def search(self, val: int) -> bool:
        return self._search(self.root, val)
    
    def _search(self, node: Optional[TreeNode], val: int) -> bool:
        if node is None:
            return False
        if val == node.val:
            return True
        return self._search(node.left if val < node.val else node.right, val)
    
    def delete(self, val: int) -> None:
        self.root = self._delete(self.root, val)
    
    def _delete(self, node: Optional[TreeNode], val: int) -> Optional[TreeNode]:
        if node is None:
            return None
        if val < node.val:
            node.left = self._delete(node.left, val)
        elif val > node.val:
            node.right = self._delete(node.right, val)
        else:
            if node.left is None:
                return node.right
            if node.right is None:
                return node.left
            # Node has two children
            min_node = self._find_min(node.right)
            node.val = min_node.val
            node.right = self._delete(node.right, min_node.val)
        return node
    
    def _find_min(self, node: TreeNode) -> TreeNode:
        while node.left:
            node = node.left
        return node
```

### 2. Graph Algorithms

```python
from collections import deque, defaultdict
from typing import List, Set

class Graph:
    def __init__(self):
        self.adj: dict[int, List[int]] = defaultdict(list)
    
    def add_edge(self, u: int, v: int):
        self.adj[u].append(v)
        self.adj[v].append(u)  # Undirected
    
    def bfs(self, start: int) -> List[int]:
        """BFS traversal."""
        visited: Set[int] = set()
        queue = deque([start])
        result = []
        
        while queue:
            node = queue.popleft()
            if node not in visited:
                visited.add(node)
                result.append(node)
                for neighbor in self.adj[node]:
                    if neighbor not in visited:
                        queue.append(neighbor)
        return result
    
    def dfs(self, start: int) -> List[int]:
        """DFS traversal."""
        visited: Set[int] = set()
        result = []
        
        def dfs_helper(node: int):
            if node in visited:
                return
            visited.add(node)
            result.append(node)
            for neighbor in self.adj[node]:
                dfs_helper(neighbor)
        
        dfs_helper(start)
        return result
    
    def topological_sort(self) -> List[int]:
        """Topological sort for DAG."""
        in_degree: dict[int, int] = defaultdict(int)
        for node in self.adj:
            for neighbor in self.adj[node]:
                in_degree[neighbor] += 1
        
        queue = deque([node for node in self.adj if in_degree[node] == 0])
        result = []
        
        while queue:
            node = queue.popleft()
            result.append(node)
            for neighbor in self.adj[node]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
        
        return result
```

### 3. Dynamic Programming Patterns

```python
from functools import lru_cache
from typing import Dict

# Memoization decorator
def memoize(func):
    cache: Dict = {}
    def wrapper(*args):
        if args not in cache:
            cache[args] = func(*args)
        return cache[args]
    return wrapper

# Fibonacci with memoization
@lru_cache(maxsize=None)
def fibonacci(n: int) -> int:
    if n <= 1:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)

# Longest Common Subsequence
def lcs(s1: str, s2: str) -> int:
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    
    return dp[m][n]

# Knapsack Problem
def knapsack(weights: List[int], values: List[int], capacity: int) -> int:
    n = len(weights)
    dp = [[0] * (capacity + 1) for _ in range(n + 1)]
    
    for i in range(1, n + 1):
        for w in range(1, capacity + 1):
            if weights[i - 1] <= w:
                dp[i][w] = max(
                    dp[i - 1][w],
                    dp[i - 1][w - weights[i - 1]] + values[i - 1]
                )
            else:
                dp[i][w] = dp[i - 1][w]
    
    return dp[n][capacity]
```

---

## Concurrency & Parallelism

### 1. Threading

```python
import threading
import time
from queue import Queue
from typing import Callable

class ThreadPool:
    def __init__(self, num_workers: int):
        self.num_workers = num_workers
        self.task_queue = Queue()
        self.workers = []
        self._shutdown = False
    
    def start(self):
        for _ in range(self.num_workers):
            worker = threading.Thread(target=self._worker)
            worker.daemon = True
            worker.start()
            self.workers.append(worker)
    
    def submit(self, func: Callable, *args, **kwargs):
        if self._shutdown:
            raise RuntimeError("ThreadPool is shutdown")
        self.task_queue.put((func, args, kwargs))
    
    def _worker(self):
        while not self._shutdown:
            try:
                func, args, kwargs = self.task_queue.get(timeout=1)
                func(*args, **kwargs)
                self.task_queue.task_done()
            except:
                continue
    
    def shutdown(self):
        self._shutdown = True
        self.task_queue.join()

# Usage
pool = ThreadPool(4)
pool.start()
for i in range(10):
    pool.submit(print, f"Task {i}")
pool.shutdown()
```

### 2. Async/Await

```python
import asyncio
from typing import List, Coroutine

# Async generator
async def async_range(start: int, stop: int):
    for i in range(start, stop):
        yield i
        await asyncio.sleep(0.1)

# Async context manager
class AsyncResource:
    async def __aenter__(self):
        print("Acquiring resource")
        await asyncio.sleep(0.1)
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        print("Releasing resource")
        await asyncio.sleep(0.1)

# Semaphore for rate limiting
async def fetch_with_limit(semaphore: asyncio.Semaphore, url: str):
    async with semaphore:
        print(f"Fetching {url}")
        await asyncio.sleep(1)
        return f"Data from {url}"

async def main():
    semaphore = asyncio.Semaphore(3)  # Max 3 concurrent
    urls = [f"url{i}" for i in range(10)]
    tasks = [fetch_with_limit(semaphore, url) for url in urls]
    results = await asyncio.gather(*tasks)
    return results

# asyncio.run(main())
```

### 3. Multiprocessing

```python
from multiprocessing import Pool, Manager, Process, Queue
import time

def worker_function(data):
    """CPU-intensive task."""
    result = sum(i ** 2 for i in range(data))
    return result

def parallel_map(func, data, num_workers=4):
    with Pool(num_workers) as pool:
        return pool.map(func, data)

# Shared state with Manager
def shared_counter_worker(counter, lock):
    with lock:
        counter.value += 1

def shared_state_example():
    manager = Manager()
    counter = manager.Value('i', 0)
    lock = manager.Lock()
    
    processes = [
        Process(target=shared_counter_worker, args=(counter, lock))
        for _ in range(10)
    ]
    
    for p in processes:
        p.start()
    for p in processes:
        p.join()
    
    print(f"Final counter: {counter.value}")  # 10
```

---

## Memory Management & Performance

### 1. Memory Optimization

```python
import sys
from __slots__ import __slots__

# Using __slots__ to reduce memory
class Point:
    __slots__ = ('x', 'y')  # Prevents __dict__ creation
    
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y

# Memory profiling
def memory_usage():
    import tracemalloc
    tracemalloc.start()
    
    # Your code here
    data = [i for i in range(1000000)]
    
    current, peak = tracemalloc.get_traced_memory()
    print(f"Current: {current / 1024 / 1024:.2f} MB")
    print(f"Peak: {peak / 1024 / 1024:.2f} MB")
    tracemalloc.stop()

# Weak references to avoid circular dependencies
import weakref

class Node:
    def __init__(self, value: int):
        self.value = value
        self._parent = None
    
    @property
    def parent(self):
        return self._parent() if self._parent else None
    
    @parent.setter
    def parent(self, node):
        self._parent = weakref.ref(node) if node else None
```

### 2. Performance Optimization

```python
# Using list comprehensions vs loops
# Fast
squares = [x ** 2 for x in range(1000)]

# Slower
squares = []
for x in range(1000):
    squares.append(x ** 2)

# Using generators for memory efficiency
def process_large_file(filename: str):
    with open(filename) as f:
        for line in f:
            yield process_line(line)

# Caching expensive computations
from functools import lru_cache

@lru_cache(maxsize=128)
def expensive_computation(n: int) -> int:
    # Simulate expensive operation
    return sum(i ** 2 for i in range(n))

# Using bisect for sorted lists
import bisect

def insert_sorted(lst: List[int], value: int):
    bisect.insort(lst, value)

# Using collections.Counter for frequency counting
from collections import Counter

def find_most_common(items: List[str], k: int = 1):
    return Counter(items).most_common(k)
```

---

## Metaclasses & Descriptors

### 1. Metaclasses

```python
class SingletonMeta(type):
    _instances = {}
    
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]

class Database(metaclass=SingletonMeta):
    def __init__(self):
        print("Database initialized")

db1 = Database()  # Database initialized
db2 = Database()  # (no output)
print(db1 is db2)  # True

# Metaclass for automatic property registration
class PropertyRegistry(type):
    def __new__(mcs, name, bases, namespace):
        cls = super().__new__(mcs, name, bases, namespace)
        cls._properties = {
            key: value for key, value in namespace.items()
            if isinstance(value, property)
        }
        return cls

class MyClass(metaclass=PropertyRegistry):
    @property
    def value(self):
        return 42
```

### 2. Descriptors

```python
class ValidatedProperty:
    def __init__(self, validator: Callable):
        self.validator = validator
        self.name = None
    
    def __set_name__(self, owner, name):
        self.name = name
    
    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        return obj.__dict__.get(self.name)
    
    def __set__(self, obj, value):
        if not self.validator(value):
            raise ValueError(f"Invalid value: {value}")
        obj.__dict__[self.name] = value

class Person:
    age = ValidatedProperty(lambda x: 0 <= x <= 150)
    email = ValidatedProperty(lambda x: '@' in str(x))
    
    def __init__(self, age: int, email: str):
        self.age = age
        self.email = email

# Usage
person = Person(25, "test@example.com")
# person.age = 200  # Raises ValueError
```

---

## Decorators & Context Managers

### 1. Advanced Decorators

```python
from functools import wraps
import time
from typing import Callable, TypeVar, ParamSpec

P = ParamSpec('P')
T = TypeVar('T')

# Decorator with arguments
def retry(max_attempts: int = 3, delay: float = 1.0):
    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_attempts - 1:
                        raise
                    time.sleep(delay)
            raise RuntimeError("Max attempts reached")
        return wrapper
    return decorator

# Class-based decorator
class Timer:
    def __init__(self, func: Callable):
        self.func = func
        wraps(func)(self)
    
    def __call__(self, *args, **kwargs):
        start = time.time()
        result = self.func(*args, **kwargs)
        elapsed = time.time() - start
        print(f"{self.func.__name__} took {elapsed:.2f}s")
        return result

# Decorator for method registration
class CommandRegistry:
    _commands = {}
    
    @classmethod
    def register(cls, name: str):
        def decorator(func):
            cls._commands[name] = func
            return func
        return decorator
    
    @classmethod
    def execute(cls, name: str, *args, **kwargs):
        if name not in cls._commands:
            raise ValueError(f"Unknown command: {name}")
        return cls._commands[name](*args, **kwargs)

@CommandRegistry.register("greet")
def greet(name: str):
    return f"Hello, {name}!"
```

### 2. Context Managers

```python
from contextlib import contextmanager, ExitStack
import threading

# Custom context manager
class FileManager:
    def __init__(self, filename: str, mode: str = 'r'):
        self.filename = filename
        self.mode = mode
        self.file = None
    
    def __enter__(self):
        self.file = open(self.filename, self.mode)
        return self.file
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.file:
            self.file.close()
        return False  # Don't suppress exceptions

# Function-based context manager
@contextmanager
def temporary_change(obj, attr, value):
    old_value = getattr(obj, attr)
    setattr(obj, attr, value)
    try:
        yield
    finally:
        setattr(obj, attr, old_value)

# Multiple context managers
def multiple_resources():
    with ExitStack() as stack:
        file1 = stack.enter_context(open('file1.txt'))
        file2 = stack.enter_context(open('file2.txt'))
        # Both files automatically closed

# Thread-safe context manager
class LockedResource:
    def __init__(self):
        self.lock = threading.Lock()
        self.data = {}
    
    def __enter__(self):
        self.lock.acquire()
        return self.data
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.lock.release()
```

---

## Type System & Annotations

### 1. Advanced Type Hints

```python
from typing import (
    TypeVar, Generic, Protocol, runtime_checkable,
    Union, Optional, Literal, TypedDict, Callable,
    List, Dict, Tuple, Set, FrozenSet, Any
)
from dataclasses import dataclass

# Generic classes
T = TypeVar('T')
K = TypeVar('K')
V = TypeVar('V')

class Stack(Generic[T]):
    def __init__(self):
        self._items: List[T] = []
    
    def push(self, item: T) -> None:
        self._items.append(item)
    
    def pop(self) -> T:
        return self._items.pop()

# TypedDict for structured data
class UserDict(TypedDict):
    name: str
    age: int
    email: str

# Literal types
def process_status(status: Literal["pending", "completed", "failed"]):
    pass

# Protocol for structural typing
@runtime_checkable
class Drawable(Protocol):
    def draw(self) -> None: ...

# Callable types
def apply_func(func: Callable[[int, int], int], a: int, b: int) -> int:
    return func(a, b)

# Union types (Python 3.10+)
def process(value: int | str | None) -> str:
    if value is None:
        return "None"
    return str(value)
```

### 2. Type Checking Patterns

```python
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from expensive_module import ExpensiveClass

def use_expensive_class() -> 'ExpensiveClass':
    # Type checker sees the type, but runtime doesn't import it
    pass

# Type narrowing
def process_value(value: int | str | None):
    if value is None:
        return  # Type checker knows value is None here
    if isinstance(value, int):
        # Type checker knows value is int here
        return value * 2
    # Type checker knows value is str here
    return value.upper()
```

---

## Error Handling & Exceptions

### 1. Custom Exceptions

```python
class ValidationError(Exception):
    """Base exception for validation errors."""
    pass

class InvalidEmailError(ValidationError):
    """Raised when email format is invalid."""
    pass

class InvalidAgeError(ValidationError):
    """Raised when age is out of valid range."""
    pass

# Exception chaining
def process_data():
    try:
        risky_operation()
    except ValueError as e:
        raise ProcessingError("Failed to process data") from e

# Context managers for error handling
from contextlib import suppress

# Suppress specific exceptions
with suppress(FileNotFoundError):
    os.remove('file.txt')

# Exception groups (Python 3.11+)
try:
    # Multiple operations that might fail
    pass
except* ValueError as eg:
    for exc in eg.exceptions:
        print(f"ValueError: {exc}")
except* KeyError as eg:
    for exc in eg.exceptions:
        print(f"KeyError: {exc}")
```

### 2. Error Recovery Patterns

```python
def retry_on_failure(max_retries: int = 3):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise
                    time.sleep(2 ** attempt)  # Exponential backoff
            raise RuntimeError("Max retries exceeded")
        return wrapper
    return decorator

# Circuit breaker pattern
class CircuitBreaker:
    def __init__(self, failure_threshold: int = 5, timeout: float = 60.0):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "closed"  # closed, open, half_open
    
    def call(self, func, *args, **kwargs):
        if self.state == "open":
            if time.time() - self.last_failure_time > self.timeout:
                self.state = "half_open"
            else:
                raise RuntimeError("Circuit breaker is open")
        
        try:
            result = func(*args, **kwargs)
            if self.state == "half_open":
                self.state = "closed"
                self.failure_count = 0
            return result
        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()
            if self.failure_count >= self.failure_threshold:
                self.state = "open"
            raise
```

---

## System Design Patterns

### 1. Producer-Consumer Pattern

```python
from queue import Queue
import threading
import time

class ProducerConsumer:
    def __init__(self, buffer_size: int = 10):
        self.queue = Queue(maxsize=buffer_size)
        self.lock = threading.Lock()
    
    def producer(self, items: List[int]):
        for item in items:
            self.queue.put(item)
            print(f"Produced: {item}")
            time.sleep(0.1)
    
    def consumer(self):
        while True:
            item = self.queue.get()
            if item is None:  # Poison pill
                break
            print(f"Consumed: {item}")
            self.queue.task_done()
            time.sleep(0.2)

# Usage
pc = ProducerConsumer()
producer_thread = threading.Thread(target=pc.producer, args=([1, 2, 3, 4, 5],))
consumer_thread = threading.Thread(target=pc.consumer)

consumer_thread.start()
producer_thread.start()
producer_thread.join()
pc.queue.put(None)  # Signal completion
consumer_thread.join()
```

### 2. Pub-Sub Pattern

```python
from typing import Dict, List, Callable
import threading

class EventBus:
    def __init__(self):
        self._subscribers: Dict[str, List[Callable]] = {}
        self._lock = threading.Lock()
    
    def subscribe(self, event_type: str, callback: Callable):
        with self._lock:
            if event_type not in self._subscribers:
                self._subscribers[event_type] = []
            self._subscribers[event_type].append(callback)
    
    def publish(self, event_type: str, *args, **kwargs):
        with self._lock:
            subscribers = self._subscribers.get(event_type, [])
        for callback in subscribers:
            callback(*args, **kwargs)

# Usage
bus = EventBus()
bus.subscribe("user.created", lambda user: print(f"User created: {user}"))
bus.subscribe("user.deleted", lambda user: print(f"User deleted: {user}"))
bus.publish("user.created", "Alice")
```

### 3. Cache Patterns

```python
from functools import lru_cache
import time
from typing import Optional

# Time-based cache
class TimedCache:
    def __init__(self, ttl: float = 60.0):
        self.cache: Dict[str, Tuple[Any, float]] = {}
        self.ttl = ttl
    
    def get(self, key: str) -> Optional[Any]:
        if key in self.cache:
            value, timestamp = self.cache[key]
            if time.time() - timestamp < self.ttl:
                return value
            del self.cache[key]
        return None
    
    def set(self, key: str, value: Any):
        self.cache[key] = (value, time.time())

# LRU Cache implementation
from collections import OrderedDict

class LRUCache:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache = OrderedDict()
    
    def get(self, key: int) -> int:
        if key not in self.cache:
            return -1
        self.cache.move_to_end(key)
        return self.cache[key]
    
    def put(self, key: int, value: int):
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)
```

---

## Interview Problem Patterns

### 1. Two Pointers

```python
def two_sum_sorted(nums: List[int], target: int) -> List[int]:
    left, right = 0, len(nums) - 1
    while left < right:
        current_sum = nums[left] + nums[right]
        if current_sum == target:
            return [left, right]
        elif current_sum < target:
            left += 1
        else:
            right -= 1
    return []

def remove_duplicates(nums: List[int]) -> int:
    if not nums:
        return 0
    write_index = 1
    for read_index in range(1, len(nums)):
        if nums[read_index] != nums[read_index - 1]:
            nums[write_index] = nums[read_index]
            write_index += 1
    return write_index
```

### 2. Sliding Window

```python
def max_sum_subarray(nums: List[int], k: int) -> int:
    if len(nums) < k:
        return 0
    window_sum = sum(nums[:k])
    max_sum = window_sum
    for i in range(k, len(nums)):
        window_sum = window_sum - nums[i - k] + nums[i]
        max_sum = max(max_sum, window_sum)
    return max_sum

def longest_substring_no_repeat(s: str) -> int:
    char_index = {}
    start = 0
    max_len = 0
    for end, char in enumerate(s):
        if char in char_index and char_index[char] >= start:
            start = char_index[char] + 1
        char_index[char] = end
        max_len = max(max_len, end - start + 1)
    return max_len
```

### 3. Backtracking

```python
def generate_permutations(nums: List[int]) -> List[List[int]]:
    result = []
    
    def backtrack(current: List[int], remaining: List[int]):
        if not remaining:
            result.append(current[:])
            return
        for i in range(len(remaining)):
            current.append(remaining[i])
            backtrack(current, remaining[:i] + remaining[i+1:])
            current.pop()
    
    backtrack([], nums)
    return result

def solve_n_queens(n: int) -> List[List[str]]:
    result = []
    board = [['.' for _ in range(n)] for _ in range(n)]
    
    def is_safe(row: int, col: int) -> bool:
        for i in range(row):
            if board[i][col] == 'Q':
                return False
            if col - (row - i) >= 0 and board[i][col - (row - i)] == 'Q':
                return False
            if col + (row - i) < n and board[i][col + (row - i)] == 'Q':
                return False
        return True
    
    def backtrack(row: int):
        if row == n:
            result.append([''.join(row) for row in board])
            return
        for col in range(n):
            if is_safe(row, col):
                board[row][col] = 'Q'
                backtrack(row + 1)
                board[row][col] = '.'
    
    backtrack(0)
    return result
```

---

## Google-Specific Topics

### 1. System Design with Python

```python
# Rate Limiter
from collections import deque
import time

class RateLimiter:
    def __init__(self, max_requests: int, window_seconds: float):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests = deque()
    
    def allow_request(self) -> bool:
        now = time.time()
        # Remove old requests outside window
        while self.requests and self.requests[0] < now - self.window_seconds:
            self.requests.popleft()
        
        if len(self.requests) < self.max_requests:
            self.requests.append(now)
            return True
        return False

# Distributed Lock (simplified)
class DistributedLock:
    def __init__(self, lock_id: str, ttl: float = 30.0):
        self.lock_id = lock_id
        self.ttl = ttl
        self.acquired_at = None
    
    def acquire(self) -> bool:
        # In real implementation, use Redis or similar
        if self.acquired_at is None:
            self.acquired_at = time.time()
            return True
        return False
    
    def release(self):
        self.acquired_at = None
```

### 2. Big Data Processing Patterns

```python
# Map-Reduce pattern
def map_reduce(data: List[Any], mapper: Callable, reducer: Callable):
    # Map phase
    mapped = [mapper(item) for item in data]
    
    # Shuffle phase (group by key)
    grouped = {}
    for key, value in mapped:
        if key not in grouped:
            grouped[key] = []
        grouped[key].append(value)
    
    # Reduce phase
    result = {key: reducer(values) for key, values in grouped.items()}
    return result

# Example: Word count
def word_count_mapper(text: str):
    words = text.split()
    return [(word, 1) for word in words]

def word_count_reducer(counts: List[int]):
    return sum(counts)
```

### 3. API Design Patterns

```python
from abc import ABC, abstractmethod
from typing import Dict, Any

class APIHandler(ABC):
    @abstractmethod
    def handle(self, request: Dict[str, Any]) -> Dict[str, Any]:
        pass

class RESTAPIHandler(APIHandler):
    def handle(self, request: Dict[str, Any]) -> Dict[str, Any]:
        method = request.get('method', 'GET')
        path = request.get('path', '/')
        
        if method == 'GET':
            return self._handle_get(path)
        elif method == 'POST':
            return self._handle_post(path, request.get('body'))
        else:
            return {'error': 'Method not allowed'}
    
    def _handle_get(self, path: str) -> Dict[str, Any]:
        return {'data': f'GET {path}'}
    
    def _handle_post(self, path: str, body: Any) -> Dict[str, Any]:
        return {'data': f'POST {path}', 'body': body}
```

---

## Key Takeaways for Google Interviews

1. **Master Python Internals**: Understand CPython, GIL, memory management
2. **Design Patterns**: Know when and how to apply common patterns
3. **Concurrency**: Threading, async/await, multiprocessing
4. **Data Structures**: Implement from scratch, know trade-offs
5. **Algorithms**: Time/space complexity, optimization techniques
6. **System Design**: Scalability, distributed systems, caching
7. **Code Quality**: Type hints, error handling, testing
8. **Performance**: Profiling, optimization, bottlenecks

**Practice Problems**:
- LeetCode: Arrays, Strings, Trees, Graphs, Dynamic Programming
- System Design: Design Twitter, Design a Cache, Design a URL Shortener
- Python-specific: Implement data structures, decorators, context managers

**Interview Tips**:
- Think out loud
- Ask clarifying questions
- Start with brute force, then optimize
- Consider edge cases
- Write clean, readable code
- Discuss time/space complexity

Good luck with your Google interview! 🚀
