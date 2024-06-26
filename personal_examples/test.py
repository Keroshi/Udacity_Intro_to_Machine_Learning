key_value_array = [
    ("b", 2),
    ("a", 3),
    ("c", 1)
]

# Sort by the first element (key)
sorted_by_key = sorted(key_value_array, key=lambda x: x[0])
print("Sorted by key:", sorted_by_key)

# Sort by the second element (value)
sorted_by_value = sorted(key_value_array, key=lambda x: x[1])
print("Sorted by value:", sorted_by_value)
