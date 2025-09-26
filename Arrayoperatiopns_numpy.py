import numpy as np

# 1. Create an array from user input
user_input = input("Enter array elements separated by spaces: ")
array = np.array([int(x) for x in user_input.split()])
print("Original Array:", array)

# 2. Arithmetic operations with user-defined value
operation_value = int(input("\nEnter a number to perform arithmetic operations: "))

add_result = array + operation_value
sub_result = array - operation_value
mul_result = array * operation_value
div_result = array / operation_value

print("\nAddition (+{}):".format(operation_value), add_result)
print("Subtraction (-{}):".format(operation_value), sub_result)
print("Multiplication (*{}):".format(operation_value), mul_result)
print("Division (/{}):".format(operation_value), div_result)

# 3. Reshape the array (only if total elements allow it)
try:
    rows = int(input("\nEnter number of rows to reshape: "))
    if len(array) % rows != 0:
        raise ValueError("Total elements do not divide evenly into the number of rows.")
    
    cols = int(len(array) / rows)
    reshaped_array = array.reshape(rows, cols)
    print("Reshaped Array ({}x{}):\n".format(rows, cols), reshaped_array)

except ValueError as e:
    print(f"Cannot reshape array: {e}")

# 4. Indexing and Slicing
try:
    index = int(input("\nEnter an index to access a specific element: "))
    if 0 <= index < len(array):
        print("Element at index {}:".format(index), array[index])
    else:
        print("Index out of range.")

    start = int(input("Enter start index for slicing: "))
    end = int(input("Enter end index for slicing: "))
    print("Slicing from index {} to {}:".format(start, end), array[start:end])

except ValueError:
    print("Invalid input. Please enter valid indices.")

# 5. Finding Max and Min
max_val = np.max(array)
min_val = np.min(array)
print("\nMaximum Value:", max_val)
print("Minimum Value:", min_val)
