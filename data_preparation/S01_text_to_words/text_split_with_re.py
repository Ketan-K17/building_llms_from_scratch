import re
text = "Hello, world. This, is a test."
result = re.split(r'([,.]|\s)', text)
result = [item for item in result if item.strip()]
print(result)


# We're omitting whitespaces here during tokenization for simplicity. But in practice, we should keep them, because in some cases it's important (e.g. writing Python Code)