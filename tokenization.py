import tiktoken

text = "youth is not a time of life"

enc = tiktoken.get_encoding("gpt2")

tokens = enc.encode(text)
print(tokens)
print(enc.decode(tokens))


enc = tiktoken.get_encoding("cl100k_base")  # used by gpt-4-turbo etc.
tokens = enc.encode("Youth is not a time of life")
print(tokens)


tokens = enc.encode("is Youth  not a time of life")
print(tokens)
