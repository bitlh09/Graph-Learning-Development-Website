import os

for p in os.environ["PATH"].split(";"):
    if '"' in p:
        print("❌ 有引号:", p)

