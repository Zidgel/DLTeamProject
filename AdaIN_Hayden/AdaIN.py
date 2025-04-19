# Hayden Schennum
# 2025-04-18

import torch






if __name__ == "__main__":

    tot = 0
    fh = open("p15a/input.txt","r")
    for line in fh:
        line = line.strip()
        line = line.split(",")
        for s in line:
            tot += get_hash_output(s)
    fh.close()

    print(tot)