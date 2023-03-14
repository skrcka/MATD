def brute_serach(string, substring):
    occurences = []
    for i in range(len(string) - len(substring) + 1):
        j = 0
        while j < len(substring) and string[i + j] == substring[j]:
            j += 1
        if j == len(substring):
            occurences.append(i)
    return occurences

def substring_search(string, substring):
    transitions = {}
    for i in range(len(substring)):
        for c in set(string + substring):
            if i < len(substring) and c == substring[i]:
                transitions[(i, c)] = i + 1
            else:
                k = min(i, len(substring))
                s = substring[:k] + c
                while substring[:k] not in s[:-1] and k > 0:
                    k -= 1
                    s = substring[:k] + c
                transitions[(i, c)] = k
        
    state, occurrences = 0, []
    for i, c in enumerate(string):
        state = transitions.get((state, c), 0)
        if state == len(substring):
            occurrences.append(i - len(substring) + 1)
    
    return occurrences

def main():
    text = "this is a test to look for subsfortr for ing for for xd for fofor fou"
    sub = "for"
    print(brute_serach(text, sub))
    print(substring_search(text, sub))

if __name__ == "__main__":
    main()