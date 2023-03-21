def get_err_count(word, pattern, max_errors=2):
    """
    Returns the number of errors in the word, if the word is within the
    maximum number of errors of the pattern. If the word is not within
    the maximum number of errors of the pattern, returns -1.

    Parameters
    ----------
    pattern : str
        The pattern to compare the word to.
    word : str
        The word to compare the pattern to.
    max_errors : int
        The maximum number of errors allowed in the word.

    Returns
    -------
    int
        The number of errors in the word, if the word is within the
        maximum number of errors of the pattern. If the word is not
        within the maximum number of errors of the pattern, returns -1.
    """
    n = len(pattern)+1  # nodes per row

    max_state = (max_errors+1) * n
    states = [(1, 0, 0)]
    best_state = max_state + 1

    while len(states):
        state = states.pop()
        if state[0] > max_state or state[0] > best_state:
            continue
        if not state[0] % n:
            if state[1] == len(pattern) and state[2] == len(word):
                best_state = state[0]
            else:
                states.append((state[0]+n, state[1], state[2]+1))
            continue
        states.append((state[0]+1+n, state[1]+1, state[2]))
        if state[2] < len(word):
            states.append((state[0]+n, state[1], state[2]+1))
            states.append((state[0]+1+n, state[1]+1, state[2]+1))
            if pattern[state[1]] == word[state[2]]:
                states.append((state[0]+1, state[1]+1, state[2]+1))

    if best_state == max_state + 1:
        return -1
    return int(best_state / n - 1)


def main():
    pattern = 'test'
    max_errors = 2
    words = ['testing', 'tester', 'test', 'tes', 'te', 't', 'other', 'xdddd', 'lmao']
    for word in words:
        errors = get_err_count(word, pattern, max_errors)
        if errors == -1:
            print(f"{word} is not within {max_errors} errors of {pattern}")
            continue
        print(f"{word} is within {errors} errors of {pattern}")


if __name__ == "__main__":
    main()
