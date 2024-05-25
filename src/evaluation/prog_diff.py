import argparse
import difflib as df


def get_token_differences(str_1, str_2):
    str_1_lines = [c for c in str_1]
    str_2_lines = [c for c in str_2]
    sm = df.SequenceMatcher(None, str_1_lines, str_2_lines)
    masked_code = [c for c in str_1]
    masks = []
    fixes = []
    idx = 0
    added_lines = {}
    duplicate_masks = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        # print('{:7}   a[{}:{}] --> b[{}:{}] {!r:>8} --> {!r}'.format(
        #     tag, i1, i2, j1, j2, str_1_lines[i1:i2], str_2_lines[j1:j2]))
        if tag == 'insert':
            added_lines[i1] = "<<mask>>" #_" + str(idx) + ">>"
            idx += 1
            masks.append(str_1_lines[i1:i2])
            fixes.append(str_2_lines[j1:j2])
        elif tag != 'equal':
            for ll in range(i1, i2):
                masked_code[ll] = "<<mask>>" #_" + str(idx) + ">>"
            duplicate_masks.extend(list(range(i1+1, i2)))
            idx += 1
            masks.append(str_1_lines[i1:i2])
            fixes.append(str_2_lines[j1:j2])
    i = len(masked_code)
    while i >= 0:
        if i in added_lines:
            masked_code.insert(i, added_lines[i])
        if i in duplicate_masks:
            del masked_code[i]
        i -= 1
    return (''.join(masked_code), [''.join(m) for m in masks], [''.join(f) for f in fixes])


def get_masked_lines(prog_1, prog_2):
    # Diff program lines one by one to get different lines
    # print(df.SequenceMatcher(None, prog_1.split(), prog_2.split()).ratio())
    prog_1_lines = prog_1.split('\n')
    prog_2_lines = prog_2.split('\n')
    sm = df.SequenceMatcher(None, prog_1_lines, prog_2_lines)
    masked_code = prog_1.split('\n')
    masks = []
    fixes = []
    idx = 0
    added_lines = {}
    duplicate_masks = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        # print('{:7}   a[{}:{}] --> b[{}:{}] {!r:>8} --> {!r}'.format(
        #     tag, i1, i2, j1, j2, prog_1_lines[i1:i2], prog_2_lines[j1:j2]))
        if tag == 'insert':
            added_lines[i1] = "<<mask>>" #_" + str(idx) + ">>"
            idx += 1
            masks.append(prog_1_lines[i1:i2])
            fixes.append(prog_2_lines[j1:j2])
        elif tag != 'equal':
            for ll in range(i1, i2):
                masked_code[ll] = "<<mask>>" #_" + str(idx) + ">>"
            duplicate_masks.extend(list(range(i1+1, i2)))
            idx += 1
            masks.append(prog_1_lines[i1:i2])
            fixes.append(prog_2_lines[j1:j2])
    i = len(masked_code)
    while i >= 0:
        if i in added_lines:
            masked_code.insert(i, added_lines[i])
        if i in duplicate_masks:
            del masked_code[i]
        i -= 1
    return ('\n'.join(masked_code), ['\n'.join(m) for m in masks], ['\n'.join(f) for f in fixes])


def get_diff_lines(prog_1, prog_2):
    # Diff program lines one by one to get different lines
    # print(df.SequenceMatcher(None, prog_1.split(), prog_2.split()).ratio())
    prog_1_lines = prog_1.split('\n')
    prog_2_lines = prog_2.split('\n')
    sm = df.SequenceMatcher(None, prog_1_lines, prog_2_lines)
    masked_code = prog_1.split('\n')
    masks = []
    fixes = []
    idx = 0
    added_lines = {}
    duplicate_masks = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        # print('{:7}   a[{}:{}] --> b[{}:{}] {!r:>8} --> {!r}'.format(
        #     tag, i1, i2, j1, j2, prog_1_lines[i1:i2], prog_2_lines[j1:j2]))
        if tag == 'insert':
            added_lines[i1] = "<<mask>>" #_" + str(idx) + ">>"
            idx += 1
            masks.append((i1, i2))
            fixes.append((j1, j2))
        elif tag != 'equal':
            for ll in range(i1, i2):
                masked_code[ll] = "<<mask>>" #_" + str(idx) + ">>"
            duplicate_masks.extend(list(range(i1+1, i2)))
            idx += 1
            masks.append((i1, i2))
            fixes.append((j1, j2))
    i = len(masked_code)
    while i >= 0:
        if i in added_lines:
            masked_code.insert(i, added_lines[i])
        if i in duplicate_masks:
            del masked_code[i]
        i -= 1
    return ('\n'.join(masked_code), masks, fixes)


def read_sample(samp):
    samp_1 = samp.split(" <||> ")
    samp_2 = samp_1[1].split(" <++> ")
    return (samp_1[0], samp_2, int(samp_1[2]), float(samp_1[3]), samp_1[4], samp_1[5] == "popular", samp_1[6], samp_1[7], samp_1[8])
