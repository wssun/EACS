def split_java_to_seqs(code_token_list):
    lf_bracket_up = 0
    idxs = []

    end_idx = 0

    for idx, i in enumerate(code_token_list):
        if i == ';' and not lf_bracket_up:
            idxs.append((end_idx, idx))
            end_idx = idx + 1
        elif i == '(':
            lf_bracket_up += 1
        elif i == ')':
            lf_bracket_up -= 1
        elif i == '{':
            idxs.append((end_idx, idx))
            end_idx = idx + 1
        elif i == '}':
            end_idx = idx + 1

    code_seqs = []
    for idx, i in enumerate(idxs):
        code_snap = code_token_list[i[0]:i[1] + 1]
        if code_snap[-1] == '{':
            code_seqs.append(' '.join(code_snap) + ' }')
        else:
            code_seqs.append(' '.join(code_snap))
    return code_seqs


# def split_python_to_seqs(code_seq, code_token_list):
#     '''
#     1. 需要去除\'''\'''的内容
#     2. 分句
#     3. 分tokens
#         1. 需要加空格的token ( ) : + - = . , * / += -= *= /= //
#         2. 但是...不分
#         3. ''和""里面的内容不分
#     '''
#     seq_lines = code_seq.splitlines()
#     seq_lines = [i.strip() for i in seq_lines]
#     seq_lines = [i for i in seq_lines if len(i) > 0]
#
#     comment_1 = False
#     comment_2 = False
#
#     comment_1_in = False
#     comment_2_in = False
#
#     cleaned_seq_lines = []
#
#     for i in seq_lines:
#         if i.startswith('#'):
#             continue
#
#         if not comment_1 and i.startswith(r'"""'):
#             comment_1 = True
#             continue
#         if comment_1 and i.startswith(r'"""'):
#             comment_1 = False
#             continue
#
#         if not comment_2 and i.startswith(r"'''"):
#             comment_2 = True
#             continue
#         if comment_2 and i.startswith(r"'''"):
#             comment_2 = False
#             continue
#
#         if comment_1 or comment_2:
#             continue
#
#         if r"'''" in i and not i.startswith(r"'''"):
#             if i.count(r"'''") == 2:
#                 s_idx = i.find(r"'''")
#                 e_idx = len(i) - i[::-1].find(r"'''")
#                 cleaned_seq = i[:s_idx] + i[e_idx:]
#                 cleaned_seq_lines.append(cleaned_seq)
#             elif i.count(r"'''") == 1 and not comment_1_in:
#                 s_idx = i.find(r"'''")
#                 cleaned_seq_lines.append(i[:s_idx])
#                 comment_1_in = True
#             # elif:
#             #     pass
#
#         if r'"""' in i:
#             pass
#
#         if comment_1_in or comment_2_in:
#             continue
#
#         seq = i
#         seq_temp = ''
#         while True:
#             if '#' in seq:
#                 quo_1 = 0
#                 quo_2 = 0
#                 last_index = seq[::-1].find('#')
#                 for j in seq[:last_index]:
#                     if j == "\'":
#                         quo_1 += 1
#
#                     if j == '\"':
#                         quo_2 += 1
#
#                 if quo_1 % 2 == 0 and quo_2 % 2 == 0:
#                     seq_temp += seq[:last_index]
#                     seq = seq_temp
#                 else:
#                     break
#             else:
#                 break
#
#         cleaned_seq_lines.append(seq)
#
#     seq_idx = 0
#     token_start_idx = 0
#     token_end_idx = 0
#
#     code_seqs = []
#     # for cl in cleaned_seq_lines:
#     #     new_cl = cl.replace('(',' ( ').replace(')',' ) ').replace('{',' { ').replace('(',' ( ')
#     #     code_seqs.append(new_cl)
#
#     for idx, cl in enumerate(code_token_list):
#         if seq_idx == len(cleaned_seq_lines) - 1:
#             code_seqs.append(' '.join(code_token_list[token_end_idx + 1:]))
#             break
#
#         if cleaned_seq_lines[seq_idx].startswith(cl):
#             token_start_idx = idx
#             continue
#         if cleaned_seq_lines[seq_idx].endswith(cl):
#             if seq_idx < len(cleaned_seq_lines) - 1 and idx < len(code_token_list) - 1:
#                 if cleaned_seq_lines[seq_idx + 1].startswith(code_token_list[idx + 1]):
#                     token_end_idx = idx
#                     code_seqs.append(' '.join(code_token_list[token_start_idx:token_end_idx + 1]))
#                     seq_idx += 1
#                 else:
#                     continue
#             else:
#                 token_end_idx = idx
#                 code_seqs.append(' '.join(code_token_list[token_start_idx:token_end_idx + 1]))
#                 seq_idx += 1
#
#     if seq_idx != len(cleaned_seq_lines) - 1:
#         print('aa')
#
#     return code_seqs

def split_python_to_seqs(code_token_list):
    # '"
    left_s_quo, left_d_quo = 0, 0

    # (
    left_par = 0

    # {
    left_brace = 0

    # [
    left_bracket = 0

    s_word = [
        '=', '+', '-', '*', '/', '%', '^', '~', '!', '|', '&', '<', '>',
        '==', '+=', '-=', '*=', '/=', '%=', '^=', '~=', '!=', '||', '&&',
        '<=', '>=',
        '.', ',', ':',
        '(', '{', '[',
        'and', 'or', 'not', 'is', 'in', 'as'
    ]

    end_idx = 0
    idxs = []

    sp_word = ['return',
               'while', 'for',
               'if', 'elif', 'else',
               'try', 'except', 'Exception', 'raise',
               'assert',
               'yield',
               'with', 'print', ]

    return_sent = 0
    for_sent = 0
    if_sent = 0
    try_sent = 0
    assert_sent = 0
    with_sent = 0
    print_sent = 0
    yield_sent = 0

    def_class_sent = False
    sp_sent = 0

    for idx, i in enumerate(code_token_list):
        if i == 'def' or i == 'class':
            def_class_sent = True

        if i == 'return':
            return_sent += 1
        elif i == 'for' or i == 'while':
            for_sent += 1
        elif i == 'if' or i == 'elif' or i == 'else':
            if_sent += 1
        elif i == 'try' or i == 'except' or i == 'Exception' or i == 'raise':
            try_sent += 1
        elif i == 'assert':
            assert_sent += 1
        elif i == 'with':
            with_sent += 1
        elif i == 'print':
            print_sent += 1
        elif i == 'yield':
            yield_sent += 1

        left_up = left_s_quo + left_d_quo + left_par + left_brace + left_bracket
        if i == ':' and left_up == 0:
            idxs.append((end_idx, idx))
            end_idx = idx + 1
            def_class_sent = False
            continue
        elif i == r"'" and left_up == 0:
            left_s_quo += 1
        elif i == r'"':
            left_d_quo += 1
        elif i == '(':
            left_par += 1
        elif i == '{':
            left_brace += 1
        elif i == '[':
            left_bracket += 1
        elif i == r"'" and left_s_quo == 1:
            left_s_quo -= 1
        elif i == r'"' and left_d_quo == 1:
            left_d_quo -= 1
        elif i == ')':
            left_par -= 1
        elif i == '}':
            left_brace -= 1
        elif i == ']':
            left_bracket -= 1

        new_left_up = left_s_quo + left_d_quo + left_par + left_brace + left_bracket
        if new_left_up == 0:
            if idx < len(code_token_list) - 1:
                if return_sent == 1:
                    return_sent += 1
                    continue
                elif return_sent == 2:
                    return_sent = 0

                if for_sent == 1:
                    for_sent += 1
                    continue
                elif for_sent == 2:
                    for_sent = 0

                if if_sent == 1:
                    if_sent += 1
                    continue
                elif if_sent == 2:
                    if_sent = 0

                if try_sent == 1:
                    try_sent += 1
                    continue
                elif try_sent == 2:
                    try_sent = 0

                if assert_sent == 1:
                    assert_sent += 1
                    continue
                elif assert_sent == 2:
                    assert_sent = 0

                if with_sent == 1:
                    with_sent += 1
                    continue
                elif with_sent == 2:
                    with_sent = 0

                if print_sent == 1:
                    print_sent += 1
                    continue
                elif print_sent == 2:
                    print_sent = 0

                if yield_sent == 1:
                    yield_sent += 1
                    continue
                elif yield_sent == 2:
                    yield_sent = 0

                if code_token_list[idx + 1] not in s_word \
                        and code_token_list[idx] not in s_word \
                        and not def_class_sent:
                    idxs.append((end_idx, idx))
                    end_idx = idx + 1
                elif code_token_list[idx + 1] not in s_word \
                        and code_token_list[idx] in ['}', ']', ']'] \
                        and not def_class_sent:
                    idxs.append((end_idx, idx))
                    end_idx = idx + 1
            else:
                idxs.append((end_idx, idx))

    code_seqs = []
    for idx, i in enumerate(idxs):
        code_snap = code_token_list[i[0]:i[1] + 1]
        if ' '.join(code_snap) == 'return' and idx < len(idxs) - 1:
            code_snap = code_token_list[i[0]:idxs[idx + 1][1] + 1]
        if ' '.join(code_snap) == 'if' and idx < len(idxs) - 1:
            code_snap = code_token_list[i[0]:idxs[idx + 1][1] + 1]
        code_seqs.append(' '.join(code_snap))
    return code_seqs


def split_go_to_seqs(code_token_list):
    # '"
    left_s_quo, left_d_quo = 0, 0

    # (
    left_par = 0

    # [
    left_bracket = 0

    s_word = [
        '=', '+', '-', '*', '/', '%', '^', '~', '!', '|', '&', '<', '>',
        '==', '+=', '-=', '*=', '/=', '%=', '^=', '~=', '!=', '||', '&&',
        '<=', '>=', ':=',
        '.', ',', ':', ';',
        '(', '{', '[',
        'range',
    ]

    func_sent = False
    return_sent = 0
    for_sent = 0
    defer_sent = 0
    var_sent = 0
    switch_sent = 0

    end_idx = 0
    idxs = []

    for idx, i in enumerate(code_token_list):
        left_up = left_par + left_bracket
        if i == '{' and left_up == 0:
            idxs.append((end_idx, idx))
            end_idx = idx + 1
            func_sent = False
            continue
        elif i == '}':
            end_idx = idx + 1
            continue
        elif i == '(':
            left_par += 1
        elif i == ')':
            left_par -= 1
        elif i == '[':
            left_bracket += 1
        elif i == ']':
            left_bracket -= 1

        if i == 'func':
            func_sent = True
        elif i == 'return':
            return_sent += 1
        elif i == 'for' or i == 'while':
            for_sent += 1
        elif i == 'defer':
            defer_sent += 1
        elif i == 'var':
            var_sent += 1
        elif i == 'switch':
            switch_sent += 1

        new_left_up = left_par + left_bracket
        if new_left_up == 0:
            if idx < len(code_token_list) - 1:
                if return_sent == 1:
                    return_sent += 1
                    continue
                elif return_sent == 2:
                    return_sent = 0

                if for_sent == 1:
                    for_sent += 1
                    continue
                elif for_sent == 2:
                    for_sent = 0

                if defer_sent == 1:
                    defer_sent += 1
                    continue
                elif defer_sent == 2:
                    defer_sent = 0

                if var_sent == 1:
                    var_sent += 1
                    continue
                elif var_sent == 2:
                    var_sent = 0

                if switch_sent == 1:
                    switch_sent += 1
                    continue
                elif switch_sent == 2:
                    switch_sent = 0

                if code_token_list[idx + 1] not in s_word \
                        and code_token_list[idx] not in s_word \
                        and not func_sent:
                    idxs.append((end_idx, idx))
                    end_idx = idx + 1
                elif code_token_list[idx + 1] not in s_word \
                        and code_token_list[idx] in ['}', ')', ']'] \
                        and not func_sent:
                    idxs.append((end_idx, idx))
                    end_idx = idx + 1
            else:
                idxs.append((end_idx, idx))

    code_seqs = []
    for idx, i in enumerate(idxs):
        code_snap = code_token_list[i[0]:i[1] + 1]
        if ' '.join(code_snap) == 'return' and idx < len(idxs) - 1:
            code_snap = code_token_list[i[0]:idxs[idx + 1][1] + 1]
        if ' '.join(code_snap) == 'if' and idx < len(idxs) - 1:
            code_snap = code_token_list[i[0]:idxs[idx + 1][1] + 1]

        if code_snap[-1] == '{':
            code_seqs.append(' '.join(code_snap) + ' }')
        else:
            code_seqs.append(' '.join(code_snap))

    return code_seqs


def split_ruby_to_seqs(code_token_list):
    # '"
    left_s_quo, left_d_quo = 0, 0

    # (
    left_par = 0

    # [
    left_bracket = 0

    # {
    left_brace = 0

    s_word = [
        '=', '+', '-', '*', '/', '%', '^', '~', '!', '|', '&', '<', '>',
        '==', '+=', '-=', '*=', '/=', '%=', '^=', '~=', '!=', '||', '&&',
        '<=', '>=', ':=', '<<', '>>',
        '.', ',', ':', ';', '::',
        '(', '{', '[',
        'do'
    ]

    def_sent = False
    return_sent = 0
    for_sent = 0
    if_sent = 0

    end_idx = 0
    idxs = []

    for idx, i in enumerate(code_token_list):
        if i == 'def':
            def_sent = True
            continue
        if i == ')':
            left_par -= 1
        left_up = left_par + left_bracket + left_brace
        if i == ')' and left_up == 0 and def_sent:
            idxs.append((end_idx, idx))
            end_idx = idx + 1
            def_sent = False
            continue
        elif i == 'end':
            end_idx = idx + 1
            continue
        elif i == '(':
            left_par += 1
        elif i == '[':
            left_bracket += 1
        elif i == ']':
            left_bracket -= 1
        elif i == '{':
            left_brace += 1
        elif i == '}':
            left_brace -= 1

        if i == 'def':
            def_sent = True
        elif i == 'return':
            return_sent += 1
        elif i == 'for' or i == 'while':
            for_sent += 1
        elif i == 'if':
            if_sent += 1

        if idx == 2 and i != '(' and def_sent:
            idxs.append((end_idx, idx))
            end_idx = idx
            def_sent = False
            continue

        new_left_up = left_par + left_bracket + left_brace
        if new_left_up == 0:
            if idx < len(code_token_list) - 1:
                if return_sent == 1:
                    return_sent += 1
                    continue
                elif return_sent == 2:
                    return_sent = 0

                if for_sent == 1:
                    for_sent += 1
                    continue
                elif for_sent == 2:
                    for_sent = 0

                if if_sent == 1:
                    if_sent += 1
                    continue
                elif if_sent == 2:
                    if_sent = 0

                if code_token_list[idx + 1] not in s_word \
                        and code_token_list[idx] not in s_word \
                        and not def_sent:
                    idxs.append((end_idx, idx))
                    end_idx = idx + 1
                elif code_token_list[idx + 1] not in s_word \
                        and code_token_list[idx] in ['}', ')', ']'] \
                        and not def_sent:
                    idxs.append((end_idx, idx))
                    end_idx = idx + 1
            else:
                idxs.append((end_idx, idx))

    code_seqs = []
    for idx, i in enumerate(idxs):
        code_snap = code_token_list[i[0]:i[1] + 1]
        if ' '.join(code_snap) == 'return' and idx < len(idxs) - 1:
            code_snap = code_token_list[i[0]:idxs[idx + 1][1] + 1]
        if ' '.join(code_snap) == 'if' and idx < len(idxs) - 1:
            code_snap = code_token_list[i[0]:idxs[idx + 1][1] + 1]

        if code_snap[-1] == '{':
            code_seqs.append(' '.join(code_snap) + ' }')
        else:
            code_seqs.append(' '.join(code_snap))

    return code_seqs


def split_php_to_seqs(code_token_list):
    lf_bracket_up = 0
    idxs = []

    end_idx = 0

    for idx, i in enumerate(code_token_list):
        if i == ';' and not lf_bracket_up:
            idxs.append((end_idx, idx))
            end_idx = idx + 1
        elif i == '(':
            lf_bracket_up += 1
        elif i == ')':
            lf_bracket_up -= 1
        elif i == '{':
            idxs.append((end_idx, idx))
            end_idx = idx + 1
        elif i == '}':
            end_idx = idx + 1

    code_seqs = []
    for idx, i in enumerate(idxs):
        code_snap = code_token_list[i[0]:i[1] + 1]
        if code_snap[-1] == '{':
            code_seqs.append(' '.join(code_snap) + ' }')
        else:
            code_seqs.append(' '.join(code_snap))
    return code_seqs
