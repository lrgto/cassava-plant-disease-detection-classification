import time


def years_secs(s):
    y = int(s // 31536000)
    s %= 31536000
    d = int(s // 86400)
    s %= 86400
    h = int(s // 3600)
    s %= 3600
    m = int(s // 60)
    s %= 60
    return y, d, h, m, s


def years_secs_str(s, labels=True):
    y, d, h, m, s = years_secs(s)
    st = '{0:.2f}s'
    if m > 0 or h > 0 or d > 0 or y > 0:
        l = 'm ' if labels else ':'
        st = '{1:1d}' + l + st
        if h > 0 or d > 0 or y > 0:
            l = 'h ' if labels else ':'
            st = '{2:1d}' + l + st
            if d > 0 or y > 0:
                l = 'd '  # if labels else ':'
                st = '{3:1d}' + l + st
                if y > 0:
                    l = 'y '  # if labels else ':'
                    st = '{4:1d}' + l + st
    return st.format(s, m, h, d, y)


# Time String Test
def time_string_test():
    print(years_secs_str(31536000))  # 1y 0d 0h 0m 0.00s
    print(years_secs_str(31535999))  # 364d 23h 59m 59.00s
    print(years_secs_str(86400))  # 1d 0h 0m 0.00s
    print(years_secs_str(86399))  # 23h 59m 59.00s
    print(years_secs_str(3600))  # 1h 0m 0.00s
    print(years_secs_str(3599))  # 59m 59.00s
    print(years_secs_str(60))  # 1m 0.00s
    print(years_secs_str(59))  # 59.00s


def start_timer(show_time=True):
    t_start = time.time()
    if show_time:
        print()
        print(time.asctime(time.localtime(t_start)))
    return t_start


def show_timer(t_start, message):
    secs = time.time() - t_start
    print(message, years_secs_str(secs))
    return secs


def step_time_str(t_start, secs, epoch, total_epochs, val, max_val):
    s = time.time() - t_start
    step_time = s - secs
    secs = s
    total = secs * max_val / val
    if epoch > 0 and total_epochs > 0:
        total = secs * (total_epochs * max_val) / ((epoch - 1) * max_val + val)
    ts = ': ' + years_secs_str(step_time) \
         + '    |= Elapsed: ' + years_secs_str(secs) \
         + '    Remaining: ' + years_secs_str(total - secs) \
         + '    Total: ' + years_secs_str(total) + ' =|'
    return secs, total, ts


def sci_form_bytes(size):
    if size >= 1099511627776:
        s = f'{size / 1099511627776:.3f} TB'
    elif size >= 1073741824:
        s = f'{size / 1073741824:.3f} GB'
    elif size >= 1048576:
        s = f'{size / 1048576:.3f} MB'
    elif size >= 1024:
        s = f'{size / 1024:.3f} KB'
    else:
        s = f'{size:d} B'
    return s


def dataset_memory_use(batch_size, width, height, d=3):
    x_size = batch_size * width * height * d * 4
    y_size = batch_size * 4
    return sci_form_bytes(x_size + y_size) + ' {' + str(x_size) + ' + ' + str(y_size) + ' bytes}'
