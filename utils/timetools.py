import time


def seconds2str(seconds:int,reduce=True):
    """convert seconds to hms format string"""
    seconds = int(seconds)
    if reduce:
        hours = seconds//3600
        seconds = seconds % 3600
        minutes = seconds//60
        seconds = seconds % 60
        h_str = '' if hours==0 else '{0}h '
        m_str = '' if minutes==0 else '{1}m '
        s_str = '{2}s' #总是显示秒的部分给人安全感
        res = (h_str+m_str+s_str).format(hours,minutes,seconds)
        return res
    else:
        #总是显示时分秒
        return time.strftime('%Hh %Mm %Ss', time.gmtime(80))