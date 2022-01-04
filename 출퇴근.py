from copy import deepcopy
import csv
import datetime as dt
import pandas as pd
import random
import sys
import time


def main():
    '''--------------------  파라미터 입력  ----------------------'''
    yr_mth = '2020-03'
    start_day = yr_mth + '-' + '01'
    end_day = yr_mth + '-' + '15'
    # 해당 월의 첫일, 마지막일 입력

    holidays = ['01', '24', '27']
    # 주말 제외한 공휴일 입력
    holidays = get_holidays(yr_mth, holidays)

    people_to_holidays = {'Nashat': ['03'], 'Nadine': [],
                          'Mohammad': ['13', '15'],
                          'Amer': [], 'Soonhee': ['07:14', '20', '22:29'],
                          'Inky': [], 'Hayat': ['07', '08'],
                          'Alysa': [], 'Kristine': ['03']
                          }
    # 개인 휴가 입력 (공휴일은 입력하지 않음)
    people_to_holidays = get_people_to_holidays(yr_mth, people_to_holidays)

    people_to_uid = {'Nashat': '0004', 'Nadine': '0017', 'Mohammad': '0006',
                     'Amer': '0005', 'Soonhee': '0007', 'Inky': '0010',
                     'Hayat': '0008', 'Alysa': '0015', 'Kristine': '0009'}

    diplomats = ['Nashat', 'Nadine']
    cleaners = ['Alysa', 'Kristine']
    peoples = diplomats + cleaners + [
        'Mohammad', 'Amer', 'Soonhee', 'Inky', 'Hayat']
    '''----------------------------------------------------------'''

    assert len(people_to_uid) == len(peoples)
    pd.set_option('display.max_colwidth', -1)
    pd.options.display.max_rows = 999

    days = get_days(start_day, end_day, holidays)
    print('타겟 Event Date\n', days)
    people_to_holidays = adjust_people_to_holidays(days, people_to_holidays)
    day_to_workers = get_day_to_workers(days, people_to_holidays)
    print('\n일자별 출석자')
    print(pd.Series(day_to_workers))

    data = {}
    cnt = 0
    for day_idx, day in enumerate(days):
        key = str(cnt).zfill(4)
        data[key] = {}
        data[key]['Event Date'] = day
        data[key]['TerminalID'] = '0001' if random.randrange(0, 100, 1)/100 < 0.9 else '0002'
        data[key]['Type'] = '1:N'
        data[key]['Result'] = 'O'

        workers = day_to_workers[day]
        disarm_diplomat = get_one_diplomat(day, diplomats, workers)
        _ = deepcopy(workers)
        _.remove(disarm_diplomat)
        data[key]['UserID'] = people_to_uid[disarm_diplomat]
        data[key]['Name'] = disarm_diplomat
        data[key]['Mode'] = 'Disarm'
        att_event_times = get_event_times(mode='Att', num=len(workers))
        data[key]['Event Time'] = att_event_times[0]

        cnt += 1
        _key = str(cnt).zfill(4)
        data[_key] = deepcopy(data[key])
        data[_key]['Mode'] = 'Att'
        key = _key

        random.shuffle(_)
        _1 = [people for people in _ if people in (diplomats + cleaners)]
        _2 = [people for people in _ if people not in (diplomats + cleaners)]
        _ = _1 + _2
        # diplomat과 cleaner를 출근 맨 앞에 강제로.

        for worker_idx, worker in enumerate(_):
            cnt += 1
            _key = str(cnt).zfill(4)
            data[_key] = deepcopy(data[key])
            data[_key]['Event Time'] = att_event_times[worker_idx+1]
            data[_key]['TerminalID'] = '0001' if random.randrange(0, 100, 1)/100 < 0.9 else '0002'
            data[_key]['UserID'] = people_to_uid[worker]
            data[_key]['Name'] = worker
            data[_key]['Mode'] = 'Att'
            key = _key
        # 나머지는 랜덤 분배

        away_diplomat = get_one_diplomat(day, diplomats, workers)
        _ = deepcopy(workers)
        _.remove(away_diplomat)
        random.shuffle(_)
        ext_event_times = get_event_times(mode='Ext', num=len(workers))
        for worker_idx, worker in enumerate(_):
            cnt += 1
            _key = str(cnt).zfill(4)
            data[_key] = deepcopy(data[key])
            data[_key]['Event Time'] = ext_event_times[worker_idx]
            data[_key]['TerminalID'] = '0001' if random.randrange(0, 100, 1)/100 < 0.9 else '0002'
            data[_key]['UserID'] = people_to_uid[worker]
            data[_key]['Name'] = worker
            data[_key]['Mode'] = 'Ext'
            key = _key

        cnt += 1
        _key = str(cnt).zfill(4)
        data[_key] = deepcopy(data[key])
        data[_key]['Event Time'] = ext_event_times[-1]
        data[_key]['TerminalID'] = '0001' if random.randrange(0, 100, 1)/100 < 0.9 else '0002'
        data[_key]['UserID'] = people_to_uid[away_diplomat]
        data[_key]['Name'] = away_diplomat
        data[_key]['Mode'] = 'Ext'
        key = _key

        cnt += 1
        _key = str(cnt).zfill(4)
        data[_key] = deepcopy(data[key])
        data[_key]['Mode'] = 'Away'
        key = _key

        cnt += 1

    print('\n엑셀 아웃풋')
    df = pd.DataFrame(data).T
    column_orders = ['Event Date', 'Event Time', 'TerminalID', 'UserID', 'Name', 'Mode', 'Type', 'Result']
    print(df[column_orders])
    df[column_orders].to_csv(end_day + '.csv')
    return


def adjust_people_to_holidays(days, people_to_holidays):
    _ = deepcopy(people_to_holidays)
    for people in people_to_holidays:
        holidays = people_to_holidays[people]
        for holiday in holidays:
            if len(holiday) == 13:
                _[people].remove(holiday)
                start_idx = int(holiday[8:10])
                end_idx = int(holiday[11:13])
                for idx in range(start_idx, end_idx+1):
                    if idx < 10:
                        day = holiday[:8] + '0' + str(idx)
                    else:
                        day = holiday[:8] + str(idx)
                    if day in days:
                        _[people].append(day)
        _[people] = sorted(list(set(_[people])))
    people_to_holidays = deepcopy(_)
    return people_to_holidays


def get_days(start_day, end_day, holidays):
    days = []
    for idx in range(int(end_day[-2:])):
        if idx + 1 < int(start_day[-2:]):
            continue
        if idx < 9:
            day = start_day[:8] + '0' + str(idx+1)
        else:
            day = start_day[:8] + str(idx+1)
        _ = dt.datetime.strptime(day, '%Y-%m-%d').weekday()
        if day not in holidays and _ not in [5, 6]:
            # 공휴일이 아니면서 주말이 아닐 때 days 만들기
            days.append(day)
    days = sorted(list(set(days)))
    return days


def get_day_to_workers(days, people_to_holidays):
    day_to_workers = {}
    _ = deepcopy(people_to_holidays)
    for day in days:
        day_to_workers[day] = list(people_to_holidays.keys())
        for people in people_to_holidays:
            holidays = people_to_holidays[people]
            for holiday in holidays:
                if holiday == day:
                    day_to_workers[day].remove(people)
                    break

    return day_to_workers


def get_event_times(mode, num):
    if mode == 'Att':
        start_time = dt.datetime.strptime('08:40:00', '%H:%M:%S')
    elif mode == 'Ext':
        start_time = dt.datetime.strptime('17:00:00', '%H:%M:%S')
    else:
        print('Mode를 선택해주세요')
        sys.exit()

    event_times = []
    time = start_time
    for idx in range(num):
        delta = int(random.randrange(600, 2000, 1) / 10)
        time = time + dt.timedelta(seconds=delta)
        event_times.append(time.strftime('%H:%M:%S'))
    return event_times


def get_holidays(yr_mth, holidays):
    return [yr_mth + '-' + holiday for holiday in holidays]


def get_one_diplomat(day, diplomats, workers):
    cnt = 0
    while True:
        cnt += 1
        worker_idx = random.randrange(0, len(diplomats), 1)
        diplomat = diplomats[worker_idx]
        if diplomat in workers:
            break
        if cnt > 100:
            print(day, ' 출근한 외교관이 없습니다.')
            sys.exit()
    return diplomat


def get_people_to_holidays(yr_mth, people_to_holidays):
    _ = {}
    for people in people_to_holidays:
        _[people] = [yr_mth + '-' + holiday for holiday in people_to_holidays[people]]
    people_to_holidays = deepcopy(_)
    return people_to_holidays


if __name__ == '__main__':
    main()
