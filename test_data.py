import qlib
from qlib.constant import REG_CN
from qlib.data import D

def main():
    qlib.init(provider_uri='~/.qlib/qlib_data/cn_data', region=REG_CN)
    cal = D.calendar()
    print('first', cal[0], 'last', cal[-1], 'total', len(cal))

    df = D.features(D.instruments('csi300'), ['$close', '$volume', '$turnover'], start_time='2020-01-01', end_time='2020-03-01')
    print(df.head())
    print('turnover nonnull', df['$turnover'].notna().sum(), 'total', len(df))

if __name__ == '__main__':
    main()
