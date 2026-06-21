# 有效买托策略 – Golden Triangle Stock Selector

基于 [Qlib](https://github.com/microsoft/qlib) 开发的多因子选股模型，通过**均线金叉三角**、**成交量突增**、**换手率**三个技术指标筛选 A 股市场出现有效买托信号的股票。

---

## 一、策略逻辑（四步漏斗）

| 步骤 | 条件 | 说明 |
|---|---|---|
| 1. 有效买托 | MA5 金叉 MA10，且 MA10 金叉 MA20 发生在相近窗口，当日 MA5 > MA10 > MA20 | 观察窗口：T, T-1, T-2 |
| 2. 成交量突增 | 金叉日成交量 ≥ 前7日平均成交量 × 1.5 | 确认有资金配合 |
| 3. 活跃度过滤 | 金叉日换手率 > 3% | 剔除流动性差的股票 |
| 4. 行业匹配（预留） | 用户输入目标行业列表 | 人工/NLP 二次筛选 |

---

## 二、快速开始

### 2.1 环境准备

```bash
# 在 Qlib 虚拟环境中安装 akshare（用于获取换手率）
pip install akshare
```

### 2.2 方式 A：零 Qlib 数据运行（AKShare 全量兜底）

如果你没有本地 Qlib 二进制数据，可以直接用 AKShare 获取 OHLCV + 换手率：

```bash
cd examples/golden_triangle
python run_selector.py \
    --fallback-akshare \
    --instruments csi300 \
    --lookback 35 \
    --output result.csv
```

> ⚠️ 全市场 5000+ 只股票逐个请求较慢，建议先用 `--instruments csi300` 或自定义股票池测试。

### 2.3 方式 B：使用 Qlib 数据（推荐）

先确保本地有 Qlib 日频数据（可通过 Yahoo Collector 或 AKShare Collector 生成）：

```bash
# 示例：通过 Yahoo 生成基础 OHLCV 数据
python scripts/dump_bin.py dump_all \
    --csv_path ~/.qlib/csv_dir \
    --qlib_dir ~/.qlib/qlib_data/cn_data

# 再通过 AKShare 补充换手率
python scripts/data_collector/akshare/collector.py \
    --incremental-days 365 \
    --output-dir ~/.qlib/akshare_turnover \
    --qlib-dir ~/.qlib/qlib_data/cn_data \
    --fields turnover
```

补充完换手率后，选股器可以直接从 Qlib 读取 `$turnover`：

```bash
python run_selector.py \
    --provider-uri ~/.qlib/qlib_data/cn_data \
    --instruments all \
    --turnover-source qlib \
    --output result.xlsx
```

### 2.4 方式 C：混合模式（Qlib OHLCV + AKShare 换手率）

如果你只有 Qlib 的基础价格数据，没有换手率，可以运行时实时拉取：

```bash
python run_selector.py \
    --provider-uri ~/.qlib/qlib_data/cn_data \
    --instruments all \
    --turnover-source akshare_hist \
    --output result.csv
```

> 注意：`akshare_hist` 模式需要逐个股票请求历史数据，全市场约 5000 只，耗时较长（预计 10–30 分钟）。

---

## 三、参数说明

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--provider-uri` | `None` | Qlib 数据目录 |
| `--instruments` | `all` | 股票池：`all`, `csi300`, `csi500` 或逗号分隔列表 |
| `--trade-date` | 今天 | 观察锚定日 T（YYYY-MM-DD） |
| `--lookback` | 35 | 回溯天数（需覆盖 MA20 + 7 日均量 + 观察窗口） |
| `--obs-window` | 3 | 观察窗口（T, T-1, T-2） |
| `--volume-multiplier` | 1.5 | 量比阈值 |
| `--turnover-threshold` | 3.0 | 换手率阈值（%） |
| `--turnover-source` | `akshare_hist` | 换手率来源：`qlib`, `akshare_spot`, `akshare_hist`, `tushare`, `csv` |
| `--turnover-csv` | `None` | 本地换手率 CSV 路径（`turnover-source=csv` 时必填） |
| `--fallback-akshare` | `False` | 完全使用 AKShare，不依赖 Qlib 数据 |
| `--industry-filter` | `None` | 行业过滤，逗号分隔关键词 |
| `--output` | `golden_triangle_result.csv` | 输出文件（支持 `.csv` 或 `.xlsx`） |

---

## 四、输出字段

| 字段 | 说明 |
|---|---|
| `instrument` | 股票代码（Qlib 格式，如 SH600000） |
| `name` | 股票名称 |
| `cross_date` | 金叉信号日期 |
| `ma5` | 金叉日 MA5 |
| `ma10` | 金叉日 MA10 |
| `ma20` | 金叉日 MA20 |
| `volume` | 金叉日成交量 |
| `avg_volume_7` | 前7日平均成交量 |
| `volume_ratio` | 量比（当日 / 前7日均量） |
| `turnover` | 金叉日换手率（%） |
| `industry` | 所属行业 |

---

## 五、模块结构

```
qlib/contrib/golden_triangle/
├── __init__.py       # 包入口
├── selector.py       # 核心选股逻辑（向量化计算）
├── data_source.py    # 数据适配器（Qlib + AKShare/Tushare/CSV）
└── strategy.py       # Qlib 回测策略

examples/golden_triangle/
├── run_selector.py   # 选股运行入口
├── run_backtest.py   # 回测运行入口
└── README.md         # 本文件

scripts/data_collector/akshare/
└── collector.py      # AKShare 换手率收集器（导入 Qlib 二进制格式）
```

---

## 六、回测

`golden_triangle` 目前是一个**选股器**，不自带仓位管理和买卖逻辑。我们额外提供了一个 Qlib 策略 `GoldenTriangleStrategy`，把它接入 Qlib 事件驱动回测引擎：

### 6.1 命令行回测

```bash
cd examples/golden_triangle

# 使用本地 Qlib 数据（已包含换手率）
python run_backtest.py \
    --provider-uri ~/.qlib/qlib_data/cn_data \
    --turnover-source qlib \
    --start 2024-01-01 \
    --end 2024-12-31 \
    --instruments csi300 \
    --max-positions 10 \
    --holding-period 5 \
    --account 1000000

# 使用 Qlib OHLCV + Tushare 历史换手率（按交易日批量拉取，较快）
python run_backtest.py \
    --provider-uri ~/.qlib/qlib_data/cn_data \
    --turnover-source tushare \
    --tushare-token your_tushare_token \
    --start 2024-01-01 \
    --end 2024-06-30 \
    --max-positions 10

# 使用 Qlib OHLCV + akshare 历史换手率
python run_backtest.py \
    --provider-uri ~/.qlib/qlib_data/cn_data \
    --turnover-source akshare_hist \
    --start 2024-01-01 \
    --end 2024-06-30 \
    --max-positions 10
```

### 6.2 回测逻辑说明

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--max-positions` | 10 | 最多同时持有几只股票 |
| `--holding-period` | 5 | 买入后最少持有几个交易日 |
| `--sell-on-exit-signal` | False | 是否股票掉出选股结果即卖出 |
| `--open-cost` / `--close-cost` | 0.0005 / 0.0015 | 买卖手续费 |
| `--limit-threshold` | 0.095 | 涨跌停限制（A 股主板） |
| `--deal-price` | `open` | 成交价字段，可改为 `close` |

回测流程：
1. 每个交易日收盘后运行 `GoldenTriangleSelector`。
2. 对当天选出的股票按可用资金**等权买入**（未持仓且不超过 `max_positions`）。
3. 持仓满 `holding-period` 个交易日即卖出；若开启 `--sell-on-exit-signal`，掉出选股结果也卖出。
4. 输出 `portfolio_metrics.csv`（每日净值）和 `indicator_metrics.csv`（绩效指标）。

> ⚠️ 性能提示：回测会**每天**调用选股器。若 `--turnover-source=akshare_hist`，每天都会对全市场逐个请求换手率，速度极慢。建议先用 `scripts/data_collector/akshare/collector.py` 把换手率导入 Qlib，再用 `--turnover-source=qlib`。

### 6.3 在 Python 中调用

```python
from qlib.contrib.golden_triangle.strategy import GoldenTriangleStrategy
import qlib
from qlib.constant import REG_CN
from qlib.backtest import backtest

qlib.init(provider_uri="~/.qlib/qlib_data/cn_data", region=REG_CN)

strategy = GoldenTriangleStrategy(
    turnover_source="qlib",
    max_positions=10,
    holding_period=5,
)

portfolio_metric, indicator_metric = backtest(
    start_time="2024-01-01",
    end_time="2024-12-31",
    strategy=strategy,
    executor={
        "class": "SimulatorExecutor",
        "module_path": "qlib.backtest.executor",
        "kwargs": {"time_per_step": "day", "generate_portfolio_metrics": True},
    },
    account=1_000_000,
    benchmark="SH000300",
    exchange_kwargs={
        "freq": "day",
        "limit_threshold": 0.095,
        "deal_price": "open",
        "open_cost": 0.0005,
        "close_cost": 0.0015,
        "min_cost": 5,
    },
)
```

---

## 七、集成到 Qlib Workflow

你也可以在 Python 代码中直接调用：

```python
import qlib
from qlib.contrib.golden_triangle import GoldenTriangleSelector, HybridDataSource

qlib.init(provider_uri="~/.qlib/qlib_data/cn_data", region=REG_CN)

# 1. Fetch data
ds = HybridDataSource(turnover_source="qlib")
df = ds.fetch(instruments="csi300", start_date="2024-01-01", end_date="2024-12-31")

# 2. Run selector
selector = GoldenTriangleSelector()
result = selector.select(df, trade_date="2024-12-31")
print(result)
```

---

## 八、注意事项

1. **停牌股 / ST 股 / 新股**
   - 程序会自动剔除成交量为 0（停牌）和上市不足 20 日的股票。
   - ST 股过滤需要提供 `st_set` 参数，或运行时通过 AKShare 获取。

2. **数据完整性**
   - 均线计算需要至少 20 个交易日数据，程序会自动过滤数据不足的股票。
   - 如果某只股票的 turnover 缺失，该行会被保留但可能无法通过换手率筛选。

3. **性能优化**
   - Qlib 的 `D.features` 可以秒级读取全市场 5000 只股票 30 天的 OHLCV。
   - AKShare 历史模式逐个请求较慢，建议首次通过 `collector.py` 导入 Qlib，之后使用 `turnover_source=qlib`。

4. **复权**
   - `run_selector.py --fallback-akshare` 默认使用**前复权**价格计算均线。
   - Qlib 本地数据默认也是复权后的，确保均线计算一致。
