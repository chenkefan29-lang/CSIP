# CSIP 城市污水处理 VR 数字孪生

SE2026-AI-04。机理仿真（ASM1/BSM1）+ 数据驱动残差修正 + 强化学习设定值优化，
Three.js/WebXR 三维厂区，MQTT 实时链路，InfluxDB 历史存储。

## 目录

| 路径 | 内容 |
|---|---|
| `csip/sim/` | 机理仿真：ASM1/BSM1、进水生成、PI 控制、异常工况与应急预案、实时引擎、MQTT/Influx 管道 |
| `csip/ai/` | `dataset`/`hybrid` 混合模型、`online` 在线推理、`rl_env`/`rl_agent` 强化学习、`train_*` 见 `scripts/` |
| `main.py` | FastAPI：状态 / 历史 / 命令 / 工况 / 混合模型信息，并托管 `web/dist` |
| `web/` | Three.js + WebXR 前端（Vite） |
| `scripts/` | 数据集生成与模型训练脚本 |
| `infra/` | InfluxDB 数据目录、Mosquitto 配置 |
| `models/` | 训练产物：`hybrid/`、`rl/` |

## 启动

```bash
# 1. 基础设施（一次性，后台常驻）
influxd --bolt-path infra/influxdb/influxd.bolt --engine-path infra/influxdb/engine \
        --sqlite-path infra/influxdb/influxd.sqlite --http-bind-address 127.0.0.1:8086
"C:/Program Files/mosquitto/mosquitto.exe" -c infra/mosquitto/mosquitto.conf

# 2. 后端 + 仿真引擎（默认预热 10 天，约 3 分钟；调试可用 WARMUP_DAYS=3）
.venv/Scripts/python.exe main.py          # http://127.0.0.1:8000

# 3. 前端（开发模式，/api 代理到 8000）
cd web && npm run dev                     # http://127.0.0.1:5173
# 或构建后由后端直接托管：cd web && npm run build，然后访问 8000
```

凭据与端口在 `.env`（Influx token、租户、MQTT 端口等），不要提交到仓库。

## 训练

```bash
.venv/Scripts/python.exe scripts/train_hybrid.py        # 混合模型，GPU 约 10 s
.venv/Scripts/python.exe scripts/train_rl.py --iters 200 --B 32 --init 2.0 2.5 240 --tag _v3
```

RL 训练约 11 分钟（B=32，200 轮）。`--tag` 控制输出文件名，最后需把选定的权重命名为
`models/rl/policy.pt` 供后端加载。报告写在 `models/rl/report*.json`。

## 验收指标与实测

| 指标 | 要求 | 实测 |
|---|---|---|
| 仿真与实测偏差 | < 10% | 混合模型测试集最大 6.36%，平均 5.01%（纯机理平均 49.75%） |
| 操作响应 | < 100 ms | 命令往返 2.2 ms（P95 2.4 ms，WebSocket 链路，含服务端执行与回显） |
| VR 帧率 | > 72 fps | 待在有独显的机器 / 头显上实测，HUD 左上角显示 fps |

偏差定义：各目标量相对平均绝对误差 `Σ|ŷ−y| / Σ|y|`，见 `csip/ai/hybrid.py::rel_mae`。

## 前端操作

- 鼠标：单击设备选中（拖拽旋转视角），弹出面板可调 kLa / 内回流 / 外回流 / 排泥。
- VR：右手柄射线 + 扳机选择与按按钮，指向地面扣扳机传送；左手腕挂操作菜单。
- HUD 卡片：出水水质对照限值、告警与可一键执行的应急预案、异常工况演练、
  强化学习开关、混合模型机理→修正值对照、帧率与操作响应。

## 已知限制

- 出水总氮在 BSM1 基准工况下本身就有约 30% 时段超过 18 mg/L，这是基准模型特性，
  不是偏差指标问题；RL 在此工况下比默认控制减少约 33% 的 TN 超标时段。
- 强化学习提升出水质量与减少超标，但以约 8% 曝气能耗为代价。
- VR 帧率只做了性能优化（固定注视点、降低帧缓冲倍率、面板重绘限频、VR 下阴影降级），
  未在真实头显上测量。
