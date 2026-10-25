# CSIP 城市污水处理 VR 数字孪生 — 组员安装指南

需要 Python 3.14（requirements.txt 是在 3.14 下导出的，其他版本可能装不上部分包）和 Node.js。InfluxDB 2.x 和 Mosquitto 是可选的：不装也能运行仿真和三维页面，只是没有历史曲线和 MQTT 推送。

---

## 1. 克隆仓库

```bash
git clone https://github.com/<你的用户名>/CSIP.git
cd CSIP
```

---

## 2. 配置环境文件

把 `.env.example` 复制一份命名为 `.env`，填入 InfluxDB 凭据：

```bash
cp .env.example .env
```

打开 `.env`，把下面两行的值填上（初次启动 InfluxDB 会让你设置，设好后填回来）：

```
INFLUX_TOKEN=你的 token
INFLUX_ADMIN_PASSWORD=你设的密码
```

其他配置项保持默认即可（`INFLUX_URL`、`MQTT_HOST` 等）。

---

## 3. 安装 Python 依赖

建议用虚拟环境，防止版本冲突：

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/Mac
source .venv/bin/activate

pip install -r requirements.txt
```

**注意：** `requirements.txt` 里的 `torch==2.11.0` 会从 PyPI 下载 CPU 版本。如果你的电脑有 NVIDIA 显卡并且想用 GPU 加速训练，安装后再运行：

```bash
pip install torch==2.11.0+cu128 --index-url https://download.pytorch.org/whl/cu128
```

没有 GPU 也能跑，只是训练混合模型时从 10 秒变成约 1 分钟。

---

## 4. 安装前端依赖

```bash
cd web
npm install
npm run build
cd ..
```

构建产物在 `web/dist/`，后端启动时会自动托管这个目录。

---

## 5. 启动 InfluxDB（后台常驻）

**首次启动：**（先建好数据目录：`mkdir -p infra/influxdb`）

```bash
# Windows（假设安装在 C:\Program Files\InfluxDB）
"C:\Program Files\InfluxDB\influxd.exe" ^
  --bolt-path infra/influxdb/influxd.bolt ^
  --engine-path infra/influxdb/engine ^
  --sqlite-path infra/influxdb/influxd.sqlite ^
  --http-bind-address 127.0.0.1:8086

# Linux/Mac
influxd --bolt-path infra/influxdb/influxd.bolt \
        --engine-path infra/influxdb/engine \
        --sqlite-path infra/influxdb/influxd.sqlite \
        --http-bind-address 127.0.0.1:8086
```

首次启动后，浏览器打开 http://127.0.0.1:8086 创建初始账号：

- **Username:** 填 `admin`（和 `.env` 里的 `INFLUX_ADMIN_USER` 一致）
- **Password:** 自己设一个，记得填回 `.env` 的 `INFLUX_ADMIN_PASSWORD`
- **Organization:** 填 `csip`（和 `.env` 里的 `INFLUX_ORG` 一致）
- **Bucket:** 填 `wwtp`（和 `.env` 里的 `INFLUX_BUCKET` 一致）

设置完成后会显示 token，复制下来，填进 `.env` 的 `INFLUX_TOKEN=`。

以后启动用同样的命令，或者在任务管理器 / systemd 里设成开机启动。

---

## 6. 启动 Mosquitto（后台常驻）

```bash
# Windows（假设安装在 C:\Program Files\mosquitto）
"C:\Program Files\mosquitto\mosquitto.exe" -c infra/mosquitto/mosquitto.conf

# Linux/Mac
mosquitto -c infra/mosquitto/mosquitto.conf
```

这是一个无认证的本地 MQTT broker，监听 1884（TCP）和 9001（WebSocket）。日志输出到终端，Ctrl+C 停止，或者后台启动。

---

## 7. 启动后端

```bash
# Windows
.venv\Scripts\python.exe main.py

# Linux/Mac
.venv/bin/python main.py
```

首次启动会预热 10 天的仿真数据，约 3 分钟。看到 `Uvicorn running on http://127.0.0.1:8000` 就成功了。

如果想加速预热，可以设环境变量：

```bash
# Windows PowerShell
$env:WARMUP_DAYS=5
.venv\Scripts\python.exe main.py

# Linux/Mac / Git Bash
WARMUP_DAYS=5 .venv/bin/python main.py
```

---

## 8. 打开浏览器

访问 http://127.0.0.1:8000 ，看到三维厂区页面就说明前后端都启动成功了。

- 点设备（二沉池、曝气池、反应器）会弹出控制面板，可以调节 kLa、内回流、外回流、排泥量。
- HUD 卡片显示出水水质、告警与应急预案、工况演练、强化学习开关、混合模型对比。

VR 模式需要有 WebXR 兼容的头显和浏览器（如 Meta Quest + Chromium 内核浏览器）。

---

## VS Code 调试（可选）

项目里已经配置好 `.vscode/launch.json`，用 VS Code 打开项目后：

1. 按 `Ctrl+Shift+P`，输入 "Python: Select Interpreter"，选 `.venv/Scripts/python.exe`（或 `.venv/bin/python`）。
2. 按 `F5`，选择"CSIP 后端"，自动用 5 天预热启动后端。
3. 可以在 Python 代码里下断点，按 `Shift+F5` 停止。

---

## 训练模型（可选）

如果想重新训练混合模型或强化学习策略：

```bash
# 混合模型（GPU 约 10 秒，CPU 约 1 分钟）
.venv/Scripts/python.exe scripts/train_hybrid.py

# 强化学习（默认 200 轮，约 11 分钟）
.venv/Scripts/python.exe scripts/train_rl.py --iters 200 --B 32 --tag _v4
```

训练完成后，模型文件保存在 `models/hybrid/` 和 `models/rl/`。后端启动时加载的是 `models/rl/policy.pt` 和 `models/hybrid/residual_gru.pt`，新训练的 RL 权重要改名成 `policy.pt` 才会生效。

---

## 常见问题

**Q: InfluxDB 启动报 "permission denied" 或 "database is locked"**  
A: `infra/influxdb/` 目录可能权限不对，或者另一个 InfluxDB 进程在占用。关掉其他 InfluxDB，删掉 `infra/influxdb/` 里的所有文件重新初始化。

**Q: 后端启动报 "ModuleNotFoundError: No module named 'torch'"**  
A: 虚拟环境没激活，或者激活的虚拟环境不是项目的 `.venv`。重新激活一次：`.venv\Scripts\activate`（Windows）/ `source .venv/bin/activate`（Linux/Mac）。

**Q: 浏览器打开后三维场景是黑的**  
A: 检查浏览器控制台（F12）有没有报错。可能是前端没构建（缺 `web/dist`）或者后端托管路径不对。重新 `cd web; npm run build`。

**Q: VR 模式进不去**  
A: 需要 WebXR 兼容的设备和浏览器。手机、PC 浏览器默认不支持。可以用 Chrome 的 WebXR Emulator 插件测试。

**Q: 想在另一台电脑（或 VR 头显）访问，但页面打不开**  
A: 默认只监听 127.0.0.1（本机）。在 `.env` 里加一行 `API_HOST=0.0.0.0`，同时把 `infra/mosquitto/mosquitto.conf` 里两处 `127.0.0.1` 也改成 `0.0.0.0`。注意：这样做后局域网内其他人也能访问，项目没有身份认证，只适合内网演示。

---

## 许可

SE2026-AI-04 课程项目，仅供学习交流。
