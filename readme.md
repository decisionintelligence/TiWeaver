## Debug
1. 修改 `train.py, eval.py` 中的 `--config_filename` 参数，直接点击右上方 `debug` 运行符号
2. 编写 `.sh` 脚本，在终端运行，如：`sh train.sh` -> 点击文件目录侧的debug运行符号
    - 配置 `.vscode/launch.json` （现有的）
    - 脚本内部：`python -m debugpy --listen localhost:8891 --wait-for-client train.py` 参数
      - `str` 参数 需要加引号 `--config_filename "config/xxx.yaml"`
      - `int` 参数 不需要加引号 `--batch_size 16`
      - `bool` 参数 `--use_gpu True`(默认设置为`False`, 当运行时出现这个参数表明为`True`；`--use_gpu False` 无法设置为`False`)
3. 若想调试具体的单个文件，直接点击运行符号会出现路径问题，在命令行中使用 `python .py` 启动
4. 根据图片调试代码，在`launch.json`文件中修改添加参数，选择要调试的点击运行即可

   ![debug](img/debug.png)
5. 
   ```bash
   # 格式：scp -P 源端口 -P 目标端口 用户@源IP:文件 用户@目标IP:路径
   # 注意：每个 -P 都要紧跟在对应的服务器前面
   scp -P 50015 /root/BAOWU_TS/humanactivity_model.ts root@172.23.166.140:/root/TiWeaver/models
   ```
## 环境配置
完整环境 `foundts`

mamba模型适用环境 `mamba`

## 运行小技巧
- 实时查看gpu使用 命令行 `gpustat -i`
   ![debug](img/gpustat.png)
- 查看运行程序 `ps auxww`
- 杀死程序 `kill -9 [pid]`
- 唤醒程序 `kill -SIGCONT [pid]`
- 切断窗口
   - 创建面板 `screen -S 面板命名`
   - 查看面板 `screen -ls`
   - 进入面板 `screen -r 面板命名`
   - 退出面板 `ctrl + a + d`
   - 关闭面板 `exit`（慎用 ！！！）
   - 强制关闭所有面板 `screen -S 面板命名 -X quit`（删除面板，且杀死所有在对应面板下运行代码，防止有些时候有些后台程序在关闭面板后没有被杀死成为僵尸）

## 文件结构

```
BAOWU_TS/
├── .vscode/                  # IDE配置目录
│   ├── launch.json           # debugger配置文件
│   └── settings.json         # vscode配置文件
├── Data_analysis/            # 数据分析相关代码
│   ├── clean.py              # 数据清洗脚本
│   └── process_logs.py       # 日志处理脚本
├── Data_Provider/            # 数据加载相关代码
├── dataset/                  # 数据目录
│   ├── align_dataset/        # 对齐后的数据集
│   ├── raw_dataset/          # 原始数据集
│   └── single_dataset/       # 单变量数据集
├── exp /                     # 实验相关代码
│   ├── exp_align_forecasting.py  # 对齐数据集的实验示例代码
│   ├── exp_basic.py          # 模型训练的基本代码组件
│   └── exp_align_rose.py     # ROSE训练、评估代码
├── img /                     # readme.md 图片目录
├── Model_Config/BAOWU /      # 模型配置目录
├── model /                   # baseline（包括：LR, ARIMA）；现有深度学习模型（包括：NODE，PatchTST等）；自己的idea（以后会有）
│   ├── layers /              # 网络结构组件， 如 Embed.py
│   ├── rose_checkpoints /    # ROSE 预训练参数
│   └── XX.py                 # 具体模型代码
├── Run/                      # 运行相关目录
│   ├── eval.py               # 评估脚本 评估，保存预测结果
│   ├── train.py              # 训练脚本 保存checkpoint，保存log
│   ├── logs/                 # 日志目录
│   ├── checkpoints/          # 模型检查点目录
│   ├── result/               # 结果CSV文件和可视化预测
│   └── scripts/              # 运行脚本
├── utils /                   # 工具函数目录
└── README.md                 # 项目说明
```

**注：项目的工作目录为BAOWU_TS**，checkpoints和logs一律保存到`./Run`下