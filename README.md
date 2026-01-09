# 视频镜头语言分析系统

一个基于大语言模型（LLM）的视频镜头语言分析系统，能够自动分析视频的镜头维度、运镜方式和镜头衔接，并生成评分报告。

## 功能特性

- 🎬 **视频帧提取**：自动从视频中提取每秒一帧
- 📷 **帧维度识别**：识别每帧的景别、拍摄角度和构图
- 🎥 **运镜方式分析**：分析相邻帧之间的运镜方式
- 🔄 **镜头衔接分析**：分析镜头之间的衔接方式
- 📊 **视频评分**：基于4个维度生成视频评分
- 📈 **Excel报告**：生成详细的Excel分析报告
- 💾 **结果保存**：自动保存分析结果，支持中断恢复
- 📁 **批量处理**：自动处理video文件夹下的所有视频

## 技术栈

- Python 3.8+
- OpenCV (cv2)：视频帧提取
- pandas：数据处理和Excel报告生成
- scikit-image：结构相似度计算
- ModelScope API：大语言模型调用（Qwen2.5-VL-7B-Instruct）

## 安装步骤

1. 克隆仓库
   ```bash
   git clone <仓库地址>
   cd video-lens-analysis
   ```

2. 创建虚拟环境
   ```bash
   python -m venv .venv
   ```

3. 激活虚拟环境
   - Windows：
     ```bash
     .venv\Scripts\activate
     ```
   - Linux/macOS：
     ```bash
     source .venv/bin/activate
     ```

4. 安装依赖
   ```bash
   pip install -r requirements.txt
   ```

5. 配置环境变量
   - 复制`.env.example`文件并重命名为`.env`
   - 填写ModelScope API相关信息

## 使用方法

1. 将视频文件放入`./video`文件夹

2. 运行分析脚本
   ```bash
   python video_analysis.py
   ```

3. 查看分析结果
   - 分析报告将保存在`video_analysis_report/`文件夹中
   - 提取的帧将保存在`video_frames/`文件夹中

## 分析维度说明

### 1. 镜头丰富度（25分）
- 基于相邻帧内容差异度计算
- 差异越大，得分越高

### 2. 景别/拍摄角度/构图丰富度（25分）
- 景别：远景、全景、中景、近景、特写
- 拍摄角度：平拍、仰拍、俯拍、斜拍
- 构图：三分法、引导线、对称、框架式、留白

### 3. 运镜方式丰富度（25分）
- 推镜头、拉镜头、摇镜头、移镜头、跟镜头、升降镜头、甩镜头

### 4. 镜头衔接多样性（25分）
- 顺序蒙太奇、平行蒙太奇、交叉蒙太奇、对比蒙太奇、象征蒙太奇、抒情蒙太奇

## 项目结构

```
.
├── llm_client.py          # LLM客户端，用于调用大语言模型
├── video_analysis.py      # 核心视频分析脚本
├── requirements.txt       # 项目依赖
├── .env                   # 环境变量配置
├── .gitignore             # Git忽略文件
└── README.md              # 项目说明文档
```

## 注意事项

1. 首次运行需要配置ModelScope API密钥
2. 分析过程中会调用LLM API，可能产生费用
3. 视频分析时间取决于视频长度和API响应速度
4. 建议使用较短的视频进行测试
5. 中断后再次运行会自动保存已分析的结果

## 许可证

MIT License

## 贡献

欢迎提交Issue和Pull Request！

## 联系方式

如有问题，请提交Issue或联系项目维护者。
