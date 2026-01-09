import os
import sys
import time
import argparse
import logging
import json
import signal
from datetime import datetime
from typing import List, Dict, Tuple

import cv2
import pandas as pd
from skimage.metrics import structural_similarity as ssim
import numpy as np
from PIL import Image
import base64

from llm_client import HelloAgentsLLM

# 全局变量，用于保存中间结果
frame_info_list = []
frame_dimensions_list = []
motion_list = []
transition_list = []
score_result = None
output_excel_path = ""
video_path = ""

# 信号处理函数
def save_partial_results():
    """
    保存已处理的结果
    """
    # 声明全局变量
    global output_excel_path, score_result
    
    if frame_info_list or frame_dimensions_list or motion_list or transition_list:
        try:
            print("\n💾 正在保存已处理结果...")
            
            # 检查output_excel_path是否已设置
            if not output_excel_path:
                print("❌ 输出路径未设置，尝试自动生成...")
                # 尝试获取视频路径并生成输出路径
                try:
                    # 从全局变量中获取video_path
                    if 'video_path' in globals():
                        vp = globals()['video_path']
                    elif frame_info_list:
                        # 从帧信息中提取视频标识
                        first_frame_path = frame_info_list[0]['文件路径']
                        # 获取目录名作为视频标识
                        vp = os.path.dirname(first_frame_path)
                    else:
                        vp = "unknown_video"
                    
                    video_basename = os.path.basename(vp)
                    import hashlib
                    safe_video_name = hashlib.md5(video_basename.encode('utf-8')).hexdigest()[:8]
                    report_output_dir = "video_analysis_report"
                    os.makedirs(report_output_dir, exist_ok=True)
                    output_excel_path = os.path.join(report_output_dir, f"{safe_video_name}_partial_report.xlsx")
                    print(f"📁 已自动生成输出路径：{output_excel_path}")
                except Exception as e:
                    print(f"❌ 无法自动生成输出路径：{e}")
                    # 使用固定的默认路径
                    report_output_dir = "video_analysis_report"
                    os.makedirs(report_output_dir, exist_ok=True)
                    output_excel_path = os.path.join(report_output_dir, "partial_report.xlsx")
                    print(f"📁 使用默认输出路径：{output_excel_path}")
            
            # 确保frame_info_list和frame_dimensions_list长度匹配
            min_length = min(len(frame_info_list), len(frame_dimensions_list))
            safe_frame_info_list = frame_info_list[:min_length]
            safe_frame_dimensions_list = frame_dimensions_list[:min_length]
            
            # 计算视频整体评分（基于现有结果）
            if safe_frame_info_list:
                try:
                    frame_paths = [frame_info["文件路径"] for frame_info in safe_frame_info_list]
                    score_result = calculate_video_score(safe_frame_dimensions_list, motion_list, transition_list, frame_paths)
                    print(f"📊 已计算视频评分：{score_result['总得分']}分")
                except Exception as e:
                    print(f"❌ 计算评分失败：{e}")
                    score_result = None
            
            # 生成分析报告（基于现有结果）
            try:
                generate_analysis_report(
                    safe_frame_info_list,
                    safe_frame_dimensions_list,
                    motion_list,
                    transition_list,
                    score_result,
                    output_excel_path
                )
                print(f"✅ 已保存分析报告：{output_excel_path}")
                return True
            except Exception as e:
                print(f"❌ 生成Excel报告失败：{e}")
                # 尝试保存更简单的结果格式
                try:
                    simple_output_path = output_excel_path.replace('.xlsx', '_simple.json')
                    simple_result = {
                        "已处理帧数": len(frame_info_list),
                        "已分析帧数": len(frame_dimensions_list),
                        "已分析运镜次数": len(motion_list),
                        "已分析衔接次数": len(transition_list),
                        "帧信息": frame_info_list,
                        "帧维度": frame_dimensions_list,
                        "运镜方式": motion_list,
                        "衔接方式": transition_list,
                        "评分结果": score_result
                    }
                    with open(simple_output_path, 'w', encoding='utf-8') as f:
                        json.dump(simple_result, f, ensure_ascii=False, indent=2)
                    print(f"✅ 已保存简单JSON结果：{simple_output_path}")
                    return True
                except Exception as e2:
                    print(f"❌ 保存JSON结果也失败：{e2}")
                    # 尝试保存最基本的文本结果
                    try:
                        text_output_path = output_excel_path.replace('.xlsx', '_basic.txt')
                        with open(text_output_path, 'w', encoding='utf-8') as f:
                            f.write(f"已处理帧数：{len(frame_info_list)}\n")
                            f.write(f"已分析帧数：{len(frame_dimensions_list)}\n")
                            f.write(f"已分析运镜次数：{len(motion_list)}\n")
                            f.write(f"已分析衔接次数：{len(transition_list)}\n")
                            if score_result:
                                f.write(f"评分结果：{score_result['总得分']}分\n")
                        print(f"✅ 已保存基本文本结果：{text_output_path}")
                        return True
                    except Exception as e3:
                        print(f"❌ 保存文本结果也失败：{e3}")
                        return False
        except Exception as e:
            print(f"❌ 保存结果时发生未知错误：{e}")
            return False
    return False

def signal_handler(sig, frame):
    """
    捕获用户中断信号，保存已处理的结果
    """
    print(f"\n⚠️  捕获到信号 {sig}，正在保存已处理结果...")
    save_partial_results()
    print("👋 程序已安全终止")
    sys.exit(0)

# 注册信号处理
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# 在程序退出时自动保存结果
import atexit
def atexit_handler():
    """
    程序退出时自动保存结果
    """
    save_partial_results()

atexit.register(atexit_handler)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('video_analysis.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def extract_frames_per_second(video_path: str, output_dir: str) -> List[Dict]:
    """
    视频帧提取函数
    
    入参：
        video_path: 视频文件路径
        output_dir: 帧保存目录
    
    出参：
        帧信息列表，包含每帧的秒数、时间戳、文件路径
    """
    logger.info(f"开始从视频 {video_path} 中提取帧")
    
    # 检查视频文件是否存在
    if not os.path.exists(video_path):
        logger.error(f"视频文件不存在: {video_path}")
        raise FileNotFoundError(f"视频文件不存在: {video_path}")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 打开视频
    cap = cv2.VideoCapture(video_path)
    
    # 获取视频基本信息
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    logger.info(f"视频基本信息：帧率={fps:.2f}, 总帧数={total_frames}, 时长={duration:.2f}秒, 分辨率={width}x{height}")
    
    # 初始化结果列表
    frame_info_list = []
    
    try:
        # 每秒提取1帧
        for second in range(int(duration) + 1):
            # 计算该秒对应的帧位置
            frame_pos = int(second * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
            
            # 读取帧
            ret, frame = cap.read()
            if not ret:
                logger.warning(f"第{second}秒帧提取失败，跳过")
                continue
            
            # 生成时间戳
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            
            # 保存帧
            frame_filename = f"frame_{second}_timestamp_{timestamp}.jpg"
            frame_path = os.path.join(output_dir, frame_filename)
            
            if cv2.imwrite(frame_path, frame):
                logger.info(f"成功保存帧：{frame_path}")
                frame_info_list.append({
                    "秒数": second,
                    "时间戳": timestamp,
                    "文件路径": frame_path
                })
            else:
                logger.warning(f"保存帧失败：{frame_path}")
        
        logger.info(f"帧提取完成，共提取 {len(frame_info_list)} 帧")
        return frame_info_list
        
    except Exception as e:
        logger.error(f"帧提取过程中发生错误：{e}")
        raise
    finally:
        # 释放视频资源
        cap.release()

def recognize_single_frame_dimensions(frame_path: str) -> Dict[str, str]:
    """
    单帧维度识别函数
    
    入参：
        frame_path: 单帧图片路径
    
    出参：
        单帧维度字典，包含景别、拍摄角度、构图、识别错误
    """
    logger.info(f"开始识别帧 {frame_path} 的维度")
    
    # 检查图片是否存在
    if not os.path.exists(frame_path):
        logger.error(f"图片文件不存在: {frame_path}")
        return {"景别": "识别失败", "拍摄角度": "识别失败", "构图": "识别失败", "识别错误": "图片文件不存在"}
    
    try:
        # 初始化LLM客户端
        llm_client = HelloAgentsLLM()
        
        # 读取图片并编码为base64
        with open(frame_path, "rb") as image_file:
            image_base64 = base64.b64encode(image_file.read()).decode('utf-8')
        
        # 定义识别提示词
        prompts = {
            "景别": "判断该视频帧的景别，仅返回结果：远景 / 全景 / 中景 / 近景 / 特写",
            "拍摄角度": "判断该视频帧的拍摄角度，仅返回结果：平拍 / 仰拍 / 俯拍 / 斜拍",
            "构图": "判断该视频帧的构图方式，仅返回结果：三分法 / 引导线 / 对称 / 框架式 / 留白"
        }
        
        result = {
            "景别": "识别失败",
            "拍摄角度": "识别失败",
            "构图": "识别失败",
            "识别错误": ""
        }
        
        # 依次识别景别、拍摄角度、构图
        for dimension, prompt in prompts.items():
            messages = [
                {
                    "role": "user", 
                    "content": [
                        {
                            "type": "text", 
                            "text": prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_base64}"
                            }
                        }
                    ]
                }
            ]
            
            response = llm_client.think(messages)
            # 添加更长的延迟，避免请求过于频繁
            time.sleep(5)
            
            if response:
                # 清理响应，只保留有效结果
                response = response.strip()
                # 验证响应是否在有效选项中
                if dimension == "景别":
                    valid_options = ["远景", "全景", "中景", "近景", "特写"]
                elif dimension == "拍摄角度":
                    valid_options = ["平拍", "仰拍", "俯拍", "斜拍"]
                elif dimension == "构图":
                    valid_options = ["三分法", "引导线", "对称", "框架式", "留白"]
                
                if any(option in response for option in valid_options):
                    # 提取有效响应
                    for option in valid_options:
                        if option in response:
                            result[dimension] = option
                            break
                else:
                    logger.warning(f"{dimension}识别结果无效：{response}")
                    result["识别错误"] += f"{dimension}识别无效；"
            else:
                logger.error(f"{dimension}识别失败")
                result["识别错误"] += f"{dimension}识别失败；"
        
        logger.info(f"帧 {frame_path} 维度识别完成：{result}")
        return result
        
    except Exception as e:
        logger.error(f"帧 {frame_path} 维度识别过程中发生错误：{e}")
        return {"景别": "识别失败", "拍摄角度": "识别失败", "构图": "识别失败", "识别错误": str(e)}

def analyze_motion_between_frames(prev_frame_info: Dict, curr_frame_info: Dict, model_call_func) -> Tuple[str, str]:
    """
    相邻帧运镜分析函数
    
    入参：
        prev_frame_info: 前一帧信息
        curr_frame_info: 当前帧信息
        model_call_func: 模型调用函数
    
    出参：
        运镜方式、分析错误信息
    """
    logger.info(f"开始分析第{prev_frame_info['秒数']}秒帧与第{curr_frame_info['秒数']}秒帧之间的运镜方式")
    
    prev_frame_path = prev_frame_info["文件路径"]
    curr_frame_path = curr_frame_info["文件路径"]
    
    # 检查图片是否存在
    if not os.path.exists(prev_frame_path) or not os.path.exists(curr_frame_path):
        logger.error("图片文件不存在，无法分析运镜")
        return "无运镜", "图片文件不存在"
    
    try:
        # 读取两张图片
        prev_img = Image.open(prev_frame_path).convert('RGB')
        curr_img = Image.open(curr_frame_path).convert('RGB')
        
        # 初始化LLM客户端
        llm_client = HelloAgentsLLM()
        
        # 编码图片为base64
        def encode_image(image_path):
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        
        prev_base64 = encode_image(prev_frame_path)
        curr_base64 = encode_image(curr_frame_path)
        
        # 构建提示词
        messages = [
            {
                "role": "user", 
                "content": [
                    {
                        "type": "text", 
                        "text": "对比这两张相邻视频帧，分析运镜方式，仅返回结果：推镜头 / 拉镜头 / 摇镜头 / 移镜头 / 跟镜头 / 升降镜头 / 甩镜头 / 无运镜"
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{prev_base64}"
                        }
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{curr_base64}"
                        }
                    }
                ]
            }
        ]
        
        # 调用模型
        response = llm_client.think(messages)
        # 添加更长的延迟，避免请求过于频繁
        time.sleep(5)
        
        if response:
            # 清理响应
            response = response.strip()
            
            # 验证响应是否有效
            valid_motions = ["推镜头", "拉镜头", "摇镜头", "移镜头", "跟镜头", "升降镜头", "甩镜头", "无运镜"]
            
            if any(motion in response for motion in valid_motions):
                # 提取有效响应
                for motion in valid_motions:
                    if motion in response:
                        logger.info(f"运镜分析完成：{motion}")
                        return motion, ""
            else:
                logger.warning(f"运镜分析结果无效：{response}，使用兜底判断")
        else:
            logger.error("运镜分析失败，使用兜底判断")
        
        # 兜底判断：基于两帧的画面位置/比例变化
        # 这里使用简单的SSIM计算来判断是否有运镜
        prev_cv = cv2.imread(prev_frame_path, cv2.IMREAD_GRAYSCALE)
        curr_cv = cv2.imread(curr_frame_path, cv2.IMREAD_GRAYSCALE)
        
        # 调整图片大小，确保相同尺寸
        prev_cv = cv2.resize(prev_cv, (640, 480))
        curr_cv = cv2.resize(curr_cv, (640, 480))
        
        # 计算SSIM
        sim_score = ssim(prev_cv, curr_cv)
        
        # 如果相似度很高，认为无运镜
        if sim_score > 0.95:
            logger.info("兜底判断：无运镜")
            return "无运镜", ""
        else:
            logger.info("兜底判断：无运镜（无法精确判断具体类型）")
            return "无运镜", "兜底判断，无法精确判断具体类型"
        
    except Exception as e:
        logger.error(f"运镜分析过程中发生错误：{e}")
        return "无运镜", str(e)

def analyze_shot_transition(frame_dimensions_list: List[Dict], motion_list: List[str]) -> Dict[str, str]:
    """
    镜头衔接方式分析函数
    
    入参：
        frame_dimensions_list: 连续帧维度数据列表
        motion_list: 运镜方式列表
    
    出参：
        衔接方式详情字典，包含具体类型、大类、错误信息
    """
    logger.info("开始分析镜头衔接方式")
    
    try:
        # 初始化LLM客户端
        llm_client = HelloAgentsLLM()
        
        # 构建分析数据
        analysis_data = {
            "帧维度数据": frame_dimensions_list,
            "运镜方式": motion_list
        }
        
        # 构建提示词
        messages = [
            {
                "role": "user", 
                "content": f"基于这些连续帧的景别、拍摄角度、构图和运镜数据，分析镜头衔接方式，仅返回结果：顺序蒙太奇 / 平行蒙太奇 / 交叉蒙太奇 / 对比蒙太奇 / 象征蒙太奇 / 抒情蒙太奇 / 无明显衔接\n\n{json.dumps(analysis_data, ensure_ascii=False)}"
            }
        ]
        
        # 调用模型
        response = llm_client.think(messages)
        # 添加更长的延迟，避免请求过于频繁
        time.sleep(5)
        
        if response:
            # 清理响应
            response = response.strip()
            
            # 验证响应是否有效
            valid_transitions = ["顺序蒙太奇", "平行蒙太奇", "交叉蒙太奇", "对比蒙太奇", "象征蒙太奇", "抒情蒙太奇", "无明显衔接"]
            
            if any(transition in response for transition in valid_transitions):
                # 提取有效响应
                for transition in valid_transitions:
                    if transition in response:
                        # 分类为叙事蒙太奇或表现蒙太奇
                        if transition in ["顺序蒙太奇", "平行蒙太奇", "交叉蒙太奇"]:
                            category = "叙事蒙太奇"
                        elif transition in ["对比蒙太奇", "象征蒙太奇", "抒情蒙太奇"]:
                            category = "表现蒙太奇"
                        else:
                            category = ""
                        
                        result = {
                            "具体类型": transition,
                            "大类": category,
                            "错误信息": ""
                        }
                        logger.info(f"镜头衔接方式分析完成：{result}")
                        return result
            else:
                logger.warning(f"镜头衔接方式分析结果无效：{response}")
        else:
            logger.error("镜头衔接方式分析失败")
        
        # 兜底结果
        logger.info("使用兜底结果：无明显衔接")
        return {"具体类型": "无明显衔接", "大类": "", "错误信息": "分析失败，使用兜底结果"}
        
    except Exception as e:
        logger.error(f"镜头衔接方式分析过程中发生错误：{e}")
        return {"具体类型": "无明显衔接", "大类": "", "错误信息": str(e)}

def calculate_video_score(
    frame_dimensions_list: List[Dict],
    motion_list: List[str],
    transition_list: List[Dict],
    frame_paths: List[str]
) -> Dict:
    """
    视频整体评分函数
    
    入参：
        frame_dimensions_list: 帧维度数据列表
        motion_list: 运镜方式列表
        transition_list: 衔接方式列表
        frame_paths: 帧图片路径列表
    
    出参：
        评分结果字典，包含4个维度的得分、计算依据、说明，及视频总得分
    """
    logger.info("开始计算视频整体评分")
    
    try:
        # 1. 镜头丰富度（25分）：基于相邻帧内容差异度
        logger.info("计算镜头丰富度得分")
        avg_similarity = 0.0
        valid_frame_pairs = 0
        
        if len(frame_paths) >= 2:
            for i in range(len(frame_paths) - 1):
                # 读取相邻两帧
                prev_img = cv2.imread(frame_paths[i], cv2.IMREAD_GRAYSCALE)
                curr_img = cv2.imread(frame_paths[i+1], cv2.IMREAD_GRAYSCALE)
                
                if prev_img is not None and curr_img is not None:
                    # 调整图片大小
                    prev_img = cv2.resize(prev_img, (640, 480))
                    curr_img = cv2.resize(curr_img, (640, 480))
                    
                    # 计算SSIM
                    sim_score = ssim(prev_img, curr_img)
                    avg_similarity += sim_score
                    valid_frame_pairs += 1
        
        if valid_frame_pairs > 0:
            avg_similarity /= valid_frame_pairs
        
        # 得分公式：25 - (平均相似度 × 25)，相似度越低，差异越大，得分越高
        shot_richness_score = max(0, min(25, 25 - (avg_similarity * 25)))
        
        # 2. 景别/拍摄角度/构图丰富度（25分）
        logger.info("计算景别/拍摄角度/构图丰富度得分")
        
        # 统计各维度的唯一类型数量
        景别_set = set()
        拍摄角度_set = set()
        构图_set = set()
        
        for frame_data in frame_dimensions_list:
            if frame_data["景别"] != "识别失败":
                景别_set.add(frame_data["景别"])
            if frame_data["拍摄角度"] != "识别失败":
                拍摄角度_set.add(frame_data["拍摄角度"])
            if frame_data["构图"] != "识别失败":
                构图_set.add(frame_data["构图"])
        
        # 计算各维度得分
        # 景别总类型5种，拍摄角度总类型4种，构图总类型5种
        景别_score = (len(景别_set) / 5) * (25 / 3)
        拍摄角度_score = (len(拍摄角度_set) / 4) * (25 / 3)
        构图_score = (len(构图_set) / 5) * (25 / 3)
        
        dimension_richness_score = max(0, min(25, 景别_score + 拍摄角度_score + 构图_score))
        
        # 3. 运镜方式丰富度（25分）
        logger.info("计算运镜方式丰富度得分")
        
        # 统计运镜方式的唯一类型数量（排除"无运镜"）
        motion_set = set([motion for motion in motion_list if motion != "无运镜"])
        motion_richness_score = max(0, min(25, (len(motion_set) / 7) * 25))
        
        # 4. 镜头衔接多样性（25分）
        logger.info("计算镜头衔接多样性得分")
        
        # 统计衔接方式的唯一类型数量（排除"无明显衔接"）
        transition_set = set([t["具体类型"] for t in transition_list if t["具体类型"] != "无明显衔接"])
        transition_diversity_score = max(0, min(25, (len(transition_set) / 6) * 25))
        
        # 计算总得分
        total_score = shot_richness_score + dimension_richness_score + motion_richness_score + transition_diversity_score
        
        # 生成评分说明
        score_result = {
            "镜头丰富度": {
                "得分": round(shot_richness_score, 2),
                "计算依据": f"基于{valid_frame_pairs}组相邻帧的内容相似度，平均相似度为{avg_similarity:.4f}",
                "说明": "镜头内容差异越大，得分越高"
            },
            "景别/拍摄角度/构图丰富度": {
                "得分": round(dimension_richness_score, 2),
                "计算依据": f"景别类型数：{len(景别_set)}/5，拍摄角度类型数：{len(拍摄角度_set)}/4，构图类型数：{len(构图_set)}/5",
                "说明": "类型越多样，得分越高"
            },
            "运镜方式丰富度": {
                "得分": round(motion_richness_score, 2),
                "计算依据": f"运镜类型数：{len(motion_set)}/7",
                "说明": "运镜类型越多，得分越高"
            },
            "镜头衔接多样性": {
                "得分": round(transition_diversity_score, 2),
                "计算依据": f"衔接方式类型数：{len(transition_set)}/6",
                "说明": "衔接方式越多样，得分越高"
            },
            "总得分": round(total_score, 2)
        }
        
        logger.info(f"视频整体评分完成：{score_result}")
        return score_result
        
    except Exception as e:
        logger.error(f"视频评分过程中发生错误：{e}")
        raise

def generate_analysis_report(
    frame_info_list: List[Dict],
    frame_dimensions_list: List[Dict],
    motion_list: List[str],
    transition_list: List[Dict],
    score_result: Dict,
    output_excel_path: str
) -> None:
    """
    结果保存与报告生成函数
    
    入参：
        frame_info_list: 帧基础信息列表
        frame_dimensions_list: 帧维度数据列表
        motion_list: 运镜方式列表
        transition_list: 衔接方式列表
        score_result: 评分结果字典
        output_excel_path: 输出Excel路径
    
    出参：
        无，仅完成文件保存
    """
    logger.info(f"开始生成分析报告，保存路径：{output_excel_path}")
    
    try:
        # 确保目录存在
        os.makedirs(os.path.dirname(output_excel_path), exist_ok=True)
        
        # 创建Excel写入器
        with pd.ExcelWriter(output_excel_path, engine='openpyxl') as writer:
            # 1. 工作表1：帧基础数据
            logger.info("生成帧基础数据工作表")
            
            frame_data_list = []
            # 确保处理的是两个列表的最小长度
            min_frames = min(len(frame_info_list), len(frame_dimensions_list))
            for i in range(min_frames):
                frame_info = frame_info_list[i]
                frame_dim = frame_dimensions_list[i]
                frame_data_list.append({
                    "帧序号": i + 1,
                    "秒数": frame_info["秒数"],
                    "时间戳": frame_info["时间戳"],
                    "帧路径": frame_info["文件路径"],
                    "景别": frame_dim["景别"],
                    "拍摄角度": frame_dim["拍摄角度"],
                    "构图": frame_dim["构图"]
                })
            
            # 处理空数据情况
            if not frame_data_list:
                df_frame_data = pd.DataFrame(columns=["帧序号", "秒数", "时间戳", "帧路径", "景别", "拍摄角度", "构图"])
            else:
                df_frame_data = pd.DataFrame(frame_data_list)
            df_frame_data.to_excel(writer, sheet_name="帧基础数据", index=False)
            
            # 2. 工作表2：运镜与衔接数据
            logger.info("生成运镜与衔接数据工作表")
            
            motion_transition_data = []
            for i in range(len(motion_list)):
                frame_range = f"{i}-{i+1}秒"
                motion = motion_list[i]
                
                # 获取对应的衔接方式
                transition = transition_list[i] if i < len(transition_list) else {"具体类型": "", "大类": ""}
                
                motion_transition_data.append({
                    "帧区间": frame_range,
                    "运镜方式": motion,
                    "衔接方式（具体类型）": transition["具体类型"],
                    "衔接方式（大类）": transition["大类"]
                })
            
            # 处理空数据情况
            if not motion_transition_data:
                df_motion_transition = pd.DataFrame(columns=["帧区间", "运镜方式", "衔接方式（具体类型）", "衔接方式（大类）"])
            else:
                df_motion_transition = pd.DataFrame(motion_transition_data)
            df_motion_transition.to_excel(writer, sheet_name="运镜与衔接数据", index=False)
            
            # 3. 工作表3：视频评分结果
            logger.info("生成视频评分结果工作表")
            
            score_data = []
            if score_result:
                for dimension, score_info in score_result.items():
                    if dimension == "总得分":
                        continue
                    
                    score_data.append({
                        "评分维度": dimension,
                        "得分（满分25）": score_info["得分"],
                        "计算依据": score_info["计算依据"],
                        "评分说明": score_info["说明"]
                    })
                
                # 添加总得分行
                score_data.append({
                    "评分维度": "总得分",
                    "得分（满分100）": score_result["总得分"],
                    "计算依据": "4个维度得分之和",
                    "评分说明": "各维度权重25分，总分100分"
                })
            
            # 处理空数据情况
            if not score_data:
                df_score = pd.DataFrame(columns=["评分维度", "得分（满分25/100）", "计算依据", "评分说明"])
            else:
                df_score = pd.DataFrame(score_data)
            df_score.to_excel(writer, sheet_name="视频评分结果", index=False)
        
        logger.info("报告生成成功")
        print(f"✅ 分析报告已成功生成：{output_excel_path}")
        
    except Exception as e:
        logger.error(f"报告生成过程中发生错误：{e}")
        raise

def main():
    """
    主函数，整合所有功能
    """
    # 记录程序开始时间
    import time
    start_time = time.perf_counter()
    
    # 使用全局变量
    global frame_info_list, frame_dimensions_list, motion_list, transition_list, score_result, output_excel_path, video_path
    
    # 初始化全局变量
    frame_info_list.clear()
    frame_dimensions_list.clear()
    motion_list.clear()
    transition_list.clear()
    score_result = None
    
    # 获取video文件夹下的所有视频文件
    video_folder = "./video"
    if not os.path.exists(video_folder):
        print(f"❌ 视频文件夹不存在：{video_folder}")
        sys.exit(1)
    
    # 支持的视频格式
    video_extensions = [".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv"]
    
    # 获取所有视频文件
    video_files = []
    for file in os.listdir(video_folder):
        file_path = os.path.join(video_folder, file)
        if os.path.isfile(file_path) and os.path.splitext(file)[1].lower() in video_extensions:
            video_files.append(file_path)
    
    if not video_files:
        print(f"❌ 视频文件夹中没有找到支持的视频文件")
        sys.exit(1)
    
    print(f"🎉 找到{len(video_files)}个视频文件，开始分析...")
    
    # 遍历所有视频文件
    for i, video_path in enumerate(video_files):
        print(f"\n📽️  开始分析第{i+1}/{len(video_files)}个视频：{os.path.basename(video_path)}")
        
        # 初始化当前视频的全局变量
        frame_info_list.clear()
        frame_dimensions_list.clear()
        motion_list.clear()
        transition_list.clear()
        score_result = None
        
        # 1. 创建输出目录
        video_basename = os.path.basename(video_path)
        video_name = os.path.splitext(video_basename)[0]
        # 使用视频名的哈希值作为目录名，避免中文编码问题
        import hashlib
        safe_video_name = hashlib.md5(video_basename.encode('utf-8')).hexdigest()[:8]
        frame_output_dir = os.path.join("video_frames", safe_video_name)
        report_output_dir = "video_analysis_report"
        
        os.makedirs(frame_output_dir, exist_ok=True)
        os.makedirs(report_output_dir, exist_ok=True)
        
        # 使用安全的文件名，避免中文编码问题
        output_excel_path = os.path.join(report_output_dir, f"{safe_video_name}_report.xlsx")
        
        try:
            try:
                # 2. 提取视频帧
                print("🎬 开始提取视频帧...")
                frame_info_list = extract_frames_per_second(video_path, frame_output_dir)
                if not frame_info_list:
                    logger.error("未提取到任何帧，跳过该视频")
                    continue
                
                # 立即保存帧提取结果
                print(f"💾 已提取{len(frame_info_list)}帧，正在保存初始结果...")
                save_partial_results()
                
                # 3. 识别单帧维度
                print("📷 开始识别帧维度...")
                for j, frame_info in enumerate(frame_info_list):
                    frame_path = frame_info["文件路径"]
                    
                    # 在处理每一帧前检查是否需要保存结果
                    if j > 0 and j % 5 == 0:
                        print(f"💾 已处理{j}帧，正在保存临时结果...")
                        save_partial_results()
                    
                    try:
                        frame_dim = recognize_single_frame_dimensions(frame_path)
                        frame_dimensions_list.append(frame_dim)
                        print(f"✅ 已处理第{j+1}帧：景别={frame_dim['景别']}，拍摄角度={frame_dim['拍摄角度']}，构图={frame_dim['构图']}")
                    except KeyboardInterrupt:
                        print(f"\n⚠️  处理第{j+1}帧时捕获到中断")
                        save_partial_results()
                        print("👋 程序已安全终止")
                        sys.exit(0)
                    except Exception as e:
                        print(f"❌ 处理第{j+1}帧时发生错误：{e}")
                        frame_dimensions_list.append({"景别": "识别失败", "拍摄角度": "识别失败", "构图": "识别失败", "识别错误": str(e)})
                
                # 处理完所有帧后保存结果
                print(f"💾 已处理所有{len(frame_dimensions_list)}帧，正在保存结果...")
                save_partial_results()
                
                # 4. 分析相邻帧运镜方式
                print("🎥 开始分析运镜方式...")
                for j in range(len(frame_info_list) - 1):
                    prev_frame = frame_info_list[j]
                    curr_frame = frame_info_list[j + 1]
                    
                    # 在处理每10个运镜前检查是否需要保存结果
                    if (j + 1) % 10 == 0:
                        print(f"💾 已分析{len(motion_list)}个运镜，正在保存临时结果...")
                        save_partial_results()
                    
                    try:
                        motion, error = analyze_motion_between_frames(prev_frame, curr_frame, None)
                        motion_list.append(motion)
                        print(f"✅ 已分析第{j+1}个运镜：{motion}")
                    except KeyboardInterrupt:
                        print(f"\n⚠️  分析第{j+1}个运镜时捕获到中断")
                        save_partial_results()
                        print("👋 程序已安全终止")
                        sys.exit(0)
                    except Exception as e:
                        print(f"❌ 分析第{j+1}个运镜时发生错误：{e}")
                        motion_list.append("无运镜")
                
                # 处理完所有运镜后保存结果
                print(f"💾 已分析所有{len(motion_list)}个运镜，正在保存结果...")
                save_partial_results()
                
                # 5. 分析镜头衔接方式
                print("🔄 开始分析镜头衔接方式...")
                # 每3个连续帧分析一次衔接方式
                for j in range(0, len(frame_dimensions_list) - 2):
                    # 在处理每5个衔接前检查是否需要保存结果
                    if (j + 1) % 5 == 0:
                        print(f"💾 已分析{len(transition_list)}个镜头衔接，正在保存临时结果...")
                        save_partial_results()
                    
                    try:
                        transition = analyze_shot_transition(frame_dimensions_list[j:j+3], motion_list[j:j+2])
                        transition_list.append(transition)
                        print(f"✅ 已分析第{j+1}个镜头衔接：{transition['具体类型']}")
                    except KeyboardInterrupt:
                        print(f"\n⚠️  分析第{j+1}个镜头衔接时捕获到中断")
                        save_partial_results()
                        print("👋 程序已安全终止")
                        sys.exit(0)
                    except Exception as e:
                        print(f"❌ 分析第{j+1}个镜头衔接时发生错误：{e}")
                        transition_list.append({"具体类型": "无明显衔接", "大类": "", "错误信息": str(e)})
                
                # 处理完所有衔接后保存结果
                print(f"💾 已分析所有{len(transition_list)}个镜头衔接，正在保存结果...")
                save_partial_results()
                
            except KeyboardInterrupt:
                print(f"\n⚠️  捕获到键盘中断")
                save_partial_results()
                print("👋 程序已安全终止")
                sys.exit(0)
            except Exception as e:
                logger.error(f"程序执行过程中发生错误：{e}")
                print(f"❌ 错误：{e}")
                # 继续执行，保存已处理的结果
            
            # 6. 计算视频整体评分（基于现有结果）
            if frame_info_list:
                frame_paths = [frame_info["文件路径"] for frame_info in frame_info_list]
                score_result = calculate_video_score(frame_dimensions_list, motion_list, transition_list, frame_paths)
            
            # 7. 生成最终分析报告
            print("📊 生成最终分析报告...")
            generate_analysis_report(
                frame_info_list,
                frame_dimensions_list,
                motion_list,
                transition_list,
                score_result,
                output_excel_path
            )
            
            logger.info("视频分析完成")
            print("🎉 视频分析完成！")
            print(f"📊 分析报告已保存至：{output_excel_path}")
            print(f"🎬 提取的帧保存至：{frame_output_dir}")
            
        except FileNotFoundError as e:
            logger.error(f"文件不存在：{e}")
            print(f"❌ 错误：文件不存在 - {e}")
            # 保存已处理结果
            save_partial_results()
        except KeyboardInterrupt:
            logger.info("捕获到键盘中断")
            print(f"\n⚠️  捕获到键盘中断")
            # 保存已处理结果
            save_partial_results()
        except Exception as e:
            logger.error(f"最终处理过程中发生错误：{e}")
            print(f"❌ 错误：{e}")
            # 再次尝试保存结果
            save_partial_results()
        finally:
            # 无论程序如何结束，都保存结果
            print(f"\n💾 当前视频处理结束，正在保存结果...")
            save_partial_results()
    
    # 计算并输出程序运行总时间
    end_time = time.perf_counter()
    total_time = end_time - start_time
    minutes = int(total_time // 60)
    seconds = total_time % 60
    print(f"\n🎉 所有视频分析完成！")
    print(f"⏱️  程序总运行时间：{minutes}分{seconds:.2f}秒")
    print("👋 程序已安全终止")

if __name__ == "__main__":
    main()