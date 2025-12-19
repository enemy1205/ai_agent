import cv2
import numpy as np
import pyrealsense2 as rs
import json
import os
from scipy.spatial.transform import Rotation as R

# ================= 配置参数 =================
# AprilGrid 参数 (单位: mm)
TAG_FAMILY = cv2.aruco.DICT_APRILTAG_36H11
TAG_ROWS = 6
TAG_COLS = 6
TAG_SIZE = 55.0       
TAG_SPACING = 16.5    

# 相机内参 (RealSense)
CAMERA_MATRIX = np.array([
    [648.493, 0, 649.185],
    [0, 647.91, 371.815],
    [0, 0, 1]
], dtype=np.float64)

DIST_COEFFS = np.array([-0.0497892, 0.0565904, -0.00093964, 0.000405942, -0.0172231], dtype=np.float64)

# 机械臂欧拉角旋转顺序 (非常重要！)
# 大多数机械臂 (如 UR, ABB, KUKA, Fanuc) 的欧拉角定义不同。
# 常见的有 'xyz' (静态轴) 或 'zyx' (动态轴)。如果不确定，通常先试 'xyz'。
ROBOT_ROTATION_ORDER = 'xyz' 

# ===========================================

def init_realsense():
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
    profile = pipeline.start(config)
    return pipeline, profile

def parse_robot_pose(input_str):
    """
    解析用户输入的机械臂位姿字符串: x, y, z, rx, ry, rz
    """
    try:
        input_str = input_str.replace('，', ',').strip()
        parts = [float(x) for x in input_str.replace(',', ' ').split()]
        if len(parts) != 6:
            print(f"⚠️ 格式错误: 需要6个数值 (x y z rx ry rz)，检测到 {len(parts)} 个")
            return None
        return parts 
    except ValueError:
        print("⚠️ 输入包含非数字字符")
        return None

def main():
    # 1. 初始化 Aruco
    aruco_dict = cv2.aruco.getPredefinedDictionary(TAG_FAMILY)
    aruco_params = cv2.aruco.DetectorParameters()
    
    # 兼容 OpenCV 不同版本的 GridBoard 创建
    try:
        board = cv2.aruco.GridBoard(
            (TAG_COLS, TAG_ROWS), TAG_SIZE, TAG_SPACING, aruco_dict
        )
    except AttributeError:
        board = cv2.aruco.GridBoard_create(
            TAG_COLS, TAG_ROWS, TAG_SIZE, TAG_SPACING, aruco_dict
        )

    # 2. 初始化相机
    pipeline, _ = init_realsense()
    
    # 存储用于标定的数据
    data_records = [] 

    print("=============================================================")
    print("   手眼标定数据采集 (Eye-in-Hand)")
    print("   目标：求解 相机 相对于 机械臂末端法兰 的位姿")
    print("=============================================================")
    print("1. 移动机械臂，确保相机能看到 AprilGrid")
    print("2. 按【空格键】暂停画面")
    print("3. 输入机械臂位姿: x y z rx ry rz (单位: mm, 度)")
    print("4. 按【Q】结束采集并计算")
    print("=============================================================")

    try:
        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame: continue
            
            frame = np.asanyarray(color_frame.get_data())
            display_frame = frame.copy()
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            corners, ids, rejected = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=aruco_params)
            
            valid_pose = False
            rvec_board, tvec_board = None, None

            if ids is not None and len(ids) > 0:
                cv2.aruco.drawDetectedMarkers(display_frame, corners, ids)
                
                # 估计标定板相对于相机的位姿
                retval, rvec_board, tvec_board = cv2.aruco.estimatePoseBoard(
                    corners, ids, board, CAMERA_MATRIX, DIST_COEFFS, None, None
                )
                
                if retval > 0:
                    valid_pose = True
                    # 绘制坐标轴 (长度100mm)
                    cv2.drawFrameAxes(display_frame, CAMERA_MATRIX, DIST_COEFFS, rvec_board, tvec_board, 100)
                    cv2.putText(display_frame, "Tag Detected! Press SPACE", (30, 50), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            cv2.imshow('Calibration Collector', display_frame)
            key = cv2.waitKey(1)
            
            # === 按下空格键，录入数据 ===
            if key == 32: 
                if not valid_pose:
                    print("⚠️ 未检测到标定板，无法记录！")
                    continue
                
                print("\n>>> [暂停] 画面已锁定。")
                cv2.imshow('Calibration Collector', display_frame) # 刷新显示
                cv2.waitKey(1)
                
                print(">>> 请输入机械臂位姿 (x y z rx ry rz): ")
                user_input = input("Input > ")
                robot_pose = parse_robot_pose(user_input)
                
                if robot_pose:
                    record = {
                        "id": len(data_records),
                        "robot_pose": robot_pose,           # 用户输入的原始数据 [x,y,z,rx,ry,rz]
                        "cam_rvec": rvec_board.flatten().tolist(), # 标定板相对相机 R (向量)
                        "cam_tvec": tvec_board.flatten().tolist()  # 标定板相对相机 T
                    }
                    data_records.append(record)
                    print(f"✅ 第 {len(data_records)} 组数据已保存。")
                else:
                    print("❌ 输入格式错误，忽略本次。")
                
                print(">>> 继续视频流...\n")

            # === 按 Q 退出 ===
            elif key & 0xFF == ord('q'):
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

    # ================= 计算部分 =================
    if len(data_records) >= 3:
        # 保存原始数据备份
        with open("hand_eye_data.json", "w") as f:
            json.dump(data_records, f, indent=4)
        
        print(f"\n采集结束，共 {len(data_records)} 组数据。正在计算...")
        perform_calibration(data_records)
    else:
        print("数据过少 (至少需要3组)，无法计算。")

def perform_calibration(records):
    """
    核心标定计算函数
    """
    R_gripper2base = [] # 机械臂末端 -> 基座 (旋转矩阵)
    t_gripper2base = [] # 机械臂末端 -> 基座 (平移向量)
    R_target2cam = []   # 标定板 -> 相机 (旋转矩阵)
    t_target2cam = []   # 标定板 -> 相机 (平移向量)

    print(f"正在处理数据，假设机械臂欧拉角顺序为: {ROBOT_ROTATION_ORDER}, 单位: 度")

    for rec in records:
        # 1. 视觉数据 (Target -> Camera)
        rvec_cam = np.array(rec['cam_rvec'])
        tvec_cam = np.array(rec['cam_tvec'])
        # 将旋转向量转换为矩阵
        R_cam_mat, _ = cv2.Rodrigues(rvec_cam)
        
        R_target2cam.append(R_cam_mat)
        t_target2cam.append(tvec_cam)

        # 2. 机械臂数据 (Gripper -> Base)
        p = rec['robot_pose'] # [x, y, z, rx, ry, rz]
        
        # 平移部分 (假设单位是 mm)
        t_robot = np.array([p[0], p[1], p[2]]).reshape(3, 1)
        
        # 旋转部分: 欧拉角 (度) -> 旋转矩阵
        # 使用 scipy 进行转换
        try:
            r = R.from_euler(ROBOT_ROTATION_ORDER, [p[3], p[4], p[5]], degrees=True)
            R_robot_mat = r.as_matrix()
        except Exception as e:
            print(f"欧拉角转换失败: {e}")
            return

        R_gripper2base.append(R_robot_mat)
        t_gripper2base.append(t_robot)

    # 3. OpenCV 手眼标定 (Tsai 方法)
    try:
        # cv2.calibrateHandEye 输入:
        # R_gripper2base, t_gripper2base: 从末端到基座的变换
        # R_target2cam, t_target2cam: 从标定板到相机的变换
        # 输出:
        # R_cam2gripper, t_cam2gripper: 从相机到末端的变换
        
        R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(
            R_gripper2base, t_gripper2base,
            R_target2cam, t_target2cam,
            method=cv2.CALIB_HAND_EYE_TSAI
        )

        print("\n========== 标定成功! ==========")
        print("结果矩阵: 相机坐标系 -> 机械臂法兰坐标系")
        print("---------------------------------------")
        print("旋转矩阵 (R):\n", R_cam2gripper)
        print("平移向量 (T) [mm]:\n", t_cam2gripper.flatten())
        
        # 构建 4x4 齐次矩阵
        H_cam2gripper = np.eye(4)
        H_cam2gripper[:3, :3] = R_cam2gripper
        H_cam2gripper[:3, 3] = t_cam2gripper.flatten()
        
        print("\n齐次变换矩阵 (Homogeneous Matrix):\n", H_cam2gripper)
        print("---------------------------------------")
        print("验证提示：")
        print(f"1. T 的模长为: {np.linalg.norm(t_cam2gripper):.2f} mm")
        print("   (理论上应接近你测量的 50mm)")
        print("2. 如果结果非常离谱，请检查 ROBOT_ROTATION_ORDER 是否为 'xyz'.")

        np.savez("hand_eye_result.npz", R=R_cam2gripper, T=t_cam2gripper, H=H_cam2gripper)
        print("结果已保存至 hand_eye_result.npz")

    except Exception as e:
        print(f"标定计算出错: {e}")

if __name__ == "__main__":
    main()