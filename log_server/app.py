#!/usr/bin/env python3
"""
集中式日志收集服务器
接收来自服务器、Jetson、NUC的日志，提供Web前端查看
"""

import os
import sqlite3
import json
import threading
from datetime import datetime, timedelta
from pathlib import Path
from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
from flask_socketio import SocketIO, emit
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__, 
            template_folder='templates',
            static_folder='static')
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

# 配置
BASE_DIR = Path(__file__).parent
DB_PATH = BASE_DIR / 'logs.db'
LOG_RETENTION_DAYS = 30  # 保留30天日志

# 数据库初始化
def init_db():
    """初始化数据库"""
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    cursor = conn.cursor()
    
    # 创建日志表
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME NOT NULL,
            level VARCHAR(10) NOT NULL,
            module VARCHAR(50) NOT NULL,
            request_id VARCHAR(20),
            message TEXT NOT NULL,
            file_path VARCHAR(200),
            line_number INTEGER,
            device_name VARCHAR(50) NOT NULL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # 创建索引
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_timestamp ON logs(timestamp)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_level ON logs(level)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_module ON logs(module)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_request_id ON logs(request_id)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_device ON logs(device_name)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_device_module ON logs(device_name, module)')
    
    conn.commit()
    conn.close()
    logger.info(f"数据库初始化完成: {DB_PATH}")

# 数据库连接池（线程安全）
db_lock = threading.Lock()

def get_db_connection():
    """获取数据库连接"""
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

# API路由

@app.route('/')
def index():
    """主页"""
    return render_template('index.html')

@app.route('/api/logs', methods=['POST'])
def receive_log():
    """接收日志（简化版，快速响应）"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "未提供JSON数据"}), 400
        
        # 验证必需字段
        required_fields = ['timestamp', 'level', 'module', 'message', 'device']
        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"缺少必需字段: {field}"}), 400
        
        # 插入数据库（使用锁保证线程安全）
        log_id = None
        try:
            with db_lock:
                conn = get_db_connection()
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO logs (timestamp, level, module, request_id, message, 
                                    file_path, line_number, device_name)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    data.get('timestamp'),
                    data.get('level'),
                    data.get('module'),
                    data.get('request_id'),
                    data.get('message'),
                    data.get('file'),
                    data.get('line'),
                    data.get('device')
                ))
                conn.commit()
                log_id = cursor.lastrowid
                conn.close()
        except Exception as db_error:
            # 数据库错误不影响响应，记录日志即可
            logger.warning(f"数据库插入失败: {db_error}")
            # 即使数据库失败，也返回成功，避免客户端重试
            log_id = 0
        
        # 通过WebSocket实时推送（非阻塞，失败不影响）
        try:
            socketio.emit('new_log', {
                'id': log_id or 0,
                'timestamp': data.get('timestamp'),
                'level': data.get('level'),
                'module': data.get('module'),
                'request_id': data.get('request_id'),
                'message': data.get('message'),
                'device': data.get('device'),
                'file': data.get('file'),
                'line': data.get('line')
            })
        except Exception:
            # WebSocket推送失败不影响响应
            pass
        
        # 快速返回成功（即使数据库失败也返回200，避免客户端重试）
        return jsonify({"success": True, "id": log_id or 0}), 200
        
    except Exception as e:
        # 严重错误才返回500
        logger.error(f"接收日志失败: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@app.route('/api/logs/query', methods=['GET'])
def query_logs():
    """查询日志"""
    try:
        # 获取查询参数
        device = request.args.get('device', '')
        module = request.args.get('module', '')
        level = request.args.get('level', '')
        request_id = request.args.get('request_id', '')
        keyword = request.args.get('keyword', '')
        start_time = request.args.get('start_time', '')
        end_time = request.args.get('end_time', '')
        page = int(request.args.get('page', 1))
        page_size = int(request.args.get('page_size', 100))
        
        # 构建查询
        conn = get_db_connection()
        cursor = conn.cursor()
        
        conditions = []
        params = []
        
        if device:
            conditions.append("device_name = ?")
            params.append(device)
        
        if module:
            conditions.append("module = ?")
            params.append(module)
        
        if level:
            conditions.append("level = ?")
            params.append(level)
        
        if request_id:
            conditions.append("request_id = ?")
            params.append(request_id)
        
        if keyword:
            conditions.append("message LIKE ?")
            params.append(f"%{keyword}%")
        
        if start_time:
            conditions.append("timestamp >= ?")
            params.append(start_time)
        
        if end_time:
            conditions.append("timestamp <= ?")
            params.append(end_time)
        
        where_clause = " AND ".join(conditions) if conditions else "1=1"
        
        # 查询总数
        count_query = f"SELECT COUNT(*) as total FROM logs WHERE {where_clause}"
        cursor.execute(count_query, params)
        total = cursor.fetchone()['total']
        
        # 查询数据
        offset = (page - 1) * page_size
        query = f'''
            SELECT * FROM logs 
            WHERE {where_clause}
            ORDER BY timestamp DESC
            LIMIT ? OFFSET ?
        '''
        params.extend([page_size, offset])
        cursor.execute(query, params)
        rows = cursor.fetchall()
        
        conn.close()
        
        # 转换为字典列表
        logs = [dict(row) for row in rows]
        
        return jsonify({
            "success": True,
            "data": logs,
            "total": total,
            "page": page,
            "page_size": page_size,
            "total_pages": (total + page_size - 1) // page_size
        }), 200
        
    except Exception as e:
        logger.error(f"查询日志失败: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@app.route('/api/logs/stats', methods=['GET'])
def get_stats():
    """获取统计信息"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # 按级别统计
        cursor.execute('''
            SELECT level, COUNT(*) as count 
            FROM logs 
            WHERE timestamp >= datetime('now', '-1 day')
            GROUP BY level
        ''')
        level_stats = {row['level']: row['count'] for row in cursor.fetchall()}
        
        # 按模块统计
        cursor.execute('''
            SELECT module, COUNT(*) as count 
            FROM logs 
            WHERE timestamp >= datetime('now', '-1 day')
            GROUP BY module
            ORDER BY count DESC
            LIMIT 20
        ''')
        module_stats = {row['module']: row['count'] for row in cursor.fetchall()}
        
        # 按设备统计
        cursor.execute('''
            SELECT device_name, COUNT(*) as count 
            FROM logs 
            WHERE timestamp >= datetime('now', '-1 day')
            GROUP BY device_name
        ''')
        device_stats = {row['device_name']: row['count'] for row in cursor.fetchall()}
        
        # 总日志数
        cursor.execute('SELECT COUNT(*) as total FROM logs')
        total_logs = cursor.fetchone()['total']
        
        # 今日日志数
        cursor.execute('''
            SELECT COUNT(*) as count 
            FROM logs 
            WHERE DATE(timestamp) = DATE('now')
        ''')
        today_logs = cursor.fetchone()['count']
        
        conn.close()
        
        return jsonify({
            "success": True,
            "stats": {
                "level": level_stats,
                "module": module_stats,
                "device": device_stats,
                "total": total_logs,
                "today": today_logs
            }
        }), 200
        
    except Exception as e:
        logger.error(f"获取统计信息失败: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@app.route('/api/modules', methods=['GET'])
def get_modules():
    """获取所有模块列表（按设备分类）"""
    try:
        device = request.args.get('device', '')
        conn = get_db_connection()
        cursor = conn.cursor()
        
        if device:
            cursor.execute('''
                SELECT DISTINCT module, device_name 
                FROM logs 
                WHERE device_name = ?
                ORDER BY device_name, module
            ''', (device,))
        else:
            cursor.execute('''
                SELECT DISTINCT module, device_name 
                FROM logs 
                ORDER BY device_name, module
            ''')
        
        rows = cursor.fetchall()
        conn.close()
        
        # 按设备分组
        modules_by_device = {}
        for row in rows:
            device_name = row['device_name']
            module = row['module']
            if device_name not in modules_by_device:
                modules_by_device[device_name] = []
            modules_by_device[device_name].append(module)
        
        return jsonify({
            "success": True,
            "modules": modules_by_device
        }), 200
        
    except Exception as e:
        logger.error(f"获取模块列表失败: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@app.route('/api/cleanup', methods=['POST'])
def cleanup_old_logs():
    """清理旧日志"""
    try:
        days = int(request.args.get('days', LOG_RETENTION_DAYS))
        cutoff_date = (datetime.now() - timedelta(days=days)).isoformat()
        
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute('DELETE FROM logs WHERE timestamp < ?', (cutoff_date,))
        deleted_count = cursor.rowcount
        conn.commit()
        conn.close()
        
        logger.info(f"清理了 {deleted_count} 条旧日志（{days}天前）")
        
        return jsonify({
            "success": True,
            "deleted_count": deleted_count
        }), 200
        
    except Exception as e:
        logger.error(f"清理日志失败: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

# WebSocket事件
@socketio.on('connect')
def handle_connect():
    """客户端连接"""
    logger.info("WebSocket客户端已连接")
    emit('connected', {'message': '已连接到日志服务器'})

@socketio.on('disconnect')
def handle_disconnect():
    """客户端断开"""
    logger.info("WebSocket客户端已断开")

# 启动时初始化
if __name__ == '__main__':
    # 确保目录存在
    BASE_DIR.mkdir(exist_ok=True)
    (BASE_DIR / 'templates').mkdir(exist_ok=True)
    (BASE_DIR / 'static').mkdir(exist_ok=True)
    
    # 初始化数据库
    init_db()
    
    # 启动服务
    port = int(os.getenv('LOG_SERVER_PORT', 8888))
    host = os.getenv('LOG_SERVER_HOST', '0.0.0.0')
    
    logger.info(f"日志服务器启动: http://{host}:{port}")
    socketio.run(app, host=host, port=port, debug=False)

