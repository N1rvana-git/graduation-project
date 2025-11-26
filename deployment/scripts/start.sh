#!/bin/bash

# 口罩检测系统启动脚本
# 用于在云服务器上启动FastAPI应用

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 日志函数
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_debug() {
    echo -e "${BLUE}[DEBUG]${NC} $1"
}

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

log_info "项目根目录: $PROJECT_ROOT"

# 切换到项目根目录
cd "$PROJECT_ROOT"

# 检查Python环境
check_python() {
    log_info "检查Python环境..."
    
    if ! command -v python3 &> /dev/null; then
        log_error "Python3 未安装"
        exit 1
    fi
    
    PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
    log_info "Python版本: $PYTHON_VERSION"
    
    # 检查是否在虚拟环境中
    if [[ "$VIRTUAL_ENV" != "" ]]; then
        log_info "当前在虚拟环境中: $VIRTUAL_ENV"
    else
        log_warn "未检测到虚拟环境，建议使用虚拟环境"
    fi
}

# 安装依赖
install_dependencies() {
    log_info "安装Python依赖..."
    
    if [ -f "requirements.txt" ]; then
        pip3 install -r requirements.txt
        log_info "依赖安装完成"
    else
        log_error "requirements.txt 文件不存在"
        exit 1
    fi
}

# 检查模型文件
check_models() {
    log_info "检查模型文件..."
    
    if [ ! -d "models/yolov5" ]; then
        log_warn "YOLOv5模型目录不存在，正在下载..."
        python3 models/download_yolov5.py
    fi
    
    # 检查预训练模型
    if [ ! -f "models/weights/yolov5s.pt" ]; then
        log_warn "预训练模型不存在，将使用默认模型"
    fi
    
    log_info "模型检查完成"
}

# 创建必要的目录
create_directories() {
    log_info "创建必要的目录..."
    
    mkdir -p logs
    mkdir -p models/weights
    mkdir -p data/images
    mkdir -p data/labels
    
    log_info "目录创建完成"
}

# 设置环境变量
setup_environment() {
    log_info "设置环境变量..."
    
    export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
    export UVICORN_APP="backend.app:app"
    export APP_ENV="${APP_ENV:-production}"
    
    log_info "环境变量设置完成"
}

# 运行测试
run_tests() {
    if [ "$1" = "--skip-tests" ]; then
        log_info "跳过测试"
        return
    fi
    
    log_info "运行API测试..."
    
    # 启动FastAPI应用（后台）
    uvicorn backend.app:app --host 0.0.0.0 --port 5001 >/dev/null 2>&1 &
    FASTAPI_PID=$!
    
    # 等待服务启动
    sleep 10
    
    # 运行测试
    if python3 tests/test_api.py --test health; then
        log_info "基础测试通过"
    else
        log_error "基础测试失败"
        kill $FASTAPI_PID 2>/dev/null || true
        exit 1
    fi
    
    # 停止测试用的FastAPI进程
    kill $FASTAPI_PID 2>/dev/null || true
    sleep 2
}

# 启动服务
start_service() {
    log_info "启动口罩检测服务..."
    
    # 设置启动参数
    HOST="${HOST:-0.0.0.0}"
    PORT="${PORT:-5000}"
    WORKERS="${WORKERS:-4}"
    
    log_info "服务配置:"
    log_info "  主机: $HOST"
    log_info "  端口: $PORT"
    log_info "  工作进程: $WORKERS"
    
    # 检查端口是否被占用
    if lsof -Pi :$PORT -sTCP:LISTEN -t >/dev/null 2>&1; then
        log_error "端口 $PORT 已被占用"
        exit 1
    fi
    
    # 启动方式选择
    if command -v gunicorn &> /dev/null; then
        log_info "使用Gunicorn(UvicornWorker)启动服务..."
        gunicorn \
            --bind $HOST:$PORT \
            --workers $WORKERS \
            --worker-class uvicorn.workers.UvicornWorker \
            --timeout 300 \
            --keep-alive 2 \
            --max-requests 1000 \
            --max-requests-jitter 100 \
            --access-logfile logs/access.log \
            --error-logfile logs/error.log \
            --log-level info \
            backend.app:app
    else
        log_warn "Gunicorn未安装，使用Uvicorn开发服务器..."
        uvicorn backend.app:app --host $HOST --port $PORT
    fi
}

# 停止服务
stop_service() {
    log_info "停止口罩检测服务..."
    
    # 查找并停止相关进程
    pkill -f "python.*backend/app.py" || true
    pkill -f "uvicorn.*backend.app:app" || true
    pkill -f "gunicorn.*backend.app:app" || true
    
    log_info "服务已停止"
}

# 重启服务
restart_service() {
    log_info "重启口罩检测服务..."
    stop_service
    sleep 3
    start_service
}

# 查看服务状态
status_service() {
    log_info "检查服务状态..."
    
     if pgrep -f "python.*backend/app.py" > /dev/null || \
         pgrep -f "uvicorn.*backend.app:app" > /dev/null || \
         pgrep -f "gunicorn.*backend.app:app" > /dev/null; then
        log_info "✅ 服务正在运行"
        
        # 检查健康状态
        if curl -f http://localhost:5000/api/health >/dev/null 2>&1; then
            log_info "✅ 服务健康检查通过"
        else
            log_warn "⚠️  服务健康检查失败"
        fi
    else
        log_warn "❌ 服务未运行"
    fi
}

# 查看日志
view_logs() {
    log_info "查看服务日志..."
    
    if [ -f "logs/error.log" ]; then
        echo "=== 错误日志 ==="
        tail -n 50 logs/error.log
    fi
    
    if [ -f "logs/access.log" ]; then
        echo "=== 访问日志 ==="
        tail -n 50 logs/access.log
    fi
}

# 显示帮助信息
show_help() {
    echo "口罩检测系统启动脚本"
    echo ""
    echo "用法: $0 [命令] [选项]"
    echo ""
    echo "命令:"
    echo "  start         启动服务"
    echo "  stop          停止服务"
    echo "  restart       重启服务"
    echo "  status        查看服务状态"
    echo "  logs          查看服务日志"
    echo "  install       安装依赖"
    echo "  test          运行测试"
    echo "  help          显示帮助信息"
    echo ""
    echo "选项:"
    echo "  --skip-tests  跳过测试"
    echo "  --host HOST   指定主机地址 (默认: 0.0.0.0)"
    echo "  --port PORT   指定端口 (默认: 5000)"
    echo "  --workers N   指定工作进程数 (默认: 4)"
    echo ""
    echo "环境变量:"
    echo "  HOST          服务主机地址"
    echo "  PORT          服务端口"
    echo "  WORKERS       工作进程数"
    echo "  APP_ENV       FastAPI环境 (development/production)"
    echo ""
    echo "示例:"
    echo "  $0 start                    # 启动服务"
    echo "  $0 start --skip-tests       # 启动服务并跳过测试"
    echo "  HOST=127.0.0.1 $0 start     # 指定主机启动"
    echo "  $0 restart                  # 重启服务"
}

# 主函数
main() {
    case "${1:-start}" in
        "start")
            check_python
            create_directories
            setup_environment
            check_models
            run_tests "$@"
            start_service
            ;;
        "stop")
            stop_service
            ;;
        "restart")
            restart_service
            ;;
        "status")
            status_service
            ;;
        "logs")
            view_logs
            ;;
        "install")
            check_python
            install_dependencies
            ;;
        "test")
            check_python
            setup_environment
            python3 tests/test_api.py
            ;;
        "help"|"-h"|"--help")
            show_help
            ;;
        *)
            log_error "未知命令: $1"
            show_help
            exit 1
            ;;
    esac
}

# 执行主函数
main "$@"