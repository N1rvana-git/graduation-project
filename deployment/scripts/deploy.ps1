# 口罩检测系统 - Docker部署脚本
# 自动化构建和部署整个应用栈

param(
    [switch]$Build = $false,
    [switch]$Start = $false,
    [switch]$Stop = $false,
    [switch]$Restart = $false,
    [switch]$Logs = $false,
    [switch]$Clean = $false
)

$ErrorActionPreference = "Stop"

# 颜色输出函数
function Write-ColorOutput($ForegroundColor) {
    $fc = $host.UI.RawUI.ForegroundColor
    $host.UI.RawUI.ForegroundColor = $ForegroundColor
    if ($args) {
        Write-Output $args
    } else {
        $input | Write-Output
    }
    $host.UI.RawUI.ForegroundColor = $fc
}

function Write-Info($message) {
    Write-ColorOutput Cyan "ℹ️ $message"
}

function Write-Success($message) {
    Write-ColorOutput Green "✅ $message"
}

function Write-Warning($message) {
    Write-ColorOutput Yellow "⚠️ $message"
}

function Write-Error($message) {
    Write-ColorOutput Red "❌ $message"
}

# 检查Docker是否可用
function Test-DockerAvailable {
    try {
        docker version | Out-Null
        return $true
    }
    catch {
        return $false
    }
}

# 构建Docker镜像
function Build-DockerImages {
    Write-Info "开始构建Docker镜像..."
    
    # 构建主应用镜像
    Write-Info "构建口罩检测应用镜像..."
    docker build -t mask-detection-app:latest .
    
    if ($LASTEXITCODE -eq 0) {
        Write-Success "应用镜像构建成功"
    } else {
        Write-Error "应用镜像构建失败"
        exit 1
    }
}

# 启动服务
function Start-Services {
    Write-Info "启动Docker Compose服务..."
    docker-compose up -d
    
    if ($LASTEXITCODE -eq 0) {
        Write-Success "服务启动成功"
        Write-Info "等待服务就绪..."
        Start-Sleep -Seconds 10
        
        # 检查服务状态
        Write-Info "检查服务状态..."
        docker-compose ps
        
        Write-Info "服务访问地址:"
        Write-ColorOutput Green "🌐 前端页面: http://localhost"
        Write-ColorOutput Green "🔧 API接口: http://localhost/api"
        Write-ColorOutput Green "❤️ 健康检查: http://localhost/api/health"
    } else {
        Write-Error "服务启动失败"
        exit 1
    }
}

# 停止服务
function Stop-Services {
    Write-Info "停止Docker Compose服务..."
    docker-compose down
    
    if ($LASTEXITCODE -eq 0) {
        Write-Success "服务已停止"
    } else {
        Write-Error "服务停止失败"
    }
}

# 重启服务
function Restart-Services {
    Write-Info "重启服务..."
    Stop-Services
    Start-Sleep -Seconds 5
    Start-Services
}

# 查看日志
function Show-Logs {
    Write-Info "显示服务日志..."
    docker-compose logs -f
}

# 清理资源
function Clean-Resources {
    Write-Warning "这将删除所有容器、镜像和卷，确定要继续吗? (y/N)"
    $confirmation = Read-Host
    
    if ($confirmation -eq 'y' -or $confirmation -eq 'Y') {
        Write-Info "停止并删除容器..."
        docker-compose down -v --remove-orphans
        
        Write-Info "删除镜像..."
        docker rmi mask-detection-app:latest -f 2>$null
        
        Write-Info "清理未使用的资源..."
        docker system prune -f
        
        Write-Success "清理完成"
    } else {
        Write-Info "取消清理操作"
    }
}

# 主逻辑
Write-Info "口罩检测系统 Docker 部署工具"
Write-Info "================================"

# 检查Docker可用性
if (-not (Test-DockerAvailable)) {
    Write-Error "Docker不可用，请先启动Docker Desktop"
    Write-Info "运行以下命令启动Docker Desktop:"
    Write-ColorOutput Yellow ".\deployment\scripts\start-docker.ps1"
    exit 1
}

# 切换到项目根目录
$projectRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location $projectRoot

# 根据参数执行相应操作
if ($Build) {
    Build-DockerImages
}
elseif ($Start) {
    Start-Services
}
elseif ($Stop) {
    Stop-Services
}
elseif ($Restart) {
    Restart-Services
}
elseif ($Logs) {
    Show-Logs
}
elseif ($Clean) {
    Clean-Resources
}
else {
    Write-Info "使用方法:"
    Write-ColorOutput White "  .\deployment\scripts\deploy.ps1 -Build    # 构建镜像"
    Write-ColorOutput White "  .\deployment\scripts\deploy.ps1 -Start    # 启动服务"
    Write-ColorOutput White "  .\deployment\scripts\deploy.ps1 -Stop     # 停止服务"
    Write-ColorOutput White "  .\deployment\scripts\deploy.ps1 -Restart  # 重启服务"
    Write-ColorOutput White "  .\deployment\scripts\deploy.ps1 -Logs     # 查看日志"
    Write-ColorOutput White "  .\deployment\scripts\deploy.ps1 -Clean    # 清理资源"
    Write-Info ""
    Write-Info "快速开始:"
    Write-ColorOutput Yellow "  1. .\deployment\scripts\deploy.ps1 -Build"
    Write-ColorOutput Yellow "  2. .\deployment\scripts\deploy.ps1 -Start"
}