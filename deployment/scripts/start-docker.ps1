# Docker Desktop启动脚本
# 用于启动Docker Desktop并等待服务就绪

Write-Host "正在启动Docker Desktop..." -ForegroundColor Green

# 检查Docker Desktop是否已安装
$dockerDesktopPath = "C:\Program Files\Docker\Docker\Docker Desktop.exe"
if (-not (Test-Path $dockerDesktopPath)) {
    Write-Host "错误: 未找到Docker Desktop，请先安装Docker Desktop" -ForegroundColor Red
    Write-Host "下载地址: https://www.docker.com/products/docker-desktop" -ForegroundColor Yellow
    exit 1
}

# 启动Docker Desktop
Write-Host "启动Docker Desktop..." -ForegroundColor Yellow
Start-Process -FilePath $dockerDesktopPath -WindowStyle Minimized

# 等待Docker服务启动
Write-Host "等待Docker服务启动..." -ForegroundColor Yellow
$maxWaitTime = 120  # 最大等待时间（秒）
$waitTime = 0

do {
    Start-Sleep -Seconds 5
    $waitTime += 5
    
    try {
        $dockerVersion = docker version --format "{{.Server.Version}}" 2>$null
        if ($dockerVersion) {
            Write-Host "Docker服务已启动，版本: $dockerVersion" -ForegroundColor Green
            break
        }
    }
    catch {
        # 继续等待
    }
    
    Write-Host "等待中... ($waitTime/$maxWaitTime 秒)" -ForegroundColor Yellow
    
    if ($waitTime -ge $maxWaitTime) {
        Write-Host "错误: Docker服务启动超时" -ForegroundColor Red
        exit 1
    }
} while ($true)

Write-Host "Docker Desktop已成功启动并准备就绪!" -ForegroundColor Green
Write-Host "现在可以运行 docker-compose up 来启动应用容器" -ForegroundColor Cyan