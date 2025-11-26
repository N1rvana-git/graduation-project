# SSL 证书配置

本目录用于存放 SSL 证书文件，以启用 HTTPS 支持。

## 📁 文件结构

```
ssl/
├── cert.pem          # SSL 证书文件
├── privkey.pem       # 私钥文件
├── chain.pem         # 证书链文件 (可选)
└── README.md         # 本说明文件
```

## 🔒 获取 SSL 证书

### 方法一: 使用 Let's Encrypt (免费)

1. **安装 Certbot**:
   ```bash
   # Ubuntu/Debian
   sudo apt-get install certbot
   
   # CentOS/RHEL
   sudo yum install certbot
   ```

2. **获取证书**:
   ```bash
   sudo certbot certonly --standalone -d yourdomain.com
   ```

3. **复制证书文件**:
   ```bash
   sudo cp /etc/letsencrypt/live/yourdomain.com/fullchain.pem ./cert.pem
   sudo cp /etc/letsencrypt/live/yourdomain.com/privkey.pem ./privkey.pem
   ```

### 方法二: 自签名证书 (开发环境)

1. **生成私钥**:
   ```bash
   openssl genrsa -out privkey.pem 2048
   ```

2. **生成证书**:
   ```bash
   openssl req -new -x509 -key privkey.pem -out cert.pem -days 365
   ```

### 方法三: 商业证书

从证书颁发机构 (CA) 购买证书，然后将证书文件放置在此目录中。

## ⚙️ 配置 HTTPS

1. **确保证书文件存在**:
   - `cert.pem` - SSL 证书
   - `privkey.pem` - 私钥

2. **修改 Nginx 配置**:
   编辑 `../default.conf`，添加 HTTPS 服务器块:

   ```nginx
   server {
       listen 443 ssl http2;
       server_name localhost;
       
       ssl_certificate /etc/nginx/ssl/cert.pem;
       ssl_certificate_key /etc/nginx/ssl/privkey.pem;
       
       # SSL 配置
       ssl_protocols TLSv1.2 TLSv1.3;
       ssl_ciphers ECDHE-RSA-AES128-GCM-SHA256:ECDHE-RSA-AES256-GCM-SHA384;
       ssl_prefer_server_ciphers off;
       
       # 其他配置...
   }
   
   # HTTP 重定向到 HTTPS
   server {
       listen 80;
       server_name localhost;
       return 301 https://$server_name$request_uri;
   }
   ```

3. **重启服务**:
   ```bash
   docker-compose restart nginx
   ```

## 🔐 安全建议

1. **文件权限**: 确保私钥文件权限为 600
   ```bash
   chmod 600 privkey.pem
   chmod 644 cert.pem
   ```

2. **定期更新**: 定期更新 SSL 证书，特别是 Let's Encrypt 证书 (90天有效期)

3. **强制 HTTPS**: 配置 HTTP 到 HTTPS 的重定向

4. **HSTS**: 启用 HTTP Strict Transport Security
   ```nginx
   add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;
   ```

## 🧪 测试 SSL 配置

1. **检查证书**:
   ```bash
   openssl x509 -in cert.pem -text -noout
   ```

2. **测试 SSL 连接**:
   ```bash
   openssl s_client -connect localhost:443
   ```

3. **在线 SSL 测试**: 使用 [SSL Labs](https://www.ssllabs.com/ssltest/) 测试

## ⚠️ 注意事项

- 开发环境可以使用自签名证书
- 生产环境建议使用受信任的 CA 证书
- 私钥文件不应提交到版本控制系统
- 定期备份证书文件