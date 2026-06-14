### Docker Compose 完整部署

##### 1. 克隆仓库
git clone https://github.com/infiniflow/ragflow.git
cd ragflow/docker

##### 2. 配置环境（可选）
编辑 .env 文件，修改端口、密码等

##### 3. 启动服务
docker compose -f docker-compose.yml up -d

##### 4. 查看日志
docker logs -f ragflow-server