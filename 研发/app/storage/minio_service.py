from __future__ import annotations
'''
这是一个 MinIO 对象存储服务封装类，用于上传文件到 MinIO（兼容 AWS S3 的对象存储服务）。
MinIO = 开源的对象存储服务器，兼容 AWS S3 API
类比理解：
MinIO 就像自建的阿里云OSS/七牛云/腾讯云COS
- 可以存文件、图片、视频
- 通过 API 上传/下载
- 自己架设，数据在自己服务器上
'''

# 导入异步IO库，提供 asyncio.Lock() 用于创建异步锁
import asyncio
# 导入MIME类型猜测库，用于根据文件扩展名判断文件类型
# mimetypes.guess_type("photo.jpg")  # 返回 ("image/jpeg", None)
# mimetypes.guess_type("document.pdf")  # 返回 ("application/pdf", None)
# 第二个返回值是编码方式
'''
content_type="image/jpeg"      # JPEG图片
content_type="application/pdf"  # PDF文档
content_type="video/mp4"        # MP4视频
content_type="text/plain"       # 文本文件
'''
import mimetypes
# 作用：导入面向对象的路径处理库
from pathlib import Path
# 作用：导入异步版本的boto3（AWS SDK），用于连接MinIO/S3
# boto3 = 同步版本
# aioboto3 = 异步版本，支持 async/await
import aioboto3

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)

# 作用：定义MinIO存储服务类
class MinioStorageService:
    # 作用：类级别的异步锁，防止多个协程同时创建同一个bucket
    # 为什么是类变量？所有实例共享同一把锁，确保跨实例的并发安全
    _bucket_lock = asyncio.Lock()
    # 作用：类级别的集合，记录已经初始化过的bucket名称
    _initialized_buckets: set[str] = set()

    def __init__(self) -> None:
        self.settings = get_settings()
        # 作用：创建aioboto3会话，用于管理S3连接
        # Session 可以重复使用，避免反复创建连接
        self._session = aioboto3.Session()

    # 定义异步上传文件方法
    async def upload_file(
        self,
        *,
        bucket: str,
        object_name: str,
        local_path: str,
        content_type: str | None = None,
        metadata: dict[str, str] | None = None,
    ) -> str:
        '''
        作用：定义参数和返回值类型
        参数	           含义	              示例
        bucket	      存储桶名称	       "avatars"
        object_name	  对象在桶中的路径	   "user_123/photo.jpg"
        local_path	  本地文件路径	   "/tmp/upload.jpg"
        content_type  MIME类型（可选）   "image/jpeg"
        metadata	 自定义元数据（可选）  {"user": "123"}  存储文件的自定义属性（不显示在文件名中）
        返回值	     文件URI	            "minio://avatars/user_123/photo.jpg"
        '''
        # 确保bucket存在（不存在则创建）  为什么需要：MinIO/S3 不能自动创建不存在的桶
        await self.ensure_bucket(bucket)
        # 作用：根据文件扩展名猜测MIME类型
        guessed_type, _ = mimetypes.guess_type(local_path)
        # 构造上传参数
        # 优先级：用户指定 > 系统猜测 > 默认二进制流
        # "application/octet-stream" 是通用二进制类型
        extra_args = {
            "ContentType": content_type or guessed_type or "application/octet-stream",
            "Metadata": metadata or {},
        }
        # 这是一个务实的混合设计 - 文件读取用同步（因为快），网络传输用异步（因为慢），两者结合达到最佳性能和代码简洁性。
        # 作用：获取S3客户端（自动管理生命周期）  外层：异步上下文管理器 - 管理S3连接
        async with self._client() as client:
            # 作用：同步方式打开本地文件（二进制读取模式）  # 中层：同步上下文管理器 - 打开本地文件
            '''
            文件 I/O 通常不需要异步：
                本地文件读取非常快（毫秒级）
                操作系统有缓存机制
                异步带来的开销可能弊大于利
            '''
            with open(local_path, "rb") as handle:
                # 作用：异步上传文件对象到MinIO  # 内层：异步操作 - 实际上传
                '''
                upload_fileobj 内部会：
                    - 循环读取 handle 的数据
                    - 分块上传到 MinIO
                    - 每次读取和上传交替进行
                '''
                await client.upload_fileobj(handle, bucket, object_name, ExtraArgs=extra_args)
        # 作用：生成文件访问URI
        # 返回类似 "minio://avatars/user_123/photo.jpg"
        uri = self.build_uri(bucket=bucket, object_name=object_name)
        logger.info("minio_file_uploaded", bucket=bucket, object_name=object_name)
        # 作用：返回URI给调用方
        return uri

    # 上传字节数据（不依赖本地文件）
    # 使用场景：内存中生成的图片、API返回的数据等
    # upload_bytes 是为小数据（几MB内）设计的便捷方法，避免频繁的磁盘 I/O，特别适合处理内存中已有的数据。
    async def upload_bytes(
        self,
        *,
        bucket: str,
        object_name: str,
        data: bytes,
        content_type: str = "application/octet-stream",
        metadata: dict[str, str] | None = None,
    ) -> str:
        '''
        # bytes 的常见来源
        text_bytes = "Hello".encode('utf-8')     # 字符串转 bytes
        json_bytes = json.dumps(data).encode()   # JSON 转 bytes
        image_bytes = image.tobytes()             # 图片对象转 bytes
        file_bytes = open("file.pdf", "rb").read()  # 读取文件（不推荐这样用）
        '''
        await self.ensure_bucket(bucket)
        async with self._client() as client:
            # 直接put字节数据（比upload_fileobj更轻量）
            '''
            轻量的原因：
            不需要文件对象包装
            不需要分块读取（适合小数据）
            代码更直接
            '''
            # 与 upload_file 不同，这里不会自动猜测类型，所以给了默认值。
            await client.put_object(
                Bucket=bucket,
                Key=object_name,
                Body=data,
                ContentType=content_type,
                Metadata=metadata or {},
            )
        uri = self.build_uri(bucket=bucket, object_name=object_name)
        logger.info("minio_bytes_uploaded", bucket=bucket, object_name=object_name, size=len(data))
        return uri

    # 确保bucket存在
    async def ensure_bucket(self, bucket: str) -> None:
        # 第一次检查（无锁，快速返回）
        if bucket in self._initialized_buckets:
            return

        # 获取锁，防止并发创建  串行化创建操作，防止并发
        async with self._bucket_lock:
            # 第二次检查（有锁，防止重复）
            if bucket in self._initialized_buckets:
                return
            async with self._client() as client:
                try:
                    # 尝试获取 bucket 信息 < 检查操作 >   如果不存在，head_bucket 会抛异常
                    await client.head_bucket(Bucket=bucket)
                except Exception:
                    # bucket 不存在，创建它
                    await client.create_bucket(Bucket=bucket)
                    logger.info("minio_bucket_created", bucket=bucket)
            self._initialized_buckets.add(bucket)
            '''
            # head_bucket: 只返回元数据，不修改任何东西
            # 成功：bucket 存在
            # 失败：bucket 不存在（或没权限）
            
            # create_bucket: 创建 bucket，是写操作
            # 如果 bucket 已存在，会抛异常 BucketAlreadyExists
            '''

    def build_raw_object_name(self, *, user_id: str, role_id: str, task_id: str, file_name: str) -> str:
        # 作用：从路径中提取纯文件名，防止路径遍历攻击。
        '''
        # 示例：恶意输入
        file_name = "../../../etc/passwd"
        Path(file_name).name  # 返回 "passwd"（去掉了路径部分）

        # 正常输入
        file_name = "report.pdf"
        Path(file_name).name  # 返回 "report.pdf"

        file_name = "C:/Users/me/photo.jpg"
        Path(file_name).name  # 返回 "photo.jpg"
        '''
        safe_name = Path(file_name).name
        return f"{user_id}/{role_id}/raw/{task_id}/{safe_name}"

    def build_parsed_object_name(self, *, user_id: str, role_id: str, task_id: str) -> str:
        return f"{user_id}/{role_id}/parsed/{task_id}/manifest.json"

    @staticmethod
    def build_uri(*, bucket: str, object_name: str) -> str:
        return f"minio://{bucket}/{object_name}"

    # 创建S3兼容客户端（私有方法，下划线开头）
    # 配置决定了目标服务器是MinIO
    def _client(self):
        return self._session.client(
            "s3",   # 这个参数告诉 aioboto3.Session 要创建哪种服务的客户端
                    # 为什么用 "s3"？
                    # MinIO 兼容的是 S3 的 API 规范
                    # 所以即使连接 MinIO，也要用 "s3" 这个服务名称
                    # 告诉 aioboto3："请按照 S3 的协议格式来通信"
            endpoint_url=self.settings.minio_endpoint_url,
            aws_access_key_id=self.settings.minio_access_key,
            aws_secret_access_key=self.settings.minio_secret_key,
            use_ssl=self.settings.minio_secure,     # SSL 是什么？
                                                    # SSL/TLS：加密协议，保护数据传输安全
                                                    # HTTPS = HTTP + SSL（网站的小锁图标）
                                                    # HTTP = 明文传输，可以被窃听
        )
    '''
    localhost:9000 = "本机的 9000 端口"
    MinIO = "正在这个端口上运行的程序"
    所以文件存储到了MinIO
    
    # aioboto3/boto3 内部支持的服务名称（部分）
    SUPPORTED_SERVICES = {
        "s3": "S3 存储服务",
        "ec2": "EC2 云服务器", 
        "lambda": "Lambda 函数计算",
        "dynamodb": "DynamoDB 数据库",
        "rds": "RDS 关系数据库",
        # ... 200+ AWS 服务
    }
    '''
