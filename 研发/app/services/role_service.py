from __future__ import annotations
'''
这是一个AI角色管理服务，实现了系统预置角色和用户自定义角色的完整 CRUD 操作，并包含智能角色识别和匹配功能。
字母	英文	    中文	    对应 HTTP 方法
C	Create	创建	       POST
R	Read	读取	       GET
U	Update	更新	     PUT/PATCH
D	Delete	删除	      DELETE

核心功能概览
  功能	              说明
角色初始化	  预置4个默认角色（律师、医生、股票分析师、历史人物）
角色列表	      获取预置角色 + 用户自定义角色
角色解析	      通过ID或名称获取角色，支持自动创建
角色检测	      根据用户问题关键词智能匹配角色
角色CRUD	      创建、更新、删除自定义角色
'''
from dataclasses import dataclass
from datetime import datetime

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logging import get_logger

logger = get_logger(__name__)

# 检查字符串中是否包含 CJK（中日韩）统一表意文字。
def _contains_cjk(value: str) -> bool:
    return any("\u4e00" <= char <= "\u9fff" for char in value)
'''
字符范围：\u4e00 到 \u9fff 是 Unicode 中的基本 CJK 统一表意文字区块（CJK Unified Ideographs）
这个范围包含最常见的中文字符（也包含部分日韩汉字）
第一个字符 \u4e00 = "一"
最后一个字符 \u9fff = 目前最新的基本汉字（如"鿿"）
any() 函数：只要字符串中至少有一个字符在这个范围内，就返回 True，否则返回 False
'''

# 修改乱码
def _repair_mojibake(value: str | None) -> str | None:
    if value is None:
        return None
    if _contains_cjk(value):
        return value
    # "cp1252", "latin1"是处理中文乱码时最常见的两种错误编码
    '''
    1. cp1252 (Windows-1252)
    Windows 系统默认的西欧编码
    最常见的乱码来源：Excel 文件、Windows 记事本保存的文件
    典型乱码示例："中文" → "ä¸­æ–‡"
    2. latin1 (ISO-8859-1)
    古老但广泛使用的西欧编码
    很多老系统、数据库默认编码
    某些 Linux 系统的默认编码
    cp1252 在前，latin1 在后的原因：
        cp1252 在现代 Windows 系统中更常见
        实际经验中，cp1252 修复成功的概率更高
        先尝试最可能的方案，提高效率
    
    正常流程（应该发生的）：
    "中文" → UTF-8编码 → [字节: E4 B8 AD E6 96 87] → UTF-8解码 → "中文"
    错误流程（实际发生的）：
    "中文" → UTF-8编码 → [字节: E4 B8 AD E6 96 87] → cp1252解码(错误!) → "ä¸­æ–‡"
                                                                  ↑
                                                        乱码字符串
    修复流程（代码做的）：
    "ä¸­æ–‡" → cp1252编码(反向操作) → [字节: E4 B8 AD E6 96 87] → UTF-8解码 → "中文"
              ↑
        source_encoding = "cp1252"
    '''
    for source_encoding in ("cp1252", "latin1"):
        try:
            repaired = value.encode(source_encoding).decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            continue
        if _contains_cjk(repaired):
            return repaired
    return value

# 预制角色定义
# 1.完整版：用于初始化数据库、关键词匹配等
# weight ： 通过为这些关键词赋予权重，去智能识别角色
DEFAULT_PRESET_ROLE_DEFINITIONS = (
    {
        "role_id": "lawyer_01",
        "name": "Civil Lawyer",
        "category": "lawyer",
        "system_prompt": "Focus on contract, labor, and civil dispute questions. Provide grounded legal information.",
        "knowledge_base_id": "kb_lawyer_default",
        "keywords": (
            {"keyword": "\u52b3\u52a8\u5408\u540c", "weight": 3},
            {"keyword": "\u5f8b\u5e08", "weight": 3},
            {"keyword": "\u516c\u53f8", "weight": 1},
            {"keyword": "\u5408\u540c", "weight": 2},
            {"keyword": "\u7ea0\u7eb7", "weight": 2},
            {"keyword": "\u8d77\u8bc9", "weight": 2},
            {"keyword": "\u6cd5\u5f8b", "weight": 2},
            {"keyword": "contract", "weight": 2},
            {"keyword": "lawsuit", "weight": 2},
            {"keyword": "dispute", "weight": 2},
            {"keyword": "legal", "weight": 2},
        ),
    },
    {
        "role_id": "doctor_01",
        "name": "General Doctor",
        "category": "doctor",
        "system_prompt": "Provide health education and triage guidance without replacing in-person diagnosis.",
        "knowledge_base_id": "kb_doctor_default",
        "keywords": (
            {"keyword": "\u75c7\u72b6", "weight": 2},
            {"keyword": "\u6cbb\u7597", "weight": 2},
            {"keyword": "\u53d1\u70ed", "weight": 3},
            {"keyword": "\u75be\u75c5", "weight": 2},
            {"keyword": "\u533b\u751f", "weight": 3},
            {"keyword": "\u533b\u7597", "weight": 2},
            {"keyword": "\u7528\u836f", "weight": 2},
            {"keyword": "\u533b\u9662", "weight": 1},
            {"keyword": "\u75bc", "weight": 1},
            {"keyword": "symptom", "weight": 2},
            {"keyword": "treatment", "weight": 2},
            {"keyword": "fever", "weight": 3},
            {"keyword": "disease", "weight": 2},
        ),
    },
    {
        "role_id": "stock_01",
        "name": "Stock Analyst",
        "category": "stock",
        "system_prompt": "Provide investment information analysis and risk reminders without guaranteeing returns.",
        "knowledge_base_id": "kb_stock_default",
        "keywords": (
            {"keyword": "\u80a1\u7968", "weight": 3},
            {"keyword": "\u57fa\u91d1", "weight": 2},
            {"keyword": "\u6295\u8d44", "weight": 2},
            {"keyword": "\u7406\u8d22", "weight": 2},
            {"keyword": "\u8bc1\u5238", "weight": 2},
            {"keyword": "\u884c\u60c5", "weight": 2},
            {"keyword": "stock", "weight": 3},
            {"keyword": "fund", "weight": 2},
            {"keyword": "invest", "weight": 2},
            {"keyword": "market", "weight": 2},
        ),
    },
    {
        "role_id": "history_01",
        "name": "Historical Figure Guide",
        "category": "history",
        "system_prompt": "Explain historical figures and events based on grounded background knowledge.",
        "knowledge_base_id": "kb_history_default",
        "keywords": (
            {"keyword": "\u5386\u53f2", "weight": 3},
            {"keyword": "\u738b\u671d", "weight": 2},
            {"keyword": "\u4eba\u7269", "weight": 2},
            {"keyword": "\u79e6\u59cb\u7687", "weight": 3},
            {"keyword": "\u674e\u4e16\u6c11", "weight": 3},
            {"keyword": "\u4f20\u8bb0", "weight": 2},
            {"keyword": "history", "weight": 3},
            {"keyword": "dynasty", "weight": 2},
            {"keyword": "biography", "weight": 2},
            {"keyword": "historical", "weight": 2},
        ),
    },
)
# 2.简化版：用于快速查询、内存缓存等
DEFAULT_PRESET_ROLES = [
    {
        "role_id": item["role_id"],
        "name": item["name"],
        "category": item["category"],
        "system_prompt": item["system_prompt"],
        "knowledge_base_id": item["knowledge_base_id"],
    }
    for item in DEFAULT_PRESET_ROLE_DEFINITIONS
]
# 3.关键词版  用于角色匹配
DEFAULT_PRESET_ROLE_KEYWORDS = [
    {
        "role_id": item["role_id"],
        "keyword": keyword["keyword"],
        "weight": int(keyword["weight"]),
    }
    for item in DEFAULT_PRESET_ROLE_DEFINITIONS
    for keyword in item["keywords"]
]


@dataclass(slots=True)
class RoleRecord:
    role_id: str
    name: str
    category: str
    role_type: str # preset / custom / auto
    system_prompt: str
    knowledge_base_id: str | None
    created_at: datetime | None = None


@dataclass(slots=True)
class KeywordRule:
    role_id: str
    keyword: str
    weight: int = 1


class RoleService:
    async def ensure_preset_roles_seeded(self, db_session: AsyncSession) -> None:
        # 使用 ON DUPLICATE KEY UPDATE 支持更新已有数据（幂等操作：同一个操作执行一次和执行多次，结果相同。）
        '''
        为什么需要幂等？
            初始化脚本可能重复执行
            服务重启可能多次调用
            分布式系统可能有重复请求
        '''
        role_stmt = text(
            """
            INSERT INTO preset_roles (id, name, category, system_prompt, knowledge_base_id, created_at, updated_at)
            VALUES (:role_id, :name, :category, :system_prompt, :knowledge_base_id, NOW(), NOW())
            ON DUPLICATE KEY UPDATE
                name = VALUES(name),
                category = VALUES(category),
                system_prompt = VALUES(system_prompt),
                knowledge_base_id = VALUES(knowledge_base_id),
                updated_at = NOW()
            """
        )
        keyword_stmt = text(
            """
            INSERT INTO preset_role_keywords (role_id, keyword, weight, is_enabled, created_at, updated_at)
            VALUES (:role_id, :keyword, :weight, 1, NOW(), NOW())
            ON DUPLICATE KEY UPDATE
                weight = VALUES(weight),
                is_enabled = VALUES(is_enabled),
                updated_at = NOW()
            """
        )

        try:
            await db_session.execute(role_stmt, DEFAULT_PRESET_ROLES)
            await db_session.commit()
        except Exception as exc:
            # 失败回滚
            await db_session.rollback()
            logger.warning("preset_roles_seed_failed", error=str(exc))
            return

        try:
            await db_session.execute(keyword_stmt, DEFAULT_PRESET_ROLE_KEYWORDS)
            await db_session.commit()
        except Exception as exc:
            await db_session.rollback()
            logger.warning("preset_role_keywords_seed_failed", error=str(exc))
            return

        logger.info(
            "preset_roles_seeded",
            role_count=len(DEFAULT_PRESET_ROLES),
            keyword_count=len(DEFAULT_PRESET_ROLE_KEYWORDS),
        )

    # 列出所有预设角色和用户自定义角色（如果有user_id）
    async def list_roles(self, db_session: AsyncSession, user_id: str | None = None) -> list[RoleRecord]:
        roles: list[RoleRecord] = []
        roles.extend(await self._list_preset_roles(db_session))
        if user_id:
            roles.extend(await self._list_custom_roles(db_session, user_id))
        return roles

    # 优先级策略：
    # ID查找：数据库preset → 数据库custom → 内存preset
    # 名称查找：数据库custom → 内存preset（防止冲突，用户自定义优先）
    async def resolve_role(
        self,
        db_session: AsyncSession,
        *,
        user_id: str,
        role_id: str | None = None,
        role_name: str | None = None,
    ) -> RoleRecord:
        '''
        resolve_role 是"查找或创建"角色，比普通查询多了一个"找不到就创建"的能力。
        resolve_role 是"给我一个能用的角色"：有就用现成的，没有就自动创建一个，保证调用方总能得到一个角色。 主要用于用户对话场景，避免因角色不存在而报错。
        '''
        if role_id:
            # 精确匹配，适合前端已知角色 ID 的场景。
            # 安全性：不会意外创建新角色。
            role = await self._get_role_by_id(db_session, user_id=user_id, role_id=role_id)
            if role is None:
                raise ValueError(f"Role not found: {role_id}")
            return role
        # 用名字查找找不到就创建是因为：提高用户的体验 且 适合动态角色场景
        if role_name:
            role = await self._get_role_by_name(db_session, user_id=user_id, role_name=role_name)
            if role:
                return role
            return await self.create_custom_role(
                db_session,
                user_id=user_id,
                name=role_name,
                system_prompt=(
                    f"You are playing the role '{role_name}'. Answer with knowledge-base grounding first. "
                    "If evidence is insufficient, say so explicitly."
                ),
                category="general",
                role_type="auto",
            )
        # 你正在扮演角色 '{role_name}'。请先基于知识库回答。如果证据不足，请明确说明。
        # 必须提供 role_id 或 role_name
        raise ValueError("Either role_id or role_name must be provided.")

    # 关键词匹配角色
    async def detect_role(
        self,
        db_session: AsyncSession,
        *,
        user_id: str,
        query: str,
    ) -> tuple[RoleRecord, str, float, str]:
        '''
        返回值详解：
        位置	类型	          说明	         示例
        0	RoleRecord	检测到的角色对象	RoleRecord(role_id="lawyer_01")
        1	str	        检测方式	        "matched" / "created"
        2	float	    置信度 (0-1)	    0.92
        3	str	        调试信息	        "matched_by_keywords:律师,合同"
        '''
        lowered = query.lower()
        scores: dict[str, tuple[int, list[str]]] = {}  # {role_id: (分数, [匹配的关键词])}
        # _list_preset_keyword_rules() 返回list[KeywordRule]
        for rule in await self._list_preset_keyword_rules(db_session):
            if rule.keyword.lower() in lowered:
                score, matched_keywords = scores.get(rule.role_id, (0, []))
                scores[rule.role_id] = (
                    score + rule.weight,  # 累加权重
                    [*matched_keywords, rule.keyword]  # 添加匹配的关键词
                )
        # 如果有匹配到的角色--找得分最大的返回
        if scores:
            # 取分数作为比较依据去算max
            role_id, (score, matched_keywords) = max(scores.items(), key=lambda item: item[1][0])
            # 获取角色完整信息
            role = await self.resolve_role(db_session, user_id=user_id, role_id=role_id)
            # 计算置信度
            '''
            为什么是 0.68 和 0.06？
            0.68 = 基础置信度（即使只匹配1个低权重词也有68%信心）
            0.06 = 每个权重单位的增益（匹配到高权重词快速提升置信度）
            min(..., 0.98) = 最高98%置信度，留2%的不确定性
            '''
            confidence = min(0.98, 0.68 + score * 0.06)
            return role, "matched", confidence, f"matched_by_keywords:{','.join(matched_keywords)}"
        # 如果没有匹配到角色--新建通用角色
        # 优化：可能会有大量的无用角色，可以优化为 如果没有匹配到角色就返回通用助手，通用助手只新建一次
        auto_role = await self.create_custom_role(
            db_session,
            user_id=user_id,
            name="Auto Assigned Assistant",
            system_prompt=(
                "You are a general assistant created automatically by the system. "
                "Prefer grounded answers from the knowledge base and user intent."
            ),
            category="general",
            role_type="auto",
        )
        return auto_role, "created", 0.55, "fallback_auto_role"

    # 用户自定义角色
    async def create_custom_role(
        self,
        db_session: AsyncSession,
        *,
        user_id: str,
        name: str,
        system_prompt: str,
        category: str = "general",
        role_type: str = "custom",
        knowledge_base_id: str | None = None,
    ) -> RoleRecord:
        role_id = await self._next_custom_role_id(db_session)
        stmt = text(
            """
            INSERT INTO custom_roles (id, user_id, name, category, role_type, system_prompt, knowledge_base_id, created_at, updated_at)
            VALUES (:id, :user_id, :name, :category, :role_type, :system_prompt, :knowledge_base_id, NOW(), NOW())
            """
        )
        await db_session.execute(
            stmt,
            {
                "id": role_id,
                "user_id": user_id,
                "name": name,
                "category": category,
                "role_type": role_type,
                "system_prompt": system_prompt,
                "knowledge_base_id": knowledge_base_id,
            },
        )
        await db_session.commit()
        logger.info("custom_role_created", user_id=user_id, role_id=role_id, role_type=role_type)
        return RoleRecord(
            role_id=role_id,
            name=name,
            category=category,
            role_type=role_type,
            system_prompt=system_prompt,
            knowledge_base_id=knowledge_base_id,
            created_at=datetime.utcnow(),
        )

    # 删除自定义角色
    async def delete_custom_role(
        self,
        db_session: AsyncSession,
        *,
        user_id: str,
        role_id: str,
    ) -> None:
        lookup_stmt = text(
            """
            SELECT id
            FROM custom_roles
            WHERE id = :role_id AND user_id = :user_id
            LIMIT 1
            """
        )
        result = await db_session.execute(lookup_stmt, {"role_id": role_id, "user_id": user_id})
        row = result.mappings().first()
        if not row:
            raise ValueError(f"Custom role not found: {role_id}")

        # 删除自定义角色表
        await db_session.execute(
            text("DELETE FROM custom_roles WHERE id = :role_id AND user_id = :user_id"),
            {"role_id": role_id, "user_id": user_id},
        )
        # 删除用户角色映射表   user_role_mapping 记录的是用户和角色之间的"使用关系"
        '''
        此表的作用
        1. 统计用户最常用哪些角色
        2. "最近使用"的角色列表
        3.清理不活跃的角色（运营需求）
        4.推荐角色（根据使用习惯）
        '''
        await db_session.execute(
            text("DELETE FROM user_role_mapping WHERE user_id = :user_id AND role_id = :role_id"),
            {"user_id": user_id, "role_id": role_id},
        )
        # 删除聊天内容
        await db_session.execute(
            text("DELETE FROM conversations WHERE user_id = :user_id AND role_id = :role_id"),
            {"user_id": user_id, "role_id": role_id},
        )
        # 删除角色文件
        await db_session.execute(
            text("DELETE FROM knowledge_files WHERE user_id = :user_id AND role_id = :role_id"),
            {"user_id": user_id, "role_id": role_id},
        )
        await db_session.commit()
        logger.info("custom_role_deleted", user_id=user_id, role_id=role_id)

    # 更新自定义角色
    async def update_custom_role(
        self,
        db_session: AsyncSession,
        *,
        user_id: str,
        role_id: str,
        name: str,
        system_prompt: str,
        category: str = "general",
        knowledge_base_id: str | None = None,
    ) -> RoleRecord:
        lookup_stmt = text(
            """
            SELECT id, created_at
            FROM custom_roles
            WHERE id = :role_id AND user_id = :user_id
            LIMIT 1
            """
        )
        result = await db_session.execute(lookup_stmt, {"role_id": role_id, "user_id": user_id})
        row = result.mappings().first()
        if not row:
            raise ValueError(f"Custom role not found: {role_id}")

        update_stmt = text(
            """
            UPDATE custom_roles
            SET name = :name,
                category = :category,
                system_prompt = :system_prompt,
                knowledge_base_id = :knowledge_base_id,
                updated_at = NOW()
            WHERE id = :role_id AND user_id = :user_id
            """
        )
        await db_session.execute(
            update_stmt,
            {
                "role_id": role_id,
                "user_id": user_id,
                "name": name,
                "category": category,
                "system_prompt": system_prompt,
                "knowledge_base_id": knowledge_base_id,
            },
        )
        await db_session.commit()
        logger.info("custom_role_updated", user_id=user_id, role_id=role_id)
        return RoleRecord(
            role_id=role_id,
            name=name,
            category=category,
            role_type="custom",
            system_prompt=system_prompt,
            knowledge_base_id=knowledge_base_id,
            created_at=row["created_at"],
        )

    async def _list_preset_roles(self, db_session: AsyncSession) -> list[RoleRecord]:
        stmt = text(
            """
            SELECT id, name, category, system_prompt, knowledge_base_id, created_at
            FROM preset_roles
            ORDER BY created_at ASC
            """
        )
        result = await db_session.execute(stmt)
        rows = result.mappings().all()
        if not rows:
            return [
                RoleRecord(
                    role_id=item["role_id"],
                    name=item["name"],
                    category=item["category"],
                    role_type="preset",
                    system_prompt=item["system_prompt"],
                    knowledge_base_id=item["knowledge_base_id"],
                    created_at=None,
                )
                for item in DEFAULT_PRESET_ROLES
            ]
        # 从数据库读，为了防止乱码，加一层_repair_mojibake处理
        return [
            RoleRecord(
                role_id=str(row["id"]),
                name=_repair_mojibake(str(row["name"])) or str(row["name"]),
                category=str(row["category"]),
                role_type="preset",
                system_prompt=_repair_mojibake(str(row["system_prompt"])) or str(row["system_prompt"]),
                knowledge_base_id=row["knowledge_base_id"],
                created_at=row["created_at"],
            )
            for row in rows
        ]

    async def _list_custom_roles(self, db_session: AsyncSession, user_id: str) -> list[RoleRecord]:
        stmt = text(
            """
            SELECT id, name, category, role_type, system_prompt, knowledge_base_id, created_at
            FROM custom_roles
            WHERE user_id = :user_id
            ORDER BY created_at DESC
            """
        )
        result = await db_session.execute(stmt, {"user_id": user_id})
        rows = result.mappings().all()
        return [
            RoleRecord(
                role_id=str(row["id"]),
                name=_repair_mojibake(str(row["name"])) or str(row["name"]),
                category=str(row["category"]),
                role_type=str(row.get("role_type") or "custom"),
                system_prompt=_repair_mojibake(str(row["system_prompt"])) or str(row["system_prompt"]),
                knowledge_base_id=row["knowledge_base_id"],
                created_at=row["created_at"],
            )
            for row in rows
        ]

    async def _list_preset_keyword_rules(self, db_session: AsyncSession) -> list[KeywordRule]:
        stmt = text(
            """
            SELECT role_id, keyword, weight
            FROM preset_role_keywords
            WHERE is_enabled = 1
            ORDER BY weight DESC, id ASC
            """
        )
        # WHERE is_enabled = 1 只查询启用的记录
        try:
            result = await db_session.execute(stmt)
        except Exception as exc:
            logger.warning("preset_role_keywords_load_failed", error=str(exc))
            # 加载失败用内存的
            return [
                KeywordRule(
                    role_id=str(item["role_id"]),
                    keyword=str(item["keyword"]),
                    weight=int(item["weight"]),
                )
                for item in DEFAULT_PRESET_ROLE_KEYWORDS
            ]

        rows = result.mappings().all()
        # 查询为空用内存的
        if not rows:
            return [
                KeywordRule(
                    role_id=str(item["role_id"]),
                    keyword=str(item["keyword"]),
                    weight=int(item["weight"]),
                )
                for item in DEFAULT_PRESET_ROLE_KEYWORDS
            ]

        return [
            KeywordRule(
                role_id=str(row["role_id"]),
                keyword=str(row["keyword"]),
                weight=int(row.get("weight") or 1),
            )
            for row in rows
        ]

    # 整个方法是按优先级查找角色:数据库预设角色 → 用户自定义角色 → 内存默认角色
    '''
    1. 数据库预设角色可以动态更新
    2. 用户自定义角色优先级更高
    3. 内存默认角色是兜底方案 内存默认一旦修改，需要重启服务
    优化：
    如果预设角色和自定义角色是同一角色的不同版本 → 用户自定义应该优先
    如果它们是不同的角色类型 → 当前顺序可能合理
    '''
    async def _get_role_by_id(self, db_session: AsyncSession, *, user_id: str, role_id: str) -> RoleRecord | None:
        preset_stmt = text(
            """
            SELECT id, name, category, system_prompt, knowledge_base_id, created_at
            FROM preset_roles
            WHERE id = :role_id
            LIMIT 1
            """
        )
        result = await db_session.execute(preset_stmt, {"role_id": role_id})
        row = result.mappings().first()
        if row:
            return RoleRecord(
                role_id=str(row["id"]),
                name=_repair_mojibake(str(row["name"])) or str(row["name"]),
                category=str(row["category"]),
                role_type="preset",
                system_prompt=_repair_mojibake(str(row["system_prompt"])) or str(row["system_prompt"]),
                knowledge_base_id=row["knowledge_base_id"],
                created_at=row["created_at"],
            )

        custom_stmt = text(
            """
            SELECT id, name, category, role_type, system_prompt, knowledge_base_id, created_at
            FROM custom_roles
            WHERE id = :role_id AND user_id = :user_id
            LIMIT 1
            """
        )
        result = await db_session.execute(custom_stmt, {"role_id": role_id, "user_id": user_id})
        row = result.mappings().first()
        if row:
            return RoleRecord(
                role_id=str(row["id"]),
                name=_repair_mojibake(str(row["name"])) or str(row["name"]),
                category=str(row["category"]),
                role_type=str(row.get("role_type") or "custom"),
                system_prompt=_repair_mojibake(str(row["system_prompt"])) or str(row["system_prompt"]),
                knowledge_base_id=row["knowledge_base_id"],
                created_at=row["created_at"],
            )

        for item in DEFAULT_PRESET_ROLES:
            if item["role_id"] == role_id:
                return RoleRecord(
                    role_id=item["role_id"],
                    name=item["name"],
                    category=item["category"],
                    role_type="preset",
                    system_prompt=item["system_prompt"],
                    knowledge_base_id=item["knowledge_base_id"],
                )
        return None

    '''
    通过 ID 查找（_get_role_by_id）
    优先级：数据库preset > 数据库custom > 内存preset
    原因：ID 是唯一的，不存在冲突，先查数据库确保获取最新数据
    
    通过名称查找（_get_role_by_name）  
    优先级：数据库custom > 内存preset
    原因：名称可能冲突，用户自定义优先；预设角色用内存镜像即可
    '''
    async def _get_role_by_name(self, db_session: AsyncSession, *, user_id: str, role_name: str) -> RoleRecord | None:
        stmt = text(
            """
            SELECT id, name, category, role_type, system_prompt, knowledge_base_id, created_at
            FROM custom_roles
            WHERE user_id = :user_id AND name = :role_name
            LIMIT 1
            """
        )
        result = await db_session.execute(stmt, {"user_id": user_id, "role_name": role_name})
        row = result.mappings().first()
        if row:
            return RoleRecord(
                role_id=str(row["id"]),
                name=_repair_mojibake(str(row["name"])) or str(row["name"]),
                category=str(row["category"]),
                role_type=str(row.get("role_type") or "custom"),
                system_prompt=_repair_mojibake(str(row["system_prompt"])) or str(row["system_prompt"]),
                knowledge_base_id=row["knowledge_base_id"],
                created_at=row["created_at"],
            )

        for item in DEFAULT_PRESET_ROLES:
            if item["name"] == role_name:
                return RoleRecord(
                    role_id=item["role_id"],
                    name=item["name"],
                    category=item["category"],
                    role_type="preset",
                    system_prompt=item["system_prompt"],
                    knowledge_base_id=item["knowledge_base_id"],
                )
        return None

    async def _next_custom_role_id(self, db_session: AsyncSession) -> str:
        stmt = text(
            """
            SELECT COALESCE(MAX(CAST(SUBSTRING(id, 8) AS UNSIGNED)), 0) AS max_suffix
            FROM custom_roles
            WHERE id LIKE 'custom_%'
            """
        )
        '''
        SUBSTRING(id, 8) 从id第8个字符开始截取
        CAST(... AS UNSIGNED) 将字符串数字转换为无符号整数 自动去除前导零
        MAX(...) 对转换后的数字取最大值
        COALESCE(..., 0) 返回参数列表中第一个非NULL的值，可以理解为将多个值合并成一个确定的值。
                         处理空值情况，如果没有匹配的记录返回0而不是NULL
        '''
        result = await db_session.execute(stmt)
        # result.scalar()  从SQL查询结果中获取第一行第一列的值
        max_suffix = int(result.scalar() or 0)
        '''
        :	格式化分隔符
        0	用0填充（而不是空格）
        5	总宽度为5位
        d	十进制整数（decimal integer）
        '''
        return f"custom_{max_suffix + 1:05d}"

