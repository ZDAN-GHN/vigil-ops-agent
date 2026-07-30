-- =====================================================
-- OnCallAgent 数据库初始化脚本
-- =====================================================
-- 用途：创建应用所需的数据库和表结构
-- 数据库：MySQL 8.0+
-- 字符集：utf8mb4
--
-- 使用方法：
--   mysql -u root -p < sql/create_table.sql
--   或在 MySQL 客户端中执行：source sql/create_table.sql
--
-- 注意：
--   1. 所有 CREATE 语句使用 IF NOT EXISTS，可重复执行
--   2. 应用启动时也会通过 SQLAlchemy 自动建表，此脚本用于手动初始化或 DBA 审核
--   3. 数据库名默认 oncall_agent，与 .env 中 MYSQL_DB 配置一致
-- =====================================================

-- 创建数据库（如果不存在）
CREATE DATABASE IF NOT EXISTS `oncall_agent`
    DEFAULT CHARACTER SET utf8mb4
    DEFAULT COLLATE utf8mb4_unicode_ci;

-- 切换到目标数据库
USE `oncall_agent`;

-- =====================================================
-- 表1: users - 用户认证表
-- 用途：存储用户登录信息，包括用户名、密码哈希、角色等
-- 对应模型：app/models/user.py -> User
-- =====================================================
CREATE TABLE IF NOT EXISTS `users` (
    `id` INT AUTO_INCREMENT PRIMARY KEY COMMENT '用户ID',
    `username` VARCHAR(64) NOT NULL COMMENT '登录用户名',
    `hashed_password` VARCHAR(256) NOT NULL COMMENT 'bcrypt哈希密码',
    `display_name` VARCHAR(128) NOT NULL DEFAULT '' COMMENT '显示名称',
    `is_active` TINYINT(1) NOT NULL DEFAULT 1 COMMENT '是否启用：1=启用，0=禁用',
    `is_admin` TINYINT(1) NOT NULL DEFAULT 0 COMMENT '是否管理员：1=管理员，0=普通用户',
    `created_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    `updated_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    
    -- 索引
    UNIQUE KEY `uk_username` (`username`),
    KEY `idx_username` (`username`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='用户认证表';

-- =====================================================
-- 表2: user_profiles - 用户画像表
-- 用途：存储跨会话的长期记忆，包括用户特征、偏好等信息
-- 对应模型：app/models/user_profile.py -> UserProfile
-- =====================================================
CREATE TABLE IF NOT EXISTS `user_profiles` (
    `user_id` VARCHAR(128) NOT NULL PRIMARY KEY COMMENT '用户唯一标识',
    `features` JSON NOT NULL DEFAULT (JSON_OBJECT()) COMMENT '用户特征画像，如 {"role": "运维工程师", "focus": "Kubernetes"}',
    `preferences` JSON NOT NULL DEFAULT (JSON_OBJECT()) COMMENT '用户偏好设置，如 {"response_style": "简洁"}',
    `summary_count` INT NOT NULL DEFAULT 0 COMMENT '累计摘要次数',
    `last_summary_at` DATETIME NULL COMMENT '最后一次摘要时间',
    `notes` TEXT NULL COMMENT '备注信息',
    `created_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    `updated_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间'
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='用户画像表（长期记忆）';

-- =====================================================
-- 表3: conversation_summaries - 对话摘要历史表
-- 用途：存储每次对话摘要的记录，用于追溯和分析
-- 对应模型：app/models/user_profile.py -> ConversationSummary
-- =====================================================
CREATE TABLE IF NOT EXISTS `conversation_summaries` (
    `id` INT AUTO_INCREMENT PRIMARY KEY COMMENT '自增主键',
    `session_id` VARCHAR(128) NOT NULL COMMENT '会话ID',
    `user_id` VARCHAR(128) NULL COMMENT '用户ID（可选，用于关联用户画像）',
    `summary` TEXT NOT NULL COMMENT '摘要内容',
    `features_extracted` JSON NOT NULL DEFAULT (JSON_OBJECT()) COMMENT '从摘要中提取的特征',
    `message_count` INT NOT NULL DEFAULT 0 COMMENT '原始消息数量',
    `created_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    
    -- 索引
    KEY `idx_session_id` (`session_id`),
    KEY `idx_user_id` (`user_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='对话摘要历史表';

-- =====================================================
-- 表4: conversation_sessions - 会话管理表
-- 用途：管理用户的对话会话，记录会话元数据。
--       用户可以查询自己拥有的会话列表，实现会话的生命周期管理。
-- 对应模型：app/models/conversation_session.py -> ConversationSession
-- =====================================================
CREATE TABLE IF NOT EXISTS `conversation_sessions` (
    `id` INT AUTO_INCREMENT PRIMARY KEY COMMENT '自增主键',
    `session_id` VARCHAR(128) NOT NULL COMMENT '会话ID（唯一，与前端传入的 id 对应）',
    `user_id` INT NOT NULL COMMENT '用户ID（关联 users.id）',
    `title` VARCHAR(256) NOT NULL DEFAULT '' COMMENT '会话标题',
    `message_count` INT NOT NULL DEFAULT 0 COMMENT '消息数量（冗余字段，便于排序）',
    `created_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    `updated_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '最后更新时间',
    `is_deleted` TINYINT(1) NOT NULL DEFAULT 0 COMMENT '是否已删除：0=未删除，1=已删除（软删除）',
    
    -- 索引
    UNIQUE KEY `uk_session_id` (`session_id`),
    KEY `idx_user_id` (`user_id`),
    KEY `idx_user_updated` (`user_id`, `updated_at`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='会话管理表（用户会话列表）';

-- =====================================================
-- 表5: conversation_histories - 对话历史表
-- 用途：存储完整的对话消息记录，作为 Redis checkpoint 的 MySQL 持久化备份。
--       当 Redis TTL 过期后，可从 MySQL 恢复对话上下文到 Redis。
-- 对应模型：app/models/conversation_history.py -> ConversationHistory
-- =====================================================
CREATE TABLE IF NOT EXISTS `conversation_histories` (
    `id` INT AUTO_INCREMENT PRIMARY KEY COMMENT '自增主键',
    `session_id` VARCHAR(128) NOT NULL COMMENT '会话ID（即 thread_id）',
    `role` VARCHAR(32) NOT NULL COMMENT '消息角色: user/assistant/system/summary',
    `content` TEXT NOT NULL COMMENT '消息内容',
    `message_order` INT NOT NULL COMMENT '消息在会话中的顺序（从 0 开始递增）',
    `tool_calls` TEXT NULL COMMENT '工具调用信息 JSON（可选）',
    `metadata_json` TEXT NULL COMMENT '额外元数据 JSON（可选）',
    `created_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    
    -- 索引
    KEY `idx_session_id` (`session_id`),
    KEY `ix_session_order` (`session_id`, `message_order`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='对话历史表（Redis checkpoint 的 MySQL 持久化备份）';

-- =====================================================
-- 初始化数据（可选）
-- =====================================================
-- 如需创建初始管理员，可执行以下语句（密码需要预先用 bcrypt 哈希）
-- 注意：应用启动时会自动创建初始管理员，通常无需手动插入
-- 
-- INSERT INTO `users` (`username`, `hashed_password`, `display_name`, `is_active`, `is_admin`)
-- VALUES ('admin', '$2b$12$...', '系统管理员', 1, 1)
-- ON DUPLICATE KEY UPDATE `username` = `username`;

-- =====================================================
-- 完成
-- =====================================================
-- 数据库和表结构创建完成
-- 可通过以下命令验证：
--   SHOW TABLES;
--   DESCRIBE users;
--   DESCRIBE user_profiles;
--   DESCRIBE conversation_summaries;
--   DESCRIBE conversation_sessions;
--   DESCRIBE conversation_histories;
