/**
 * 认证模块
 * 基于 HttpOnly Cookie 的认证管理
 *
 * Token 由后端通过 Set-Cookie 设置，JS 无法读取（防 XSS）。
 * 所有 fetch 请求需携带 credentials: 'include' 以自动附带 Cookie。
 */

class AuthManager {
    constructor() {
        this.apiBaseUrl = '/api/auth';
        this.storageKeys = {
            user: 'user_info',
        };

        // Token 刷新定时器
        this.refreshTimer = null;

        // 初始化
        this.initTokenRefresh();
    }

    /**
     * 登录
     * @param {string} username - 用户名
     * @param {string} password - 密码
     * @returns {Promise<Object>} 登录结果
     */
    async login(username, password) {
        try {
            const response = await fetch(`${this.apiBaseUrl}/login`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                credentials: 'include',  // 接收 Set-Cookie
                body: JSON.stringify({ username, password })
            });

            if (!response.ok) {
                const error = await response.json().catch(() => ({}));
                throw new Error(error.detail || '登录失败，请检查用户名和密码');
            }

            const data = await response.json();

            // 仅存储用户信息（token 由 Cookie 管理，JS 不可见）
            this.setUser(data.user);

            // 启动自动刷新
            this.initTokenRefresh();

            return {
                success: true,
                user: data.user
            };
        } catch (error) {
            console.error('登录失败:', error);
            return {
                success: false,
                error: error.message
            };
        }
    }

    /**
     * 登出
     */
    async logout() {
        try {
            await fetch(`${this.apiBaseUrl}/logout`, {
                method: 'POST',
                credentials: 'include',  // 自动携带 Cookie
            });
        } catch (error) {
            console.error('登出请求失败:', error);
        }

        // 清除本地用户信息
        this.clearUser();

        // 清除刷新定时器
        if (this.refreshTimer) {
            clearTimeout(this.refreshTimer);
            this.refreshTimer = null;
        }

        // 跳转到登录页
        window.location.href = '/login';
    }

    /**
     * 刷新 access token
     * 后端从 HttpOnly Cookie 中读取 refresh_token，签发新 access_token 并设置 Cookie
     */
    async refreshToken() {
        try {
            const response = await fetch(`${this.apiBaseUrl}/refresh`, {
                method: 'POST',
                credentials: 'include',  // 自动携带 refresh_token Cookie
            });

            if (!response.ok) {
                throw new Error('刷新 token 失败');
            }

            return true;
        } catch (error) {
            console.error('刷新 token 失败:', error);
            this.handleAuthError();
            return false;
        }
    }

    /**
     * 初始化 token 自动刷新
     */
    initTokenRefresh() {
        // 清除已有的定时器
        if (this.refreshTimer) {
            clearTimeout(this.refreshTimer);
        }

        // 每 25 分钟刷新一次（提前 5 分钟）
        const refreshInterval = 25 * 60 * 1000;

        this.refreshTimer = setTimeout(() => {
            this.refreshToken().then(() => {
                this.initTokenRefresh();
            });
        }, refreshInterval);
    }

    /**
     * 获取用户信息
     */
    getUser() {
        const userStr = localStorage.getItem(this.storageKeys.user);
        if (userStr) {
            try {
                return JSON.parse(userStr);
            } catch (e) {
                return null;
            }
        }
        return null;
    }

    /**
     * 设置用户信息
     */
    setUser(user) {
        localStorage.setItem(this.storageKeys.user, JSON.stringify(user));
    }

    /**
     * 清除用户信息
     */
    clearUser() {
        localStorage.removeItem(this.storageKeys.user);
    }

    /**
     * 检查是否已登录
     * Cookie 模式下，通过检查用户信息是否存在来判断
     */
    isLoggedIn() {
        const user = this.getUser();
        return user !== null;
    }

    /**
     * 检查认证状态，未登录则跳转
     */
    checkAuth() {
        if (!this.isLoggedIn()) {
            this.handleAuthError();
            return false;
        }
        return true;
    }

    /**
     * 处理认证错误
     */
    handleAuthError() {
        this.clearUser();
        window.location.href = '/login';
    }

    /**
     * 发送带认证的请求
     * Cookie 会自动携带，无需手动注入 Authorization header
     * @param {string} url - 请求 URL
     * @param {Object} options - fetch 选项
     */
    async authFetch(url, options = {}) {
        // 确保已登录
        if (!this.isLoggedIn()) {
            this.handleAuthError();
            throw new Error('未登录');
        }

        // 合并 headers，确保 Content-Type 存在
        const headers = {
            'Content-Type': 'application/json',
            ...options.headers
        };

        try {
            const response = await fetch(url, {
                ...options,
                headers,
                credentials: 'include',  // 自动携带 Cookie
            });

            // 如果返回 401，尝试刷新 token 后重试
            if (response.status === 401) {
                const refreshed = await this.refreshToken();
                if (refreshed) {
                    // 重试请求
                    return fetch(url, {
                        ...options,
                        headers,
                        credentials: 'include',
                    });
                } else {
                    this.handleAuthError();
                    throw new Error('认证失败');
                }
            }

            return response;
        } catch (error) {
            throw error;
        }
    }
}

// 创建全局实例
const authManager = new AuthManager();

// 导出
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { AuthManager, authManager };
}
