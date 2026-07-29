/**
 * 认证模块
 * 封装登录、token 管理、自动刷新等逻辑
 */

class AuthManager {
    constructor() {
        this.apiBaseUrl = '/api/auth';
        this.storageKeys = {
            accessToken: 'access_token',
            refreshToken: 'refresh_token',
            user: 'user_info',
            tokenExpiry: 'token_expiry'
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
                body: JSON.stringify({ username, password })
            });

            if (!response.ok) {
                const error = await response.json().catch(() => ({}));
                throw new Error(error.detail || '登录失败，请检查用户名和密码');
            }

            const data = await response.json();
            
            // 存储 tokens
            this.setTokens(data.access_token, data.refresh_token);
            this.setUser(data.user);
            
            // 设置 token 过期时间（30分钟后）
            const expiryTime = Date.now() + 30 * 60 * 1000;
            localStorage.setItem(this.storageKeys.tokenExpiry, expiryTime.toString());
            
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
        const refreshToken = this.getRefreshToken();
        
        if (refreshToken) {
            try {
                await fetch(`${this.apiBaseUrl}/logout`, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({ refresh_token: refreshToken })
                });
            } catch (error) {
                console.error('登出请求失败:', error);
            }
        }
        
        // 清除本地存储
        this.clearTokens();
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
     */
    async refreshToken() {
        const refreshToken = this.getRefreshToken();
        
        if (!refreshToken) {
            this.handleAuthError();
            return false;
        }

        try {
            const response = await fetch(`${this.apiBaseUrl}/refresh`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ refresh_token: refreshToken })
            });

            if (!response.ok) {
                throw new Error('刷新 token 失败');
            }

            const data = await response.json();
            
            // 更新 access token
            localStorage.setItem(this.storageKeys.accessToken, data.access_token);
            
            // 更新过期时间
            const expiryTime = Date.now() + 30 * 60 * 1000;
            localStorage.setItem(this.storageKeys.tokenExpiry, expiryTime.toString());
            
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
     * 获取 access token
     */
    getAccessToken() {
        return localStorage.getItem(this.storageKeys.accessToken);
    }

    /**
     * 获取 refresh token
     */
    getRefreshToken() {
        return localStorage.getItem(this.storageKeys.refreshToken);
    }

    /**
     * 设置 tokens
     */
    setTokens(accessToken, refreshToken) {
        localStorage.setItem(this.storageKeys.accessToken, accessToken);
        localStorage.setItem(this.storageKeys.refreshToken, refreshToken);
    }

    /**
     * 清除 tokens
     */
    clearTokens() {
        localStorage.removeItem(this.storageKeys.accessToken);
        localStorage.removeItem(this.storageKeys.refreshToken);
        localStorage.removeItem(this.storageKeys.tokenExpiry);
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
     */
    isLoggedIn() {
        const accessToken = this.getAccessToken();
        const expiryTime = localStorage.getItem(this.storageKeys.tokenExpiry);
        
        if (!accessToken || !expiryTime) {
            return false;
        }
        
        // 检查 token 是否过期
        if (Date.now() > parseInt(expiryTime)) {
            return false;
        }
        
        return true;
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
        this.clearTokens();
        this.clearUser();
        window.location.href = '/login';
    }

    /**
     * 获取带认证头的请求配置
     */
    getAuthHeaders() {
        const accessToken = this.getAccessToken();
        return {
            'Authorization': `Bearer ${accessToken}`,
            'Content-Type': 'application/json'
        };
    }

    /**
     * 发送带认证的请求
     * @param {string} url - 请求 URL
     * @param {Object} options - fetch 选项
     */
    async authFetch(url, options = {}) {
        // 确保已登录
        if (!this.isLoggedIn()) {
            this.handleAuthError();
            throw new Error('未登录');
        }

        const headers = {
            ...this.getAuthHeaders(),
            ...options.headers
        };

        try {
            const response = await fetch(url, {
                ...options,
                headers
            });

            // 如果返回 401，尝试刷新 token
            if (response.status === 401) {
                const refreshed = await this.refreshToken();
                if (refreshed) {
                    // 重试请求
                    return fetch(url, {
                        ...options,
                        headers: {
                            ...this.getAuthHeaders(),
                            ...options.headers
                        }
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
