// Main application JavaScript for Financial Assistant UI
class FinancialAssistantApp {
    constructor() {
        this.apiBaseUrl = window.location.origin;
        this.currentChatId = null;
        this.chatHistory = [];
        this.isTyping = false;
        this.theme = localStorage.getItem('theme') || 'light';
        this.isAuthenticated = false;
        this.attachedFiles = [];

        this.init();
    }

    init() {
        this.checkAuthentication();
    }

    async checkAuthentication() {
        try {
            // Add session token to Authorization header if available in localStorage
            const sessionToken = localStorage.getItem('session_token');
            const headers = {};
            if (sessionToken) {
                headers['Authorization'] = `Bearer ${sessionToken}`;
            }

            const response = await fetch(`${this.apiBaseUrl}/auth/status`, {
                headers: headers
            });
            const data = await response.json();

            if (data.authenticated) {
                this.isAuthenticated = true;
                this.setupEventListeners();
                this.loadTheme();
                // Add small delay to ensure DOM is ready
                setTimeout(() => {
                    this.checkSystemHealth();
                }, 100);
                this.loadRecentMemories();
                this.autoResizeTextarea();
                this.setupLogoutButton();
            } else {
                // Try to authenticate automatically
                console.log('Not authenticated, attempting automatic login...');
                const loginSuccess = await this.reauthenticate();
                if (loginSuccess) {
                    // Retry authentication check
                    await this.checkAuthentication();
                } else {
                    // Redirect to login page
                    window.location.href = '/login';
                }
            }
        } catch (error) {
            console.error('Auth check failed:', error);
            window.location.href = '/login';
        }
    }

    async logout() {
        try {
            // Add session token to Authorization header if available
            const sessionToken = localStorage.getItem('session_token');
            const headers = {};
            if (sessionToken) {
                headers['Authorization'] = `Bearer ${sessionToken}`;
            }

            const response = await fetch(`${this.apiBaseUrl}/auth/logout`, {
                method: 'POST',
                headers: headers
            });

            const data = await response.json();
            if (data.success) {
                // Clear any local session data
                document.cookie = 'session_token=; expires=Thu, 01 Jan 1970 00:00:00 UTC; path=/;';
                localStorage.removeItem('session_token');

                // Redirect to login page
                window.location.href = '/login';
            }
        } catch (error) {
            console.error('Logout failed:', error);
            // Clear local data and redirect to login page even if logout API fails
            document.cookie = 'session_token=; expires=Thu, 01 Jan 1970 00:00:00 UTC; path=/;';
            localStorage.removeItem('session_token');
            window.location.href = '/login';
        }
    }

    setupLogoutButton() {
        // Find the settings button and add logout functionality
        const settingsBtn = document.getElementById('settingsBtn');
        if (settingsBtn) {
            settingsBtn.addEventListener('click', () => {
                this.showSettingsModal();
            });
        }
    }

    showSettingsModal() {
        // Create a simple settings modal with logout option
        const modalHtml = `
            <div class="modal-overlay" id="settingsModal">
                <div class="modal-content">
                    <div class="modal-header">
                        <h3><i class="fas fa-cog"></i> Settings</h3>
                        <button class="modal-close" onclick="this.closest('.modal-overlay').remove()">
                            <i class="fas fa-times"></i>
                        </button>
                    </div>
                    <div class="modal-body">
                        <div class="settings-section">
                            <h4><i class="fas fa-sign-out-alt"></i> Session</h4>
                            <button class="logout-btn" onclick="app.logout()">
                                <i class="fas fa-sign-out-alt"></i>
                                Logout
                            </button>
                        </div>
                        <div class="settings-section">
                            <h4><i class="fas fa-info-circle"></i> About</h4>
                            <p>Financial Assistant v1.0.0</p>
                            <p>Authenticated with secure session management</p>
                        </div>
                    </div>
                </div>
            </div>
        `;

        // Add modal styles if not already present
        if (!document.querySelector('#modal-styles')) {
            const style = document.createElement('style');
            style.id = 'modal-styles';
            style.textContent = `
                .modal-overlay {
                    position: fixed;
                    top: 0;
                    left: 0;
                    right: 0;
                    bottom: 0;
                    background: rgba(0, 0, 0, 0.5);
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    z-index: 1000;
                }
                .modal-content {
                    background: white;
                    border-radius: 12px;
                    padding: 0;
                    max-width: 400px;
                    width: 90%;
                    max-height: 80vh;
                    overflow-y: auto;
                    box-shadow: 0 20px 40px rgba(0, 0, 0, 0.2);
                }
                .modal-header {
                    padding: 1.5rem;
                    border-bottom: 1px solid #e2e8f0;
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                }
                .modal-header h3 {
                    margin: 0;
                    color: #2d3748;
                }
                .modal-close {
                    background: none;
                    border: none;
                    font-size: 1.2rem;
                    cursor: pointer;
                    color: #718096;
                    padding: 0.5rem;
                    border-radius: 6px;
                    transition: all 0.3s ease;
                }
                .modal-close:hover {
                    background: #f7fafc;
                    color: #2d3748;
                }
                .modal-body {
                    padding: 1.5rem;
                }
                .settings-section {
                    margin-bottom: 1.5rem;
                }
                .settings-section:last-child {
                    margin-bottom: 0;
                }
                .settings-section h4 {
                    margin: 0 0 0.75rem 0;
                    color: #4a5568;
                    font-size: 0.9rem;
                }
                .logout-btn {
                    background: linear-gradient(135deg, #f56565 0%, #e53e3e 100%);
                    color: white;
                    border: none;
                    padding: 0.75rem 1.5rem;
                    border-radius: 8px;
                    cursor: pointer;
                    font-weight: 500;
                    transition: all 0.3s ease;
                    display: flex;
                    align-items: center;
                    gap: 0.5rem;
                }
                .logout-btn:hover {
                    transform: translateY(-1px);
                    box-shadow: 0 4px 12px rgba(245, 101, 101, 0.3);
                }
            `;
            document.head.appendChild(style);
        }

        // Add modal to page
        document.body.insertAdjacentHTML('beforeend', modalHtml);
    }

    setupEventListeners() {
        // Chat functionality
        const sendBtn = document.getElementById('sendBtn');
        const messageInput = document.getElementById('messageInput');

        sendBtn.addEventListener('click', () => this.sendMessage());
        messageInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                this.sendMessage();
            }
        });

        messageInput.addEventListener('input', () => {
            this.updateCharCount();
            this.toggleSendButton();
            this.autoResizeTextarea();
        });

        // Memory management
        document.getElementById('storeMemoryBtn').addEventListener('click', () => this.storeMemory());
        document.getElementById('refreshMemoriesBtn').addEventListener('click', () => this.loadRecentMemories());

        // UI controls
        document.getElementById('newChatBtn').addEventListener('click', () => this.newChat());
        document.getElementById('menuToggle').addEventListener('click', () => this.toggleSidebar());
        document.getElementById('themeToggle').addEventListener('click', () => this.toggleTheme());
        document.getElementById('settingsBtn').addEventListener('click', () => this.showSettings());

        // Close sidebar when clicking outside on mobile
        document.addEventListener('click', (e) => {
            const sidebar = document.getElementById('sidebar');
            const menuToggle = document.getElementById('menuToggle');

            if (window.innerWidth <= 768 &&
                !sidebar.contains(e.target) &&
                !menuToggle.contains(e.target) &&
                sidebar.classList.contains('open')) {
                sidebar.classList.remove('open');
            }
        });

        // Handle window resize
        window.addEventListener('resize', () => {
            if (window.innerWidth > 768) {
                document.getElementById('sidebar').classList.remove('open');
            }
        });

        // Model switching
        const modelSwitchBtn = document.getElementById('modelSwitchBtn');
        const modelDropdown = document.getElementById('modelDropdown');

        if (modelSwitchBtn) {
            modelSwitchBtn.addEventListener('click', () => this.switchModel());
        }

        if (modelDropdown) {
            modelDropdown.addEventListener('change', () => {
                // Enable switch button only when model is different from current
                const currentModel = modelDropdown.value;
                // Auto-switch when selection changes
                if (currentModel) {
                    this.switchModel();
                }
            });
        }

        // File upload functionality
        const fileUploadBtn = document.getElementById('fileUploadBtn');
        const fileInput = document.getElementById('fileInput');

        if (fileUploadBtn && fileInput) {
            fileUploadBtn.addEventListener('click', () => {
                fileInput.click();
            });

            fileInput.addEventListener('change', (e) => {
                this.handleFileSelect(e.target.files);
            });
        }
    }

    async sendMessage() {
        const messageInput = document.getElementById('messageInput');
        const message = messageInput.value.trim();

        if (!message && this.attachedFiles.length === 0) return;

        // Hide welcome message
        this.hideWelcomeMessage();

        // Add user message to chat
        this.addMessage('user', message);

        // Clear input
        messageInput.value = '';
        this.updateCharCount();
        this.toggleSendButton();
        this.autoResizeTextarea();

        // Use streaming by default
        await this.sendMessageStream(message);
    }

    async sendMessageStream(message) {
        this.isTyping = true;

        try {
            // Upload files first if any
            let uploadedFiles = [];
            if (this.attachedFiles.length > 0) {
                this.showToast('Uploading files...', 'info');
                uploadedFiles = await this.uploadFiles();

                if (uploadedFiles.length === 0) {
                    this.showToast('Failed to upload files', 'error');
                    this.isTyping = false;
                    return;
                }
            }

            // Clear files after upload
            this.clearFiles();

            // Create assistant message that will be updated with streaming content
            const messageId = this.addStreamingMessage('assistant', '');

            // Add session token to Authorization header if available
            const sessionToken = localStorage.getItem('session_token');
            const headers = {
                'Content-Type': 'application/json',
            };
            if (sessionToken) {
                headers['Authorization'] = `Bearer ${sessionToken}`;
            }

            const requestBody = {
                message: message,
                model: 'phi3',
                memory_context: 3
            };

            // Include uploaded files in the request
            if (uploadedFiles.length > 0) {
                requestBody.files = uploadedFiles;
            }

            const response = await fetch(`${this.apiBaseUrl}/chat/stream`, {
                method: 'POST',
                headers: headers,
                body: JSON.stringify(requestBody)
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            let accumulatedContent = '';

            while (true) {
                const { done, value } = await reader.read();

                if (done) break;

                const chunk = decoder.decode(value);
                const lines = chunk.split('\n');

                for (const line of lines) {
                    if (line.startsWith('data: ')) {
                        const data = line.slice(6);

                        if (data === '[DONE]') {
                            // Stream finished
                            this.finalizeStreamingMessage(messageId, accumulatedContent);
                            this.isTyping = false;
                            return;
                        }

                        try {
                            const parsed = JSON.parse(data);

                            // Handle error chunks
                            if (parsed.error) {
                                if (parsed.choices && parsed.choices[0] && parsed.choices[0].delta && parsed.choices[0].delta.content) {
                                    accumulatedContent += parsed.choices[0].delta.content;
                                    this.updateStreamingMessage(messageId, accumulatedContent);
                                }
                                this.finalizeStreamingMessage(messageId, accumulatedContent);
                                this.isTyping = false;
                                return;
                            }

                            if (parsed.choices && parsed.choices[0]) {
                                const delta = parsed.choices[0].delta;

                                if (delta.content) {
                                    // Check if content contains an error message from Ollama
                                    if (delta.content.includes('ERROR:')) {
                                        accumulatedContent += delta.content;
                                        this.updateStreamingMessage(messageId, accumulatedContent);
                                        this.finalizeStreamingMessage(messageId, accumulatedContent);
                                        this.isTyping = false;
                                        return;
                                    }
                                    accumulatedContent += delta.content;
                                    this.updateStreamingMessage(messageId, accumulatedContent);
                                }
                            }
                        } catch (e) {
                            // Skip invalid JSON
                            continue;
                        }
                    }
                }
            }

            this.finalizeStreamingMessage(messageId, accumulatedContent);
            this.isTyping = false;

        } catch (error) {
            this.isTyping = false;
            console.error('Streaming API error:', error);

            // Fall back to non-streaming API
            this.sendMessageFallback(message);
        }
    }

    async sendMessageFallback(message) {
        // Show typing indicator for fallback
        this.showTypingIndicator();

        try {
            const response = await this.apiCall('/chat', {
                message: message,
                model: 'phi3',
                memory_context: 3
            });

            this.hideTypingIndicator();

            if (response.error) {
                this.addMessage('system', `Error: ${response.error}`, 'error');
            } else {
                this.addMessage('assistant', response.response, null, {
                    model: response.model,
                    timestamp: response.timestamp,
                    memory_used: response.memory_used
                });
            }
        } catch (error) {
            this.hideTypingIndicator();
            this.addMessage('system', 'Failed to connect to the server. Please check if the service is running.', 'error');
            console.error('Chat API error:', error);
        }
    }

    addMessage(role, content, type = null, metadata = {}) {
        const chatContainer = document.getElementById('chatContainer');
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${role}`;

        const avatar = this.createAvatar(role);
        const messageContent = this.createMessageContent(role, content, metadata);

        messageDiv.appendChild(avatar);
        messageDiv.appendChild(messageContent);

        chatContainer.appendChild(messageDiv);
        chatContainer.scrollTop = chatContainer.scrollHeight;

        // Add to chat history
        this.chatHistory.push({
            role,
            content,
            timestamp: new Date().toISOString(),
            metadata
        });
    }

    addStreamingMessage(role, initialContent = '') {
        const chatContainer = document.getElementById('chatContainer');
        const messageDiv = document.createElement('div');
        const messageId = `msg-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;

        messageDiv.className = `message ${role} streaming`;
        messageDiv.id = messageId;

        const avatar = this.createAvatar(role);
        const messageContent = this.createStreamingMessageContent(role, initialContent);

        messageDiv.appendChild(avatar);
        messageDiv.appendChild(messageContent);

        chatContainer.appendChild(messageDiv);
        chatContainer.scrollTop = chatContainer.scrollHeight;

        // Add to chat history
        this.chatHistory.push({
            id: messageId,
            role,
            content: initialContent,
            timestamp: new Date().toISOString(),
            streaming: true
        });

        return messageId;
    }

    updateStreamingMessage(messageId, content) {
        const messageDiv = document.getElementById(messageId);
        if (!messageDiv) return;

        const textDiv = messageDiv.querySelector('.message-text');
        if (textDiv) {
            // Remove cursor if it exists
            const cursor = textDiv.querySelector('.streaming-cursor');
            if (cursor) {
                textDiv.removeChild(cursor);
            }

            // Update content
            textDiv.textContent = content;

            // Re-add cursor
            const newCursor = document.createElement('span');
            newCursor.className = 'streaming-cursor';
            newCursor.innerHTML = '|';
            newCursor.style.animation = 'pulse 1s infinite';
            textDiv.appendChild(newCursor);

            // Update chat history
            const historyItem = this.chatHistory.find(item => item.id === messageId);
            if (historyItem) {
                historyItem.content = content;
            }

            // Auto-scroll to keep latest content visible
            const chatContainer = document.getElementById('chatContainer');
            chatContainer.scrollTop = chatContainer.scrollHeight;
        }
    }

    finalizeStreamingMessage(messageId, content) {
        const messageDiv = document.getElementById(messageId);
        if (!messageDiv) return;

        // Remove streaming class
        messageDiv.classList.remove('streaming');

        // Add final metadata
        const messageContent = messageDiv.querySelector('.message-content');
        const metaDiv = document.createElement('div');
        metaDiv.className = 'message-meta';

        const timestamp = document.createElement('span');
        timestamp.textContent = new Date().toLocaleTimeString();

        metaDiv.appendChild(timestamp);
        messageContent.appendChild(metaDiv);

        // Update chat history
        const historyItem = this.chatHistory.find(item => item.id === messageId);
        if (historyItem) {
            historyItem.content = content;
            historyItem.streaming = false;
            historyItem.timestamp = new Date().toISOString();
        }

        // Show a subtle animation to indicate completion
        messageDiv.style.animation = 'fadeIn 0.3s ease';
    }

    createStreamingMessageContent(role, content) {
        const contentDiv = document.createElement('div');
        contentDiv.className = 'message-content';

        const roleDiv = document.createElement('div');
        roleDiv.className = 'message-role';
        roleDiv.textContent = role.charAt(0).toUpperCase() + role.slice(1);

        const textDiv = document.createElement('div');
        textDiv.className = 'message-text';
        textDiv.textContent = content;

        // Add cursor indicator for streaming
        const cursor = document.createElement('span');
        cursor.className = 'streaming-cursor';
        cursor.innerHTML = '|';
        cursor.style.animation = 'pulse 1s infinite';
        textDiv.appendChild(cursor);

        contentDiv.appendChild(roleDiv);
        contentDiv.appendChild(textDiv);

        return contentDiv;
    }

    createAvatar(role) {
        const avatar = document.createElement('div');
        avatar.className = 'message-avatar';

        if (role === 'user') {
            avatar.innerHTML = '<i class="fas fa-user"></i>';
        } else if (role === 'assistant') {
            avatar.innerHTML = '<i class="fas fa-robot"></i>';
        } else {
            avatar.innerHTML = '<i class="fas fa-exclamation-triangle"></i>';
        }

        return avatar;
    }

    createMessageContent(role, content, metadata) {
        const contentDiv = document.createElement('div');
        contentDiv.className = 'message-content';

        const roleDiv = document.createElement('div');
        roleDiv.className = 'message-role';
        roleDiv.textContent = role.charAt(0).toUpperCase() + role.slice(1);

        const textDiv = document.createElement('div');
        textDiv.className = 'message-text';
        textDiv.textContent = content;

        contentDiv.appendChild(roleDiv);
        contentDiv.appendChild(textDiv);

        // Add metadata if available
        if (metadata && Object.keys(metadata).length > 0) {
            const metaDiv = document.createElement('div');
            metaDiv.className = 'message-meta';

            if (metadata.memory_used) {
                const memoryIndicator = document.createElement('span');
                memoryIndicator.className = 'memory-indicator';
                memoryIndicator.innerHTML = '<i class="fas fa-brain"></i> Memory used';
                metaDiv.appendChild(memoryIndicator);
            }

            if (metadata.timestamp) {
                const timestamp = document.createElement('span');
                timestamp.textContent = new Date(metadata.timestamp).toLocaleTimeString();
                metaDiv.appendChild(timestamp);
            }

            contentDiv.appendChild(metaDiv);
        }

        return contentDiv;
    }

    showTypingIndicator() {
        this.isTyping = true;
        const chatContainer = document.getElementById('chatContainer');

        const typingDiv = document.createElement('div');
        typingDiv.className = 'message assistant typing-message';
        typingDiv.id = 'typingIndicator';

        const avatar = this.createAvatar('assistant');
        const contentDiv = document.createElement('div');
        contentDiv.className = 'message-content';

        const roleDiv = document.createElement('div');
        roleDiv.className = 'message-role';
        roleDiv.textContent = 'Assistant';

        const typingIndicator = document.createElement('div');
        typingIndicator.className = 'typing-indicator';
        typingIndicator.innerHTML = '<span></span><span></span><span></span>';

        contentDiv.appendChild(roleDiv);
        contentDiv.appendChild(typingIndicator);
        typingDiv.appendChild(avatar);
        typingDiv.appendChild(contentDiv);

        chatContainer.appendChild(typingDiv);
        chatContainer.scrollTop = chatContainer.scrollHeight;
    }

    hideTypingIndicator() {
        this.isTyping = false;
        const typingIndicator = document.getElementById('typingIndicator');
        if (typingIndicator) {
            typingIndicator.remove();
        }
    }

    async storeMemory() {
        const key = document.getElementById('memoryKey').value.trim();
        const value = document.getElementById('memoryValue').value.trim();
        const category = document.getElementById('memoryCategory').value;

        if (!key || !value) {
            this.showToast('Please fill in both key and value fields', 'error');
            return;
        }

        try {
            const response = await this.apiCall('/memorize', {
                key,
                value,
                category
            });

            if (response.error) {
                this.showToast(`Failed to store memory: ${response.error}`, 'error');
            } else {
                this.showToast('Memory stored successfully!', 'success');

                // Clear form
                document.getElementById('memoryKey').value = '';
                document.getElementById('memoryValue').value = '';

                // Refresh memories list
                this.loadRecentMemories();
            }
        } catch (error) {
            this.showToast('Failed to store memory', 'error');
            console.error('Memory API error:', error);
        }
    }

    async loadRecentMemories() {
        const memoriesList = document.getElementById('memoriesList');
        memoriesList.innerHTML = '<div class="loading">Loading memories...</div>';

        try {
            const memories = await this.getRecentMemories();

            if (memories.length === 0) {
                memoriesList.innerHTML = '<div class="loading">No memories stored yet</div>';
                return;
            }

            memoriesList.innerHTML = '';
            memories.forEach(memory => {
                const memoryItem = document.createElement('div');
                memoryItem.className = 'memory-item';

                const keyDiv = document.createElement('div');
                keyDiv.className = 'memory-key';
                keyDiv.textContent = memory.key;

                const valueDiv = document.createElement('div');
                valueDiv.className = 'memory-value';
                valueDiv.textContent = memory.value;

                const metaDiv = document.createElement('div');
                metaDiv.className = 'memory-meta';
                metaDiv.innerHTML = `
                    <span class="category">${memory.category}</span>
                    <span class="time">${new Date(memory.timestamp).toLocaleDateString()}</span>
                `;

                memoryItem.appendChild(keyDiv);
                memoryItem.appendChild(valueDiv);
                memoryItem.appendChild(metaDiv);

                memoriesList.appendChild(memoryItem);
            });
        } catch (error) {
            memoriesList.innerHTML = '<div class="error-message">Failed to load memories</div>';
            console.error('Memories load error:', error);
        }
    }

    async getRecentMemories() {
        try {
            const response = await this.apiCall('/memories?n=10', null, 'GET');
            return response || [];
        } catch (error) {
            console.error('Failed to fetch recent memories:', error);
            return [];
        }
    }

    async checkSystemHealth() {
        try {
            console.log('Checking system health...');

            // Wait for DOM to be ready if needed
            await this.waitForDOMReady();

            // Check if DOM elements are ready
            const modelDropdown = document.getElementById('modelDropdown');
            const memoryStatus = document.getElementById('memoryStatus');
            const apiStatus = document.getElementById('apiStatus');

            console.log('DOM elements found:', {
                modelDropdown: !!modelDropdown,
                memoryStatus: !!memoryStatus,
                apiStatus: !!apiStatus
            });

            if (!memoryStatus || !apiStatus) {
                console.warn('Critical DOM elements not found, retrying in 1 second...');
                setTimeout(() => this.checkSystemHealth(), 1000);
                return;
            }

            const response = await this.apiCall('/health', null, 'GET');
            console.log('Health response:', response);

            if (response.error) {
                this.updateSystemStatus('error', 'API Error');
                return;
            }

            // Load available models and update dropdown
            await this.loadModels();

            // Model status
            if (response.model && modelDropdown) {
                console.log('Setting model dropdown to:', response.model);
                modelDropdown.value = response.model;
            }

            // Memory status
            if (response.memory_status && memoryStatus) {
                console.log('Setting memory status to:', response.memory_status);
                memoryStatus.textContent = response.memory_status;
                memoryStatus.className = response.memory_status === 'ok' ? 'status-value online' : 'status-value offline';
            } else if (memoryStatus) {
                memoryStatus.textContent = 'Unknown';
                memoryStatus.className = 'status-value offline';
            }

            // API status
            if (apiStatus) {
                apiStatus.textContent = 'Online';
                apiStatus.className = 'status-value online';
            }

            console.log('System health updated successfully');

        } catch (error) {
            console.error('Health check error:', error);
            // Don't call updateSystemStatus here as it might also fail
            console.error('Error details:', error.message);
        }
    }

    async waitForDOMReady() {
        // Simple wait function for DOM elements
        let attempts = 0;
        const maxAttempts = 10;

        while (attempts < maxAttempts) {
            const memoryStatus = document.getElementById('memoryStatus');
            const apiStatus = document.getElementById('apiStatus');

            if (memoryStatus && apiStatus) {
                console.log('DOM elements are ready');
                return;
            }

            console.log(`Waiting for DOM elements... attempt ${attempts + 1}/${maxAttempts}`);
            await new Promise(resolve => setTimeout(resolve, 100));
            attempts++;
        }

        console.warn('DOM elements not found after maximum attempts');
    }

    updateSystemStatus(status, message) {
        try {
            const memoryStatus = document.getElementById('memoryStatus');
            const apiStatus = document.getElementById('apiStatus');

            if (memoryStatus) {
                memoryStatus.textContent = message;
            } else {
                console.warn('memoryStatus element not found');
            }

            if (apiStatus) {
                apiStatus.textContent = message;
                apiStatus.className = `status-value ${status}`;
            } else {
                console.warn('apiStatus element not found');
            }
        } catch (error) {
            console.error('Error in updateSystemStatus:', error);
        }
    }

    newChat() {
        // Clear chat history
        this.chatHistory = [];
        this.currentChatId = null;

        // Clear messages but keep welcome message
        const chatContainer = document.getElementById('chatContainer');
        chatContainer.innerHTML = '';

        // Show welcome message again
        this.showWelcomeMessage();
    }

    hideWelcomeMessage() {
        const welcomeMessage = document.querySelector('.welcome-message');
        if (welcomeMessage) {
            welcomeMessage.remove();
        }
    }

    showWelcomeMessage() {
        const chatContainer = document.getElementById('chatContainer');

        // Check if welcome message already exists
        if (chatContainer.querySelector('.welcome-message')) {
            return;
        }

        const welcomeDiv = document.createElement('div');
        welcomeDiv.className = 'welcome-message';
        welcomeDiv.innerHTML = `
            <div class="welcome-icon">
                <i class="fas fa-chart-line"></i>
            </div>
            <h2>Welcome to your Financial Assistant</h2>
            <p>I can help you with market analysis, trading strategies, risk management, and more. Ask me anything about financial markets!</p>

            <div class="example-prompts">
                <div class="prompt-example" onclick="app.sendPrompt('Analyze the current gold market trends')">
                    <i class="fas fa-search"></i>
                    <span>Analyze the current gold market trends</span>
                </div>
                <div class="prompt-example" onclick="app.sendPrompt('What are the key risk management principles?')">
                    <i class="fas fa-shield-alt"></i>
                    <span>What are the key risk management principles?</span>
                </div>
                <div class="prompt-example" onclick="app.sendPrompt('Explain forex correlation analysis')">
                    <i class="fas fa-link"></i>
                    <span>Explain forex correlation analysis</span>
                </div>
                <div class="prompt-example" onclick="app.sendPrompt('Best technical indicators for day trading')">
                    <i class="fas fa-chart-bar"></i>
                    <span>Best technical indicators for day trading</span>
                </div>
            </div>
        `;

        chatContainer.appendChild(welcomeDiv);
    }

    sendPrompt(prompt) {
        const messageInput = document.getElementById('messageInput');
        messageInput.value = prompt;
        this.updateCharCount();
        this.toggleSendButton();
        this.sendMessage();
    }

    toggleSidebar() {
        const sidebar = document.getElementById('sidebar');
        sidebar.classList.toggle('open');
    }

    toggleTheme() {
        this.theme = this.theme === 'light' ? 'dark' : 'light';
        document.documentElement.setAttribute('data-theme', this.theme);
        localStorage.setItem('theme', this.theme);

        const themeIcon = document.querySelector('#themeToggle i');
        themeIcon.className = this.theme === 'light' ? 'fas fa-moon' : 'fas fa-sun';
    }

    loadTheme() {
        document.documentElement.setAttribute('data-theme', this.theme);
        const themeIcon = document.querySelector('#themeToggle i');
        themeIcon.className = this.theme === 'light' ? 'fas fa-moon' : 'fas fa-sun';
    }

    showSettings() {
        this.showToast('Settings panel coming soon!', 'warning');
    }

    updateCharCount() {
        const messageInput = document.getElementById('messageInput');
        const charCount = document.getElementById('charCount');
        charCount.textContent = messageInput.value.length;
    }

    toggleSendButton() {
        const messageInput = document.getElementById('messageInput');
        const sendBtn = document.getElementById('sendBtn');
        sendBtn.disabled = !messageInput.value.trim() || this.isTyping;
    }

    autoResizeTextarea() {
        const textarea = document.getElementById('messageInput');
        textarea.style.height = 'auto';
        textarea.style.height = Math.min(textarea.scrollHeight, 120) + 'px';
    }

    showToast(message, type = 'success') {
        const toast = document.getElementById('toast');
        const toastMessage = document.getElementById('toastMessage');
        const toastIcon = toast.querySelector('i');

        toastMessage.textContent = message;
        toast.className = `toast ${type}`;

        // Update icon based on type
        if (type === 'error') {
            toastIcon.className = 'fas fa-exclamation-circle';
        } else if (type === 'warning') {
            toastIcon.className = 'fas fa-exclamation-triangle';
        } else {
            toastIcon.className = 'fas fa-check-circle';
        }

        toast.classList.add('show');

        setTimeout(() => {
            toast.classList.remove('show');
        }, 3000);
    }

    async apiCall(endpoint, data = null, method = 'POST') {
        const options = {
            method,
            headers: {
                'Content-Type': 'application/json',
            }
        };

        // Add authorization header if available
        const sessionToken = localStorage.getItem('session_token');
        if (sessionToken) {
            options.headers['Authorization'] = `Bearer ${sessionToken}`;
        }

        if (data) {
            options.body = JSON.stringify(data);
        }

        const response = await fetch(this.apiBaseUrl + endpoint, options);

        if (response.status === 401) {
            // Authentication failed, try to re-authenticate
            console.warn('Authentication failed, attempting to re-authenticate...');
            const reauthSuccess = await this.reauthenticate();
            if (reauthSuccess) {
                // Retry the original request with new token
                const newSessionToken = localStorage.getItem('session_token');
                if (newSessionToken) {
                    options.headers['Authorization'] = `Bearer ${newSessionToken}`;
                    const retryResponse = await fetch(this.apiBaseUrl + endpoint, options);
                    if (!retryResponse.ok) {
                        throw new Error(`HTTP error! status: ${retryResponse.status}`);
                    }
                    return await retryResponse.json();
                }
            }
            // If re-authentication fails, redirect to login
            window.location.href = '/login';
            throw new Error('Authentication failed');
        }

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        return await response.json();
    }

    showLoading() {
        document.getElementById('loadingOverlay').classList.add('show');
    }

    hideLoading() {
        document.getElementById('loadingOverlay').classList.remove('show');
    }

    async reauthenticate() {
        try {
            console.log('Attempting to re-authenticate...');
            const response = await fetch(this.apiBaseUrl + '/auth/login', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ password: 'admin123' })
            });

            if (response.ok) {
                const data = await response.json();
                if (data.success && data.session_token) {
                    localStorage.setItem('session_token', data.session_token);
                    console.log('Re-authentication successful');
                    return true;
                }
            }
            console.log('Re-authentication failed');
            return false;
        } catch (error) {
            console.error('Re-authentication error:', error);
            return false;
        }
    }

    async loadModels() {
        try {
            console.log('Loading models...');
            const response = await this.apiCall('/models', null, 'GET');
            const modelDropdown = document.getElementById('modelDropdown');

            console.log('Models response:', response);

            if (response.available_models && modelDropdown) {
                // Clear existing options
                modelDropdown.innerHTML = '';

                // Add available models as options
                response.available_models.forEach(model => {
                    const option = document.createElement('option');
                    option.value = model;
                    option.textContent = model;
                    if (model === response.current_model) {
                        option.selected = true;
                    }
                    modelDropdown.appendChild(option);
                });

                console.log(`Loaded ${response.available_models.length} models:`, response.available_models);
            } else {
                console.warn('No models available or dropdown not found:', {
                    available_models: !!response.available_models,
                    modelDropdown: !!modelDropdown
                });
            }
        } catch (error) {
            console.error('Failed to load models:', error);
            const modelDropdown = document.getElementById('modelDropdown');
            if (modelDropdown) {
                modelDropdown.innerHTML = '<option value="">Failed to load models</option>';
            }
        }
    }

    async switchModel() {
        const modelDropdown = document.getElementById('modelDropdown');
        const switchBtn = document.getElementById('modelSwitchBtn');

        if (!modelDropdown || !modelDropdown.value) {
            this.showToast('Please select a model', 'error');
            return;
        }

        const selectedModel = modelDropdown.value;

        // Show loading state
        if (switchBtn) {
            switchBtn.classList.add('loading');
            switchBtn.disabled = true;
        }

        try {
            const response = await this.apiCall('/models/switch', { model_name: selectedModel });

            if (response.success) {
                this.showToast(response.message, 'success');

                // Update button state
                if (switchBtn) {
                    switchBtn.classList.remove('loading');
                    switchBtn.classList.add('success');
                    setTimeout(() => {
                        switchBtn.classList.remove('success');
                    }, 2000);
                }

                console.log(`Successfully switched to model: ${selectedModel}`);
            } else {
                this.showToast(response.message, 'error');

                // Reset button state
                if (switchBtn) {
                    switchBtn.classList.remove('loading');
                    switchBtn.classList.add('error');
                    setTimeout(() => {
                        switchBtn.classList.remove('error');
                    }, 2000);
                }
            }
        } catch (error) {
            console.error('Failed to switch model:', error);
            this.showToast('Failed to switch model', 'error');

            // Reset button state
            if (switchBtn) {
                switchBtn.classList.remove('loading');
                switchBtn.classList.add('error');
                setTimeout(() => {
                    switchBtn.classList.remove('error');
                }, 2000);
            }
        } finally {
            // Re-enable button
            if (switchBtn) {
                switchBtn.disabled = false;
            }
        }
    }

    handleFileSelect(files) {
        if (!files || files.length === 0) return;

        const fileUploadArea = document.getElementById('fileUploadArea');
        const filePreviewContainer = document.getElementById('filePreviewContainer');
        const fileInfo = document.getElementById('fileInfo');
        const fileCount = document.getElementById('fileCount');

        // Show upload area
        fileUploadArea.style.display = 'block';

        // Process each file
        Array.from(files).forEach(file => {
            const fileId = Date.now() + '_' + Math.random().toString(36).substr(2, 9);

            // Create file object
            const fileObj = {
                id: fileId,
                file: file,
                name: file.name,
                size: file.size,
                type: file.type,
                lastModified: file.lastModified
            };

            // Add to attached files
            this.attachedFiles.push(fileObj);

            // Create preview element
            const preview = this.createFilePreview(fileObj);
            filePreviewContainer.appendChild(preview);
        });

        // Update file info
        this.updateFileInfo();

        // Clear file input
        document.getElementById('fileInput').value = '';
    }

    createFilePreview(fileObj) {
        const preview = document.createElement('div');
        preview.className = 'file-preview';
        preview.dataset.fileId = fileObj.id;

        if (fileObj.type.startsWith('image/')) {
            // Create image preview
            const img = document.createElement('img');
            img.className = 'file-preview-image';
            img.src = URL.createObjectURL(fileObj.file);
            img.onload = () => URL.revokeObjectURL(img.src); // Clean up memory
            preview.appendChild(img);
        } else {
            // Create file icon
            const icon = document.createElement('div');
            icon.className = 'file-preview-icon';
            icon.innerHTML = this.getFileIcon(fileObj.type);
            icon.style.fontSize = '24px';
            icon.style.color = '#666';
            preview.appendChild(icon);
        }

        // Add file info
        const info = document.createElement('div');
        info.className = 'file-preview-info';

        const name = document.createElement('div');
        name.className = 'file-preview-name';
        name.textContent = fileObj.name;
        info.appendChild(name);

        const size = document.createElement('div');
        size.className = 'file-preview-size';
        size.textContent = this.formatFileSize(fileObj.size);
        info.appendChild(size);

        preview.appendChild(info);

        // Add remove button
        const removeBtn = document.createElement('button');
        removeBtn.className = 'file-preview-remove';
        removeBtn.innerHTML = '×';
        removeBtn.onclick = () => this.removeFile(fileObj.id);
        preview.appendChild(removeBtn);

        return preview;
    }

    getFileIcon(fileType) {
        if (fileType.includes('pdf')) return '📄';
        if (fileType.includes('text')) return '📝';
        if (fileType.includes('csv') || fileType.includes('excel')) return '📊';
        if (fileType.includes('word') || fileType.includes('document')) return '📄';
        if (fileType.includes('image')) return '🖼️';
        if (fileType.includes('zip') || fileType.includes('rar')) return '📦';
        return '📎';
    }

    formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }

    removeFile(fileId) {
        // Remove from attached files
        this.attachedFiles = this.attachedFiles.filter(file => file.id !== fileId);

        // Remove preview element
        const preview = document.querySelector(`[data-file-id="${fileId}"]`);
        if (preview) {
            preview.remove();
        }

        // Update file info
        this.updateFileInfo();

        // Hide upload area if no files
        if (this.attachedFiles.length === 0) {
            document.getElementById('fileUploadArea').style.display = 'none';
        }
    }

    updateFileInfo() {
        const fileInfo = document.getElementById('fileInfo');
        const fileCount = document.getElementById('fileCount');

        if (this.attachedFiles.length > 0) {
            fileInfo.style.display = 'flex';
            fileCount.textContent = this.attachedFiles.length;
        } else {
            fileInfo.style.display = 'none';
        }
    }

    async uploadFiles() {
        if (this.attachedFiles.length === 0) return [];

        const uploadedFiles = [];

        for (const fileObj of this.attachedFiles) {
            try {
                const formData = new FormData();
                formData.append('file', fileObj.file);

                const response = await fetch(`${this.apiBaseUrl}/api/upload`, {
                    method: 'POST',
                    headers: {
                        'Authorization': `Bearer ${localStorage.getItem('session_token')}`
                    },
                    body: formData
                });

                if (response.ok) {
                    const result = await response.json();
                    uploadedFiles.push({
                        name: fileObj.name,
                        type: fileObj.type,
                        size: fileObj.size,
                        url: result.url,
                        content: result.content,
                        file_id: result.file_id
                    });
                } else {
                    console.error(`Failed to upload ${fileObj.name}`);
                }
            } catch (error) {
                console.error(`Error uploading ${fileObj.name}:`, error);
            }
        }

        return uploadedFiles;
    }

    clearFiles() {
        // Clear attached files
        this.attachedFiles = [];

        // Clear preview container
        const filePreviewContainer = document.getElementById('filePreviewContainer');
        if (filePreviewContainer) {
            filePreviewContainer.innerHTML = '';
        }

        // Hide upload area
        document.getElementById('fileUploadArea').style.display = 'none';

        // Update file info
        this.updateFileInfo();
    }
}

// Initialize the app when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.app = new FinancialAssistantApp();
});

// Global function for prompt examples
function sendPrompt(prompt) {
    if (window.app) {
        window.app.sendPrompt(prompt);
    }
}