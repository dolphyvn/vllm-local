// Main application JavaScript for Financial Assistant UI
class FinancialAssistantApp {
    constructor() {
        this.apiBaseUrl = window.location.origin;
        this.currentChatId = null;
        this.chatHistory = [];
        this.isTyping = false;
        this.theme = localStorage.getItem('theme') || 'light';

        this.init();
    }

    init() {
        this.setupEventListeners();
        this.loadTheme();
        this.checkSystemHealth();
        this.loadRecentMemories();
        this.autoResizeTextarea();
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
    }

    async sendMessage() {
        const messageInput = document.getElementById('messageInput');
        const message = messageInput.value.trim();

        if (!message || this.isTyping) return;

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
            // Create assistant message that will be updated with streaming content
            const messageId = this.addStreamingMessage('assistant', '');

            const response = await fetch(`${this.apiBaseUrl}/chat/stream`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    message: message,
                    model: 'phi3',
                    memory_context: 3
                })
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

                            if (parsed.choices && parsed.choices[0]) {
                                const delta = parsed.choices[0].delta;

                                if (delta.content) {
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
            const response = await this.apiCall('/health', null, 'GET');
            console.log('Health response:', response);

            if (response.error) {
                this.updateSystemStatus('error', 'API Error');
                return;
            }

            // Update individual status fields
            const modelStatus = document.getElementById('modelStatus');
            const memoryStatus = document.getElementById('memoryStatus');
            const apiStatus = document.getElementById('apiStatus');

            // Model status
            if (response.model) {
                modelStatus.textContent = response.model;
                modelStatus.className = 'status-value online';
            } else {
                modelStatus.textContent = 'Unknown';
                modelStatus.className = 'status-value offline';
            }

            // Memory status
            if (response.memory_status) {
                memoryStatus.textContent = response.memory_status;
                memoryStatus.className = response.memory_status === 'ok' ? 'status-value online' : 'status-value offline';
            } else {
                memoryStatus.textContent = 'Unknown';
                memoryStatus.className = 'status-value offline';
            }

            // API status
            apiStatus.textContent = 'Online';
            apiStatus.className = 'status-value online';

            console.log('System health updated successfully');

        } catch (error) {
            console.error('Health check error:', error);
            this.updateSystemStatus('error', 'Offline');
        }
    }

    updateSystemStatus(status, message) {
        document.getElementById('modelStatus').textContent = message;
        document.getElementById('memoryStatus').textContent = message;

        const apiStatus = document.getElementById('apiStatus');
        apiStatus.textContent = message;
        apiStatus.className = `status-value ${status}`;
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

        if (data) {
            options.body = JSON.stringify(data);
        }

        const response = await fetch(this.apiBaseUrl + endpoint, options);

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