# Web UI Setup and Usage Guide

## 🌐 Overview

Your Local Financial Assistant now includes a modern, ChatGPT-like web interface that provides an intuitive way to interact with your AI assistant. The web UI features real-time chat, memory management, system monitoring, and responsive design for both desktop and mobile devices.

## ✨ Key Features

### 💬 **Chat Interface**
- Modern, ChatGPT-like conversation interface
- Real-time message streaming with typing indicators
- Example prompts for quick start
- Message history with timestamps
- Memory context indicators

### 🧠 **Memory Management**
- Store important trading rules and information
- Organize memories by categories (trading, strategy, risk management)
- View recent memories in the sidebar
- Persistent storage using ChromaDB

### 📊 **System Monitoring**
- Real-time system status display
- Model information and memory system health
- API connectivity status
- Service health checks

### 🎨 **User Experience**
- Light/Dark theme toggle
- Responsive design for mobile and desktop
- Smooth animations and transitions
- Keyboard shortcuts (Enter to send, Shift+Enter for new line)
- Character counting for messages

## 🚀 Quick Start

### Prerequisites
- All backend services running (vLLM + FastAPI)
- Modern web browser (Chrome, Firefox, Safari, Edge)

### Access the Web UI

1. **Start the services:**
   ```bash
   ./run.sh start
   ```

2. **Open your browser and navigate to:**
   ```
   http://localhost:8080
   ```

3. **Alternative access points:**
   - Main UI: `http://localhost:8080/`
   - API Docs: `http://localhost:8080/docs`
   - Health Check: `http://localhost:8080/health`

## 🖥️ Interface Tour

### Main Components

1. **Sidebar (Left Panel)**
   - **New Chat Button**: Start a fresh conversation
   - **System Status**: Monitor model, memory, and API health
   - **Memory Management**: Store and view important information
   - **Recent Memories**: Quick access to stored information

2. **Chat Area (Center)**
   - **Welcome Screen**: Example prompts and getting started info
   - **Messages**: Conversation history with user and assistant messages
   - **Typing Indicators**: Visual feedback when AI is responding

3. **Input Area (Bottom)**
   - **Message Input**: Type your questions and commands
   - **Send Button**: Submit messages (or press Enter)
   - **Character Count**: Track message length (max 4000 characters)

4. **Header (Top)**
   - **Menu Toggle**: Show/hide sidebar on mobile
   - **Theme Toggle**: Switch between light and dark themes
   - **Settings Button**: Access configuration options (coming soon)

## 🎯 Using the Web UI

### Basic Chat
1. Type your question in the input field
2. Press Enter or click the send button
3. View the AI response in the chat area
4. Continue the conversation naturally

### Example Prompts
Click on any example prompt in the welcome screen:
- "Analyze the current gold market trends"
- "What are the key risk management principles?"
- "Explain forex correlation analysis"
- "Best technical indicators for day trading"

### Memory Management

#### Storing Information
1. **Fill in the memory form** in the sidebar:
   - **Key**: Short identifier (e.g., "risk_management")
   - **Value**: Detailed information (e.g., "Never risk more than 2% per trade")
   - **Category**: Choose from General, Trading, Strategy, or Risk Management

2. **Click "Store Memory"** to save

3. **Confirmation**: Success message appears and memories list updates

#### Viewing Memories
- Recent memories appear in the sidebar automatically
- Click the refresh icon to update the list
- Memories show key, value, category, and timestamp

### System Monitoring
The system status panel shows:
- **Model**: Currently loaded AI model (Phi-3 Mini)
- **Memory**: ChromaDB memory system status
- **API**: Connection status to backend services

### Theme Customization
1. Click the moon/sun icon in the header
2. Choose between Light and Dark themes
3. Preference is saved automatically for future visits

## 📱 Mobile Usage

### Responsive Design
- The interface automatically adapts to mobile screens
- Sidebar becomes a slide-out menu
- Touch-friendly buttons and inputs

### Mobile Navigation
1. **Menu Button**: Tap the hamburger icon (☰) to open/close sidebar
2. **Sending Messages**: Use the send button or tap "Go" on keyboard
3. **Scrolling**: Swipe up/down to browse conversation history

## 🔧 Advanced Features

### Keyboard Shortcuts
- **Enter**: Send message
- **Shift + Enter**: New line in message input
- **Escape**: Close sidebar (mobile)

### Message Features
- **Memory Context**: Shows when AI uses stored memory in responses
- **Timestamps**: Each message includes time information
- **Character Limit**: 4000 characters per message with live counter

### Error Handling
- **Connection Errors**: Clear error messages if services are unavailable
- **Input Validation**: Helpful warnings for invalid inputs
- **Loading States**: Visual feedback during processing

## 🛠️ Troubleshooting

### Common Issues

#### "Failed to connect to the server"
- **Solution**: Ensure vLLM and FastAPI services are running
- **Check**: Run `./run.sh status` to verify services

#### "Web UI template not found"
- **Solution**: Ensure all files are in the correct directories
- **Check**: Verify `templates/index.html` and `static/` files exist

#### Memory not saving
- **Solution**: Check ChromaDB connection and permissions
- **Check**: Review `chroma_db/` directory permissions

#### Slow response times
- **Solution**: Check GPU availability and model loading
- **Check**: Monitor system resources and vLLM logs

### Debug Information

#### Browser Console
1. Press F12 to open developer tools
2. Check Console tab for JavaScript errors
3. Look at Network tab for failed API requests

#### Server Logs
```bash
# View FastAPI logs
tail -f fastapi.log

# View vLLM logs
tail -f vllm.log
```

#### Health Check
```bash
curl http://localhost:8080/health
```

## 🔄 API Integration

The web UI uses the same API endpoints as the programmatic interface:

### Chat Endpoint
```javascript
POST /chat
{
  "message": "Analyze gold market trends",
  "model": "phi3",
  "memory_context": 3
}
```

### Memory Endpoint
```javascript
POST /memorize
{
  "key": "trading_rule",
  "value": "Never trade during news",
  "category": "trading"
}
```

### Memories Endpoint
```javascript
GET /memories?n=10
```

## 🎨 Customization

### Theming
- Edit `static/css/style.css` for custom colors and styles
- Modify CSS variables in the `:root` selector
- Add custom themes by extending the theme system

### Adding Features
- JavaScript logic in `static/js/app.js`
- HTML structure in `templates/index.html`
- API endpoints in `main.py`

### Branding
- Update logo and text in the sidebar header
- Modify welcome message content
- Customize example prompts for your use case

## 📊 Performance Tips

### Browser Optimization
- Use modern browsers for best performance
- Clear cache if experiencing issues
- Disable unnecessary browser extensions

### System Performance
- Ensure adequate RAM (16GB+ recommended)
- Use GPU acceleration for vLLM when available
- Monitor system resource usage

### Network Performance
- Use localhost for fastest response times
- Check network latency if accessing remotely
- Consider SSL setup for production use

## 🔒 Security Considerations

### Local Development
- Interface is designed for local use only
- No authentication required for local access
- Consider firewall rules for network access

### Production Deployment
- Add authentication mechanisms
- Implement rate limiting
- Use HTTPS/SSL certificates
- Configure CORS appropriately

## 📱 Browser Compatibility

### Supported Browsers
- ✅ Chrome 90+
- ✅ Firefox 88+
- ✅ Safari 14+
- ✅ Edge 90+

### Required Features
- JavaScript ES6+
- CSS Grid and Flexbox
- Fetch API
- Local Storage

## 🆕 Upcoming Features

### Planned Enhancements
- **Conversation History**: Save and load chat sessions
- **Export Functionality**: Download conversations and memories
- **Advanced Settings**: Configure model parameters
- **File Upload**: Analyze documents and images
- **Voice Input**: Speech-to-text functionality
- **Multiple Models**: Switch between different AI models

### UI Improvements
- **Customizable Layout**: Resize panels and arrange workspace
- **Keyboard Shortcuts**: More hotkeys for power users
- **Search Functionality**: Find information in chat history
- **Markdown Support**: Rich text formatting in responses

## 📞 Support

### Getting Help
1. **Check this guide** for common solutions
2. **Review the logs** for error messages
3. **Test the API** directly using curl or test_client.py
4. **Check GitHub issues** for known problems

### Contributing
- Report bugs via GitHub issues
- Suggest features with detailed descriptions
- Submit pull requests for improvements
- Share feedback and user experience

---

## 🎉 Enjoy Your Web UI!

You now have a powerful, modern web interface for your Local Financial Assistant. The interface combines the sophistication of ChatGPT with specialized financial analysis capabilities, all running locally on your machine.

Start exploring the features, store important trading knowledge, and enjoy the seamless conversation experience with your AI financial assistant!