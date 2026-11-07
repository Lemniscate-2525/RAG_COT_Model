/**
 * Puter AI Chat Application
 * Built with Puter.js — Serverless AI Chat without API Keys
 * https://developer.puter.com
 */

// Application State
const state = {
    currentSessionId: null,
    sessions: [],
    settings: {
        model: 'gpt-4o-mini',
        systemPrompt: 'You are an intelligent, helpful, and concise AI assistant powered by Puter.js.',
        temperature: 0.7,
        maxTokens: 2000,
    },
    isGenerating: false,
    abortController: null,
    attachedFile: null, // { name, type, dataUrl, puterPath }
    currentUser: null,
};

// Storage keys
const STORAGE_KEY_SESSIONS = 'puter_ai_chat_sessions';
const STORAGE_KEY_SETTINGS = 'puter_ai_chat_settings';

// DOM Elements
const elements = {
    sidebar: document.getElementById('sidebar'),
    sidebarToggleBtn: document.getElementById('sidebarToggleBtn'),
    newChatBtn: document.getElementById('newChatBtn'),
    sessionsList: document.getElementById('sessionsList'),
    modelSelect: document.getElementById('modelSelect'),
    mobileModelSelect: document.getElementById('mobileModelSelect'),
    authBtn: document.getElementById('authBtn'),
    authBtnText: document.getElementById('authBtnText'),
    cloudSyncIndicator: document.getElementById('cloudSyncIndicator'),
    cloudSyncText: document.getElementById('cloudSyncText'),
    clearAllBtn: document.getElementById('clearAllBtn'),
    messagesContainer: document.getElementById('messagesContainer'),
    emptyState: document.getElementById('emptyState'),
    scrollToBottomBtn: document.getElementById('scrollToBottomBtn'),
    chatForm: document.getElementById('chatForm'),
    promptInput: document.getElementById('promptInput'),
    sendBtn: document.getElementById('sendBtn'),
    stopBtn: document.getElementById('stopBtn'),
    attachBtn: document.getElementById('attachBtn'),
    fileInput: document.getElementById('fileInput'),
    attachmentPreview: document.getElementById('attachmentPreview'),
    attachmentThumb: document.getElementById('attachmentThumb'),
    attachmentFileIcon: document.getElementById('attachmentFileIcon'),
    attachmentName: document.getElementById('attachmentName'),
    removeAttachmentBtn: document.getElementById('removeAttachmentBtn'),
    typingStatus: document.getElementById('typingStatus'),
    settingsBtn: document.getElementById('settingsBtn'),
    settingsModal: document.getElementById('settingsModal'),
    closeSettingsBtn: document.getElementById('closeSettingsBtn'),
    saveSettingsBtn: document.getElementById('saveSettingsBtn'),
    systemPromptInput: document.getElementById('systemPromptInput'),
    tempSlider: document.getElementById('tempSlider'),
    tempValue: document.getElementById('tempValue'),
    maxTokensInput: document.getElementById('maxTokensInput'),
};

// Initialize Application
document.addEventListener('DOMContentLoaded', async () => {
    initMarkdown();
    setupEventListeners();
    await checkAuthStatus();
    await loadSettings();
    await loadSessions();

    if (state.sessions.length === 0) {
        createNewSession();
    } else {
        switchSession(state.sessions[0].id);
    }
});

// Configure Marked & Highlight.js
function initMarkdown() {
    marked.setOptions({
        highlight: function(code, lang) {
            const language = highlight.getLanguage(lang) ? lang : 'plaintext';
            return highlight.highlight(code, { language }).value;
        },
        breaks: true,
        gfm: true
    });
}

// Check Puter Auth Status
async function checkAuthStatus() {
    try {
        if (typeof puter !== 'undefined' && puter.auth) {
            const isSignedIn = puter.auth.isSignedIn();
            if (isSignedIn) {
                const user = await puter.auth.getUser();
                state.currentUser = user;
                elements.authBtnText.textContent = user.username || 'Signed In';
                elements.cloudSyncIndicator.className = 'w-2 h-2 rounded-full bg-emerald-400 inline-block';
                elements.cloudSyncText.textContent = 'Puter Cloud Sync';
                return;
            }
        }
    } catch (e) {
        console.warn('Puter Auth Check Notice:', e);
    }
    elements.authBtnText.textContent = 'Sign In with Puter';
    elements.cloudSyncIndicator.className = 'w-2 h-2 rounded-full bg-amber-400 inline-block';
    elements.cloudSyncText.textContent = 'Local Guest Mode';
}

// Authentication Toggle
async function handleAuthClick() {
    try {
        if (state.currentUser) {
            if (confirm(`Signed in as ${state.currentUser.username}. Do you want to sign out?`)) {
                await puter.auth.signOut();
                state.currentUser = null;
                elements.authBtnText.textContent = 'Sign In with Puter';
                elements.cloudSyncIndicator.className = 'w-2 h-2 rounded-full bg-amber-400 inline-block';
                elements.cloudSyncText.textContent = 'Local Guest Mode';
            }
        } else {
            await puter.auth.signIn();
            await checkAuthStatus();
            // Sync sessions to Puter KV store upon login
            await saveSessions();
        }
    } catch (e) {
        console.error('Auth error:', e);
    }
}

// Load & Save Settings (Using Puter KV when available, localStorage fallback)
async function loadSettings() {
    try {
        let saved = null;
        if (state.currentUser && puter.kv) {
            saved = await puter.kv.get(STORAGE_KEY_SETTINGS);
        }
        if (!saved) {
            saved = localStorage.getItem(STORAGE_KEY_SETTINGS);
        }
        if (saved) {
            const parsed = typeof saved === 'string' ? JSON.parse(saved) : saved;
            state.settings = { ...state.settings, ...parsed };
        }
    } catch (e) {
        console.warn('Could not load settings:', e);
    }

    elements.modelSelect.value = state.settings.model || 'gpt-4o-mini';
    elements.mobileModelSelect.value = state.settings.model || 'gpt-4o-mini';
    elements.systemPromptInput.value = state.settings.systemPrompt || '';
    elements.tempSlider.value = state.settings.temperature ?? 0.7;
    elements.tempValue.textContent = elements.tempSlider.value;
    elements.maxTokensInput.value = state.settings.maxTokens || 2000;
}

async function saveSettings() {
    state.settings.model = elements.modelSelect.value;
    state.settings.systemPrompt = elements.systemPromptInput.value.trim();
    state.settings.temperature = parseFloat(elements.tempSlider.value);
    state.settings.maxTokens = parseInt(elements.maxTokensInput.value, 10) || 2000;

    localStorage.setItem(STORAGE_KEY_SETTINGS, JSON.stringify(state.settings));
    if (state.currentUser && puter.kv) {
        try {
            await puter.kv.set(STORAGE_KEY_SETTINGS, JSON.stringify(state.settings));
        } catch (e) {
            console.warn('Failed saving settings to Puter KV:', e);
        }
    }
    closeSettings();
}

// Session Management (Puter KV Store + LocalStorage fallback)
async function loadSessions() {
    try {
        let data = null;
        if (state.currentUser && puter.kv) {
            data = await puter.kv.get(STORAGE_KEY_SESSIONS);
        }
        if (!data) {
            data = localStorage.getItem(STORAGE_KEY_SESSIONS);
        }
        if (data) {
            state.sessions = typeof data === 'string' ? JSON.parse(data) : data;
        }
    } catch (e) {
        console.warn('Could not load sessions:', e);
        state.sessions = [];
    }
    renderSessionsList();
}

async function saveSessions() {
    localStorage.setItem(STORAGE_KEY_SESSIONS, JSON.stringify(state.sessions));
    if (state.currentUser && puter.kv) {
        try {
            await puter.kv.set(STORAGE_KEY_SESSIONS, JSON.stringify(state.sessions));
        } catch (e) {
            console.warn('Failed saving sessions to Puter KV:', e);
        }
    }
    renderSessionsList();
}

function createNewSession() {
    const newSession = {
        id: 'session_' + Date.now(),
        title: 'New Conversation',
        model: state.settings.model,
        createdAt: new Date().toISOString(),
        messages: []
    };
    state.sessions.unshift(newSession);
    saveSessions();
    switchSession(newSession.id);
}

function switchSession(sessionId) {
    state.currentSessionId = sessionId;
    const session = state.sessions.find(s => s.id === sessionId);
    if (!session) return;

    // Update model dropdown to match session's preferred model if specified
    if (session.model) {
        elements.modelSelect.value = session.model;
        elements.mobileModelSelect.value = session.model;
    }

    renderSessionsList();
    renderMessages();

    // Close sidebar on mobile upon switching
    if (window.innerWidth < 768) {
        elements.sidebar.classList.add('-translate-x-full');
    }
}

function deleteSession(sessionId, event) {
    if (event) event.stopPropagation();
    state.sessions = state.sessions.filter(s => s.id !== sessionId);
    saveSessions();
    if (state.currentSessionId === sessionId) {
        if (state.sessions.length > 0) {
            switchSession(state.sessions[0].id);
        } else {
            createNewSession();
        }
    }
}

function clearAllSessions() {
    if (confirm('Are you sure you want to delete all chat history?')) {
        state.sessions = [];
        saveSessions();
        createNewSession();
    }
}

// Render Sidebar Sessions
function renderSessionsList() {
    elements.sessionsList.innerHTML = '';
    state.sessions.forEach(session => {
        const isActive = session.id === state.currentSessionId;
        const div = document.createElement('div');
        div.className = `group flex items-center justify-between p-2.5 rounded-xl cursor-pointer text-xs transition ${
            isActive 
                ? 'bg-sky-500/10 text-sky-400 font-medium border border-sky-500/20' 
                : 'text-slate-300 hover:bg-slate-900 border border-transparent'
        }`;
        div.onclick = () => switchSession(session.id);

        const left = document.createElement('div');
        left.className = 'flex items-center gap-2 truncate flex-1 mr-2';
        left.innerHTML = `
            <i class="fa-regular fa-message text-[11px] ${isActive ? 'text-sky-400' : 'text-slate-500'}"></i>
            <span class="truncate">${escapeHtml(session.title)}</span>
        `;

        const deleteBtn = document.createElement('button');
        deleteBtn.className = 'text-slate-500 hover:text-red-400 opacity-0 group-hover:opacity-100 transition px-1';
        deleteBtn.title = 'Delete Chat';
        deleteBtn.innerHTML = '<i class="fa-solid fa-trash-can text-[10px]"></i>';
        deleteBtn.onclick = (e) => deleteSession(session.id, e);

        div.appendChild(left);
        div.appendChild(deleteBtn);
        elements.sessionsList.appendChild(div);
    });
}

// Render Messages
function renderMessages() {
    const currentSession = state.sessions.find(s => s.id === state.currentSessionId);
    if (!currentSession || currentSession.messages.length === 0) {
        elements.emptyState.classList.remove('hidden');
        elements.messagesContainer.querySelectorAll('.message-row').forEach(el => el.remove());
        return;
    }

    elements.emptyState.classList.add('hidden');
    elements.messagesContainer.querySelectorAll('.message-row').forEach(el => el.remove());

    currentSession.messages.forEach(msg => {
        appendMessageElement(msg.role, msg.content, msg.attachment, false);
    });

    scrollToBottom();
}

function appendMessageElement(role, content, attachment = null, isStreaming = false) {
    const isUser = role === 'user';
    const row = document.createElement('div');
    row.className = `message-row flex gap-3 max-w-3xl mx-auto w-full ${isUser ? 'justify-end' : 'justify-start'}`;

    // Avatar
    const avatar = document.createElement('div');
    avatar.className = `w-7 h-7 rounded-lg shrink-0 flex items-center justify-center text-xs shadow-md ${
        isUser 
            ? 'bg-sky-600 text-white order-2' 
            : 'bg-gradient-to-tr from-sky-500 to-indigo-600 text-white order-1'
    }`;
    avatar.innerHTML = isUser ? '<i class="fa-solid fa-user"></i>' : '<i class="fa-solid fa-wand-magic-sparkles"></i>';

    // Content Bubble
    const bubble = document.createElement('div');
    bubble.className = `rounded-2xl px-4 py-3 text-sm leading-relaxed overflow-hidden relative shadow-md ${
        isUser 
            ? 'bg-sky-600 text-white order-1 rounded-tr-sm max-w-[85%]' 
            : 'bg-slate-800/90 border border-slate-700/80 text-slate-100 order-2 rounded-tl-sm w-full'
    }`;

    // Render attachment preview if attached
    if (attachment) {
        const attachBox = document.createElement('div');
        attachBox.className = 'mb-2.5 p-2 bg-slate-950/40 rounded-xl flex items-center gap-2 border border-white/10 text-xs';
        if (attachment.type.startsWith('image/')) {
            attachBox.innerHTML = `
                <img src="${attachment.dataUrl}" class="w-10 h-10 object-cover rounded-lg border border-white/20">
                <span class="font-medium truncate">${escapeHtml(attachment.name)}</span>
            `;
        } else {
            attachBox.innerHTML = `
                <i class="fa-solid fa-file text-sky-400 ml-1"></i>
                <span class="font-medium truncate">${escapeHtml(attachment.name)}</span>
            `;
        }
        bubble.appendChild(attachBox);
    }

    const textDiv = document.createElement('div');
    textDiv.className = isUser ? 'whitespace-pre-wrap' : 'prose-chat';
    if (isUser) {
        textDiv.textContent = content;
    } else {
        textDiv.innerHTML = renderMarkdown(content);
        if (isStreaming) {
            textDiv.classList.add('streaming-cursor');
        }
    }
    bubble.appendChild(textDiv);

    // Assistant toolbar (Copy button & model badge)
    if (!isUser && !isStreaming) {
        const toolbar = document.createElement('div');
        toolbar.className = 'mt-2 pt-2 border-t border-slate-700/50 flex items-center justify-between text-[11px] text-slate-400';
        toolbar.innerHTML = `
            <span class="flex items-center gap-1 font-mono text-[10px] text-slate-400">
                <i class="fa-solid fa-microchip text-[9px] text-sky-400"></i> ${escapeHtml(elements.modelSelect.value)}
            </span>
            <button class="copy-btn hover:text-white transition flex items-center gap-1">
                <i class="fa-regular fa-copy"></i> <span>Copy</span>
            </button>
        `;
        toolbar.querySelector('.copy-btn').onclick = () => copyText(content, toolbar.querySelector('.copy-btn'));
        bubble.appendChild(toolbar);
    }

    row.appendChild(avatar);
    row.appendChild(bubble);
    elements.messagesContainer.appendChild(row);

    addCodeCopyButtons(bubble);
    return { row, bubble, textDiv };
}

// Send Message Handler
async function handleChatSubmit(e) {
    if (e) e.preventDefault();
    if (state.isGenerating) return;

    const rawPrompt = elements.promptInput.value.trim();
    if (!rawPrompt && !state.attachedFile) return;

    const currentSession = state.sessions.find(s => s.id === state.currentSessionId);
    if (!currentSession) return;

    const userMessageContent = rawPrompt || (state.attachedFile ? `[Uploaded file: ${state.attachedFile.name}]` : '');
    const attachmentCopy = state.attachedFile ? { ...state.attachedFile } : null;

    // Reset input immediately
    elements.promptInput.value = '';
    elements.promptInput.style.height = 'auto';
    clearAttachment();

    // Auto-name session on first user message
    if (currentSession.messages.length === 0) {
        currentSession.title = userMessageContent.slice(0, 32) + (userMessageContent.length > 32 ? '...' : '');
    }

    // Add user message to session
    currentSession.messages.push({
        role: 'user',
        content: userMessageContent,
        attachment: attachmentCopy
    });

    elements.emptyState.classList.add('hidden');
    appendMessageElement('user', userMessageContent, attachmentCopy, false);
    scrollToBottom();

    // Prepare assistant response streaming
    state.isGenerating = true;
    setUIGenerating(true);
    state.abortController = new AbortController();

    const selectedModel = elements.modelSelect.value;
    currentSession.model = selectedModel;

    // Create streaming assistant message bubble
    const { bubble, textDiv } = appendMessageElement('assistant', '', null, true);
    let fullResponseText = '';

    try {
        // Construct message payload
        const messagesPayload = [];
        if (state.settings.systemPrompt) {
            messagesPayload.push({
                role: 'system',
                content: state.settings.systemPrompt
            });
        }

        // Build history
        for (const msg of currentSession.messages) {
            if (msg.attachment && msg.attachment.type.startsWith('image/')) {
                // Multimodal image support
                messagesPayload.push({
                    role: msg.role,
                    content: [
                        { type: 'text', text: msg.content },
                        { type: 'image_url', image_url: { url: msg.attachment.dataUrl } }
                    ]
                });
            } else {
                messagesPayload.push({
                    role: msg.role,
                    content: msg.content
                });
            }
        }

        // Call Puter.js AI Chat with Streaming
        const completion = await puter.ai.chat(messagesPayload, {
            model: selectedModel,
            stream: true,
            temperature: state.settings.temperature,
            max_tokens: state.settings.maxTokens,
            signal: state.abortController.signal
        });

        // Read stream chunks
        for await (const part of completion) {
            if (part?.text) {
                fullResponseText += part.text;
                textDiv.innerHTML = renderMarkdown(fullResponseText);
                addCodeCopyButtons(textDiv);
                scrollToBottom();
            }
        }

    } catch (err) {
        if (err.name === 'AbortError') {
            fullResponseText += '\n\n*(Generation stopped by user)*';
        } else {
            console.error('Puter AI error:', err);
            fullResponseText += `\n\n> ⚠️ **Error:** ${err.message || 'Failed to complete request with Puter AI.'}`;
        }
        textDiv.innerHTML = renderMarkdown(fullResponseText);
    } finally {
        textDiv.classList.remove('streaming-cursor');
        state.isGenerating = false;
        state.abortController = null;
        setUIGenerating(false);

        // Append assistant toolbar
        const toolbar = document.createElement('div');
        toolbar.className = 'mt-2 pt-2 border-t border-slate-700/50 flex items-center justify-between text-[11px] text-slate-400';
        toolbar.innerHTML = `
            <span class="flex items-center gap-1 font-mono text-[10px] text-slate-400">
                <i class="fa-solid fa-microchip text-[9px] text-sky-400"></i> ${escapeHtml(selectedModel)}
            </span>
            <button class="copy-btn hover:text-white transition flex items-center gap-1">
                <i class="fa-regular fa-copy"></i> <span>Copy</span>
            </button>
        `;
        toolbar.querySelector('.copy-btn').onclick = () => copyText(fullResponseText, toolbar.querySelector('.copy-btn'));
        bubble.appendChild(toolbar);

        // Save assistant response
        currentSession.messages.push({
            role: 'assistant',
            content: fullResponseText
        });
        await saveSessions();
    }
}

// Stop Generation Handler
function handleStopGeneration() {
    if (state.abortController) {
        state.abortController.abort();
    }
}

// File Attachment Handling
function handleFileSelect(e) {
    const file = e.target.files[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = function(event) {
        state.attachedFile = {
            name: file.name,
            type: file.type,
            size: file.size,
            dataUrl: event.target.result
        };

        elements.attachmentName.textContent = file.name;
        if (file.type.startsWith('image/')) {
            elements.attachmentThumb.src = event.target.result;
            elements.attachmentThumb.classList.remove('hidden');
            elements.attachmentFileIcon.classList.add('hidden');
        } else {
            elements.attachmentThumb.classList.add('hidden');
            elements.attachmentFileIcon.classList.remove('hidden');
        }
        elements.attachmentPreview.classList.remove('hidden');
    };
    reader.readAsDataURL(file);
}

function clearAttachment() {
    state.attachedFile = null;
    elements.fileInput.value = '';
    elements.attachmentPreview.classList.add('hidden');
    elements.attachmentThumb.src = '';
}

// UI State Toggles
function setUIGenerating(generating) {
    if (generating) {
        elements.sendBtn.classList.add('hidden');
        elements.stopBtn.classList.remove('hidden');
        elements.typingStatus.classList.remove('hidden');
        elements.promptInput.disabled = true;
    } else {
        elements.sendBtn.classList.remove('hidden');
        elements.stopBtn.classList.add('hidden');
        elements.typingStatus.classList.add('hidden');
        elements.promptInput.disabled = false;
        elements.promptInput.focus();
    }
}

// Settings Modal Handlers
function openSettings() {
    elements.settingsModal.classList.remove('hidden');
}

function closeSettings() {
    elements.settingsModal.classList.add('hidden');
}

// Markdown & Code Copy Helpers
function renderMarkdown(text) {
    if (!text) return '';
    try {
        return marked.parse(text);
    } catch (e) {
        return escapeHtml(text);
    }
}

function addCodeCopyButtons(container) {
    const preBlocks = container.querySelectorAll('pre:not([data-copy-bound])');
    preBlocks.forEach(pre => {
        pre.setAttribute('data-copy-bound', 'true');
        const wrapper = document.createElement('div');
        wrapper.className = 'code-block-wrapper';
        pre.parentNode.insertBefore(wrapper, pre);
        wrapper.appendChild(pre);

        const btn = document.createElement('button');
        btn.className = 'code-copy-btn';
        btn.innerHTML = '<i class="fa-regular fa-copy"></i> <span>Copy</span>';
        btn.onclick = () => {
            const code = pre.querySelector('code')?.innerText || pre.innerText;
            copyText(code, btn);
        };
        wrapper.appendChild(btn);
    });
}

function copyText(text, btnElement) {
    navigator.clipboard.writeText(text).then(() => {
        const originalHtml = btnElement.innerHTML;
        btnElement.innerHTML = '<i class="fa-solid fa-check text-emerald-400"></i> <span>Copied!</span>';
        setTimeout(() => {
            btnElement.innerHTML = originalHtml;
        }, 2000);
    });
}

function escapeHtml(str) {
    if (!str) return '';
    return str
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#039;');
}

function scrollToBottom() {
    elements.messagesContainer.scrollTop = elements.messagesContainer.scrollHeight;
}

// Setup Event Listeners
function setupEventListeners() {
    // Sidebar toggle for mobile
    elements.sidebarToggleBtn.onclick = () => {
        elements.sidebar.classList.toggle('-translate-x-full');
    };

    // New Chat
    elements.newChatBtn.onclick = () => createNewSession();

    // Clear All
    elements.clearAllBtn.onclick = () => clearAllSessions();

    // Model selection sync between desktop & mobile
    elements.modelSelect.onchange = (e) => {
        elements.mobileModelSelect.value = e.target.value;
        state.settings.model = e.target.value;
        const s = state.sessions.find(x => x.id === state.currentSessionId);
        if (s) s.model = e.target.value;
        saveSessions();
    };
    elements.mobileModelSelect.onchange = (e) => {
        elements.modelSelect.value = e.target.value;
        state.settings.model = e.target.value;
        const s = state.sessions.find(x => x.id === state.currentSessionId);
        if (s) s.model = e.target.value;
        saveSessions();
    };

    // Puter Auth
    elements.authBtn.onclick = handleAuthClick;

    // Chat form submit
    elements.chatForm.onsubmit = handleChatSubmit;
    elements.stopBtn.onclick = handleStopGeneration;

    // Prompt input auto-expand & enter handling
    elements.promptInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            handleChatSubmit();
        }
    });

    elements.promptInput.addEventListener('input', () => {
        elements.promptInput.style.height = 'auto';
        elements.promptInput.style.height = Math.min(elements.promptInput.scrollHeight, 200) + 'px';
    });

    // File attachments
    elements.attachBtn.onclick = () => elements.fileInput.click();
    elements.fileInput.onchange = handleFileSelect;
    elements.removeAttachmentBtn.onclick = clearAttachment;

    // Starter Prompts
    document.querySelectorAll('.starter-prompt').forEach(btn => {
        btn.onclick = () => {
            const prompt = btn.querySelector('.text-\\[11px\\]').textContent.trim();
            elements.promptInput.value = prompt;
            handleChatSubmit();
        };
    });

    // Settings Modal
    elements.settingsBtn.onclick = openSettings;
    elements.closeSettingsBtn.onclick = closeSettings;
    elements.saveSettingsBtn.onclick = saveSettings;
    elements.settingsModal.onclick = (e) => {
        if (e.target === elements.settingsModal) closeSettings();
    };

    elements.tempSlider.oninput = (e) => {
        elements.tempValue.textContent = e.target.value;
    };

    // Scroll button visibility
    elements.messagesContainer.onscroll = () => {
        const threshold = 150;
        const fromBottom = elements.messagesContainer.scrollHeight - elements.messagesContainer.scrollTop - elements.messagesContainer.clientHeight;
        if (fromBottom > threshold) {
            elements.scrollToBottomBtn.classList.remove('hidden');
        } else {
            elements.scrollToBottomBtn.classList.add('hidden');
        }
    };

    elements.scrollToBottomBtn.onclick = scrollToBottom;
}

