# Puter.js AI Chat Application

A serverless, multi-model AI Chat application built with **[Puter.js](https://developer.puter.com)** that lets you chat with cutting-edge AI models (GPT-4o, Claude 3.5 Sonnet, Gemini 2.0 Flash, DeepSeek, and Mistral) with **zero API keys or configuration**.

---

## Features

- **No API Keys Needed:** Powered by Puter.js serverless AI gateway (`puter.ai.chat()`).
- **Multi-Model Support:** Switch instantly between models:
  - `gpt-4o-mini` (OpenAI - Fast & versatile)
  - `gpt-4o` (OpenAI - Flagship)
  - `claude-3-5-sonnet` (Anthropic - Advanced reasoning)
  - `claude-3-5-haiku` (Anthropic - Ultra fast)
  - `gemini-2.0-flash` (Google - High throughput)
  - `deepseek-chat` (DeepSeek V3)
  - `mistral-large` (Mistral)
- **Real-Time Token Streaming:** Responses stream smoothly word-by-word with live markdown formatting and code highlighting.
- **Multimodal Image Support:** Attach images to analyze, OCR, and discuss with vision-capable models.
- **Puter Cloud Sync & Auth:** Sign in with your Puter account (`puter.auth.signIn()`) to automatically sync conversation history across devices via Puter KV Store (`puter.kv.set()`), with automatic local fallback for guest mode.
- **Markdown & Code Highlighting:** Syntax highlighted code blocks with one-click copy buttons.
- **Customizable System Prompt & Temperature:** Tweak temperature, max tokens, and instructions in the Settings drawer.
- **Session Management:** Create, switch, and delete multiple conversations.

---

## Security & Architecture Note

Per official Puter.js specifications:
> *An application using Puter.js should be served by an HTTP server (not a `file://` URL) to run properly. Puter identifies your app by its web origin.*

---

## How to Run Locally

### Option 1: Using the Included Python Runner
```bash
python puter_chat/server.py
```
This starts an HTTP server at `http://localhost:3000` and automatically opens your default browser.

### Option 2: Using Any Standard HTTP Server
```bash
# Using Python
cd puter_chat
python -m http.server 3000

# Using Node.js
npx serve puter_chat
```
Navigate to `http://localhost:3000`.

---

## Attribution

This application is built with Puter.js.
Visit [developer.puter.com](https://developer.puter.com) for official documentation, SDKs, and platform capabilities.
