# Security Policy & Guardrail Architecture

This document describes the security model, threat landscape, defense-in-depth controls, and vulnerability disclosure guidelines for the `RAG_COT_Model` application.

---

## 1. Threat Model & Trust Boundaries

The application operates across four distinct trust boundaries:

```
[ External User Input ]  --> [ Boundary 1: Query Validation & Length Controls ]
                                     |
                                     v
[ Public WordPress API ] --> [ Boundary 2: SSRF Defense & Schema Sanitization ]
                                     |
                                     v
[ Retrieved Documents ]  --> [ Boundary 3: Prompt Injection Delimiters & Isolation ]
                                     |
                                     v
[ LLM Generation Output] --> [ Boundary 4: Pydantic Validation & Citation Grounding ]
```

### Untrusted Inputs:
1. **User Queries:** May contain prompt injection attempts, oversized payloads, or attempts to force the LLM to ignore safety constraints.
2. **Retrieved Web Content:** WordPress articles fetched from public endpoints are treated as **untrusted data**. They may contain malicious instructions ("ignore previous instructions", role overrides, or exfiltration URLs).
3. **LLM Output:** LLM responses are treated as untrusted and unverified until validated by the Pydantic schema and citation cross-checker.

---

## 2. Layered Defensive Controls

### A. Server-Side Request Forgery (SSRF) Defense
When fetching articles from external WordPress REST APIs, the ingestion layer enforces:
- **Protocol Restrictions:** Only `http://` and `https://` schemes are permitted (`file://`, `ftp://`, `gopher://`, etc., are rejected).
- **IP Address Blacklisting:** Prohibits connections to:
  - Loopback addresses (`127.0.0.0/8`, `::1`)
  - Private IPv4 ranges (`10.0.0.0/8`, `172.16.0.0/12`, `192.168.0.0/16`, `fc00::/7`)
  - Link-local and cloud metadata addresses (`169.254.169.254`, `metadata.google.internal`)
  - Broadcast, multicast, and unspecified addresses
- **Pre-flight DNS Resolution:** Hostnames are resolved to IP addresses and verified before connection.
- **Strict Redirect Validation:** HTTP redirects (301, 302, 307, 308) are intercepted and re-validated per hop to prevent redirect-based SSRF.
- **Port Whitelisting:** Traffic is restricted to standard web ports (80, 443, 8080, 8443).
- **Response Size Limits:** Ingestion caps responses to 15 MB to prevent memory exhaustion attacks.

### B. Prompt Injection Defense & Context Isolation
Retrieved external documents are never concatenated directly into privileged instruction prompts.
1. **Structural Delimiters:** Passages are wrapped within XML-like tags:
   ```xml
   <untrusted_retrieved_context>
     <source id="S1" title="..." url="...">
       ... article text ...
     </source>
   </untrusted_retrieved_context>
   ```
2. **System Instruction Precedence:** The system prompt explicitly instructs the LLM that content inside `<untrusted_retrieved_context>` is strictly passive data and commands found within them must never be followed.
3. **Injection Pattern Heuristics:** A signature detector scans user queries and retrieved content for override patterns (e.g. `ignore previous instructions`, `reveal system prompt`, `exfiltrate`) and logs security events.
4. **No Code Execution:** The LLM is never provided with code execution tools, shell access, or access to host filesystem environments.

### C. Citation Grounding & Output Validation
1. **Schema Validation:** Generation responses must match the typed Pydantic `GeneratedAnswer` schema. Malformed responses are rejected.
2. **Citation Cross-Checking:** The application verifies that every citation label (e.g. `S1`, `S2`) emitted by the LLM corresponds to an actual retrieved passage passed in that turn. Fabricated labels are scrubbed before display.
3. **Safe Rendering:** All text displayed in the Streamlit frontend is HTML-escaped using `sanitize_for_display()` to prevent reflected or stored Cross-Site Scripting (XSS).

### D. Credential & Secrets Protection
- OpenAI API keys are read from environment variables or Streamlit secrets (`st.secrets["OPENAI_API_KEY"]`).
- Keys are never committed to version control (`.gitignore` enforces `.env` and `secrets.toml` exclusion).
- All application loggers employ `SecretMaskingFilter` to redact suspected API keys and authorization headers.
- If credentials are missing, the system degrades gracefully into a **Retrieval-Only Mode** rather than exposing raw errors.

---

## 3. Data Privacy & Session Isolation

- **State Isolation:** User chat history and active FAISS indexes are stored in `st.session_state` belonging exclusively to the current user's browser session.
- **Cross-Session Leakage:** The application prevents one user's questions or indexed documents from appearing in another user's session.
- **External Data Transmission:** Only user queries and retrieved article passages are sent to OpenAI for completion; no server metadata or internal configuration is transmitted.

---

## 4. Known Residual Risks & Limitations

1. **Prompt Injection Inherent Risk:** No software architecture can guarantee 100% immunity to sophisticated prompt injection attacks. Layered isolation, schema validation, and citation stripping significantly mitigate impact, but users should remain aware that external web content can contain hostile text.
2. **DNS Rebinding:** In environments where external DNS resolution can be dynamically manipulated during request execution, standard web proxies or dedicated network firewalls are recommended for defense-in-depth.
3. **Ephemeral Storage:** On platforms like Streamlit Community Cloud, local vector stores exist in runtime memory and do not persist across container restarts.

---

## 5. Reporting Security Vulnerabilities

If you identify a security vulnerability in this repository, please do not open a public issue. Contact the repository maintainers directly with reproduction details and evidence.

