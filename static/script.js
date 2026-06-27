const chatBox = document.getElementById("messages");
const userInput = document.getElementById("userInput");

function getTime() {
    return new Date().toLocaleTimeString("en-PK", {
        hour: "2-digit",
        minute: "2-digit"
    });
}

// Optional: if backend/user name injected somewhere later
window.userName = window.userName || document.getElementById("greeting-name")?.textContent?.trim() || "User";

function escapeHtml(text) {
    const div = document.createElement("div");
    div.textContent = text;
    return div.innerHTML;
}

function formatText(text) {
    if (!text) return "";
    let safe = escapeHtml(text);
    safe = safe.replace(/(\d{4}-\d{7}|\d{10,11})/g, "<strong>$1</strong>");
    safe = safe.replace(/\n/g, "<br>");
    return safe;
}

function addMessage(text, role, type = "normal") {
    const msgs = document.getElementById("messages");
    if (!msgs) return;

    const div = document.createElement("div");
    div.className = "message " + role;

    const initials = role === "bot"
        ? "EA"
        : (window.userName ? window.userName[0].toUpperCase() : "U");

    let extraStyle = "";
    if (type === "emergency" || type === "crisis") {
        extraStyle = ' style="border:1px solid rgba(217,106,53,0.35); background:#fff8f0;"';
    }

    div.innerHTML = `
        <div class="msg-avatar">${initials}</div>
        <div class="msg-body">
            <div class="msg-bubble"${extraStyle}>${formatText(text)}</div>
            <div class="msg-time">${getTime()}</div>
        </div>
    `;

    msgs.appendChild(div);
    msgs.scrollTop = msgs.scrollHeight;
}

function addTyping() {
    const msgs = document.getElementById("messages");
    if (!msgs) return;

    removeTyping();

    const div = document.createElement("div");
    div.className = "message bot";
    div.id = "typing-msg";
    div.innerHTML = `
        <div class="msg-avatar">EA</div>
        <div class="msg-body">
            <div class="typing-bubble"><span></span><span></span><span></span></div>
        </div>
    `;
    msgs.appendChild(div);
    msgs.scrollTop = msgs.scrollHeight;
}

function removeTyping() {
    const el = document.getElementById("typing-msg");
    if (el) el.remove();
}

function autoResize(el) {
    el.style.height = "auto";
    el.style.height = Math.min(el.scrollHeight, 120) + "px";
}

function handleKey(event) {
    if (event.key === "Enter" && !event.shiftKey) {
        event.preventDefault();
        sendMessage();
    }
}

async function sendToBackend(message, callback) {
    try {
        const response = await fetch("/chat", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({ message })
        });

        let data = {};
        try {
            data = await response.json();
        } catch (e) {
            callback("I am having trouble understanding the server response right now. Please try again.", "normal");
            return;
        }

        if (!response.ok) {
            const msg = data.response || data.error || "Something went wrong while sending your message.";
            callback(msg, data.type || "normal");
            return;
        }

        callback(data.response || "I am here with you.", data.type || "normal");
    } catch (error) {
        console.error("Chat error:", error);
        callback("I am having trouble connecting right now. Please try again.", "normal");
    }
}

async function initChatSession() {
    try {
        const response = await fetch("/init", {
            method: "GET",
            headers: {
                "Accept": "application/json"
            }
        });

        if (!response.ok) return;

        const data = await response.json();

        // Agar backend user ka naam bhej raha ho
        if (data.name) {
            window.userName = data.name;
        }

        // Welcome state par pehla bot message auto show nahi karna
        // kyun ke tum chahti ho pehle dashboard se greeting aaye,
        // phir user chat start kare.
        // Lekin returning user ho aur backend initial guidance dena chahe,
        // to hum usay first message ke baad show kar sakte hain if needed.
        window.initialBotResponse = data.response || "";
    } catch (error) {
        console.error("Init error:", error);
    }
}

async function resetChat() {
    try {
        await fetch("/reset", { method: "POST" });
    } catch (error) {
        console.error("Reset error:", error);
    }

    if (typeof chatStarted !== "undefined") {
        chatStarted = false;
    }

    const statusPill = document.querySelector(".status-pill");
    if (statusPill) statusPill.classList.remove("visible");

    const messages = document.getElementById("messages");
    if (messages) messages.innerHTML = "";

    const chatState = document.getElementById("chat-state");
    if (chatState) chatState.style.display = "none";

    const bottomInput = document.getElementById("bottom-input");
    if (bottomInput) bottomInput.style.display = "none";

    const welcomeState = document.getElementById("welcome-state");
    if (welcomeState) {
        welcomeState.style.display = "flex";
        welcomeState.style.opacity = "1";
    }

    const centerInput = document.getElementById("center-input");
    if (centerInput) centerInput.value = "";

    if (userInput) {
        userInput.value = "";
        userInput.style.height = "auto";
    }

    await initChatSession();
}

window.addEventListener("load", async () => {
    await initChatSession();

    const centerInput = document.getElementById("center-input");
    if (centerInput) centerInput.focus();
});