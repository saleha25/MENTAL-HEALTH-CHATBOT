const chatBox    = document.getElementById("chatBox");
const userInput  = document.getElementById("userInput");
const sendBtn    = document.getElementById("sendBtn");
const typingInd  = document.getElementById("typingIndicator");

function getTime() {
    const now = new Date();
    return now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
}

function addMessage(text, sender, type = "normal") {
    const div = document.createElement("div");

    let typeClass = "";
    if (type === "emergency") typeClass = "emergency-message";
    else if (type === "crisis") typeClass = "crisis-message";

    div.className = `message ${sender}-message ${typeClass}`;
    div.innerHTML = `
        <div class="message-content">${formatText(text)}</div>
        <div class="message-time">${getTime()}</div>
    `;
    chatBox.appendChild(div);
    chatBox.scrollTop = chatBox.scrollHeight;
}

function formatText(text) {
    // Bold phone numbers
    text = text.replace(/(\d{4}-\d{7}|\d{10,11})/g, '<strong>$1</strong>');
    // Line breaks
    text = text.replace(/\n/g, '<br>');
    return text;
}

function showTyping() {
    typingInd.style.display = "flex";
    chatBox.scrollTop = chatBox.scrollHeight;
}

function hideTyping() {
    typingInd.style.display = "none";
}

async function sendMessage() {
    const message = userInput.value.trim();
    if (!message) return;

    // Add user message
    addMessage(message, "user");
    userInput.value = "";
    userInput.style.height = "auto";
    sendBtn.disabled = true;

    // Show typing
    showTyping();

    try {
        const response = await fetch("/chat", {
            method:  "POST",
            headers: { "Content-Type": "application/json" },
            body:    JSON.stringify({ message }),
        });

        const data = await response.json();
        hideTyping();

        // Small delay for natural feel
        setTimeout(() => {
            addMessage(data.response, "bot", data.type || "normal");
            sendBtn.disabled = false;
            userInput.focus();
        }, 300);

    } catch (error) {
        hideTyping();
        addMessage("I am having trouble connecting right now. Please try again.", "bot");
        sendBtn.disabled = false;
    }
}

function handleKey(event) {
    if (event.key === "Enter" && !event.shiftKey) {
        event.preventDefault();
        sendMessage();
    }
}

function autoResize(el) {
    el.style.height = "auto";
    el.style.height = Math.min(el.scrollHeight, 120) + "px";
}

async function resetChat() {
    await fetch("/reset", { method: "POST" });
    chatBox.innerHTML = `
        <div class="message bot-message">
            <div class="message-content">Hello. I am here to listen and support you. What is your name?</div>
            <div class="message-time">${getTime()}</div>
        </div>
    `;
    userInput.value = "";
    userInput.focus();
}

// Focus input on load
window.onload = () => userInput.focus();