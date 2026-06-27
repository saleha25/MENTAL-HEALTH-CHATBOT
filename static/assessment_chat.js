const assessmentChatBox = document.getElementById("assessmentChatBox");
const assessmentInput = document.getElementById("assessmentInput");
const assessmentSendBtn = document.getElementById("assessmentSendBtn");
const assessmentTyping = document.getElementById("assessmentTyping");
const assessmentOptions = document.getElementById("assessmentOptions");
const optionGrid = document.getElementById("optionGrid");

function getAssessmentTime() {
    const now = new Date();
    return now.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}

function formatAssessmentText(text) {
    text = text.replace(/(\d{4}-\d{7}|\d{10,11})/g, "<strong>$1</strong>");
    // Section headers like ── Personal Guidance ──
    text = text.replace(/──\s*(.+?)\s*──/g, "<strong>$1</strong>");
    // Markdown links [label](url)
    text = text.replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2" style="color:#0F6E56;font-weight:600;">$1</a>');
    text = text.replace(/\n/g, "<br>");
    return text;
}

function addAssessmentMessage(text, sender = "bot") {
    const div = document.createElement("div");
    div.className = `assessment-message ${sender}`;
    div.innerHTML = `
        <div class="assessment-message-content">${formatAssessmentText(text)}</div>
        <div class="assessment-message-time">${getAssessmentTime()}</div>
    `;
    assessmentChatBox.appendChild(div);
    assessmentChatBox.scrollTop = assessmentChatBox.scrollHeight;
}

function showAssessmentTyping() {
    assessmentTyping.style.display = "flex";
    assessmentChatBox.scrollTop = assessmentChatBox.scrollHeight;
}

function hideAssessmentTyping() {
    assessmentTyping.style.display = "none";
}

function renderOptions(options) {
    optionGrid.innerHTML = "";
    if (!options || options.length === 0) {
        assessmentOptions.style.display = "none";
        return;
    }

    options.forEach(opt => {
        const btn = document.createElement("button");
        btn.className = "option-btn";
        btn.textContent = opt.label;
        btn.onclick = () => {
            addAssessmentMessage(opt.label, "user");
            sendAssessmentPayload(opt.value);
        };
        optionGrid.appendChild(btn);
    });

    assessmentOptions.style.display = "block";
}

async function initAssessmentChat() {
    showAssessmentTyping();
    try {
        const res = await fetch("/assessment_chat_init");
        const data = await res.json();
        hideAssessmentTyping();

        if (data.response) addAssessmentMessage(data.response, "bot");
        renderOptions(data.options || []);
    } catch (err) {
        hideAssessmentTyping();
        addAssessmentMessage("Unable to start assessment right now. Please try again.", "bot");
    }
}

async function sendAssessmentPayload(answerText) {
    assessmentSendBtn.disabled = true;
    showAssessmentTyping();

    try {
        const res = await fetch("/assessment_chat", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ message: answerText })
        });

        const data = await res.json();
        hideAssessmentTyping();

        setTimeout(() => {
            if (data.response) addAssessmentMessage(data.response, "bot");
            renderOptions(data.options || []);
            assessmentSendBtn.disabled = false;
            assessmentInput.focus();
        }, 250);
    } catch (err) {
        hideAssessmentTyping();
        addAssessmentMessage("There was a problem processing your assessment. Please try again.", "bot");
        assessmentSendBtn.disabled = false;
    }
}

function sendAssessmentMessage() {
    const message = assessmentInput.value.trim();
    if (!message) return;

    addAssessmentMessage(message, "user");
    assessmentInput.value = "";
    assessmentInput.style.height = "auto";
    sendAssessmentPayload(message);
}

function handleAssessmentKey(event) {
    if (event.key === "Enter" && !event.shiftKey) {
        event.preventDefault();
        sendAssessmentMessage();
    }
}

function autoResizeAssessment(el) {
    el.style.height = "auto";
    el.style.height = Math.min(el.scrollHeight, 120) + "px";
}

async function resetAssessment() {
    await fetch("/assessment_chat_reset", { method: "POST" });
    assessmentChatBox.innerHTML = "";
    assessmentOptions.style.display = "none";
    assessmentInput.value = "";
    initAssessmentChat();
}

window.onload = () => {
    assessmentInput.focus();
    initAssessmentChat();
};