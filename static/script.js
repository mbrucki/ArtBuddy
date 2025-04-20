// static/script.js

document.addEventListener("DOMContentLoaded", () => {
    const chatBox     = document.getElementById("chat-box");
    const userInput   = document.getElementById("user-input");
    const sendButton  = document.getElementById("send-button");
    const sessionElem = document.getElementById("session-id");
  
    // --- Add Message Function --- 
    function addMessage(text, sender) {
        console.log(`addMessage called for sender: ${sender}, text:`, text); // Log entry
        const messageDiv = document.createElement('div');
        messageDiv.classList.add('message');
        // Add sender-specific class (user-message, bot-message, error-message)
        const senderClass = sender === 'user' ? 'user-message' : 
                           (sender === 'bot' ? 'bot-message' : 'error-message');
        messageDiv.classList.add(senderClass);
        messageDiv.textContent = text;
        chatBox.appendChild(messageDiv);
        chatBox.scrollTop = chatBox.scrollHeight;
    }
    
    // --- Append Error Function (using addMessage) --- 
    function appendError(message) {
        addMessage(message, 'error'); // Use addMessage for errors
    }
  
    async function sendMessage() {
      const text = userInput.value.trim();
      if (!text) return;

      // 1) Echo user message using addMessage
      addMessage(text, 'user');
      userInput.value = "";

      // --- Typing indicator ---
      const typingDiv = document.createElement("div");
      typingDiv.className = "message bot-message typing-indicator";
      typingDiv.innerHTML = `<span class="dot"></span><span class="dot"></span><span class="dot"></span>`;
      chatBox.appendChild(typingDiv);
      chatBox.scrollTop = chatBox.scrollHeight;

      // 2) Build the correct payload
      const payload = {
          session_id: sessionElem.value,
          message:  text // Ensure this key matches what Flask expects
      };
      console.log("→ Sending payload:", payload);

      let resp, data;
      try {
          // 3) Fire the request - Ensure URL is /send_message
          resp = await fetch("/send_message", { // <-- CORRECT URL HERE
              method:  "POST",
              headers: { "Content-Type": "application/json" },
              body:    JSON.stringify(payload),
          });
      } catch (networkErr) {
          console.error("Network error:", networkErr);
          typingDiv.remove();
          return appendError("Network error — please try again.");
      }

      console.log("← HTTP status:", resp.status);

      try {
          // 4) Parse JSON
          data = await resp.json();
          console.log("← Response JSON:", data);
      } catch (parseErr) {
          console.error("Parse error:", parseErr);
          typingDiv.remove();
          return appendError("Invalid server response.");
      }

      // Remove typing indicator before showing real reply or error
      typingDiv.remove();

      // 5) Handle HTTP errors
      if (!resp.ok) {
          const msg = data.error || `Error ${resp.status}`;
          return appendError(msg);
      }

      // 6) Everything's good: show the bot reply using addMessage
      // Ensure we are using the correct key 'response' from the Flask app
      addMessage(data.response || "Error: Empty response from server.", 'bot');

  }
  
    sendButton.addEventListener("click", sendMessage);
    userInput.addEventListener("keydown", e => {
      if (e.key === "Enter") sendMessage();
    });
  });
  