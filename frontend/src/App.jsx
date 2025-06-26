    import React, { useState, useEffect, useRef } from 'react';

    // Main App component
    function App() {
      const [messages, setMessages] = useState([]);
      const [inputValue, setInputValue] = useState('');
      const messagesEndRef = useRef(null); // Ref for scrolling to the bottom of messages

      // Scroll to the latest message whenever messages update
      useEffect(() => {
        messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
      }, [messages]);

      // Function to send message and get AI response from Flask API
      const handleSendMessage = async () => {
        if (inputValue.trim() === '') return;

        const userMessage = { sender: 'user', text: inputValue };
        setMessages((prevMessages) => [...prevMessages, userMessage]);
        setInputValue(''); // Clear input immediately

        // Show typing indicator
        const typingMessage = { sender: 'ai', text: 'Thinking...', isTyping: true };
        setMessages((prevMessages) => [...prevMessages, typingMessage]);

        try {
          // --- Actual Model API Call ---
          // This endpoint should point to your local Flask API server (or deployed server later)
          const API_ENDPOINT = "http://localhost:5000/generate"; 

          console.log(`Sending prompt to API: ${inputValue}`);
          const response = await fetch(API_ENDPOINT, {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({ prompt: inputValue })
          });

          // Check if the API call was successful
          if (!response.ok) {
              const errorData = await response.json();
              throw new Error(`API error: ${response.status} - ${errorData.error || 'Unknown error'}`);
          }

          const data = await response.json();
          // The API returns a JSON object like {"prompt": "...", "generated_script": "..."}
          // We reformat it to match the expected display format in the ChatMessage component.
          const aiResponseText = `### Instruction:\n${inputValue}\n\n### Response:\n\`\`\`bash\n${data.generated_script}\n\`\`\``;
          // --- End of Actual Model API Call ---

          setMessages((prevMessages) => {
            // Remove typing indicator and add final response
            const newMessages = prevMessages.filter(msg => !msg.isTyping);
            newMessages.push({ sender: 'ai', text: aiResponseText });
            return newMessages;
          });

        } catch (error) {
          console.error('Error sending message or getting AI response:', error);
          setMessages((prevMessages) => {
            // Remove typing indicator and add error message
            const newMessages = prevMessages.filter(msg => !msg.isTyping);
            newMessages.push({ sender: 'ai', text: `Error generating script: ${error.message}. Please check the Flask API terminal for details.` });
            return newMessages;
          });
        }
      };

      return (
        <div className="flex flex-col h-screen w-full bg-gray-100 dark:bg-gray-900 text-gray-900 dark:text-gray-100 font-inter rounded-lg p-4">
          {/* Header */}
          <div className="flex-shrink-0 p-4 bg-gradient-to-r from-blue-600 to-purple-600 text-white rounded-t-lg shadow-md mb-4">
            <h1 className="text-3xl font-bold text-center">Bash Script Chatbot</h1>
            <p className="text-center text-sm mt-1 opacity-90">
              Ask me to generate Bash scripts!
            </p>
          </div>

          {/* Chat Window */}
          <div className="flex-1 overflow-y-auto p-4 space-y-4 bg-white dark:bg-gray-800 rounded-lg shadow-inner custom-scrollbar">
            {messages.length === 0 && (
              <div className="text-center text-gray-500 dark:text-gray-400 py-10">
                Type a request to generate a Bash script (e.g., "create a directory", "list files").
              </div>
            )}
            {messages.map((msg, index) => (
              <ChatMessage key={index} message={msg} />
            ))}
            <div ref={messagesEndRef} /> {/* For auto-scrolling */}
          </div>

          {/* Input Area */}
          <div className="flex-shrink-0 p-4 bg-white dark:bg-gray-800 rounded-b-lg shadow-md mt-4">
            <div className="flex items-center space-x-3">
              <input
                type="text"
                className="flex-1 p-3 border border-gray-300 dark:border-gray-700 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 dark:bg-gray-700 dark:text-gray-100 transition duration-200"
                placeholder="Type your request here..."
                value={inputValue}
                onChange={(e) => setInputValue(e.target.value)}
                onKeyPress={(e) => {
                  if (e.key === 'Enter') {
                    handleSendMessage();
                  }
                }}
              />
              <button
                onClick={handleSendMessage}
                className="px-6 py-3 bg-blue-600 hover:bg-blue-700 text-white font-semibold rounded-lg shadow-lg transform transition duration-200 hover:scale-105 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2"
              >
                Send
              </button>
            </div>
          </div>
        </div>
      );
    }

    // ChatMessage component for displaying individual messages
    const ChatMessage = ({ message }) => {
      const isUser = message.sender === 'user';
      const messageClass = isUser
        ? 'bg-blue-500 dark:bg-blue-700 text-white self-end rounded-br-none'
        : 'bg-gray-200 dark:bg-gray-700 text-gray-900 dark:text-gray-100 self-start rounded-bl-none';

      // Function to extract and format Bash code
      const formatMessageContent = (text) => {
        // Regex to find the ### Response: block and extract code
        const responseRegex = /### Response:\s*```bash\n([\s\S]*?)\n```/;
        const match = text.match(responseRegex);

        if (match && match[1]) {
          const instructionPart = text.substring(0, match.index).replace('### Instruction:', '').trim();
          const codePart = match[1].trim();
          return (
            <div className="flex flex-col">
              {instructionPart && (
                <p className="text-sm italic text-gray-700 dark:text-gray-300 mb-2">
                  <span className="font-semibold not-italic">Prompt:</span> {instructionPart}
                </p>
              )}
              <pre className="bg-gray-800 dark:bg-gray-900 text-green-300 p-3 rounded-lg overflow-x-auto text-xs sm:text-sm">
                <code>{codePart}</code>
              </pre>
              <button
                onClick={() => {
                  // Using document.execCommand('copy') for better iframe compatibility
                  const textArea = document.createElement("textarea");
                  textArea.value = codePart;
                  document.body.appendChild(textArea);
                  textArea.select();
                  try {
                    document.execCommand('copy');
                    alert('Bash script copied to clipboard!'); // Using alert for simplicity, would use custom modal in production
                  } catch (err) {
                    console.error('Failed to copy text (execCommand): ', err);
                    alert('Failed to copy script.');
                  } finally {
                    document.body.removeChild(textArea);
                  }
                }}
                className="mt-2 px-3 py-1 bg-green-500 hover:bg-green-600 text-white text-xs font-semibold rounded-md self-end transition duration-200"
              >
                Copy Code
              </button>
            </div>
          );
        }
        // Fallback for simple text messages (like typing indicator)
        return <p className="text-sm sm:text-base">{text}</p>;
      };

      return (
        <div
          className={`flex ${isUser ? 'justify-end' : 'justify-start'}`}
        >
          <div
            className={`max-w-[75%] px-4 py-3 rounded-xl shadow-md flex flex-col ${messageClass} ${message.isTyping ? 'animate-pulse' : ''}`}
          >
            {message.isTyping ? (
              <p className="text-sm italic">Thinking...</p>
            ) : (
              formatMessageContent(message.text)
            )}
          </div>
        </div>
      );
    };

export default App;
