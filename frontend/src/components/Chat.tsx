import { useState, useRef, useEffect } from "react";

interface Message {
  from: string;
  message: string;
}

interface ChatProps {
  messages: Message[];
  onSend: (message: string) => void;
  disabled?: boolean;
}

export function Chat({ messages, onSend, disabled }: ChatProps) {
  const [input, setInput] = useState("");
  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || disabled) return;
    onSend(input.trim());
    setInput("");
  };

  return (
    <div className="chat">
      <div className="chat-messages">
        {messages.map((msg, i) => (
          <div key={i} className={`chat-msg ${msg.from === "dj" ? "dj" : "user"}`}>
            <span className="chat-sender">{msg.from === "dj" ? "DJ" : msg.from}</span>
            <p>{msg.message}</p>
          </div>
        ))}
        <div ref={messagesEndRef} />
      </div>
      <form className="chat-input" onSubmit={handleSubmit}>
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder={disabled ? "Connecting..." : "Talk to the DJ..."}
          disabled={disabled}
        />
        <button type="submit" disabled={disabled || !input.trim()}>
          Send
        </button>
      </form>
    </div>
  );
}
