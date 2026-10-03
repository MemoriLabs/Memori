import { useState } from "react";

type Message = { role: "user" | "assistant"; content: string };

export default function App() {
  const [userId, setUserId] = useState("user_123");
  const [input, setInput] = useState("");
  const [messages, setMessages] = useState<Message[]>([]);
  const [loading, setLoading] = useState(false);

  const sendMessage = async () => {
    if (!input.trim()) return;
    const userMsg: Message = { role: "user", content: input };
    setMessages((m) => [...m, userMsg]);
    setInput("");
    setLoading(true);

    try {
      const res = await fetch("http://localhost:4000/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ userId, message: input }),
      });
      const data = await res.json();
      setMessages((m) => [...m, { role: "assistant", content: data.reply }]);
    } catch (err) {
      setMessages((m) => [
        ...m,
        { role: "assistant", content: "Error contacting server." },
      ]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ maxWidth: 600, margin: "40px auto", fontFamily: "sans-serif" }}>
      <h2>Memori Support Chat Demo</h2>
      <label>
        User ID:{" "}
        <input value={userId} onChange={(e) => setUserId(e.target.value)} />
      </label>
      <p style={{ fontSize: 12, color: "#666" }}>
        Switch this to a different ID to see Memori recall different memories per user.
      </p>

      <div style={{ border: "1px solid #ddd", padding: 12, minHeight: 300 }}>
        {messages.map((m, i) => (
          <div
            key={i}
            style={{ margin: "8px 0", textAlign: m.role === "user" ? "right" : "left" }}
          >
            <b>{m.role === "user" ? "You" : "Gemini"}:</b> {m.content}
          </div>
        ))}
        {loading && <div>Thinking…</div>}
      </div>

      <div style={{ marginTop: 12, display: "flex", gap: 8 }}>
        <input
          style={{ flex: 1 }}
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && sendMessage()}
          placeholder="Ask something, or tell it a preference..."
        />
        <button onClick={sendMessage}>Send</button>
      </div>
    </div>
  );
}
