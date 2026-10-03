import express from "express";
import cors from "cors";
import dotenv from "dotenv";
import { GoogleGenAI } from "@google/genai";
import { Memori } from "@memorilabs/memori";

dotenv.config();

const app = express();
app.use(cors());
app.use(express.json());

const client = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY as string });

// Wrap the client with Memori so calls are captured/recalled automatically
const mem = new Memori().llm.register(client);

app.post("/chat", async (req, res) => {
  const { userId, message } = req.body;
console.log("Received userId:", userId, "| message:", message);
  if (!userId || !message) {
    return res.status(400).json({ error: "userId and message are required" });
  }

  // Attribute this interaction to this specific user + this app/process
  mem.attribution(userId, "support-chat-demo");

  try {
    const response = await client.models.generateContent({
      model: "gemini-3.6-flash",
      contents: message,
    });

    res.json({ reply: response.text });
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: "Something went wrong" });
  }
});

app.post("/new-session", (req, res) => {
  mem.resetSession();
  res.json({ status: "session reset" });
});

const port = process.env.PORT || 4000;
app.listen(port, () => console.log(`Server running on port ${port}`));
